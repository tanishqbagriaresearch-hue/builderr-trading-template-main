"""
Evaluate ALL agent files using the sample_regimes.json.gz data.
Outputs a ranked leaderboard of every strategy across all 3 sample regimes.

Bar format: [ts_str, open, high, low, close, volume]

Run:  python evaluate_all_agents.py
"""
from __future__ import annotations

import importlib.util
import sys
import json
import gzip
import traceback
from pathlib import Path
from statistics import mean, pstdev
from math import sqrt
from typing import Any

# ── Which files to test ───────────────────────────────────────────────────────
SKIP = {
    "preview.py", "evaluate_all_agents.py", "local_test.py",
    "full_test.py", "fairness_tests.py", "selfcheck.py",
    "strategy_selftest.py", "build_universe.py", "live_runner.py",
}
AGENT_FILES = sorted(
    f for f in Path(".").glob("*.py")
    if f.name not in SKIP and not f.name.startswith("_")
)

REGIMES_FILE = Path("sample_regimes.json.gz")

# ── Load regime data ─────────────────────────────────────────────────────────
def load_regimes() -> dict:
    with gzip.open(REGIMES_FILE, "rt", encoding="utf-8") as f:
        return json.load(f)

# Bar list → dict that agents expect
def bar_to_dict(bar: list) -> dict:
    return {"ts": bar[0], "open": bar[1], "high": bar[2],
            "low": bar[3], "close": bar[4], "volume": bar[5]}

# ── Load an agent module ──────────────────────────────────────────────────────
def load_agent(path: Path):
    spec = importlib.util.spec_from_file_location(f"ag_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── Reset any module-level globals an agent may keep ─────────────────────────
def reset_agent_globals(mod):
    """Best-effort reset of common global state patterns agents use."""
    for attr in list(vars(mod).keys()):
        val = getattr(mod, attr, None)
        if attr.startswith("_last") or attr.startswith("_prev") or attr.startswith("_state"):
            if isinstance(val, dict):
                setattr(mod, attr, {})
            elif val is None or isinstance(val, str):
                setattr(mod, attr, None)
            elif isinstance(val, (int, float)):
                setattr(mod, attr, 0)
            elif isinstance(val, list):
                setattr(mod, attr, [])

# ── Simulate one regime ───────────────────────────────────────────────────────
def simulate_regime(decide_fn, bars_by_ticker_raw: dict, starting_cash: float = 100_000) -> list[float]:
    """
    bars_by_ticker_raw: {ticker: [[ts, o, h, l, c, v], ...]}
    Returns daily equity curve.
    """
    # Convert raw bars to dict-bars per ticker
    bars_dict: dict[str, list[dict]] = {
        t: [bar_to_dict(b) for b in bars]
        for t, bars in bars_by_ticker_raw.items()
    }

    # All unique trading days
    all_days = sorted({b["ts"][:10] for bars in bars_dict.values() for b in bars})

    cash = float(starting_cash)
    positions: dict[str, dict] = {}  # ticker -> {qty, avg_cost}
    equity_curve = [cash]

    for day in all_days:
        # Rolling window: all bars up to and including today
        market_state: dict[str, list[dict]] = {}
        last_prices: dict[str, float] = {}
        for ticker, bars in bars_dict.items():
            window = [b for b in bars if b["ts"][:10] <= day]
            if window:
                market_state[ticker] = window
                last_prices[ticker] = float(window[-1]["close"])

        portfolio_state = {
            "cash": cash,
            "positions": [
                {"ticker": t, "quantity": p["qty"], "avg_cost": p["avg_cost"]}
                for t, p in positions.items() if p["qty"] > 0
            ],
            "last_prices": last_prices,
        }

        try:
            orders = decide_fn(market_state, portfolio_state, cash) or []
        except Exception:
            orders = []

        # Execute: sells first, then buys
        sells = [o for o in orders if str(o.get("side","")).lower() == "sell"]
        buys  = [o for o in orders if str(o.get("side","")).lower() == "buy"]

        for order in sells:
            ticker = str(order.get("ticker","")).upper()
            qty    = int(order.get("quantity", 0))
            price  = last_prices.get(ticker, 0.0)
            if qty <= 0 or price <= 0:
                continue
            held = int(positions.get(ticker, {}).get("qty", 0))
            qty  = min(qty, held)
            if qty <= 0:
                continue
            cash += qty * price * 0.9995   # tiny slippage
            positions[ticker]["qty"] -= qty

        for order in buys:
            ticker = str(order.get("ticker","")).upper()
            qty    = int(order.get("quantity", 0))
            price  = last_prices.get(ticker, 0.0)
            if qty <= 0 or price <= 0:
                continue
            cost = qty * price * 1.0005
            if cost > cash * 0.999:
                qty  = int(cash * 0.999 / (price * 1.0005))
                cost = qty * price * 1.0005
            if qty <= 0:
                continue
            cash -= cost
            pos = positions.setdefault(ticker, {"qty": 0, "avg_cost": price})
            total = pos["qty"] + qty
            pos["avg_cost"] = (pos["qty"] * pos["avg_cost"] + qty * price) / total
            pos["qty"] = total

        # Mark-to-market
        eq = cash
        for ticker, pos in positions.items():
            price = last_prices.get(ticker, pos["avg_cost"])
            eq += pos["qty"] * price
        equity_curve.append(eq)

    return equity_curve

# ── Performance metrics ───────────────────────────────────────────────────────
def calc_metrics(curve: list[float]) -> dict:
    if len(curve) < 3:
        return {"ret": 0.0, "maxdd": 0.0, "sharpe": 0.0, "calmar": 0.0}

    ret = (curve[-1] / curve[0]) - 1.0

    peak, maxdd = curve[0], 0.0
    for v in curve:
        peak = max(peak, v)
        dd   = (peak - v) / peak if peak > 0 else 0.0
        maxdd = max(maxdd, dd)

    daily = [(curve[i] / curve[i-1]) - 1.0 for i in range(1, len(curve))]
    avg_r  = mean(daily)
    std_r  = pstdev(daily) if len(daily) > 1 else 0.0
    sharpe = (avg_r / std_r * sqrt(252)) if std_r > 1e-9 else 0.0
    calmar = ret / maxdd if maxdd > 0.001 else (50.0 if ret > 0 else 0.0)

    return {"ret": ret, "maxdd": maxdd, "sharpe": sharpe, "calmar": calmar}

# ── Safety checks (mirrors preview.py admission gates) ───────────────────────
def safety_check(regime_results: dict) -> tuple[bool, str]:
    for regime_name, m in regime_results.items():
        if m["maxdd"] >= 0.50:
            return False, f"Blow-up in {regime_name}: MaxDD={m['maxdd']:.1%}"
    return True, "OK"

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print(" BUILDERR TRADING AGENT — FULL EVALUATION SUITE")
    print("=" * 80)
    print(f"\nLoading regime data from {REGIMES_FILE} ...")
    regimes_data = load_regimes()
    regime_names = list(regimes_data.keys())
    print(f"Regimes: {regime_names}")
    print(f"Testing {len(AGENT_FILES)} agent files...\n")

    results = {}

    for agent_path in AGENT_FILES:
        name = agent_path.name
        print(f"  [{name}]", end=" ", flush=True)

        try:
            mod = load_agent(agent_path)
        except Exception as e:
            print(f"LOAD ERROR: {e}")
            continue

        if not hasattr(mod, "decide"):
            print("SKIP (no decide())")
            continue

        regime_results = {}
        failed = False

        for regime_name in regime_names:
            regime_meta = regimes_data[regime_name]
            bars_raw    = regime_meta["bars"]   # {ticker: [[ts,o,h,l,c,v],...]}

            try:
                reset_agent_globals(mod)
                curve   = simulate_regime(mod.decide, bars_raw)
                metrics = calc_metrics(curve)
                regime_results[regime_name] = metrics
            except Exception as e:
                print(f"\n    SIM ERROR ({regime_name}): {e}")
                traceback.print_exc()
                failed = True
                break

        if failed:
            continue

        admitted, reason = safety_check(regime_results)
        calmars   = [r["calmar"] for r in regime_results.values()]
        dds       = [r["maxdd"]  for r in regime_results.values()]
        rets      = [r["ret"]    for r in regime_results.values()]
        sharpes   = [r["sharpe"] for r in regime_results.values()]

        avg_calmar = mean(calmars)
        worst_dd   = max(dds)
        avg_ret    = mean(rets)
        avg_sharpe = mean(sharpes)

        results[name] = {
            "regimes":     regime_results,
            "avg_calmar":  avg_calmar,
            "avg_ret":     avg_ret,
            "avg_sharpe":  avg_sharpe,
            "worst_dd":    worst_dd,
            "admitted":    admitted,
            "fail_reason": reason,
        }
        tag = "PASS" if admitted else "FAIL"
        print(f"{tag}  calmar={avg_calmar:+.2f}  ret={avg_ret:+.1%}  maxdd={worst_dd:.1%}  sharpe={avg_sharpe:.2f}")

    # ── Leaderboard ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" LEADERBOARD — Ranked by Average Calmar Ratio")
    print("=" * 80)
    ranked = sorted(results.items(), key=lambda x: x[1]["avg_calmar"], reverse=True)

    print(f"\n{'Rank':<5} {'Agent':<35} {'AvgCalmar':>10} {'AvgRet':>8} {'AvgSharpe':>10} {'WorstDD':>8} {'Status':>7}")
    print("-" * 80)
    for i, (name, data) in enumerate(ranked, 1):
        adm = "✓ PASS" if data["admitted"] else "✗ FAIL"
        print(f"{i:<5} {name:<35} {data['avg_calmar']:>10.2f} {data['avg_ret']:>7.1%} "
              f"{data['avg_sharpe']:>10.2f} {data['worst_dd']:>7.1%} {adm:>7}")

    # ── Per-regime breakdown ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" PER-REGIME CALMAR — Top 10 Agents")
    print("=" * 80)
    top10 = [name for name, _ in ranked[:10]]
    w = 18
    hdr = f"{'Agent':<35}" + "".join(f"{r[:w]:>{w}}" for r in regime_names)
    print(hdr)
    print("-" * 80)
    for name in top10:
        row = f"{name:<35}"
        for r in regime_names:
            c = results[name]["regimes"].get(r, {}).get("calmar", 0)
            row += f"{c:>{w}.2f}"
        print(row)

    # ── What makes the best strategy ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" INSIGHT: Per-Regime Best Agent")
    print("=" * 80)
    for r in regime_names:
        best = max(results.items(), key=lambda x: x[1]["regimes"].get(r, {}).get("calmar", -999))
        m = best[1]["regimes"][r]
        print(f"  {r:<25} → {best[0]}  (Calmar={m['calmar']:.2f}, Ret={m['ret']:+.1%}, DD={m['maxdd']:.1%})")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = Path("evaluation_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n Full results saved → {out}")
    print(" Done.\n")

    return ranked

if __name__ == "__main__":
    main()
