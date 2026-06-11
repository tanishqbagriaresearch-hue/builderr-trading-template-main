import json

with open('evaluation_results.json') as f:
    d = json.load(f)

ranked = sorted(d.items(), key=lambda x: x[1]['avg_calmar'], reverse=True)
regimes = ['calm_uptrend', 'moderate_selloff', 'vol_spike_snapback']

print('LEADERBOARD - Ranked by Average Calmar Ratio')
print(f"{'Rank':<5} {'Agent':<35} {'AvgCalmar':>10} {'AvgRet':>8} {'AvgSharpe':>10} {'WorstDD':>8} {'Status':>6}")
print('-' * 80)
for i, (name, data) in enumerate(ranked, 1):
    adm = 'PASS' if data['admitted'] else 'FAIL'
    print(f"{i:<5} {name:<35} {data['avg_calmar']:>10.2f} {data['avg_ret']:>7.1%} {data['avg_sharpe']:>10.2f} {data['worst_dd']:>7.1%} {adm:>6}")

print()
print('PER-REGIME CALMAR BREAKDOWN')
header = f"{'Agent':<35}" + ''.join(f"{r[:18]:>20}" for r in regimes)
print(header)
print('-' * 95)
for name, _ in ranked[:12]:
    row = f"{name:<35}"
    for r in regimes:
        c = d[name]['regimes'].get(r, {}).get('calmar', 0)
        row += f"{c:>20.2f}"
    print(row)

print()
print('BEST AGENT PER REGIME:')
for r in regimes:
    best = max(d.items(), key=lambda x: x[1]['regimes'].get(r, {}).get('calmar', -999))
    m = best[1]['regimes'][r]
    print(f"  {r:<25} -> {best[0]}   Calmar={m['calmar']:.2f}  Ret={m['ret']:+.1%}  DD={m['maxdd']:.1%}")
