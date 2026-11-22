from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "regime_fc"
SERIES = ["NSEI", "ICICIBANK.NS", "TCS.NS", "RELIANCE.NS", "MARUTI.NS", "HINDUNILVR.NS"]
SPECS = ["GARCH-normal", "GJR-normal"]
qlike = lambda s, h: s / h + np.log(h)


def dm_test(l1, l2):
    d = np.asarray(l1) - np.asarray(l2)
    T = len(d)
    lags = int(np.floor(T ** (1 / 3)))
    dc = d - d.mean()
    lrv = np.sum(dc * dc) / T
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * np.sum(dc[k:] * dc[:-k]) / T
    stat = d.mean() / np.sqrt(lrv / T) * np.sqrt((T - 1) / T)
    return stat, 2 * (1 - stats.t.cdf(abs(stat), df=T - 1))


rows, stab, split = [], [], []
for spec in SPECS:
    for v in ["A", "B"]:
        for t in SERIES:
            if t == "NSEI" and v == "B":
                continue                       # identical to A for the index itself
            f = OUT / f"{t.replace('^', '')}_{v}_{spec}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f, index_col=0, parse_dates=True)
            lg = pd.read_csv(OUT / f"{t.replace('^', '')}_{v}_{spec}_log.csv")
            s = d["ret"] ** 2
            L = {c: qlike(s, d[c]) for c in ["h_pool", "h_hard", "h_soft", "h_soft_smooth"]}
            for name, a, b in [("hard vs pooled", "h_pool", "h_hard"), ("soft vs pooled", "h_pool", "h_soft"),
                               ("soft-SMOOTHED vs pooled", "h_pool", "h_soft_smooth"),
                               ("look-ahead bias: filtered vs smoothed", "h_soft", "h_soft_smooth")]:
                stat, p = dm_test(L[a], L[b])
                rows.append({"spec": spec, "variant": v, "series": t, "test": name,
                             "dQLIKE": L[a].mean() - L[b].mean(), "DM": stat, "p": p})
            stab.append({"spec": spec, "variant": v, "series": t, "refits": len(lg),
                         "high_days_first": lg.n_high.iloc[0], "high_days_min": lg.n_high.min(), "high_days_max": lg.n_high.max(),
                         "sd_high_first": lg.hmm_sd_high.iloc[0], "sd_high_last": lg.hmm_sd_high.iloc[-1],
                         "sd_low_first": lg.hmm_sd_low.iloc[0], "sd_low_last": lg.hmm_sd_low.iloc[-1],
                         "fallback_high": lg.fallback_high.mean(), "fallback_low": lg.fallback_low.mean()})
            mk = d["p_high"] > 0.5
            for nm, m in [("high-regime days", mk), ("calm days", ~mk)]:
                if m.sum() >= 30:
                    stat, p = dm_test(L["h_pool"][m], L["h_soft"][m])
                    split.append({"spec": spec, "variant": v, "series": t, "subset": nm, "n": int(m.sum()),
                                  "QLIKE_pool": L["h_pool"][m].mean(), "QLIKE_soft": L["h_soft"][m].mean(), "DM": stat, "p": p})

R = pd.DataFrame(rows)
R["p_holm"] = np.nan
for _, g in R.groupby(["spec", "variant", "test"]):
    R.loc[g.index, "p_holm"] = multipletests(g.p, method="holm")[1]
R.to_csv(RES / "regime_tests.csv", index=False)
pd.DataFrame(stab).to_csv(RES / "regime_stability.csv", index=False)
pd.DataFrame(split).to_csv(RES / "regime_split.csv", index=False)

star = lambda a, p: f"{a:+.2f}" + ("**" if p < .01 else "*" if p < .05 else "")
pd.set_option("display.width", 250)
for (spec, v), g in R.groupby(["spec", "variant"]):
    g = g.assign(cell=[star(a, b) for a, b in zip(g.DM, g.p)])
    pv = g.pivot(index="test", columns="series", values="cell")
    print(f"\n=== {spec} | variant {v}: DM (+ = second model has LOWER QLIKE; * p<.05, ** p<.01, unadjusted) ===")
    print(pv[[c for c in SERIES if c in pv.columns]].to_string())

R["sig_better"] = (R.p < .05) & (R.DM > 0)
R["holm_better"] = (R.p_holm < .05) & (R.DM > 0)
R["sig_worse"] = (R.p < .05) & (R.DM < 0)
R["holm_worse"] = (R.p_holm < .05) & (R.DM < 0)
print("\nCounts of significant tests (raw vs Holm across series), both specs pooled:")
print(R.groupby(["variant", "test"])[["sig_better", "holm_better", "sig_worse", "holm_worse"]].sum())

S = pd.DataFrame(stab)
print("\nRegime stability across refits (variant B, first spec): high-regime days and HMM state sd (%)")
print(S[(S.variant == "B") & (S.spec == SPECS[0])].drop(columns=["spec", "variant"]).round(2).to_string(index=False))
print("\nQLIKE by regime subset (spec 1):")
print(pd.DataFrame(split).query("spec == @SPECS[0]").drop(columns="spec").round(3).to_string(index=False))