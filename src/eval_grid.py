from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
RES = ROOT / "results"
FC, PAR = RES / "forecasts", RES / "params"
names = [cfg["tickers"]["index"]] + list(cfg["tickers"]["stocks"].values())
clean = lambda t: t.replace("^", "")
VARS, DISTS = ["GARCH", "GJR", "EGARCH", "APARCH"], ["normal", "t", "skewt"]


def qlike(s, h):
    return s / h + np.log(h)


def dm_test(l1, l2):
    """Positive stat => model 1 has HIGHER loss, i.e. model 2 (challenger) is better."""
    d = np.asarray(l1) - np.asarray(l2)
    T = len(d)
    lags = int(np.floor(T ** (1 / 3)))
    dc = d - d.mean()
    lrv = np.sum(dc * dc) / T
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * np.sum(dc[k:] * dc[:-k]) / T
    stat = d.mean() / np.sqrt(lrv / T) * np.sqrt((T - 1) / T)
    return stat, 2 * (1 - stats.t.cdf(abs(stat), df=T - 1))


# observable high-volatility flag: previous-day India VIX vs fit-period 75th percentile
vix = pd.read_csv(ROOT / "data/processed/INDIAVIX.csv", index_col="Date", parse_dates=True)["vix"]
vix_lag = vix.ffill().shift(1)
first = pd.read_csv(FC / f"{clean(names[0])}_GARCH-normal.csv", index_col=0, parse_dates=True)
test_start = first.index[0]
thr = vix_lag[vix_lag.index < test_start].quantile(0.75)
print(f"Test start {test_start.date()} | VIX threshold (75th pct of pre-test data): {thr:.2f}")

rows = []
for t in names:
    F = {(v, d): pd.read_csv(FC / f"{clean(t)}_{v}-{d}.csv", index_col=0, parse_dates=True)
         for v in VARS for d in DISTS}
    base = F[("GARCH", "normal")]
    s = base["ret"] ** 2
    v_lag = vix_lag.reindex(base.index)
    ok = v_lag.notna()
    masks = {"all": pd.Series(True, index=base.index),
             "high_vix": ok & (v_lag > thr), "calm": ok & (v_lag <= thr)}

    pairs = []
    for v in ["GJR", "EGARCH", "APARCH"]:
        for d in DISTS:
            pairs.append(("RQ2", (("GARCH", d), (v, d))))
    for v in VARS:
        pairs += [("RQ3", ((v, "normal"), (v, "t"))), ("RQ3", ((v, "normal"), (v, "skewt"))),
                  ("RQ3", ((v, "t"), (v, "skewt")))]

    for fam, (ref, chal) in pairs:
        l_ref, l_chal = qlike(s, F[ref]["h"]), qlike(s, F[chal]["h"])
        for name, m in masks.items():
            m = m.values
            stat, p = dm_test(l_ref[m], l_chal[m])
            rows.append({"family": fam, "series": clean(t), "reference": "-".join(ref),
                         "challenger": "-".join(chal), "sample": name, "n": int(m.sum()),
                         "dQLIKE": l_ref[m].mean() - l_chal[m].mean(), "DM": stat, "p": p})

R = pd.DataFrame(rows)
R["p_holm"] = np.nan
for (fam, smp), g in R.groupby(["family", "sample"]):
    R.loc[g.index, "p_holm"] = multipletests(g["p"], method="holm")[1]
R.to_csv(RES / "grid_dm_tests.csv", index=False)

star = lambda stat, p: f"{stat:+.2f}" + ("**" if p < 0.01 else "*" if p < 0.05 else "")
pd.set_option("display.width", 220)


def show(fam, sample):
    g = R[(R.family == fam) & (R["sample"] == sample)].copy()
    g["cell"] = [star(a, b) for a, b in zip(g.DM, g.p)]
    g["row"] = g.reference + "  ->  " + g.challenger
    print(f"\n{fam}, sample = {sample} (+ = challenger has LOWER QLIKE; * p<.05, ** p<.01; unadjusted)")
    print(g.pivot(index="row", columns="series", values="cell").to_string())


for smp in ["all", "high_vix", "calm"]:
    show("RQ2", smp)
show("RQ3", "all")

print("\nSummary: tests where challenger is better and p < 0.05, raw vs Holm-adjusted")
sm = R.assign(better=R.DM > 0).groupby(["family", "sample"]).apply(
    lambda g: pd.Series({"tests": len(g), "raw_sig_better": int(((g.p < .05) & g.better).sum()),
                         "holm_sig_better": int(((g.p_holm < .05) & g.better).sum())}))
print(sm)
print("\nHigh-VIX / calm day counts per series:")
print(R[(R.family == "RQ2") & (R.reference == "GARCH-normal") & (R.challenger == "GJR-normal")].pivot(index="series", columns="sample", values="n"))
# parameter summaries across refits
prow = []
for t in names:
    for v in VARS:
        for d in DISTS:
            p = pd.read_csv(PAR / f"{clean(t)}_{v}-{d}.csv")
            r = {"series": clean(t), "spec": f"{v}-{d}"}
            g = [c for c in p.columns if c.startswith("gamma")]
            if g:
                r["gamma_mean"] = p[g[0]].mean()
                r["gamma_leverage_share"] = (p[g[0]] < 0).mean() if v == "EGARCH" else (p[g[0]] > 0).mean()
            if "delta" in p.columns:
                r["aparch_delta_mean"] = p["delta"].mean()
            if "nu" in p.columns:
                r["nu_mean"] = p["nu"].mean()
            if "lambda" in p.columns:
                r["skew_lambda_mean"] = p["lambda"].mean()
            if v == "GARCH" and "alpha[1]" in p.columns:
                r["persistence_alpha_beta"] = (p["alpha[1]"] + p["beta[1]"]).mean()
            prow.append(r)
P = pd.DataFrame(prow)
P.to_csv(RES / "params_summary.csv", index=False)
print("\nLeverage parameter (share of refits with the leverage sign: GJR/APARCH gamma>0, EGARCH gamma<0):")
lev = P.dropna(subset=["gamma_mean"]).assign(v=lambda x: x.spec.str.split("-").str[0])
print(lev.groupby(["v", "series"]).gamma_leverage_share.mean().unstack().round(2))
print("\nGARCH persistence (alpha+beta), mean over refits:")
print(P.dropna(subset=["persistence_alpha_beta"]).groupby("series").persistence_alpha_beta.mean().round(3))
print("\nStudent-t degrees of freedom nu, mean over refits (smaller = fatter tails):")
print(P.dropna(subset=["nu_mean"]).groupby("series").nu_mean.mean().round(1))