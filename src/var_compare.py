from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy import stats
from arch.univariate import SkewStudent
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
RES = ROOT / "results"
FC, PAR, RF = RES / "forecasts", RES / "params", RES / "regime_fc"
names = [cfg["tickers"]["index"]] + list(cfg["tickers"]["stocks"].values())
clean = lambda t: t.replace("^", "")
VARS, DISTS, PS = ["GARCH", "GJR", "EGARCH", "APARCH"], ["normal", "t", "skewt"], [0.01, 0.05]
sk = SkewStudent()


def std_q(p, dist, nu=None, lam=None):
    if dist == "normal":
        return stats.norm.ppf(p)
    if dist == "t":
        return stats.t.ppf(p, nu) * np.sqrt((nu - 2) / nu)
    return float(np.asarray(sk.ppf(np.array([p]), [nu, lam]))[0])


def qmap(par, p, dist):
    col = "nu" if dist == "t" else "eta"
    return {int(r["refit_origin"]): std_q(p, dist, r[col], r["lambda"] if dist == "skewt" else None)
            for _, r in par.iterrows()}


def quantile(origin, par, p, dist):
    if dist == "normal":
        return np.full(len(origin), stats.norm.ppf(p))
    return origin.map(qmap(par, p, dist)).values


def pinball(r, q, p):
    return (p - (r < q).astype(float)) * (r - q)


def dm_test(l1, l2):
    """Positive => model 1 has higher loss (model 2 better)."""
    d = np.asarray(l1) - np.asarray(l2)
    T = len(d)
    lags = int(np.floor(T ** (1 / 3)))
    dc = d - d.mean()
    lrv = np.sum(dc * dc) / T
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * np.sum(dc[k:] * dc[:-k]) / T
    stat = d.mean() / np.sqrt(lrv / T) * np.sqrt((T - 1) / T)
    return stat, 2 * (1 - stats.t.cdf(abs(stat), df=T - 1))


rows = []


def add(fam, t, p, ref, chal, l_ref, l_chal):
    stat, pv = dm_test(l_ref, l_chal)
    rows.append({"family": fam, "series": clean(t), "p_level": p, "reference": ref, "challenger": chal,
                 "dPinball": l_ref.mean() - l_chal.mean(), "DM": stat, "p": pv})


for t in names:
    L = {}
    for v in VARS:
        for d in DISTS:
            f = pd.read_csv(FC / f"{clean(t)}_{v}-{d}.csv", index_col=0, parse_dates=True)
            par = pd.read_csv(PAR / f"{clean(t)}_{v}-{d}.csv")
            for p in PS:
                q = f["mu"].values + np.sqrt(f["h"].values) * quantile(f["refit_origin"], par, p, d)
                L[(v, d, p)] = pinball(f["ret"].values, q, p)
    for p in PS:
        for v in ["GJR", "EGARCH", "APARCH"]:
            for d in DISTS:
                add("RQ2", t, p, f"GARCH-{d}", f"{v}-{d}", L[("GARCH", d, p)], L[(v, d, p)])
        for v in VARS:
            for a, b in [("normal", "t"), ("normal", "skewt"), ("t", "skewt")]:
                add("RQ3", t, p, f"{v}-{a}", f"{v}-{b}", L[(v, a, p)], L[(v, b, p)])

    # RQ4: regime-aware vs pooled VaR, Normal quantile and Student-t quantile (nu from pooled GARCH-t refits)
    part = pd.read_csv(PAR / f"{clean(t)}_GARCH-t.csv")
    for variant in ["A", "B"]:
        if t == cfg["tickers"]["index"] and variant == "B":
            continue
        g = pd.read_csv(RF / f"{clean(t)}_{variant}_GARCH-normal.csv", index_col=0, parse_dates=True)
        for p in PS:
            for qname, dist in [("normalq", "normal"), ("tq", "t")]:
                z = quantile(g["refit_origin"], part, p, dist)
                base = pinball(g["ret"].values, g["mu"].values + np.sqrt(g["h_pool"].values) * z, p)
                for col in ["h_hard", "h_soft"]:
                    ch = pinball(g["ret"].values, g["mu"].values + np.sqrt(g[col].values) * z, p)
                    add(f"RQ4-{qname}-{variant}", t, p, "pooled", col, base, ch)

R = pd.DataFrame(rows)
R["p_holm"] = np.nan
for _, g in R.groupby(["family", "p_level"]):
    R.loc[g.index, "p_holm"] = multipletests(g.p, method="holm")[1]
R.to_csv(RES / "var_pinball_tests.csv", index=False)

star = lambda a, p: f"{a:+.2f}" + ("**" if p < .01 else "*" if p < .05 else "")
pd.set_option("display.width", 250)
for (fam, pl), g in R.groupby(["family", "p_level"]):
    g = g.assign(cell=[star(a, b) for a, b in zip(g.DM, g.p)], row=g.reference + " -> " + g.challenger)
    print(f"\n=== {fam} | VaR level {pl:.0%}: DM on pinball loss (+ = challenger better; * p<.05, ** p<.01, unadjusted) ===")
    print(g.pivot(index="row", columns="series", values="cell").to_string())

R["better"] = (R.p < .05) & (R.DM > 0)
R["better_holm"] = (R.p_holm < .05) & (R.DM > 0)
R["worse"] = (R.p < .05) & (R.DM < 0)
R["worse_holm"] = (R.p_holm < .05) & (R.DM < 0)
print("\nSignificant tests by family (raw vs Holm):")
print(R.groupby(["family", "p_level"]).agg(tests=("p", "size"), better=("better", "sum"), better_holm=("better_holm", "sum"),
                                            worse=("worse", "sum"), worse_holm=("worse_holm", "sum")).to_string())