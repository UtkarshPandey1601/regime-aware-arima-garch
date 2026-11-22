from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy import stats
from scipy.special import xlogy
from arch.univariate import SkewStudent

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


def var_line(f, par, p, dist):
    if dist == "normal":
        q = np.full(len(f), stats.norm.ppf(p))
    else:
        nu_col = "nu" if dist == "t" else "eta"
        qs = {}
        for _, r in par.iterrows():
            lam = r["lambda"] if dist == "skewt" else None
            qs[int(r["refit_origin"])] = std_q(p, dist, r[nu_col], lam)
        q = f["refit_origin"].map(qs).values
    return f["mu"].values + np.sqrt(f["h"].values) * q


def kupiec(x, N, p):
    ph = x / N
    lr = max(-2 * ((xlogy(N - x, 1 - p) + xlogy(x, p)) - (xlogy(N - x, 1 - ph) + xlogy(x, ph))), 0.0)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def christ_ind(hit):
    h = hit.astype(int)
    a, b = h[:-1], h[1:]
    n00, n01 = ((a == 0) & (b == 0)).sum(), ((a == 0) & (b == 1)).sum()
    n10, n11 = ((a == 1) & (b == 0)).sum(), ((a == 1) & (b == 1)).sum()
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def backtest(ret, var, p):
    hit = ret < var
    N, x = len(ret), int(hit.sum())
    lk, pk = kupiec(x, N, p)
    li, pi_ = christ_ind(hit)
    pcc = 1 - stats.chi2.cdf(lk + li, 2)
    e250 = x * 250 / N
    zone = "green" if e250 <= 4 else "yellow" if e250 <= 9 else "red"
    return {"N": N, "exceptions": x, "rate": x / N, "expected": p * N, "kupiec_p": pk,
            "christ_ind_p": pi_, "cc_p": pcc, "exc_per_250": e250, "basel_zone": zone if p == 0.01 else ""}


rows = []
for t in names:
    for v in VARS:
        for d in DISTS:
            f = pd.read_csv(FC / f"{clean(t)}_{v}-{d}.csv", index_col=0, parse_dates=True)
            par = pd.read_csv(PAR / f"{clean(t)}_{v}-{d}.csv")
            for p in PS:
                rows.append({"series": clean(t), "model": f"{v}-{d}", "dist": d, "var_model": v, "p": p,
                             **backtest(f["ret"].values, var_line(f, par, p, d), p)})
R = pd.DataFrame(rows)
R.to_csv(RES / "var_backtests.csv", index=False)

# regime-aware VaR (Normal quantile on the blended variance), GARCH-normal
rr = []
for variant in ["A", "B"]:
    for t in names:
        if t == cfg["tickers"]["index"] and variant == "B":
            continue
        f = pd.read_csv(RF / f"{clean(t)}_{variant}_GARCH-normal.csv", index_col=0, parse_dates=True)
        for col in ["h_pool", "h_hard", "h_soft"]:
            for p in PS:
                var = f["mu"].values + np.sqrt(f[col].values) * stats.norm.ppf(p)
                rr.append({"variant": variant, "series": clean(t), "forecast": col, "p": p, **backtest(f["ret"].values, var, p)})
RR = pd.DataFrame(rr)
RR.to_csv(RES / "var_regime_backtests.csv", index=False)

pd.set_option("display.width", 250)
for p in PS:
    g = R[R.p == p]
    print(f"\nException RATE at p = {p:.0%} (target {p:.3f})")
    print(g.pivot(index="model", columns="series", values="rate").round(4).to_string())
    print(f"\nKupiec p-values at p = {p:.0%} (below 0.05 = coverage rejected)")
    print(g.pivot(index="model", columns="series", values="kupiec_p").round(3).to_string())

print("\nCoverage by innovation distribution (across 4 variance models x 6 series = 24 per cell)")
S = R.groupby(["p", "dist"]).agg(mean_rate=("rate", "mean"),
                                 kupiec_not_rejected=("kupiec_p", lambda s: (s > .05).mean()),
                                 ind_not_rejected=("christ_ind_p", lambda s: (s > .05).mean()),
                                 cc_not_rejected=("cc_p", lambda s: (s > .05).mean()))
print(S.round(3).to_string())
print("\nBasel zones at 1% (exceptions scaled to 250 days), count by distribution")
print(R[R.p == 0.01].groupby(["dist", "basel_zone"]).size().unstack(fill_value=0).to_string())
print("\nRegime-aware VaR coverage (GARCH-normal, Normal quantile): exception rate and Kupiec p")
for p in PS:
    g = RR[RR.p == p].assign(c=lambda x: x.rate.round(4).astype(str) + " (p=" + x.kupiec_p.round(2).astype(str) + ")")
    print(f"\n p = {p:.0%}")
    print(g.pivot(index=["variant", "forecast"], columns="series", values="c").to_string())