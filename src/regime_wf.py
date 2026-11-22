import argparse, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy import stats
from scipy.stats import norm
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from statsmodels.tsa.arima.model import ARIMA
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
FINAL, RES = ROOT / "data" / "processed" / "final", ROOT / "results"
OUT = RES / "regime_fc"
OUT.mkdir(parents=True, exist_ok=True)
FC = RES / "forecasts"

TRAIN_FRAC, REFIT, K, N_INIT, MIN_OBS = 0.60, 20, 2, 3, 250
orders = json.loads((RES / "arma_orders.json").read_text())
VARS = {"GARCH": dict(vol="GARCH", p=1, o=0, q=1), "GJR": dict(vol="GARCH", p=1, o=1, q=1),
        "EGARCH": dict(vol="EGARCH", p=1, o=1, q=1), "APARCH": dict(vol="APARCH", p=1, o=1, q=1)}
ALL = [cfg["tickers"]["index"]] + list(cfg["tickers"]["stocks"].values())
MKT = cfg["tickers"]["index"]
clean = lambda t: t.replace("^", "")
qlike = lambda s, h: s / h + np.log(h)


def make(resid, spec):
    v, d = spec.split("-")
    return arch_model(resid, mean="Zero", dist=d, rescale=False, **VARS[v])


def pers(p, spec):
    if not spec.startswith(("GARCH", "GJR")):
        return np.nan
    g = p.get("gamma[1]", 0.0) if spec.startswith("GJR") else 0.0
    return p.get("alpha[1]", np.nan) + p.get("beta[1]", np.nan) + 0.5 * g


def fit_params(resid, spec):
    """Returns (params, ok). ok=False if the fit failed, did not converge or is non-stationary."""
    try:
        g = make(resid, spec).fit(disp="off")
        ok = g.convergence_flag == 0 and np.isfinite(pers(g.params, spec) if not np.isnan(pers(g.params, spec)) else 0.0) \
            and (np.isnan(pers(g.params, spec)) or pers(g.params, spec) < 1.0)
        return g.params, bool(ok)
    except Exception:
        return None, False


def fit_hmm(x):
    best = None
    for seed in range(N_INIT):
        m = GaussianHMM(n_components=K, covariance_type="diag", n_iter=200, tol=1e-4,
                        min_covar=0.05, random_state=seed)
        try:
            m.fit(x.reshape(-1, 1))
            ll = m.score(x.reshape(-1, 1))
        except Exception:
            continue
        if best is None or ll > best[1]:
            best = (m, ll)
    if best is None:
        return None
    m = best[0]
    var = np.array([np.ravel(m.covars_[k])[0] for k in range(K)])
    o = np.argsort(var)                                   # label states by variance
    return m.startprob_[o], m.transmat_[np.ix_(o, o)], m.means_[o, 0], var[o], o, m


def forward_filter(x, pi, A, mu, var):
    """Filtered P(S_t | x_1..x_t). Uses data up to t only."""
    B = np.maximum(norm.pdf(x[:, None], loc=mu[None, :], scale=np.sqrt(var)[None, :]), 1e-300)
    f = np.zeros((len(x), len(pi)))
    a = pi * B[0]
    f[0] = a / a.sum()
    for t in range(1, len(x)):
        a = (f[t - 1] @ A) * B[t]
        f[t] = a / a.sum()
    return f


def run(r, r_mkt, order, spec, variant, n_start, label=""):
    x, n = r.values, len(r)
    xr = r_mkt.values if variant == "B" else x
    out, log, last_hmm, last_pool = [], [], None, None
    for i in range(n_start, n, REFIT):
        end = min(i + REFIT, n)
        arma = ARIMA(x[:i], order=(order[0], 0, order[1]), trend="c").fit()
        resid = np.asarray(arma.apply(x[:end]).resid)
        hm = fit_hmm(xr[:i]) or last_hmm
        if hm is None:
            raise RuntimeError("first HMM fit failed")
        last_hmm = hm
        pi, A, hmu, hvar, o, model = hm
        filt = forward_filter(xr[:end], pi, A, hmu, hvar)          # filtered: no future data
        smooth = model.predict_proba(xr.reshape(-1, 1))[:, o]       # smoothed: USES FUTURE (bias check only)
        sf = filt[:i].argmax(1)                                     # in-sample regime labels, filtered

        pool, ok = fit_params(resid[:i], spec)
        if not ok:
            if last_pool is None:
                raise RuntimeError(f"{spec}: pooled fit failed at first origin")
            pool = last_pool
        last_pool = pool
        h_pool = np.asarray(make(resid[:end], spec).fix(pool).conditional_volatility)[i:end] ** 2

        hk, fb, nk, pk = [], [], [], []
        for k in range(K):
            idx = np.where(sf == k)[0]
            nk.append(len(idx))
            pk_params, ok_k = (fit_params(resid[idx], spec) if len(idx) >= MIN_OBS else (None, False))
            if ok_k:
                hk.append(np.asarray(make(resid[:end], spec).fix(pk_params).conditional_volatility)[i:end] ** 2)
                pk.append(pers(pk_params, spec)); fb.append(0)
            else:                                                    # pooled fallback, flagged
                hk.append(h_pool); pk.append(pers(pool, spec)); fb.append(1)
        hk = np.array(hk)                                            # (K, L)

        js = np.arange(i, end)
        pf, ps = filt[js - 1] @ A, smooth[js - 1] @ A                # next-day regime probabilities
        L = np.arange(len(js))
        out.append(pd.DataFrame({
            "ret": x[i:end], "mu": x[i:end] - resid[i:end], "h_pool": h_pool,
            "h_hard": hk[pf.argmax(1), L], "h_soft": (pf * hk.T).sum(1),
            "h_hard_smooth": hk[ps.argmax(1), L], "h_soft_smooth": (ps * hk.T).sum(1),
            "p_high": pf[:, -1], "refit_origin": i}, index=r.index[i:end]))
        log.append({"refit_origin": i, "date": r.index[i], "n_low": nk[0], "n_high": nk[1],
                    "fallback_low": fb[0], "fallback_high": fb[1], "pers_pool": pers(pool, spec),
                    "pers_low": pk[0], "pers_high": pk[1], "hmm_sd_low": np.sqrt(hvar[0]), "hmm_sd_high": np.sqrt(hvar[1])})
        if ((i - n_start) // REFIT) % 10 == 0:
            print(f"   {label} origin {r.index[i].date()} (high-regime days in sample: {nk[1]}, fallbacks: {fb})", flush=True)
    return pd.concat(out), pd.DataFrame(log)


def load(t):
    return pd.read_csv(FINAL / f"{clean(t)}.csv", index_col="Date", parse_dates=True)["ret"]


def leakage_test(spec):
    t = "ICICIBANK.NS"
    r, rm = load(t), load(MKT)
    n0 = int(len(r) * TRAIN_FRAC)
    for variant in ["A", "B"]:
        b, bm = r.iloc[: n0 + 60], rm.iloc[: n0 + 60]
        p, pm = b.copy(), bm.copy()
        p.iloc[n0 + 30:] = p.iloc[n0 + 30:] * 5 + 7
        pm.iloc[n0 + 30:] = pm.iloc[n0 + 30:] * 5 + 7
        a, _ = run(b, bm, orders[t]["bic"], spec, variant, n0)
        c, _ = run(p, pm, orders[t]["bic"], spec, variant, n0)
        for col in ["h_pool", "h_hard", "h_soft"]:
            assert np.allclose(a[col].iloc[:30], c[col].iloc[:30], rtol=1e-6), f"LEAKAGE in {col} (variant {variant})"
        diff = not np.allclose(a["h_hard_smooth"].iloc[:30], c["h_hard_smooth"].iloc[:30], rtol=1e-6)
        print(f"Leakage test passed, variant {variant}. (smoothed forecasts changed with future data: {diff}; expected True)")


def task(t, variant, spec, force):
    f = OUT / f"{clean(t)}_{variant}_{spec}.csv"
    if f.exists() and not force:
        return t, variant, "cached", 0.0
    r, rm = load(t), load(MKT)
    t0 = time.time()
    fc, lg = run(r, rm, orders[t]["bic"], spec, variant, int(len(r) * TRAIN_FRAC), label=f"{clean(t)}/{variant}")
    fc.to_csv(f)
    lg.to_csv(OUT / f"{clean(t)}_{variant}_{spec}_log.csv", index=False)
    return t, variant, "done", time.time() - t0


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


def evaluate(names, specs, variants):
    rows = []
    for spec in specs:
        for v in variants:
            for t in names:
                f = OUT / f"{clean(t)}_{v}_{spec}.csv"
                if not f.exists():
                    continue
                d = pd.read_csv(f, index_col=0, parse_dates=True)
                lg = pd.read_csv(OUT / f"{clean(t)}_{v}_{spec}_log.csv")
                s = d["ret"] ** 2
                L = {c: qlike(s, d[c]) for c in ["h_pool", "h_hard", "h_soft", "h_hard_smooth", "h_soft_smooth"]}
                dh, ph = dm_test(L["h_pool"], L["h_hard"])
                dsf, ps_ = dm_test(L["h_pool"], L["h_soft"])
                chk = np.nan
                bf = FC / f"{clean(t)}_{spec}.csv"
                if bf.exists():
                    b = pd.read_csv(bf, index_col=0, parse_dates=True)
                    chk = float((b["h"] / d["h_pool"] - 1).abs().max())
                rows.append({"spec": spec, "variant": v, "series": clean(t),
                             "QLIKE_pool": L["h_pool"].mean(), "QLIKE_hard": L["h_hard"].mean(), "QLIKE_soft": L["h_soft"].mean(),
                             "DM_hard": dh, "p_hard": ph, "DM_soft": dsf, "p_soft": ps_,
                             "QLIKE_hard_smooth": L["h_hard_smooth"].mean(), "QLIKE_soft_smooth": L["h_soft_smooth"].mean(),
                             "leak_gain_soft": L["h_soft"].mean() - L["h_soft_smooth"].mean(),
                             "pers_pool": lg.pers_pool.mean(), "pers_low": lg.pers_low.mean(), "pers_high": lg.pers_high.mean(),
                             "fallback_high_share": lg.fallback_high.mean(), "fallback_low_share": lg.fallback_low.mean(),
                             "pool_vs_baseline_maxreldiff": chk})
    R = pd.DataFrame(rows)
    if R.empty:
        return
    R.to_csv(RES / "regime_eval.csv", index=False)
    pd.set_option("display.width", 250)
    for (spec, v), g in R.groupby(["spec", "variant"]):
        print(f"\n=== spec {spec} | variant {v}  (DM>0 = regime-aware has LOWER QLIKE than pooled) ===")
        print(g.drop(columns=["spec", "variant"]).set_index("series").round(4).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("series", nargs="*")
    ap.add_argument("--spec", nargs="*", default=["GARCH-normal"])
    ap.add_argument("--variants", nargs="*", default=["B", "A"])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-leak", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    names = [t for t in ALL if not a.series or clean(t) in a.series]
    if not a.eval_only:
        if not a.no_leak:
            leakage_test(a.spec[0])
        jobs = [(t, v, s) for s in a.spec for v in a.variants for t in names]
        print(f"{len(jobs)} tasks, {a.jobs} worker(s)")
        for t, v, st, sec in Parallel(n_jobs=a.jobs, return_as="generator")(delayed(task)(t, v, s, a.force) for t, v, s in jobs):
            print(f"{clean(t):14s} variant {v} {st:7s} {sec:6.0f}s", flush=True)
    evaluate(names, a.spec, a.variants)