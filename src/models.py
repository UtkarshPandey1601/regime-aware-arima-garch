import argparse, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
FINAL, RES = ROOT / "data" / "processed" / "final", ROOT / "results"
FC, PAR = RES / "forecasts", RES / "params"
FC.mkdir(parents=True, exist_ok=True)
PAR.mkdir(parents=True, exist_ok=True)

TRAIN_FRAC, REFIT = 0.60, 20
orders = json.loads((RES / "arma_orders.json").read_text())

VARS = {
    "GARCH":  dict(vol="GARCH",  p=1, o=0, q=1),
    "GJR":    dict(vol="GARCH",  p=1, o=1, q=1),
    "EGARCH": dict(vol="EGARCH", p=1, o=1, q=1),
    "APARCH": dict(vol="APARCH", p=1, o=1, q=1),
}
DISTS = ["normal", "t", "skewt"]
SPECS = [f"{v}-{d}" for v in VARS for d in DISTS]
ALL = [cfg["tickers"]["index"]] + list(cfg["tickers"]["stocks"].values())
clean = lambda t: t.replace("^", "")


def make(resid, spec):
    v, d = spec.split("-")
    return arch_model(resid, mean="Zero", dist=d, rescale=False, **VARS[v])


def run_spec(r, order, spec, n_start, refit=REFIT):
    """One-day-ahead forecasts for days n_start..n-1; uses only data before each day."""
    x, n = r.values, len(r)
    fc, prm, last = [], [], None
    for i in range(n_start, n, refit):
        end = min(i + refit, n)
        arma = ARIMA(x[:i], order=(order[0], 0, order[1]), trend="c").fit()
        resid = np.asarray(arma.apply(x[:end]).resid)
        status, ll, params = "ok", np.nan, None
        try:
            g = make(resid[:i], spec).fit(disp="off")
            params, ll = g.params, g.loglikelihood
            if g.convergence_flag != 0:
                status = "nonconverged"
        except Exception:
            status = "error"
        if status != "ok" and last is not None:
            params = last                       # fall back to previous refit, flagged
        elif params is None:
            raise RuntimeError(f"{spec}: first fit failed")
        h = np.asarray(make(resid[:end], spec).fix(params).conditional_volatility)[i:end] ** 2
        if not np.all(np.isfinite(h)) and last is not None and params is not last:
            params, status = last, status + "+badforecast"
            h = np.asarray(make(resid[:end], spec).fix(params).conditional_volatility)[i:end] ** 2
        last = params
        fc.append(pd.DataFrame({"ret": x[i:end], "mu": x[i:end] - resid[i:end], "h": h,
                                "refit_origin": i}, index=r.index[i:end]))
        prm.append({"refit_origin": i, "date": r.index[i], "status": status,
                    "loglik": ll, **params.to_dict()})
    return pd.concat(fc), pd.DataFrame(prm)


def leakage_test(r, order, n_start, spec="APARCH-skewt"):
    base = r.iloc[: n_start + 60]
    pert = base.copy()
    pert.iloc[n_start + 30:] = pert.iloc[n_start + 30:] * 5 + 7
    a, _ = run_spec(base, order, spec, n_start)
    b, _ = run_spec(pert, order, spec, n_start)
    assert np.allclose(a[["mu", "h"]].iloc[:30].values, b[["mu", "h"]].iloc[:30].values, rtol=1e-8), \
        "LEAKAGE detected"
    print(f"Leakage test passed ({spec}).")


def task(t, spec, force):
    out = FC / f"{clean(t)}_{spec}.csv"
    if out.exists() and not force:
        return t, spec, "cached", 0.0, 0
    r = pd.read_csv(FINAL / f"{clean(t)}.csv", index_col="Date", parse_dates=True)["ret"]
    t0 = time.time()
    f, p = run_spec(r, orders[t]["bic"], spec, int(len(r) * TRAIN_FRAC))
    f.to_csv(out)
    p.to_csv(PAR / f"{clean(t)}_{spec}.csv", index=False)
    return t, spec, "done", time.time() - t0, int((p.status != "ok").sum())


def summary(names, specs):
    rows = []
    for t in names:
        for s in specs:
            f = FC / f"{clean(t)}_{s}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f, index_col=0)
            p = pd.read_csv(PAR / f"{clean(t)}_{s}.csv")
            rows.append({"series": clean(t), "spec": s,
                         "QLIKE": (d.ret ** 2 / d.h + np.log(d.h)).mean(),
                         "flagged_refits": int((p.status != "ok").sum())})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(RES / "grid_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nMean QLIKE (lower is better):")
    print(df.pivot(index="spec", columns="series", values="QLIKE").round(4))
    print("\nFlagged refits (non-converged / errors):")
    print(df.pivot(index="spec", columns="series", values="flagged_refits"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("series", nargs="*", help="e.g. NSEI TCS.NS (default: all six)")
    ap.add_argument("--specs", nargs="*", default=SPECS)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-leak", action="store_true")
    a = ap.parse_args()

    names = [t for t in ALL if not a.series or clean(t) in a.series]
    if not a.no_leak and clean(names[0]) == "NSEI":
        r = pd.read_csv(FINAL / "NSEI.csv", index_col="Date", parse_dates=True)["ret"]
        leakage_test(r, orders[names[0]]["bic"], int(len(r) * TRAIN_FRAC))

    jobs = [(t, s) for t in names for s in a.specs]
    print(f"{len(jobs)} tasks, {a.jobs} parallel worker(s)")
    gen = Parallel(n_jobs=a.jobs, return_as="generator")(
        delayed(task)(t, s, a.force) for t, s in jobs)
    for t, s, st, sec, bad in gen:
        print(f"{clean(t):14s} {s:14s} {st:7s} {sec:6.0f}s  flagged refits: {bad}", flush=True)

    summary(names, a.specs)