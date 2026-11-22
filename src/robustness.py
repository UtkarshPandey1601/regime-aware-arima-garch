import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as M
from stats_utils import qlike, dm_test

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"; OUT = RES / "robust"; OUT.mkdir(exist_ok=True)
clean = lambda t: t.replace("^", "")


def one(t, refit5):
    M.VARS["GARCH21"] = dict(vol="GARCH", p=2, o=0, q=1)       # set inside the worker process
    r = pd.read_csv(M.FINAL / f"{clean(t)}.csv", index_col="Date", parse_dates=True)["ret"]
    n0 = int(len(r) * M.TRAIN_FRAC)
    base = pd.read_csv(RES / "forecasts" / f"{clean(t)}_GARCH-normal.csv", index_col=0, parse_dates=True)
    s = base["ret"] ** 2
    jobs = [("aic_arma", M.orders[t]["aic"], "GARCH-normal", 20), ("garch21", M.orders[t]["bic"], "GARCH21-normal", 20)]
    if refit5 and t == "^NSEI":
        jobs.append(("refit5", M.orders[t]["bic"], "GARCH-normal", 5))
    rows = []
    for tag, order, spec, refit in jobs:
        f = OUT / f"{clean(t)}_{tag}.csv"
        if f.exists():
            fc = pd.read_csv(f, index_col=0, parse_dates=True)
        else:
            fc, _ = M.run_spec(r, order, spec, n0, refit=refit)
            fc.to_csv(f)
        stat, p = dm_test(qlike(s, base["h"]), qlike(s, fc["h"]))
        rows.append({"series": clean(t), "check": tag, "QLIKE_base": qlike(s, base["h"]).mean(),
                     "QLIKE_check": qlike(s, fc["h"]).mean(), "DM_(+ check better)": stat, "p": p})
    # range-proxy check of RQ1: Parkinson proxy rescaled with fit-block data only
    proc = pd.read_csv(M.FINAL / f"{clean(t)}.csv", index_col="Date", parse_dates=True)
    n_fit = n0 - int(n0 * 0.15)
    c = (proc["ret"] ** 2).iloc[:n_fit].mean() / proc["parkinson"].iloc[:n_fit].mean()
    b0 = pd.read_csv(RES / "forecasts" / f"{clean(t)}_baseline.csv", index_col=0, parse_dates=True)
    pr = (proc["parkinson"] * c).reindex(b0.index)
    stat, p = dm_test(qlike(pr, b0["h_armabench"]), qlike(pr, b0["h_garch"]))
    rows.append({"series": clean(t), "check": "RQ1_parkinson_proxy", "QLIKE_base": qlike(pr, b0["h_armabench"]).mean(),
                 "QLIKE_check": qlike(pr, b0["h_garch"]).mean(), "DM_(+ check better)": stat, "p": p})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--refit5", action="store_true", help="also refit Nifty every 5 days (slow)")
    a = ap.parse_args()
    names = [M.cfg["tickers"]["index"]] + list(M.cfg["tickers"]["stocks"].values())
    res = Parallel(n_jobs=a.jobs)(delayed(one)(t, a.refit5) for t in names)
    R = pd.DataFrame([x for r in res for x in r])
    R.to_csv(RES / "robustness.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.round(4).to_string(index=False))
    print("\nFor 'aic_arma', 'garch21' and 'refit5' the base is the main GARCH-normal run; "
          "for 'RQ1_parkinson_proxy' the base is the ARMA-only benchmark and the check is GARCH(1,1).")