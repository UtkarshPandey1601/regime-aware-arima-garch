import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import run_spec, SPECS, orders, FINAL, TRAIN_FRAC, ALL, clean, FC, RES

VAL = RES / "forecasts_val"
VAL.mkdir(parents=True, exist_ok=True)
VAL_FRAC = 0.15
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


def val_task(t, spec):
    out = VAL / f"{clean(t)}_{spec}.csv"
    if out.exists():
        return t, spec, "cached"
    r = pd.read_csv(FINAL / f"{clean(t)}.csv", index_col="Date", parse_dates=True)["ret"]
    n_train = int(len(r) * TRAIN_FRAC)
    n_fit = n_train - int(n_train * VAL_FRAC)
    f, _ = run_spec(r.iloc[:n_train], orders[t]["bic"], spec, n_fit)   # no test-period data touched
    f.to_csv(out)
    return t, spec, "done"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    for t, s, st in Parallel(n_jobs=a.jobs, return_as="generator")(delayed(val_task)(t, s) for t in ALL for s in SPECS):
        print(clean(t), s, st, flush=True)

    rows = []
    for t in ALL:
        v = {}
        for s in SPECS:
            d = pd.read_csv(VAL / f"{clean(t)}_{s}.csv", index_col=0)
            v[s] = qlike(d["ret"] ** 2, d["h"]).mean()
        best = min(v, key=v.get)
        te = {}
        for s in SPECS:
            d = pd.read_csv(FC / f"{clean(t)}_{s}.csv", index_col=0)
            te[s] = qlike(d["ret"] ** 2, d["h"])
        base = te["GARCH-normal"]
        stat, p = dm_test(base, te[best])
        oracle = min(te, key=lambda k: te[k].mean())
        rank = int(pd.Series({k: x.mean() for k, x in te.items()}).rank()[best])
        rows.append({"series": clean(t), "n_val": len(d) if False else len(pd.read_csv(VAL / f"{clean(t)}_{best}.csv")),
                     "val_best": best, "val_QLIKE_best": v[best], "val_QLIKE_garch_normal": v["GARCH-normal"],
                     "test_QLIKE_chosen": te[best].mean(), "test_QLIKE_garch_normal": base.mean(),
                     "DM_chosen_vs_garch": stat, "p": p, "test_rank_of_chosen_(1=best of 12)": rank, "test_oracle_best": oracle})
    R = pd.DataFrame(rows)
    R.to_csv(RES / "validation_selection.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nModel chosen on the validation block, evaluated on the untouched test period:")
    print(R.round(4).T.to_string())