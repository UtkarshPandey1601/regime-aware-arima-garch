from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
RES = ROOT / "results"
FC = RES / "forecasts"


def qlike(s, h):            # equals QLIKE up to a constant independent of the forecast
    return s / h + np.log(h)


def dm_test(l1, l2):
    """H0: equal expected loss. Positive stat => model 1 has HIGHER loss (model 2 better)."""
    d = np.asarray(l1) - np.asarray(l2)
    T = len(d)
    lags = int(np.floor(T ** (1 / 3)))
    dc = d - d.mean()
    lrv = np.sum(dc * dc) / T
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * np.sum(dc[k:] * dc[:-k]) / T
    stat = d.mean() / np.sqrt(lrv / T) * np.sqrt((T - 1) / T)   # Harvey-Leybourne-Newbold
    return stat, 2 * (1 - stats.t.cdf(abs(stat), df=T - 1))


names = [cfg["tickers"]["index"]] + list(cfg["tickers"]["stocks"].values())
rows = []
for t in names:
    f = pd.read_csv(FC / f"{t.replace('^', '')}_baseline.csv", index_col=0, parse_dates=True)
    s = f["ret"] ** 2
    a, b = f["h_armabench"], f["h_garch"]
    stat, p = dm_test(qlike(s, a), qlike(s, b))      # positive => GARCH better
    stat_m, p_m = dm_test((s - a).abs(), (s - b).abs())
    rows.append({"series": t,
                 "QLIKE_arma": qlike(s, a).mean(), "QLIKE_garch": qlike(s, b).mean(),
                 "DM_QLIKE": stat, "DM_QLIKE_p": p,
                 "MAE_arma": (s - a).abs().mean(), "MAE_garch": (s - b).abs().mean(),
                 "DM_MAE": stat_m, "DM_MAE_p": p_m})

out = pd.DataFrame(rows).set_index("series")
out.to_csv(RES / "rq1_baseline.csv")
pd.set_option("display.width", 200)
print(out.round(4).T)
print("\nPositive DM stat = GARCH(1,1) has lower loss than ARMA-only.")