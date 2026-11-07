import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
FINAL, RES = ROOT / "data" / "processed" / "final", ROOT / "results"

TRAIN_FRAC, VAL_FRAC = 0.60, 0.15      # validation = last 15% of the training window
MAX_ORDER = 3
names = [cfg["tickers"]["index"]] + list(cfg["tickers"]["stocks"].values())

split_rows, arma_rows, orders = [], [], {}

for t in names:
    df = pd.read_csv(FINAL / (t.replace("^", "") + ".csv"), index_col="Date", parse_dates=True)
    r = df["ret"]
    n = len(r)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n_train * VAL_FRAC)
    n_fit = n_train - n_val            # ARMA/HMM/GARCH selection fits on this part
    fit, val, test = r.iloc[:n_fit], r.iloc[n_fit:n_train], r.iloc[n_train:]

    # high-volatility count in the training window, using only training-window thresholds
    thr = r.iloc[:n_train].abs().quantile(0.90)
    big = int((r.iloc[:n_train].abs() > thr).sum())
    absr = r.iloc[:n_train].abs()
    crisis_days = int((absr > 2 * r.iloc[:n_train].std()).sum())

    split_rows.append({
        "series": t, "n": n,
        "fit_start": fit.index[0].date(), "fit_end": fit.index[-1].date(), "n_fit": len(fit),
        "val_end": val.index[-1].date(), "n_val": len(val),
        "test_start": test.index[0].date(), "test_end": test.index[-1].date(), "n_test": len(test),
        "train_days_above_90pct_abs": big,
        "train_days_above_2sd": crisis_days,
    })

    # ARMA grid on the fit block only
    best = {}
    for p in range(MAX_ORDER + 1):
        for q in range(MAX_ORDER + 1):
            try:
                m = ARIMA(fit.values, order=(p, 0, q), trend="c").fit()
                arma_rows.append({"series": t, "p": p, "q": q, "aic": m.aic, "bic": m.bic})
            except Exception as e:
                arma_rows.append({"series": t, "p": p, "q": q, "aic": np.nan, "bic": np.nan})

grid = pd.DataFrame(arma_rows)
splits = pd.DataFrame(split_rows)

for t in names:
    g = grid[grid.series == t].dropna()
    b = g.loc[g.bic.idxmin()]
    a = g.loc[g.aic.idxmin()]
    orders[t] = {"bic": [int(b.p), int(b.q)], "aic": [int(a.p), int(a.q)]}

grid.to_csv(RES / "arma_grid.csv", index=False)
splits.to_csv(RES / "splits_summary.csv", index=False)
(RES / "arma_orders.json").write_text(json.dumps(orders, indent=2))

pd.set_option("display.width", 220)
print(splits.T)
print("\nChosen ARMA orders (p, q):")
for t, o in orders.items():
    print(f"  {t:15s} BIC {tuple(o['bic'])}   AIC {tuple(o['aic'])}")