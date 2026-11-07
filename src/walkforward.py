import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
FINAL, RES = ROOT / "data" / "processed" / "final", ROOT / "results"
OUT = RES / "forecasts"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_FRAC, REFIT, BENCH_WIN = 0.60, 20, 20
orders = json.loads((RES / "arma_orders.json").read_text())


def run_walkforward(r, order, n_start, refit=REFIT):
    """One-day-ahead forecasts for days n_start..n-1. Uses only data before each day."""
    x = r.values
    n = len(x)
    parts = []
    for i in range(n_start, n, refit):
        end = min(i + refit, n)
        # 1) ARMA fitted on data before day i only
        arma = ARIMA(x[:i], order=(order[0], 0, order[1]), trend="c").fit()
        resid = np.asarray(arma.apply(x[:end]).resid)   # fixed params, filtered forward
        # 2) GARCH(1,1) fitted on residuals before day i only
        g = arch_model(resid[:i], mean="Zero", vol="GARCH", p=1, q=1,
                       dist="normal", rescale=False).fit(disp="off")
        # 3) fixed GARCH params applied forward through the block
        gm = arch_model(resid[:end], mean="Zero", vol="GARCH", p=1, q=1,
                        dist="normal", rescale=False).fix(g.params)
        h = gm.forecast(horizon=1, start=i - 1, reindex=False).variance.iloc[:, 0].values[: end - i]
        mu = x[i:end] - resid[i:end]                     # one-step mean forecast
        bench = np.array([np.mean(resid[t - BENCH_WIN:t] ** 2) for t in range(i, end)])
        parts.append(pd.DataFrame({
            "ret": x[i:end], "mu": mu, "h_garch": h, "h_armabench": bench,
            "refit_origin": i, "converged": int(g.convergence_flag == 0)},
            index=r.index[i:end]))
        if ((i - n_start) // refit) % 10 == 0:
            print(f"   block starting {r.index[i].date()} done", flush=True)
    return pd.concat(parts)


def leakage_test(r, order, n_start):
    """Garbage in the future must not change earlier forecasts."""
    base = r.iloc[: n_start + 60]
    pert = base.copy()
    pert.iloc[n_start + 30:] = pert.iloc[n_start + 30:] * 5 + 7
    a = run_walkforward(base, order, n_start)
    b = run_walkforward(pert, order, n_start)
    cols = ["mu", "h_garch", "h_armabench"]
    same = np.allclose(a[cols].iloc[:30].values, b[cols].iloc[:30].values, rtol=1e-8)
    assert same, "LEAKAGE: forecasts changed when future data was altered!"
    print("Leakage test passed.")


def losses(f, col):
    s, h = f["ret"] ** 2, f[col]
    return {"MAE": (s - h).abs().mean(), "RMSE": np.sqrt(((s - h) ** 2).mean()),
            "QLIKE": (s / h - np.log(s / h) - 1).mean()}


names = [cfg["tickers"]["index"]] + list(cfg["tickers"]["stocks"].values())
if len(sys.argv) > 1:
    names = [t for t in names if t.replace("^", "") in sys.argv[1:]]

for t in names:
    print(f"\n=== {t} ===  ARMA order (BIC): {orders[t]['bic']}")
    r = pd.read_csv(FINAL / (t.replace("^", "") + ".csv"), index_col="Date",
                    parse_dates=True)["ret"]
    n_start = int(len(r) * TRAIN_FRAC)
    if t == names[0]:
        leakage_test(r, orders[t]["bic"], n_start)
    f = run_walkforward(r, orders[t]["bic"], n_start)
    f.to_csv(OUT / f"{t.replace('^', '')}_baseline.csv")
    print(f"forecasts: {len(f)} | non-converged GARCH fits: {int((1 - f.groupby('refit_origin').converged.first()).sum())}")
    print(pd.DataFrame({"GARCH(1,1)": losses(f, "h_garch"),
                        "ARMA-only": losses(f, "h_armabench")}).round(4))