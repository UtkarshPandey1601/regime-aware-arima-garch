import warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
FINAL = ROOT / "data" / "processed" / "final"
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

names = [cfg["tickers"]["index"]] + list(cfg["tickers"]["stocks"].values())
LAGS = 10
rows = []

for t in names:
    df = pd.read_csv(FINAL / (t.replace("^", "") + ".csv"), index_col="Date", parse_dates=True)
    r = df["ret"].dropna()
    jb, jb_p = stats.jarque_bera(r)
    lb_r = acorr_ljungbox(r, lags=[LAGS]).iloc[0]
    lb_r2 = acorr_ljungbox(r ** 2, lags=[LAGS]).iloc[0]
    arch_lm, arch_p, _, _ = het_arch(r - r.mean(), nlags=LAGS)
    rows.append({
        "series": t, "n": len(r), "mean": r.mean(), "std": r.std(),
        "min": r.min(), "max": r.max(),
        "skew": stats.skew(r), "excess_kurt": stats.kurtosis(r),
        "JB": jb, "JB_p": jb_p,
        "ADF_returns_p": adfuller(r, autolag="AIC")[1],
        "ADF_logprice_p": adfuller(np.log(df["adj_close"]), autolag="AIC")[1],
        "KPSS_returns_p": kpss(r, regression="c", nlags="auto")[1],
        f"LB{LAGS}_ret_p": lb_r["lb_pvalue"],
        f"LB{LAGS}_ret2_p": lb_r2["lb_pvalue"],
        f"ARCH_LM{LAGS}": arch_lm, "ARCH_LM_p": arch_p,
    })

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(df["adj_close"]); ax[0, 0].set_title(f"{t} adjusted close")
    ax[0, 1].plot(r, lw=0.6); ax[0, 1].set_title("Log returns (%)")
    ax[1, 0].plot(r ** 2, lw=0.6); ax[1, 0].set_title("Squared returns")
    plot_acf(r ** 2, lags=40, ax=ax[1, 1]); ax[1, 1].set_title("ACF of squared returns")
    fig.tight_layout()
    fig.savefig(FIG / f"{t.replace('^', '')}_diagnostics.png", dpi=130)
    plt.close(fig)

out = pd.DataFrame(rows).set_index("series")
out.to_csv(RES / "diagnostics_summary.csv")
pd.set_option("display.width", 200)
print(out.round(4).T)

print("\nACCEPTANCE: ARCH effects (ARCH-LM p < 0.05) in every series?",
      bool((out["ARCH_LM_p"] < 0.05).all()))