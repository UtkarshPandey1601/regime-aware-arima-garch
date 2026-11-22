import sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_utils import qlike

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
RES = ROOT / "results"
names = [cfg["tickers"]["index"]] + list(cfg["tickers"]["stocks"].values())
clean = lambda t: t.replace("^", "")
SPECS = [f"{v}-{d}" for v in ["GARCH", "GJR", "EGARCH", "APARCH"] for d in ["normal", "t", "skewt"]]
ALPHA, B, BLOCK = 0.10, 500, 10


def mcs(L, alpha=ALPHA, B=B, block=BLOCK, seed=1):
    """Hansen-Lunde-Nason MCS, T_max statistic, moving-block bootstrap. Returns p-value per model."""
    X = L.values; T, m = X.shape
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=(B, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % T).reshape(B, -1)[:, :T]
    alive, pv, pmax = list(range(m)), np.ones(m), 0.0
    while len(alive) > 1:
        sub = X[:, alive]
        D = sub - sub.mean(1, keepdims=True)
        dbar = D.mean(0)
        bm = D[idx].mean(1)
        sd = bm.std(0, ddof=1)
        t = dbar / sd
        p = (((bm - dbar) / sd).max(1) >= t.max()).mean()
        pmax = max(pmax, p)
        worst = alive[int(np.argmax(t))]
        pv[worst] = pmax
        alive.remove(worst)
    return pd.Series(pv, index=L.columns)


rows = []
for t in names:
    # family 1: the 12 specifications
    L = pd.DataFrame({s: (lambda d: qlike(d["ret"] ** 2, d["h"]).values)(
        pd.read_csv(RES / "forecasts" / f"{clean(t)}_{s}.csv", index_col=0)) for s in SPECS})
    for s, p in mcs(L).items():
        rows.append({"family": "12 specs", "series": clean(t), "model": s, "p_mcs": p, "in_set": p >= ALPHA})
    # family 2: pooled vs regime-aware (GARCH-normal)
    cols = {}
    for var in ["A", "B"]:
        f = RES / "regime_fc" / f"{clean(t)}_{var}_GARCH-normal.csv"
        if f.exists() and not (t == cfg["tickers"]["index"] and var == "B"):
            d = pd.read_csv(f, index_col=0)
            s = d["ret"] ** 2
            cols.setdefault("pooled", qlike(s, d["h_pool"]).values)
            cols[f"{var}-hard"], cols[f"{var}-soft"] = qlike(s, d["h_hard"]).values, qlike(s, d["h_soft"]).values
    for s, p in mcs(pd.DataFrame(cols)).items():
        rows.append({"family": "regime", "series": clean(t), "model": s, "p_mcs": p, "in_set": p >= ALPHA})

R = pd.DataFrame(rows)
R.to_csv(RES / "mcs.csv", index=False)
pd.set_option("display.width", 220)
for fam, g in R.groupby("family"):
    pv = g.pivot(index="model", columns="series", values="p_mcs")
    inn = g.pivot(index="model", columns="series", values="in_set").fillna(False).astype(bool)
    txt = pv.round(3).astype(str)
    txt = (txt + "*").where(inn, txt).where(pv.notna(), "-")
    print(f"\n=== Model Confidence Set, {fam} (alpha = {ALPHA}; * = in the set) ===")
    print(txt.to_string())
    print("models in set per series:", g[g.in_set].groupby("series").size().to_dict())