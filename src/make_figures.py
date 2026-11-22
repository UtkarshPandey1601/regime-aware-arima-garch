from pathlib import Path
import numpy as np, pandas as pd, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
RES = ROOT / "results"; FIG = RES / "figures"; FIG.mkdir(parents=True, exist_ok=True)
NAVY, MID, GREY, GREEN, AMBER = "#0B2A5B", "#2F6DB5", "#6B7C99", "#2E7D32", "#C77700"
names = [cfg["tickers"]["index"]] + list(cfg["tickers"]["stocks"].values())
clean = lambda t: t.replace("^", "")
short = {"NSEI": "Nifty 50", "ICICIBANK.NS": "ICICI", "TCS.NS": "TCS", "RELIANCE.NS": "Reliance", "MARUTI.NS": "Maruti", "HINDUNILVR.NS": "HUL"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / f"report_{name}.png", dpi=200); plt.close(fig); print("saved", name)


# 1. RQ1: ARMA-only vs GARCH(1,1)
q = pd.read_csv(RES / "rq1_baseline.csv", index_col=0)
fig, ax = plt.subplots(figsize=(7, 3.6)); x = np.arange(len(q)); w = 0.38
ax.bar(x - w / 2, q.QLIKE_arma, w, color=GREY, label="ARMA-only (rolling 20-day)")
ax.bar(x + w / 2, q.QLIKE_garch, w, color=NAVY, label="GARCH(1,1)")
for i, (p, v) in enumerate(zip(q.DM_QLIKE_p, q.QLIKE_garch)):
    if p < 0.05:
        ax.text(i + w / 2, v + 0.03, "*", ha="center", fontsize=14)
ax.set_xticks(x); ax.set_xticklabels([short[clean(i)] for i in q.index])
ax.set_ylabel("Mean QLIKE (lower is better)"); ax.legend(frameon=False)
ax.set_title("RQ1: * = significant at 5% (Diebold-Mariano)")
save(fig, "rq1_qlike")

# 2. 12-spec grid: % QLIKE difference from GARCH-normal
g = pd.read_csv(RES / "grid_summary.csv")
order = [f"{v}-{d}" for v in ["GARCH", "GJR", "EGARCH", "APARCH"] for d in ["normal", "t", "skewt"]]
pv = g.pivot(index="spec", columns="series", values="QLIKE").loc[order, [clean(t) for t in names]]
rel = (pv / pv.loc["GARCH-normal"] - 1) * 100
fig, ax = plt.subplots(figsize=(7, 4.8))
im = ax.imshow(rel.values, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
ax.set_xticks(range(rel.shape[1])); ax.set_xticklabels([short[c] for c in rel.columns])
ax.set_yticks(range(rel.shape[0])); ax.set_yticklabels(rel.index)
for i in range(rel.shape[0]):
    for j in range(rel.shape[1]):
        ax.text(j, i, f"{rel.values[i, j]:+.1f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, label="% QLIKE vs GARCH-normal (negative = better)")
ax.set_title("12 specifications, test period (no significance implied)")
save(fig, "grid_heatmap")

# 3. News-impact curves from mean parameters
def nic(v, P, s2, e):
    if v == "GARCH":
        return P["omega"] + P["alpha[1]"] * e ** 2 + P["beta[1]"] * s2
    if v == "GJR":
        return P["omega"] + (P["alpha[1]"] + P["gamma[1]"] * (e < 0)) * e ** 2 + P["beta[1]"] * s2
    if v == "EGARCH":
        z = e / np.sqrt(s2)
        return np.exp(P["omega"] + P["alpha[1]"] * (np.abs(z) - np.sqrt(2 / np.pi)) + P["gamma[1]"] * z + P["beta[1]"] * np.log(s2))
    d = P["delta"]
    return (P["omega"] + P["alpha[1]"] * (np.abs(e) - P["gamma[1]"] * e) ** d + P["beta[1]"] * s2 ** (d / 2)) ** (2 / d)

try:
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, t in zip(axs, ["NSEI", "ICICIBANK.NS"]):
        for v, c in zip(["GARCH", "GJR", "EGARCH", "APARCH"], [GREY, NAVY, MID, AMBER]):
            P = pd.read_csv(RES / "params" / f"{t}_{v}-normal.csv").mean(numeric_only=True)
            s2 = pd.read_csv(RES / "forecasts" / f"{t}_{v}-normal.csv")["h"].mean()
            e = np.linspace(-4 * np.sqrt(s2), 4 * np.sqrt(s2), 200)
            ax.plot(e, nic(v, P, s2, e), color=c, label=v)
        ax.set_title(short[clean(t)]); ax.set_xlabel("Shock (%)"); ax.set_ylabel("Next-day variance")
    axs[0].legend(frameon=False)
    fig.suptitle("News-impact curves (mean parameters over refits, Normal innovations)")
    save(fig, "news_impact")
except KeyError as err:
    print("News-impact figure skipped, parameter column missing:", err)

# 4. Nifty regimes vs India VIX
f = pd.read_csv(RES / "hmm" / "NSEI_filtered.csv", index_col=0, parse_dates=True)
vix = pd.read_csv(ROOT / "data/processed/INDIAVIX.csv", index_col="Date", parse_dates=True)["vix"].reindex(f.index)
pcol = [c for c in f.columns if c.startswith("p")][-1]
fig, axs = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
axs[0].fill_between(f.index, f[pcol], color=NAVY, alpha=0.8); axs[0].set_ylabel("P(high-volatility state)")
axs[1].plot(vix.index, vix, color=AMBER, lw=0.8); axs[1].set_ylabel("India VIX")
for a in axs:
    a.axvline(pd.Timestamp("2020-03-03"), color="k", ls="--", lw=0.8)
axs[0].set_title("Nifty HMM filtered regime probability (HMM fitted on first 2,097 days) and India VIX; dashed = test start")
save(fig, "regimes_vs_vix")

# 5. Look-ahead bias
t = pd.read_csv(RES / "regime_tests.csv")
b = t[(t.test == "look-ahead bias: filtered vs smoothed") & (t.spec == "GARCH-normal")]
pb = b.pivot(index="series", columns="variant", values="dQLIKE").reindex([clean(n) for n in names])
fig, ax = plt.subplots(figsize=(7, 3.6)); x = np.arange(len(pb)); w = 0.38
ax.bar(x - w / 2, pb["A"], w, color=NAVY, label="A: own-series HMM")
ax.bar(x + w / 2, pb["B"], w, color=MID, label="B: Nifty regimes")
ax.set_xticks(x); ax.set_xticklabels([short[i] for i in pb.index])
ax.set_ylabel("QLIKE(filtered) - QLIKE(smoothed)"); ax.legend(frameon=False)
ax.set_title("Apparent gain from look-ahead (upper bound: smoothed over the full sample)")
save(fig, "lookahead_bias")

# 6. VaR exception rates by innovation distribution
v = pd.read_csv(RES / "var_backtests.csv")
fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, p in zip(axs, [0.01, 0.05]):
    data = [v[(v.p == p) & (v.dist == d)].rate.values * 100 for d in ["normal", "t", "skewt"]]
    ax.boxplot(data, tick_labels=["Normal", "Student-t", "Skewed-t"])
    ax.axhline(p * 100, color="r", ls="--", lw=1)
    ax.set_title(f"VaR {int(p * 100)}%: exception rate (%), target dashed"); ax.set_ylabel("Exceptions (%)")
save(fig, "var_exception_rates")