import warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy.stats import norm, spearmanr
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
FINAL, RES = ROOT / "data" / "processed" / "final", ROOT / "results"
OUT = RES / "hmm"
OUT.mkdir(parents=True, exist_ok=True)
names = [cfg["tickers"]["index"]] + list(cfg["tickers"]["stocks"].values())
clean = lambda t: t.replace("^", "")
TRAIN_FRAC, VAL_FRAC, KS, N_INIT, MIN_DUR = 0.60, 0.15, [2, 3, 4], 10, 10


def fit_hmm(x, K):
    best = None
    for seed in range(N_INIT):
        m = GaussianHMM(n_components=K, covariance_type="diag", n_iter=200, tol=1e-4,min_covar=0.05, random_state=seed)
        try:
            m.fit(x)
            ll = m.score(x)
        except Exception:
            continue
        if best is None or ll > best[1]:
            best = (m, ll)
    return best


def ordered_params(m):
    K = m.n_components
    var = np.array([np.ravel(m.covars_[k])[0] for k in range(K)])
    o = np.argsort(var)
    return m.startprob_[o], m.transmat_[np.ix_(o, o)], m.means_[o, 0], var[o], o


def forward_filter(x, pi, A, mu, var):
    """Filtered probabilities P(S_t | x_1..x_t): uses data up to t only."""
    B = np.maximum(norm.pdf(x[:, None], loc=mu[None, :], scale=np.sqrt(var)[None, :]), 1e-300)
    filt = np.zeros((len(x), len(pi)))
    a = pi * B[0]
    filt[0] = a / a.sum()
    for t in range(1, len(x)):
        a = (filt[t - 1] @ A) * B[t]
        filt[t] = a / a.sum()
    return filt


qlike = lambda s, h: s / h + np.log(h)
vix = pd.read_csv(ROOT / "data/processed/INDIAVIX.csv", index_col="Date", parse_dates=True)["vix"]
rows = []

for t in names:
    r = pd.read_csv(FINAL / f"{clean(t)}.csv", index_col="Date", parse_dates=True)["ret"]
    n = len(r)
    n_train = int(n * TRAIN_FRAC)
    n_fit = n_train - int(n_train * VAL_FRAC)
    x = r.values
    xf = x[:n_fit].reshape(-1, 1)

    cand = {}
    for K in KS:
        res = fit_hmm(xf, K)
        if res is None:
            continue
        m, ll = res
        pi, A, mu, var, o = ordered_params(m)
        dur = 1 / (1 - np.diag(A))
        npar = 2 * K + K * (K - 1) + (K - 1)
        cand[K] = dict(m=m, bic=-2 * ll + npar * np.log(n_fit), min_dur=dur.min())
    ok = {K: c for K, c in cand.items() if c["min_dur"] >= MIN_DUR} or cand
    K = min(ok, key=lambda k: ok[k]["bic"])
    m = cand[K]["m"]
    if cand[K]["min_dur"] < MIN_DUR:
        print(f"  WARNING: no K met the {MIN_DUR}-day minimum duration; chose lowest BIC (min duration {cand[K]['min_dur']:.1f}d)")
    pi, A, mu, var, o = ordered_params(m)
    print(f"\n=== {t} === BIC by K: " + ", ".join(f"K={k}: {c['bic']:.0f} (min dur {c['min_dur']:.1f}d)" for k, c in cand.items()) + f"  -> chose K={K}")
    print("state sd (%):", np.sqrt(var).round(2), "| expected duration (days):", (1 / (1 - np.diag(A))).round(1))
    print("transition matrix:\n", A.round(3))

    filt = forward_filter(x, pi, A, mu, var)
    smooth = m.predict_proba(x.reshape(-1, 1))[:, o]            # uses the FUTURE
    assert np.allclose(filt[-1], smooth[-1], atol=1e-6), "forward filter check failed"

    sf, ss = filt.argmax(1), smooth.argmax(1)
    test = np.arange(n) >= n_train
    counts_fit = np.bincount(sf[:n_fit], minlength=K)
    counts_test = np.bincount(sf[test], minlength=K)
    print("days per state, fit block (filtered):", counts_fit, "| test period:", counts_test)
    print(f"highest-variance state has {counts_fit[-1]} fit-block days (plan needs about 250 for stable regime-specific fits)")
    print(f"filtered vs smoothed label agreement: all {np.mean(sf == ss):.3f}, test {np.mean(sf[test] == ss[test]):.3f}")

    top = filt[:, -1]
    v = vix.reindex(r.index)
    mk = v.notna().values
    rho = spearmanr(top[mk], v.values[mk])[0]
    covid = (r.index >= "2020-03-01") & (r.index <= "2020-04-30")
    print(f"P(highest state): mean in test {top[test].mean():.2f}, Mar-Apr 2020 {top[covid].mean():.2f} | Spearman with India VIX {rho:.2f}")

    # first look at look-ahead bias: regime-mixture one-day variance forecast, static HMM parameters
    hf = (filt[:-1] @ A) @ var          # forecast for t+1 from information up to t
    hs = (smooth[:-1] @ A) @ var        # same, but smoothed probabilities (leaks the future)
    tgt = x[1:] ** 2
    tm = np.arange(1, n) >= n_train
    qf, qs = qlike(tgt[tm], hf[tm]).mean(), qlike(tgt[tm], hs[tm]).mean()
    print(f"mixture-variance QLIKE on test: filtered {qf:.4f} | smoothed (leaky) {qs:.4f} | leakage 'gain' {qf - qs:+.4f}")

    pd.DataFrame(filt, index=r.index, columns=[f"p{k}" for k in range(K)]).assign(state=sf).to_csv(OUT / f"{clean(t)}_filtered.csv")
    rows.append({"series": clean(t), "K": K, "fit_days_top_state": int(counts_fit[-1]), "P_top_test": top[test].mean(),
                 "P_top_covid": top[covid].mean(), "rho_vix": rho, "qlike_filtered": qf, "qlike_smoothed": qs})

pd.DataFrame(rows).to_csv(RES / "hmm_summary.csv", index=False)
print("\n", pd.DataFrame(rows).round(3).to_string(index=False))