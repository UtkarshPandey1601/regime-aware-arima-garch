import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stats_utils import qlike, dm_test, kupiec, christoffersen_ind, forward_filter

NAMES = ["NSEI", "ICICIBANK.NS", "TCS.NS", "RELIANCE.NS", "MARUTI.NS", "HINDUNILVR.NS"]


def test_qlike_known_values():
    assert np.isclose(qlike(1.0, 1.0), 1.0)
    assert np.isclose(qlike(4.0, 2.0), 2 + np.log(2))


def test_qlike_minimised_at_truth():
    h = np.linspace(0.3, 3, 400)
    assert abs(h[np.argmin(qlike(1.0, h))] - 1.0) < 0.02


def test_qlike_finite_on_zero_return():
    assert np.isfinite(qlike(0.0, 1.5))


def test_dm_sign_and_power():
    rng = np.random.default_rng(0)
    base = rng.normal(size=1500) ** 2
    better = base - 0.3 + rng.normal(scale=0.2, size=1500)
    stat, p = dm_test(base, better)
    assert stat > 0 and p < 0.01
    stat2, _ = dm_test(better, base)
    assert np.isclose(stat2, -stat)


def test_dm_null_is_not_extreme():
    rng = np.random.default_rng(1)
    stat, p = dm_test(rng.normal(size=1500) ** 2, rng.normal(size=1500) ** 2)
    assert abs(stat) < 4


def test_kupiec_exact_and_reject():
    _, p_ok = kupiec(10, 1000, 0.01)
    _, p_bad = kupiec(40, 1000, 0.01)
    assert p_ok > 0.99 and p_bad < 0.001


def test_christoffersen_detects_clustering():
    spread = np.zeros(1000, int); spread[::100] = 1
    clustered = np.zeros(1000, int); clustered[500:510] = 1
    assert christoffersen_ind(clustered)[1] < 0.01
    assert christoffersen_ind(spread)[1] > 0.05


PI = np.array([0.5, 0.5]); A = np.array([[0.95, 0.05], [0.10, 0.90]])
MU = np.array([0.0, 0.0]); VAR = np.array([1.0, 9.0])


def test_filter_rows_sum_to_one_and_no_lookahead():
    x = np.random.default_rng(2).normal(size=600) * 2
    full = forward_filter(x, PI, A, MU, VAR)
    part = forward_filter(x[:300], PI, A, MU, VAR)
    assert np.allclose(full.sum(1), 1)
    assert np.allclose(full[:300], part)          # adding future data must not change past filtered probabilities


def test_filter_matches_hmmlearn_on_last_day():
    hmmlearn = pytest.importorskip("hmmlearn.hmm")
    x = np.random.default_rng(3).normal(size=400) * 2
    m = hmmlearn.GaussianHMM(n_components=2, covariance_type="diag")
    m.startprob_, m.transmat_ = PI, A
    m.means_, m.covars_ = MU.reshape(-1, 1), VAR.reshape(-1, 1)
    smooth_last = m.predict_proba(x.reshape(-1, 1))[-1]
    assert np.allclose(forward_filter(x, PI, A, MU, VAR)[-1], smooth_last, atol=1e-6)


def test_final_data_integrity():
    fr = {t: pd.read_csv(ROOT / "data/processed/final" / f"{t}.csv", index_col="Date", parse_dates=True) for t in NAMES}
    idx = fr["NSEI"].index
    for t, d in fr.items():
        assert len(d) == 4112 and d["ret"].notna().all() and d.index.equals(idx), t
        assert d.index.is_monotonic_increasing and not d.index.duplicated().any()


def test_raw_data_matches_manifest():
    man = json.loads((ROOT / "data/raw/manifest.json").read_text())
    for k, v in man.items():
        if k.startswith("_"):
            continue
        assert hashlib.sha256((ROOT / "data/raw" / v["file"]).read_bytes()).hexdigest() == v["sha256"], k


def test_walkforward_has_no_leakage():
    from models import run_spec, orders, FINAL
    r = pd.read_csv(FINAL / "NSEI.csv", index_col="Date", parse_dates=True)["ret"]
    n0 = int(len(r) * 0.60)
    base = r.iloc[: n0 + 60]
    pert = base.copy(); pert.iloc[n0 + 30:] = pert.iloc[n0 + 30:] * 5 + 7
    a, _ = run_spec(base, orders["^NSEI"]["bic"], "GARCH-normal", n0)
    b, _ = run_spec(pert, orders["^NSEI"]["bic"], "GARCH-normal", n0)
    assert np.allclose(a[["mu", "h"]].iloc[:30].values, b[["mu", "h"]].iloc[:30].values, rtol=1e-8)