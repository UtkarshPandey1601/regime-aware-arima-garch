import numpy as np
from scipy import stats
from scipy.special import xlogy
from scipy.stats import norm


def qlike(s, h):
    """QLIKE up to a constant that does not depend on the forecast: s/h + ln h."""
    return s / h + np.log(h)


def dm_test(l1, l2):
    """Diebold-Mariano with HAC variance and Harvey-Leybourne-Newbold correction.
    Positive statistic => model 1 has HIGHER loss (model 2 is better)."""
    d = np.asarray(l1, float) - np.asarray(l2, float)
    T = len(d)
    lags = int(np.floor(T ** (1 / 3)))
    dc = d - d.mean()
    lrv = np.sum(dc * dc) / T
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * np.sum(dc[k:] * dc[:-k]) / T
    stat = d.mean() / np.sqrt(lrv / T) * np.sqrt((T - 1) / T)
    return stat, 2 * (1 - stats.t.cdf(abs(stat), df=T - 1))


def kupiec(x, N, p):
    ph = x / N
    lr = max(-2 * ((xlogy(N - x, 1 - p) + xlogy(x, p)) - (xlogy(N - x, 1 - ph) + xlogy(x, ph))), 0.0)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def christoffersen_ind(hit):
    h = np.asarray(hit).astype(int)
    a, b = h[:-1], h[1:]
    n00, n01 = ((a == 0) & (b == 0)).sum(), ((a == 0) & (b == 1)).sum()
    n10, n11 = ((a == 1) & (b == 0)).sum(), ((a == 1) & (b == 1)).sum()
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def forward_filter(x, pi, A, mu, var):
    """Filtered P(S_t | x_1..x_t): uses data up to t only."""
    B = np.maximum(norm.pdf(np.asarray(x)[:, None], loc=mu[None, :], scale=np.sqrt(var)[None, :]), 1e-300)
    f = np.zeros((len(x), len(pi)))
    a = pi * B[0]
    f[0] = a / a.sum()
    for t in range(1, len(x)):
        a = (f[t - 1] @ A) * B[t]
        f[t] = a / a.sum()
    return f