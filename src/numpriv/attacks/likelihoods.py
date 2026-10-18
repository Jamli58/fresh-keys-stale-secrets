"""Per-field log-likelihoods log p(shown y | true x) for each defense, vectorized over candidate x.

Kerckhoffs: the attacker knows each defense's algorithm and parameters, not its key.
"""

from functools import lru_cache

import numpy as np

from numpriv.defenses import OrderPreserving
from numpriv.defenses.metric_dp_official import bin_log_probs, bucket, output_grid
from numpriv.defenses.order_preserving import SIGNED_LOG_LIMIT

D2_SIM_SECRETS = 4000
D2_SIM_SEED = 12345  # the attacker's own simulation, independent of the defense's randomness
D2_GRID_POINTS = 801


def d3_log_likelihood(y: float, xs: np.ndarray, eps: float, lo: float, hi: float) -> np.ndarray:
    """Truncated-Laplace density of Prεεmpt's value mechanism (scale 2/eps) on [lo, hi]."""
    x = np.clip(np.asarray(xs, float), lo, hi)
    b = 2.0 / eps
    z = b * (2.0 - np.exp(-(x - lo) / b) - np.exp(-(hi - x) / b))
    return -np.abs(y - x) / b - np.log(z)


def d3o_log_likelihood(y: float, xs: np.ndarray, eps: float) -> np.ndarray:
    """log P(shown y | true x) for Prεεmpt's released money sanitizer (D3o): sum over the bins of
    x's bucket whose shown value is y; -inf if y is not on that bucket's grid."""
    x = np.asarray(xs, float)
    v = np.trunc(x)
    out = np.full(x.shape, -np.inf)
    for lo, hi in {bucket(float(t)) for t in np.unique(v)}:
        in_bucket = np.array([bucket(float(t)) == (lo, hi) for t in v])
        hits = np.flatnonzero(np.array(output_grid(lo, hi)) == round(y))
        if not in_bucket.any() or hits.size == 0:
            continue
        logp = bin_log_probs(v[in_bucket], lo, hi, eps)
        out[in_bucket] = np.logaddexp.reduce(logp[:, hits], axis=1)
    return out



HALF_CENT = 0.005


def d1p_log_likelihood(y: float, xs: np.ndarray, k_low: float, k_high: float) -> np.ndarray:
    """Independent log-uniform key per field, shown in whole cents: y = round(k x, 2).

    P(y | x) = P(k in [(y - 0.005)/x, (y + 0.005)/x] and [k_low, k_high]) for log-uniform k,
    which stays finite for a shown 0.00.
    """
    x = np.asarray(xs, float)
    with np.errstate(divide="ignore"):
        lo = np.maximum(k_low, (y - HALF_CENT) / x)
        hi = np.minimum(k_high, (y + HALF_CENT) / x)
        mass = np.where(hi > lo, np.log(hi / lo) / np.log(k_high / k_low), 0.0)
        return np.log(mass)


@lru_cache(maxsize=1)
def _d2_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and sd of log h(x) over random secrets, on a grid of the signed-log axis t."""
    rng = np.random.default_rng(D2_SIM_SEED)
    defense = OrderPreserving()
    knots = np.array([defense.sample_secret(rng, ()).knot_log_targets for _ in range(D2_SIM_SECRETS)])
    knot_t = np.linspace(-SIGNED_LOG_LIMIT, SIGNED_LOG_LIMIT, knots.shape[1])
    grid_t = np.linspace(-SIGNED_LOG_LIMIT, SIGNED_LOG_LIMIT, D2_GRID_POINTS)
    idx = np.clip(np.searchsorted(knot_t, grid_t, side="right") - 1, 0, knots.shape[1] - 2)
    frac = (grid_t - knot_t[idx]) / (knot_t[idx + 1] - knot_t[idx])
    log_y = knots[:, idx] + frac * (knots[:, idx + 1] - knots[:, idx])
    return grid_t, log_y.mean(axis=0), np.maximum(log_y.std(axis=0), 1e-6)


def d2_log_y_moments(xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid_t, mean, sd = _d2_table()
    x = np.asarray(xs, float)
    t = np.clip(np.sign(x) * np.log1p(np.abs(x)), -SIGNED_LOG_LIMIT, SIGNED_LOG_LIMIT)
    return np.interp(t, grid_t, mean), np.interp(t, grid_t, sd)


def d2_log_likelihood(y: float, xs: np.ndarray) -> np.ndarray:
    """Gaussian approximation of log h(x) under the known distribution of D2's random monotone map."""
    mean, sd = d2_log_y_moments(xs)
    return -0.5 * ((np.log(y) - mean) / sd) ** 2 - np.log(sd)


# ---- D2 joint model: one secret h is shared by all fields of a prompt -------------------------
D2_PAIR_JITTER = 1e-8  # covers display rounding (>= 6 significant digits) and keeps cov invertible


@lru_cache(maxsize=None)
def _d2_knot_moments(law: tuple[float, float, int, float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean vector and covariance matrix of the log-target knots, plus the knot positions on t,
    for the map law ``law`` (OrderPreserving.law; None = the default law)."""
    rng = np.random.default_rng(D2_SIM_SEED + 1)
    defense = OrderPreserving(*law) if law is not None else OrderPreserving()
    knots = np.array([defense.sample_secret(rng, ()).knot_log_targets for _ in range(4 * D2_SIM_SECRETS)])
    knot_t = np.linspace(-SIGNED_LOG_LIMIT, SIGNED_LOG_LIMIT, knots.shape[1])
    return knots.mean(axis=0), np.cov(knots, rowvar=False), knot_t


def _interp_weights(xs: np.ndarray, law=None) -> tuple[np.ndarray, np.ndarray]:
    """log h(x) = (1 - f) K[j] + f K[j + 1]: return (j, f) for each x."""
    _, _, knot_t = _d2_knot_moments(law)
    x = np.asarray(xs, float)
    t = np.clip(np.sign(x) * np.log1p(np.abs(x)), -SIGNED_LOG_LIMIT, SIGNED_LOG_LIMIT)
    j = np.clip(np.searchsorted(knot_t, t, side="right") - 1, 0, knot_t.size - 2)
    return j, (t - knot_t[j]) / (knot_t[j + 1] - knot_t[j])


def _cross_cov(j1, f1, j2, f2, law=None) -> np.ndarray:
    _, cov, _ = _d2_knot_moments(law)
    return ((1 - f1) * (1 - f2) * cov[j1, j2] + (1 - f1) * f2 * cov[j1, j2 + 1]
            + f1 * (1 - f2) * cov[j1 + 1, j2] + f1 * f2 * cov[j1 + 1, j2 + 1])


def _law(defense: OrderPreserving | None):
    return None if defense is None or defense.law == OrderPreserving().law else defense.law


def d2_pair_moments(ps: np.ndarray, cs: np.ndarray, defense: OrderPreserving | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean (n, 2) and covariance (n, 2, 2) of (log h(p), log h(c)) under D2's secret distribution
    (``defense``'s map law; None = the default law)."""
    law = _law(defense)
    mu, _, _ = _d2_knot_moments(law)
    jp, fp = _interp_weights(ps, law)
    jc, fc = _interp_weights(cs, law)
    mean = np.stack([(1 - fp) * mu[jp] + fp * mu[jp + 1], (1 - fc) * mu[jc] + fc * mu[jc + 1]], axis=1)
    vpp = _cross_cov(jp, fp, jp, fp, law)
    vcc, vpc = _cross_cov(jc, fc, jc, fc, law), _cross_cov(jp, fp, jc, fc, law)
    cov = np.empty((mean.shape[0], 2, 2))
    cov[:, 0, 0], cov[:, 1, 1] = vpp + D2_PAIR_JITTER, vcc + D2_PAIR_JITTER
    cov[:, 0, 1] = cov[:, 1, 0] = vpc
    return mean, cov


def d2_pair_log_likelihood(y_price: float, y_cost: float, ps: np.ndarray, cs: np.ndarray,
                           defense: OrderPreserving | None = None) -> np.ndarray:
    """Bivariate Gaussian log-likelihood of the shown (price, cost) pair under one shared secret h."""
    mean, cov = d2_pair_moments(ps, cs, defense)
    d = np.log([y_price, y_cost])[None, :] - mean
    det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] ** 2
    quad = (cov[:, 1, 1] * d[:, 0] ** 2 - 2 * cov[:, 0, 1] * d[:, 0] * d[:, 1] + cov[:, 0, 0] * d[:, 1] ** 2) / det
    return -0.5 * quad - 0.5 * np.log(det)
