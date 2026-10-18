"""Turn P_pub quantile summaries into parametric priors, plus a weighted-median helper."""

import numpy as np
from scipy import stats

from numpriv.priors import Summary

IQR_TO_SIGMA = 1.349  # IQR of a standard normal
MIN_SIGMA = 0.05


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def fit_log_normal(s: Summary) -> tuple[float, float]:
    """(mu, sigma) of log X from median and IQR."""
    sigma = (np.log(s.q75) - np.log(s.q25)) / IQR_TO_SIGMA
    return float(np.log(s.median)), float(max(sigma, MIN_SIGMA))


def fit_logit_normal(s: Summary) -> tuple[float, float]:
    """(mu, sigma) of logit M from median and IQR."""
    sigma = (_logit(s.q75) - _logit(s.q25)) / IQR_TO_SIGMA
    return _logit(s.median), float(max(sigma, MIN_SIGMA))


def t_scale_from_normal_sigma(sigma: float, df: float) -> float:
    """Student-t scale with the same IQR as a normal with standard deviation ``sigma``."""
    return float(sigma * stats.norm.ppf(0.75) / stats.t.ppf(0.75, df))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if weights.sum() <= 0:
        raise ValueError("weights must have positive total")
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, 0.5 * cum[-1])])


def _best_window(sorted_x: np.ndarray, sorted_w: np.ndarray, width: float) -> int:
    """Index i maximizing the weight in [x_i, x_i + width] (an optimal window can start at a sample)."""
    cum = np.concatenate(([0.0], np.cumsum(sorted_w)))
    right = np.searchsorted(sorted_x, sorted_x + width, side="right")
    return int(np.argmax(cum[right] - cum[np.arange(sorted_x.size)]))


def hit_optimal_absolute(values: np.ndarray, weights: np.ndarray, tol: float) -> float:
    """Estimate g maximizing the weighted share of values with |value - g| <= tol (margin hit rule)."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if weights.sum() <= 0:
        raise ValueError("weights must have positive total")
    order = np.argsort(values)
    x, w = values[order], weights[order]
    return float(x[_best_window(x, w, 2.0 * tol)] + tol)


def hit_optimal_relative(values: np.ndarray, weights: np.ndarray, tol: float) -> float:
    """Estimate g maximizing the weighted share of positive values c with |g - c| <= tol * c (cost hit rule).

    The hit set {c : g / (1 + tol) <= c <= g / (1 - tol)} is a window of width log((1 + tol) / (1 - tol)) in log c.
    """
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if weights.sum() <= 0:
        raise ValueError("weights must have positive total")
    order = np.argsort(values)
    x, w = np.log(values[order]), weights[order]
    return float(np.exp(x[_best_window(x, w, float(np.log((1 + tol) / (1 - tol))))]) * (1 + tol))
