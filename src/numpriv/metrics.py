"""Leakage metrics: hit rates, MAE, and firm-clustered bootstrap confidence intervals."""

import numpy as np


def hit_rate(
    estimates: np.ndarray, truths: np.ndarray, abs_tol: float | None = None, rel_tol: float | None = None
) -> float:
    """Share of estimates within ``abs_tol`` (absolute) or ``rel_tol`` (relative to truth)."""
    if (abs_tol is None) == (rel_tol is None):
        raise ValueError("specify exactly one of abs_tol or rel_tol")
    err = np.abs(np.asarray(estimates, float) - np.asarray(truths, float))
    bound = abs_tol if abs_tol is not None else rel_tol * np.abs(np.asarray(truths, float))
    return float(np.mean(err <= bound))


def mean_abs_error(estimates: np.ndarray, truths: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(estimates, float) - np.asarray(truths, float))))


def bootstrap_ci(
    values: np.ndarray, groups: np.ndarray, n_boot: int = 10_000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile CI of the mean, resampling whole groups (firms) with replacement."""
    values, groups = np.asarray(values, float), np.asarray(groups)
    unique = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(unique, size=unique.size, replace=True)
        means[b] = values[np.concatenate([index[g] for g in picked])].mean()
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, (alpha, 1.0 - alpha))
    return float(lo), float(hi)
