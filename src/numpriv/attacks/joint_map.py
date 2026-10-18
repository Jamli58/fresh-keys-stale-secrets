"""Joint shared-map attacker for D2: uses every product value shown under the same secret map.

The pair-only multi-query attacker (multi_query.py) scores product 0's (price, cost) pair alone. Under D2 the
other products a prompt shows pass through the same map h, so their shown values carry information about h and
hence about product 0. This attacker models all product prices and costs shown under one map jointly: their
log h values are approximated as multivariate Gaussian (mean and covariance from the simulated knot moments of
the map law, as for the pair model), and the unknown values of products other than product 0 are integrated
over their oracle class priors by Monte Carlo. Fields without a prior in the attacker's table (volumes,
overhead, quotes, order sizes) are not modelled.

Groups of values that share a map: under fresh maps each query is its own group; under a reused map the distinct
products shown anywhere in the sequence form one group (deterministic repeats are counted once). The other
products' values are unchanged across queries, so one set of Monte Carlo draws z_s per product is shared by all
groups: L(theta) = mean_s prod_groups q_g(Y_g | theta, z_s), i.e. groups are conditionally independent given
theta AND the persistent nuisance values, and the integral over z is taken once over the whole history.
A group that shows only product 0 does not depend on z and is scored by the pair model, so on such queries the
attacker equals the pair-only attacker.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.attacks.likelihoods import D2_PAIR_JITTER, _d2_knot_moments, _interp_weights, _law
from numpriv.attacks.multi_query import MultiQueryEstimate
from numpriv.attacks.prior_fit import weighted_median
from numpriv.attacks.view import AttackerView
from numpriv.defenses import OrderPreserving

LOG = logging.getLogger(__name__)
NUISANCE_CHUNK = 25
REGIMES = ("fresh", "reuse")


def d2_joint_moments(xs: np.ndarray, defense: OrderPreserving | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean (B, m) and covariance (B, m, m) of log h at the m points of each row of xs (B, m)."""
    law = _law(defense)
    mu, cov_k, _ = _d2_knot_moments(law)
    xs = np.asarray(xs, float)
    j, f = (a.reshape(xs.shape) for a in _interp_weights(xs.ravel(), law))
    mean = (1 - f) * mu[j] + f * mu[j + 1]
    ja, jb = j[:, :, None], j[:, None, :]
    fa, fb = f[:, :, None], f[:, None, :]
    cov = ((1 - fa) * (1 - fb) * cov_k[ja, jb] + (1 - fa) * fb * cov_k[ja, jb + 1]
           + fa * (1 - fb) * cov_k[ja + 1, jb] + fa * fb * cov_k[ja + 1, jb + 1])
    cov = cov + D2_PAIR_JITTER * np.eye(xs.shape[1])
    return mean, cov


def _mvn_log_density(y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(cov)
    diff = (y[None, :] - mean)[..., None]
    sol = np.linalg.solve(chol, diff)[..., 0]
    log_det = 2.0 * np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=-1)
    return -0.5 * (sol ** 2).sum(axis=-1) - 0.5 * log_det - 0.5 * y.size * np.log(2 * np.pi)


def _shown_products(view: AttackerView) -> dict[int, tuple[float, float]]:
    vals = {f.field_id: f.value for f in view.shown}
    return {i: (vals[f"price:{i}"], vals[f"cost:{i}"]) for i in range(view.n_products)}


@dataclass(frozen=True)
class JointMapAttack:
    solver: ClassicalBayesAttack  # frozen A1+ (use_oracle=True) with an OrderPreserving defense
    regime: str
    n_nuisance: int = 400  # Monte Carlo draws of the persistent other-product values
    seed: int = 777
    # Observation smoothing (sd in log units) added to every modelled value. Within one map interval log h is
    # affine in t, so values sharing an interval are almost collinear and the exact Gaussian model is nearly
    # singular; its integral over the other products' priors is then too thin for Monte Carlo. The smoothing
    # makes the integral resolvable (same order as the reference likelihood's kernel bandwidth, 0.03).
    obs_sd: float = 0.03

    def __post_init__(self) -> None:
        if not isinstance(self.solver.defense, OrderPreserving):
            raise ValueError("JointMapAttack is for D2 (OrderPreserving)")
        if not self.solver.use_oracle:
            raise ValueError("build the solver with use_oracle=True")
        if self.regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}, got {self.regime!r}")
        if not self.obs_sd > 0:
            raise ValueError(f"obs_sd must be > 0, got {self.obs_sd}")

    def _smoothed(self, cov: np.ndarray) -> np.ndarray:
        return cov + self.obs_sd ** 2 * np.eye(cov.shape[-1])

    def _groups(self, views: tuple[AttackerView, ...]) -> list[tuple[dict[int, tuple[float, float]], tuple[str, ...]]]:
        if self.regime == "fresh":
            return [(_shown_products(v), v.oracle.product_classes) for v in views]
        union, classes = {}, {}
        for v in views:
            for i, pc in _shown_products(v).items():
                union.setdefault(i, pc)
                classes.setdefault(i, v.oracle.product_classes[i])
        return [(union, tuple(classes[i] for i in sorted(classes)))]

    def _nuisance(self, sector: str, classes: dict[int, str], rng: np.random.Generator) -> dict[int, np.ndarray]:
        """Shared Monte Carlo draws (S, 2) of (price, cost) for every product other than product 0."""
        s = self.solver
        out = {}
        for i in sorted(classes):
            comp = s.component(sector, classes[i])
            size = (2, self.n_nuisance)
            z = rng.standard_normal(size) if s.prior_df is None else rng.standard_t(s.prior_df, size=size)
            price = np.exp(comp.price[0] + comp.price[1] * z[0])
            margin = 1 / (1 + np.exp(-(comp.margin[0] + comp.margin[1] * z[1])))
            out[i] = np.stack([price, price * (1 - margin)], axis=1)
        return out

    def _group_matrix(self, shown, others, nuisance, price, cost) -> np.ndarray:
        """(N, S) log q_g(Y_g | theta_n, z_s) for a group that shows product 0 and the products in ``others``."""
        p0, c0 = shown[0]
        y = np.log([p0, c0] + [v for i in others for v in shown[i]])
        nuis = np.concatenate([nuisance[i] for i in others], axis=1)  # (S, 2 * len(others))
        blocks = []
        for start in range(0, self.n_nuisance, NUISANCE_CHUNK):
            chunk = nuis[start:start + NUISANCE_CHUNK]
            n, k = price.size, chunk.shape[0]
            xs = np.concatenate([np.repeat(np.stack([price, cost], axis=1), k, axis=0),
                                 np.tile(chunk, (n, 1))], axis=1)
            mean, cov = d2_joint_moments(xs, self.solver.defense)
            blocks.append(_mvn_log_density(y, mean, self._smoothed(cov)).reshape(n, k))
        return np.concatenate(blocks, axis=1)

    def _target_draws(self, views):
        s = self.solver
        comp = s.component(views[0].oracle.sector, views[0].oracle.product_classes[0])
        z = s.prior_draws()
        price = np.exp(comp.price[0] + comp.price[1] * z[0])
        margin = 1 / (1 + np.exp(-(comp.margin[0] + comp.margin[1] * z[1])))
        return price, margin, price * (1 - margin)

    def log_likelihood(self, views: tuple[AttackerView, ...]) -> np.ndarray:
        """log L(theta_n) for the solver's prior draws of product 0 (persistent nuisance integrated once)."""
        if not views:
            raise ValueError("need at least one query")
        price, _, cost = self._target_draws(views)
        groups = self._groups(views)
        classes = {}
        for v in views:
            for i in range(1, v.n_products):
                classes.setdefault(i, v.oracle.product_classes[i])
        nuisance = self._nuisance(views[0].oracle.sector, classes, np.random.default_rng(self.seed))
        ll_free = np.zeros_like(price)
        ll_shared = None
        for shown, _ in groups:
            others = sorted(i for i in shown if i != 0)
            if not others:
                p0, c0 = shown[0]
                ll_free = ll_free + self.solver.pair_log_likelihood(views[0], p0, c0, price, cost)
                continue
            m = self._group_matrix(shown, others, nuisance, price, cost)
            ll_shared = m if ll_shared is None else ll_shared + m
        if ll_shared is None:
            return ll_free
        return ll_free + logsumexp(ll_shared, axis=1) - np.log(self.n_nuisance)

    def attack(self, views: tuple[AttackerView, ...]) -> MultiQueryEstimate:
        ll = self.log_likelihood(views)
        _, margin, cost = self._target_draws(views)
        norm = logsumexp(ll)
        if np.isfinite(norm):
            w = np.exp(ll - norm)
        else:
            LOG.warning("no prior sample consistent with %d queries; using the prior", len(views))
            w = np.ones_like(ll)
        return MultiQueryEstimate(weighted_median(margin, w), weighted_median(cost, w))
