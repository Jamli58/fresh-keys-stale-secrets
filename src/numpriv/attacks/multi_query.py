"""B2: A1+ over a firm's linked sequence of queries; target is product 0 (shown in every query).

Same frozen prior, draws and likelihoods as the single-query A1+, so n = 1 reproduces it.
  D1 (shared scaling): every query fixes the price ratios exactly; key k_j of query j is
     k_1 * r_j with r_j = shown price_0 in j / shown price_0 in query 1 (r_j = 1 under reuse).
     Grid posterior over log k_1 = sum of the distinct products' price priors, restricted to
     the k_1 for which every k_j lies in the public key range [k_low, k_high]. Fresh keys
     therefore shrink the support; a reused key adds only the products new queries show.
  D1' / D2 / D3: product 0's log-likelihood summed over its observations. Under a reused
     key the repeats are deterministic copies and are counted once; D3 releases are
     independent noise draws and all count.
Simplifications carried over from A1+: overhead and quote fields are unused, and D2's
shared map is not exploited across different values.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.attacks.prior_fit import weighted_median
from numpriv.attacks.view import AttackerView
from numpriv.defenses import NoDefense, SharedScaling
from numpriv.sequences import REGIMES

LOG = logging.getLogger(__name__)
HALF_CENT = 0.005
_M_CLIP = 1e-4


@dataclass(frozen=True)
class MultiQueryEstimate:
    margin: float
    unit_cost: float
    key: float | None = None  # D1: estimate of the first query's key


def key_support(grid: np.ndarray, log_ratios: np.ndarray, k_low: float, k_high: float,
                tol: np.ndarray | float = 0.0) -> np.ndarray:
    """Mask of log k_1 values for which every k_1 * r_j stays inside [k_low, k_high] (up to tol)."""
    shifted = grid[None, :] + np.asarray(log_ratios, float)[:, None]
    t = np.broadcast_to(np.asarray(tol, float), np.shape(log_ratios))[:, None]
    return np.all((shifted >= np.log(k_low) - t) & (shifted <= np.log(k_high) + t), axis=0)


def _values(view: AttackerView) -> dict[str, float]:
    return {f.field_id: f.value for f in view.shown}


@dataclass(frozen=True)
class MultiQueryAttack:
    solver: ClassicalBayesAttack  # frozen A1 config with use_oracle=True
    regime: str

    def __post_init__(self) -> None:
        if not self.solver.use_oracle:
            raise ValueError("B2 runs A1+: build the solver with use_oracle=True")
        if self.regime not in REGIMES:
            raise ValueError(f"regime must be one of {sorted(REGIMES)}, got {self.regime!r}")

    def attack(self, views: tuple[AttackerView, ...]) -> MultiQueryEstimate:
        if not views:
            raise ValueError("need at least one query")
        first = _values(views[0])
        p0, c0 = first["price:0"], first["cost:0"]
        if isinstance(self.solver.defense, NoDefense):
            return MultiQueryEstimate((p0 - c0) / p0, c0)
        if isinstance(self.solver.defense, SharedScaling):
            return self._shared_scaling(views)
        return self._sampling(views)

    def _shared_scaling(self, views: tuple[AttackerView, ...]) -> MultiQueryEstimate:
        d, s = self.solver.defense, self.solver
        grid = np.linspace(np.log(d.k_low), np.log(d.k_high), s.grid_points)
        ref = _values(views[0])["price:0"]
        log_ratios, tols, scaled_prices = [], [], {}
        for view in views:
            vals = _values(view)
            ratio = vals["price:0"] / ref
            log_ratios.append(np.log(ratio))
            tols.append(HALF_CENT / vals["price:0"] + HALF_CENT / ref)
            for i, cls in enumerate(view.oracle.product_classes):
                scaled_prices.setdefault(i, (vals[f"price:{i}"] / ratio, cls))
        sector = views[0].oracle.sector
        log_post = np.zeros_like(grid)
        for p_scaled, cls in scaled_prices.values():
            mu, scale = s.component(sector, cls).price
            log_post += s.log_prior(np.log(p_scaled) - grid, mu, scale)
        mask = key_support(grid, np.array(log_ratios), d.k_low, d.k_high, np.array(tols))
        if not mask.any():
            LOG.warning("no key is consistent with every query; ignoring the key-range constraint")
            mask = np.ones_like(mask)
        log_post = np.where(mask, log_post, -np.inf)
        w = np.exp(log_post - logsumexp(log_post))
        k_hat = float(np.exp(weighted_median(grid, w)))
        first = _values(views[0])
        margin = float(np.clip(1 - first["cost:0"] / first["price:0"], _M_CLIP, 1 - _M_CLIP))
        return MultiQueryEstimate(margin, first["cost:0"] / k_hat, k_hat)

    def _sampling(self, views: tuple[AttackerView, ...]) -> MultiQueryEstimate:
        s = self.solver
        comp = s.component(views[0].oracle.sector, views[0].oracle.product_classes[0])
        z = s.prior_draws()
        price = np.exp(comp.price[0] + comp.price[1] * z[0])
        margin = 1 / (1 + np.exp(-(comp.margin[0] + comp.margin[1] * z[1])))
        cost = price * (1 - margin)
        observed = views[:1] if self.regime == "reuse" else views
        ll = np.zeros_like(price)
        for view in observed:
            vals = _values(view)
            ll = ll + s.pair_log_likelihood(view, vals["price:0"], vals["cost:0"], price, cost)
        norm = logsumexp(ll)
        if np.isfinite(norm):
            w = np.exp(ll - norm)
        else:
            LOG.warning("no prior sample consistent with %d queries; using the prior", len(observed))
            w = np.ones_like(ll)
        return MultiQueryEstimate(weighted_median(margin, w), weighted_median(cost, w))
