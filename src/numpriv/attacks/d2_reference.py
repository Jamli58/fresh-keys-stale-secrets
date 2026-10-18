"""Simulation-based reference likelihood for D2 (validates the attacker's Gaussian approximation).

The A1+ attacker scores a shown (price, cost) pair under D2 with a bivariate Gaussian approximation of
(log h(p), log h(c)) over the random map h. This module replaces that approximation by a kernel density
estimate over simulated maps: for each candidate (p, c), the shown log pair is compared with
(log h_m(p), log h_m(c)) for m = 1..n_maps maps drawn from the same public law, with a Gaussian kernel
of the given bandwidth. As n_maps grows and the bandwidth shrinks this converges to the exact likelihood
of the displayed values (up to their >= 6 significant digits), so it serves as a stronger reference
inference for the same prior and decision rule.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.defenses import OrderPreserving
from numpriv.defenses.order_preserving import SIGNED_LOG_LIMIT

MAP_CHUNK = 2000  # maps per vectorized block (memory: chunk x candidates x 4 bytes x 2)


@dataclass(frozen=True)
class SimulatedMapLikelihood:
    defense: OrderPreserving
    n_maps: int = 20_000
    bandwidth: float = 0.03
    seed: int = 54321  # the attacker's own simulation, independent of the defense's randomness
    _knots: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.n_maps < 1:
            raise ValueError(f"n_maps must be >= 1, got {self.n_maps}")
        if not self.bandwidth > 0:
            raise ValueError(f"bandwidth must be > 0, got {self.bandwidth}")
        rng = np.random.default_rng(self.seed)
        knots = np.array([self.defense.sample_secret(rng, ()).knot_log_targets for _ in range(self.n_maps)],
                         dtype=np.float32)
        object.__setattr__(self, "_knots", knots)

    def _positions(self, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        knot_t = np.linspace(-SIGNED_LOG_LIMIT, SIGNED_LOG_LIMIT, self._knots.shape[1])
        x = np.asarray(xs, float)
        t = np.clip(np.sign(x) * np.log1p(np.abs(x)), -SIGNED_LOG_LIMIT, SIGNED_LOG_LIMIT)
        j = np.clip(np.searchsorted(knot_t, t, side="right") - 1, 0, knot_t.size - 2)
        return j, ((t - knot_t[j]) / (knot_t[j + 1] - knot_t[j])).astype(np.float32)

    def pair_log_likelihood(self, y_price: float, y_cost: float, ps: np.ndarray, cs: np.ndarray) -> np.ndarray:
        """log of the kernel density of (log y_price, log y_cost) given each candidate (p, c)."""
        jp, fp = self._positions(ps)
        jc, fc = self._positions(cs)
        up, uc = np.float32(np.log(y_price)), np.float32(np.log(y_cost))
        inv = np.float32(1.0 / (2.0 * self.bandwidth ** 2))
        blocks = []
        for start in range(0, self.n_maps, MAP_CHUNK):
            k = self._knots[start:start + MAP_CHUNK]
            lp = (1 - fp) * k[:, jp] + fp * k[:, jp + 1]
            lc = (1 - fc) * k[:, jc] + fc * k[:, jc + 1]
            blocks.append(logsumexp(-((up - lp) ** 2 + (uc - lc) ** 2) * inv, axis=0))
        log_sum = logsumexp(np.stack(blocks), axis=0)
        return log_sum - np.log(self.n_maps) - 2.0 * np.log(self.bandwidth * np.sqrt(2.0 * np.pi))


@dataclass(frozen=True)
class SimulatedD2BayesAttack(ClassicalBayesAttack):
    """A1+ with D2's Gaussian pair likelihood replaced by the simulation-based reference likelihood."""

    reference: SimulatedMapLikelihood | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.defense, OrderPreserving) or self.reference is None:
            raise ValueError("SimulatedD2BayesAttack needs an OrderPreserving defense and a reference likelihood")
        if self.reference.defense.law != self.defense.law:
            raise ValueError("the reference likelihood must simulate the defense's own map law")

    def _pair_log_lik(self, view, p_shown: float, c_shown: float, price: np.ndarray, cost: np.ndarray) -> np.ndarray:
        return self.reference.pair_log_likelihood(p_shown, c_shown, price, cost)
