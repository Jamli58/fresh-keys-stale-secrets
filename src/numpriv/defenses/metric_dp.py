"""D3: value-token sanitization of Prεεmpt (NDSS'26, arXiv 2504.05147v2).

ℓ1 exponential mechanism, density ∝ exp(-ε'|x-y|/2) on [lo, hi] (continuous
truncated-Laplace form, scale b = 2/ε'), which is ε'-metric-DP. The prompt
budget ε is split evenly: ε' = ε / t, t = number of value tokens (paper Alg. 1).
Logged assumptions (research contract):
  A-D3-1 continuous form of the mechanism; outputs rounded for display (post-processing);
  A-D3-2 domain predefined per field kind (defaults below);
  A-D3-3 distance measured in the number's displayed unit;
  A-D3-4 years and days are not value tokens (left unchanged).
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from numpriv.fields import Disguised, NumField, disguised

DEFAULT_DOMAIN = MappingProxyType({"money": (0.0, 50_000.0), "count": (0.0, 20_000.0), "percent": (0.0, 100.0)})
_DISPLAY_DECIMALS = MappingProxyType({"money": 2, "count": 0, "percent": 2})


@dataclass(frozen=True)
class MetricDP:
    epsilon: float
    domain: Mapping[str, tuple[float, float]] = field(default=DEFAULT_DOMAIN)
    name: str = "D3_prempt_metric_dp"

    def __post_init__(self) -> None:
        if not self.epsilon > 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")

    @property
    def value_kinds(self) -> frozenset[str]:
        return frozenset(self.domain)

    def per_token_epsilon(self, fields: tuple[NumField, ...]) -> float:
        t = sum(1 for f in fields if f.kind in self.value_kinds)
        return self.epsilon / max(t, 1)

    def sample_secret(self, rng: np.random.Generator, fields: tuple[NumField, ...]) -> None:
        return None  # fresh randomness per release; nothing to reuse

    def apply(self, fields: tuple[NumField, ...], secret: None, rng: np.random.Generator) -> Disguised:
        eps = self.per_token_epsilon(fields)
        new_values = {}
        for f in fields:
            if f.kind in self.value_kinds:
                lo, hi = self.domain[f.kind]
                y = self.sample_value(f.value, eps, lo, hi, rng)
                new_values[f.field_id] = float(np.clip(round(y, _DISPLAY_DECIMALS[f.kind]), lo, hi))
        return disguised(fields, new_values)

    @staticmethod
    def sample_value(x: float, eps: float, lo: float, hi: float, rng: np.random.Generator) -> float:
        x = float(np.clip(x, lo, hi))
        b = 2.0 / eps
        u_lo, u_hi = _laplace_cdf(lo, x, b), _laplace_cdf(hi, x, b)
        u = rng.uniform(u_lo, u_hi)
        y = x + b * np.log(2.0 * u) if u < 0.5 else x - b * np.log(2.0 * (1.0 - u))
        return float(np.clip(y, lo, hi))

    @staticmethod
    def log_density(y: float, x: float, eps: float, lo: float, hi: float) -> float:
        x = float(np.clip(x, lo, hi))
        b = 2.0 / eps
        z = b * (2.0 - np.exp(-(x - lo) / b) - np.exp(-(hi - x) / b))
        return float(-abs(y - x) / b - np.log(z))


def _laplace_cdf(y: float, x: float, b: float) -> float:
    return 0.5 * np.exp((y - x) / b) if y < x else 1.0 - 0.5 * np.exp(-(y - x) / b)
