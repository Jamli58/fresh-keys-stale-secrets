"""D1 shared scaling and D1' per-field scaling of money fields.

Scaled money is shown in whole cents, like ordinary prices. Showing more
digits would expose the originals' cent grid: every shown/k would have to be a
whole number of cents, which pins down k (code review, M0). The cost is that
inverting a shown or derived value is exact only up to rounding.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from numpriv.fields import Disguised, NumField, disguised

SCALED_KINDS = frozenset({"money"})


def _to_cents(value: float) -> float:
    return float(round(value, 2))


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


@dataclass(frozen=True)
class SharedKey:
    k: float


@dataclass(frozen=True)
class SharedScaling:
    """D1: every money field multiplied by one secret k ~ LogUniform[k_low, k_high]."""

    k_low: float = 0.1
    k_high: float = 10.0
    name: str = "D1_shared_scaling"

    def sample_secret(self, rng: np.random.Generator, fields: tuple[NumField, ...]) -> SharedKey:
        return SharedKey(_log_uniform(rng, self.k_low, self.k_high))

    def apply(self, fields: tuple[NumField, ...], secret: SharedKey, rng: np.random.Generator) -> Disguised:
        return disguised(
            fields, {f.field_id: _to_cents(f.value * secret.k) for f in fields if f.kind in SCALED_KINDS}
        )

    def invert_money(self, value: float, secret: SharedKey) -> float:
        """Invert a derived money-denominated quantity (degree-1 in money)."""
        return value / secret.k


@dataclass(frozen=True)
class PerFieldKey:
    k_by_field: Mapping[str, float]


@dataclass(frozen=True)
class PerFieldScaling:
    """D1': an independent k per money field — removes the shared-key margin invariant."""

    k_low: float = 0.1
    k_high: float = 10.0
    name: str = "D1p_per_field_scaling"

    def sample_secret(self, rng: np.random.Generator, fields: tuple[NumField, ...]) -> PerFieldKey:
        keys = {f.field_id: _log_uniform(rng, self.k_low, self.k_high) for f in fields if f.kind in SCALED_KINDS}
        return PerFieldKey(MappingProxyType(keys))

    def apply(self, fields: tuple[NumField, ...], secret: PerFieldKey, rng: np.random.Generator) -> Disguised:
        return disguised(
            fields,
            {f.field_id: _to_cents(f.value * secret.k_by_field[f.field_id]) for f in fields if f.kind in SCALED_KINDS},
        )
