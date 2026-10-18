"""Typed numeric fields extracted from a prompt, plus a disguised view with its restore map."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

FieldKind = Literal["money", "count", "year", "day", "percent"]
DISPLAY_SIG_DIGITS = 6


@dataclass(frozen=True)
class NumField:
    field_id: str
    kind: FieldKind
    value: float


@dataclass(frozen=True)
class Disguised:
    """What the cloud sees (``fields``) plus the client-only map back to the originals."""

    fields: tuple[NumField, ...]
    restore: Mapping[str, float]

    def recover(self) -> tuple[NumField, ...]:
        return tuple(NumField(f.field_id, f.kind, self.restore[f.field_id]) for f in self.fields)


def disguised(original: tuple[NumField, ...], new_values: Mapping[str, float]) -> Disguised:
    """Build a Disguised view; fields missing from ``new_values`` pass through unchanged."""
    fields = tuple(NumField(f.field_id, f.kind, new_values.get(f.field_id, f.value)) for f in original)
    restore = MappingProxyType({f.field_id: f.value for f in original})
    return Disguised(fields, restore)


def round_sig(x: float, digits: int = DISPLAY_SIG_DIGITS) -> float:
    if x == 0 or not np.isfinite(x):
        return float(x)
    return float(round(x, digits - 1 - int(np.floor(np.log10(abs(x))))))
