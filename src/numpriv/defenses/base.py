"""Common interface for numeric disguises.

A defense separates the reusable *secret* (key) from per-prompt application,
so the key-reuse regime (B2) is just calling ``apply`` with the same secret.
"""

from typing import Any, Protocol

import numpy as np

from numpriv.fields import Disguised, NumField, disguised


class Defense(Protocol):
    name: str

    def sample_secret(self, rng: np.random.Generator, fields: tuple[NumField, ...]) -> Any: ...

    def apply(self, fields: tuple[NumField, ...], secret: Any, rng: np.random.Generator) -> Disguised: ...


class NoDefense:
    """D0: plaintext numbers (upper bound on leakage)."""

    name = "D0_none"

    def sample_secret(self, rng: np.random.Generator, fields: tuple[NumField, ...]) -> None:
        return None

    def apply(self, fields: tuple[NumField, ...], secret: None, rng: np.random.Generator) -> Disguised:
        return disguised(fields, {})
