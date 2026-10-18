"""B2: one firm's sequence of queries to the provider, disguised under a key regime.

A firm asks n questions in a row, cycling through the task templates; product 0
appears in every one, so it is the B2 target. The provider links the queries by
account (worst case, plan B2). Regimes:
  fresh  a new secret per query (the defense's default);
  reuse  one secret for the whole sequence (D1 / D1' / D2 key reuse);
  repeat D3 has no key: every release draws fresh noise.
A reused secret is sampled on the union of the sequence's fields, so it covers
fields that only some templates show (overhead, supplier quote).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from numpriv.attacks import make_view
from numpriv.attacks.view import AttackerView
from numpriv.records import Firm
from numpriv.templates import TASKS, PromptSpec, build_prompt

REGIMES = frozenset({"fresh", "reuse", "repeat"})


@dataclass(frozen=True)
class DisguisedSequence:
    specs: tuple[PromptSpec, ...]
    views: tuple[AttackerView, ...]  # with the A1+ oracle (sector and product classes)
    secrets: tuple[Any, ...]  # one per query (fresh / repeat) or a single shared one (reuse)


def build_sequence(firm: Firm, n_queries: int, start: int) -> tuple[PromptSpec, ...]:
    if n_queries < 1:
        raise ValueError(f"n_queries must be >= 1, got {n_queries}")
    return tuple(build_prompt(TASKS[(start + j) % len(TASKS)], firm, cues=True) for j in range(n_queries))


def _union_fields(specs: tuple[PromptSpec, ...]) -> tuple:
    seen = {}
    for spec in specs:
        for f in spec.fields:
            seen.setdefault(f.field_id, f)
    return tuple(seen.values())


def disguise_sequence(specs: tuple[PromptSpec, ...], defense, regime: str,
                      rng: np.random.Generator) -> DisguisedSequence:
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {sorted(REGIMES)}, got {regime!r}")
    if regime == "reuse":
        secret = defense.sample_secret(rng, _union_fields(specs))
        secrets = (secret,)
        per_query = [secret] * len(specs)
    else:
        per_query = [defense.sample_secret(rng, spec.fields) for spec in specs]
        secrets = tuple(per_query)
    views = tuple(
        make_view(spec, defense.apply(spec.fields, secret, rng).fields, defense.name, with_oracle=True)
        for spec, secret in zip(specs, per_query)
    )
    return DisguisedSequence(tuple(specs), views, secrets)
