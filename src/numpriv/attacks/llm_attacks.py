"""A2 (LLM, prose only) and A3 (LLM end-to-end) attackers.

Both return point estimates plus 80% intervals. A model failure is not dropped:
the product gets NaN estimates and a status, and scoring counts it as a miss.
"""

import math
from dataclasses import dataclass

from numpriv.attacks.llm_prompts import (
    END_TO_END_SYSTEM,
    PROSE_ONLY_SYSTEM,
    end_to_end_user,
    estimate_schema,
    prose_only_user,
)
from numpriv.attacks.view import AttackerView, Estimate
from numpriv.llm import LLMClient, LLMError

END_TO_END_MAX_TOKENS = 1024
_MARGIN_CLIP = (0.0, 0.999)
_NAN = float("nan")


def _missing(i: int, status: str) -> Estimate:
    return Estimate(i, _NAN, _NAN, status=status)


def _ordered(a: float, b: float, c: float) -> tuple[float, float, float]:
    lo, mid, hi = sorted((a, b, c))
    return lo, mid, hi


def _margin(pct: float) -> float:
    return min(max(pct / 100.0, _MARGIN_CLIP[0]), _MARGIN_CLIP[1])


def parse_estimates(obj: dict, n_products: int) -> tuple[Estimate, ...]:
    """Schema-valid LLM output -> Estimates. Non-finite values or non-positive costs mark that product invalid."""
    out = []
    for i in range(n_products):
        m = [float(obj[f"p{i}_margin_pct{s}"]) for s in ("", "_lo", "_hi")]
        c = [float(obj[f"p{i}_unit_cost{s}"]) for s in ("", "_lo", "_hi")]
        if not all(math.isfinite(v) for v in m + c) or min(c) <= 0:
            out.append(_missing(i, "invalid_output"))
            continue
        m_lo, _, m_hi = _ordered(*m)
        c_lo, _, c_hi = _ordered(*c)
        out.append(Estimate(i, _margin(m[0]), c[0], (_margin(m_lo), _margin(m_hi)), (c_lo, c_hi)))
    return tuple(out)


def _ask(llm: LLMClient, system: str, user: str, n_products: int, reasoning: bool,
         max_tokens: int | None) -> tuple[Estimate, ...]:
    try:
        obj = llm.generate_json(system, user, estimate_schema(n_products, reasoning), max_tokens=max_tokens)
    except LLMError:  # already logged by the client; recorded per product as a miss
        return tuple(_missing(i, "llm_failed") for i in range(n_products))
    return parse_estimates(obj, n_products)


@dataclass(frozen=True)
class ProseOnlyAttack:
    """A2: reads the prompt with every number masked, so its output does not depend on the defense."""

    llm: LLMClient
    name: str = "A2_llm_prose_only"

    def attack(self, view: AttackerView) -> tuple[Estimate, ...]:
        return _ask(self.llm, PROSE_ONLY_SYSTEM, prose_only_user(view), view.n_products, False, None)


@dataclass(frozen=True)
class EndToEndAttack:
    """A3: reads the full disguised prompt plus a description of the defense algorithm."""

    llm: LLMClient
    defense: object
    name: str = "A3_llm_end_to_end"
    max_tokens: int = END_TO_END_MAX_TOKENS

    def attack(self, view: AttackerView) -> tuple[Estimate, ...]:
        if view.defense_name != self.defense.name:
            raise ValueError(f"attack built for {self.defense.name}, view is {view.defense_name}")
        user = end_to_end_user(view, self.defense)
        return _ask(self.llm, END_TO_END_SYSTEM, user, view.n_products, True, self.max_tokens)
