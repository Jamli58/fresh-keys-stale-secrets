"""A4: hybrid attacker, LLM prior -> the frozen A1 Bayesian solver.

The LLM reads the *masked* prose only (never the numbers, so the numbers are not
counted twice) and returns the sector, each product's class, the city's price
level, the price effect of the firm's positioning, and a margin shift. The two price
answers multiply into one firm price multiplier. These pick and shift P_pub
components; the disguised numbers then enter through the same solver as A1/A1+.
With the true sector and classes and no shift, A4 reproduces A1+ exactly, so
Delta = A4 - A1+ measures what the LLM adds beyond the oracle.
Unknown answers fall back to A1's mixture; a failed LLM call falls back to A1.
"""

import logging
import math
from dataclasses import dataclass, replace

import numpy as np

from numpriv.attacks.bayes import ClassicalBayesAttack, Hypotheses, PriorComponent
from numpriv.attacks.llm_prompts import CLASS_LABELS, UNKNOWN, product_note, quoted
from numpriv.attacks.view import AttackerView, Estimate
from numpriv.config import SECTORS
from numpriv.llm import LLMClient, LLMError

LOG = logging.getLogger(__name__)
PRICE_MULTIPLIER_RANGE = (0.25, 4.0)
CITY_PRICE_INDEX_RANGE = (50.0, 200.0)  # 100 = US average; BEA metro RPPs span roughly 80-120
POSITIONING_PRICE_PCT_RANGE = (-60.0, 150.0)
MARGIN_SHIFT_RANGE_PP = (-40.0, 40.0)
MARGIN_MEDIAN_CLIP = (0.01, 0.97)  # the generator's margin bounds
ALL_CLASSES = tuple(c.name for spec in SECTORS.values() for c in spec.product_classes)


def _sector_lines() -> str:
    lines = []
    for key, spec in SECTORS.items():
        classes = "; ".join(f"{c.name} = {CLASS_LABELS[c.name]}" for c in spec.product_classes)
        lines.append(f"- {key}: {spec.display_name} (classes: {classes})")
    return "\n".join(lines)


EXTRACTION_SYSTEM = (
    "You are an analyst at a cloud AI provider. You read messages that small businesses send to an AI assistant. "
    "Every number in the message is hidden as [number]. From the words alone, fill in:\n"
    "- sector: one of the sector ids below, or 'unknown' if the message does not reveal it;\n"
    "- p<i>_class: each product's class id (a class of that sector), or 'unknown';\n"
    "- city_price_index: the overall price level of the city the firm is based in, on a scale where the US "
    "average is 100 (use 100 if the message names no city);\n"
    "- positioning_price_pct: by how many percent this firm's prices differ from a typical small business "
    "selling the same products in the same city, because of its market positioning (0 = typical; negative = "
    "cheaper);\n"
    "- margin_shift_pp: how its gross margin differs from the typical margin in its sector, in percentage "
    "points (0 = typical).\n"
    f"Sectors:\n{_sector_lines()}\nAnswer only with the requested JSON."
)


def extraction_schema(n_products: int) -> dict:
    props: dict[str, dict] = {"sector": {"type": "string", "enum": [*SECTORS, UNKNOWN]}}
    props.update({f"p{i}_class": {"type": "string", "enum": [*ALL_CLASSES, UNKNOWN]} for i in range(n_products)})
    props.update({name: {"type": "number"} for name in ("city_price_index", "positioning_price_pct", "margin_shift_pp")})
    return {"type": "object", "properties": props, "required": list(props)}


@dataclass(frozen=True)
class LLMPrior:
    sector: str | None
    classes: tuple[str | None, ...]
    log_price_shift: float = 0.0
    margin_shift: float = 0.0  # fraction, added to the class median margin


def _clip(x: float, bounds: tuple[float, float], neutral: float) -> float:
    return min(max(x, bounds[0]), bounds[1]) if math.isfinite(x) else neutral


def parse_prior(obj: dict, n_products: int) -> LLMPrior:
    sector = obj["sector"] if obj["sector"] in SECTORS else None
    classes = tuple(obj.get(f"p{i}_class") if obj.get(f"p{i}_class") in ALL_CLASSES else None
                    for i in range(n_products))
    city = _clip(float(obj["city_price_index"]), CITY_PRICE_INDEX_RANGE, 100.0) / 100.0
    positioning = 1.0 + _clip(float(obj["positioning_price_pct"]), POSITIONING_PRICE_PCT_RANGE, 0.0) / 100.0
    mult = _clip(city * positioning, PRICE_MULTIPLIER_RANGE, 1.0)
    shift = _clip(float(obj["margin_shift_pp"]), MARGIN_SHIFT_RANGE_PP, 0.0) / 100.0
    return LLMPrior(sector, classes, float(np.log(mult)), shift)


def _shifted(comp: PriorComponent, prior: LLMPrior) -> PriorComponent:
    mu_p, s_p = comp.price
    mu_m, s_m = comp.margin
    if prior.margin_shift != 0.0:
        median = 1.0 / (1.0 + math.exp(-mu_m)) + prior.margin_shift
        median = min(max(median, MARGIN_MEDIAN_CLIP[0]), MARGIN_MEDIAN_CLIP[1])
        mu_m = math.log(median / (1.0 - median))
    return replace(comp, price=(mu_p + prior.log_price_shift, s_p), margin=(mu_m, s_m))


def build_hypotheses(solver: ClassicalBayesAttack, prior: LLMPrior) -> Hypotheses:
    """Known sector/class -> that P_pub component; unknown -> A1's uniform mixture. Shifts apply to all."""
    by_sector: dict[str, list[str]] = {}
    for sector, cls in solver.table.by_class:
        by_sector.setdefault(sector, []).append(cls)
    sectors = [prior.sector] if prior.sector in by_sector else list(by_sector)
    hyps: Hypotheses = {}
    for s in sectors:
        per_product = []
        for cls in prior.classes:
            names = [cls] if prior.sector == s and (s, cls) in solver.table.by_class else sorted(by_sector[s])
            per_product.append([_shifted(solver.component(s, c), prior) for c in names])
        hyps[s] = per_product
    return hyps


@dataclass(frozen=True)
class HybridAttack:
    llm: LLMClient
    solver: ClassicalBayesAttack  # frozen A1 config, use_oracle=False
    name: str = "A4_hybrid"

    def extract(self, view: AttackerView) -> LLMPrior | None:
        user = f"Message:\n{quoted(view.masked_text)}\n\n{product_note(view.n_products)}"
        try:
            obj = self.llm.generate_json(EXTRACTION_SYSTEM, user, extraction_schema(view.n_products))
        except LLMError:
            return None
        return parse_prior(obj, view.n_products)

    def attack(self, view: AttackerView) -> tuple[Estimate, ...]:
        if self.solver.use_oracle:
            raise ValueError("A4 must not use the oracle; build its solver with use_oracle=False")
        prior = self.extract(view)
        status = "ok" if prior is not None else "llm_failed_fallback"
        if prior is None:
            LOG.warning("A4 prior extraction failed for a %s prompt; falling back to A1", view.task)
            prior = LLMPrior(None, (None,) * view.n_products)
        est = self.solver.solve_with(view, build_hypotheses(self.solver, prior))
        return tuple(replace(e, status=status) for e in est)
