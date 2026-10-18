"""Five SME task templates rendered from a firm, with cued and cue-stripped variants.

Templates use ``{field_id}`` placeholders (ids contain ':' so rendering is a
regex substitution, not ``str.format``). Cue-stripped prompts replace the firm
name, sector, and product names with generic wording (Block 3 ablation).
"""

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable

import numpy as np

from numpriv.config import SECTORS
from numpriv.fields import NumField
from numpriv.records import Firm, Product

REPORT_YEAR = 2025
SUPPLIER_QUOTE_FACTOR = 0.93
# Per-firm quantities (not constants: a constant is a known rank anchor under D2),
# drawn from grids that avoid D2's year range [1990, 2030] and special numbers 28-31.
_ORDER_QTY_GRID = tuple(range(100, 1000, 50))
_MIN_ORDER_GRID = tuple(q for q in range(500, 5001, 100) if not 1990 <= q <= 2030)
_LETTERS = "ABCDEFGH"
_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


@dataclass(frozen=True)
class PromptSpec:
    task: str
    firm_id: str
    sector: str  # ground-truth metadata; only oracle attackers (A0, A1+) may read it
    cues: bool
    template: str
    fields: tuple[NumField, ...]
    products: tuple[Product, ...]


@dataclass(frozen=True)
class _Names:
    firm: str
    sector: str
    products: tuple[str, ...]


# M0.5 cues (fixed before M2). No digits, no sector or product-class words.
POSITIONING_SENTENCES = MappingProxyType(
    {
        "premium": "We position ourselves as a premium, high-quality option and our customers pay for it.",
        "standard": "We are a typical local business with mid-range prices.",
        "discount": "We compete on low prices and high volume.",
    }
)


def _context(firm: Firm, cues: bool) -> str:
    """City + positioning sentence for cued prompts; nothing for cue-stripped ones."""
    if not cues:
        return ""
    return f"{firm.name} is based in {firm.city}. {POSITIONING_SENTENCES[firm.positioning]} "


def _names(firm: Firm, products: tuple[Product, ...], cues: bool) -> _Names:
    if cues:
        return _Names(firm.name, f"a small {SECTORS[firm.sector].display_name} business", tuple(p.name for p in products))
    return _Names("Company X", "a small business", tuple(f"Product {_LETTERS[i]}" for i in range(len(products))))


def _firm_quantity(firm: Firm, grid: tuple[int, ...], stream: int) -> int:
    rng = np.random.default_rng((int(firm.firm_id.lstrip("F")), stream))
    return int(grid[rng.integers(len(grid))])


def _price_cost(i: int, p: Product) -> tuple[NumField, NumField]:
    return NumField(f"price:{i}", "money", p.price), NumField(f"cost:{i}", "money", p.unit_cost)


def _quote_pricing(firm: Firm, n: _Names) -> tuple[str, tuple[NumField, ...], int]:
    p = firm.products[0]
    text = (
        f"I run {n.firm}, {n.sector}. For {n.products[0]} we charge {{price:0}} per unit and it costs us "
        f"{{cost:0}} per unit. We sell about {{volume:0}} units a month and fixed costs are {{overhead}} per month. "
        "A customer wants {order_qty} units. What price should I quote to keep at least our current margin?"
    )
    fields = (*_price_cost(0, p), NumField("volume:0", "count", p.monthly_volume[0]),
              NumField("overhead", "money", firm.monthly_overhead),
              NumField("order_qty", "count", _firm_quantity(firm, _ORDER_QTY_GRID, 0)))
    return text, fields, 1


def _margin_review(firm: Firm, n: _Names) -> tuple[str, tuple[NumField, ...], int]:
    k = min(3, len(firm.products))
    lines = " ".join(f"{n.products[i]}: price {{price:{i}}}, unit cost {{cost:{i}}}." for i in range(k))
    text = (
        f"We are {n.firm}, {n.sector}. Here are our main products. {lines} "
        "Which product has the weakest margin, and what would you change first?"
    )
    fields = tuple(f for i in range(k) for f in _price_cost(i, firm.products[i]))
    return text, fields, k


def _break_even(firm: Firm, n: _Names) -> tuple[str, tuple[NumField, ...], int]:
    p = firm.products[0]
    text = (
        f"{n.firm} is {n.sector}. {n.products[0]} sells for {{price:0}} and costs {{cost:0}} per unit. "
        "Our fixed costs are {overhead} per month. How many units do we need to sell each month to break even?"
    )
    return text, (*_price_cost(0, p), NumField("overhead", "money", firm.monthly_overhead)), 1


def _sales_summary(firm: Firm, n: _Names) -> tuple[str, tuple[NumField, ...], int]:
    p = firm.products[0]
    text = (
        f"Summarize first-quarter sales for {n.firm}, {n.sector} (report year {{year:0}}). {n.products[0]}: price "
        "{price:0}, unit cost {cost:0}; units sold in January {volume:0:0}, February {volume:0:1}, March {volume:0:2}. "
        "Give revenue, gross profit and the trend."
    )
    volumes = tuple(NumField(f"volume:0:{m}", "count", p.monthly_volume[m]) for m in range(3))
    return text, (*_price_cost(0, p), *volumes, NumField("year:0", "year", REPORT_YEAR)), 1


def _supplier_comparison(firm: Firm, n: _Names) -> tuple[str, tuple[NumField, ...], int]:
    p = firm.products[0]
    quote = round(p.unit_cost * SUPPLIER_QUOTE_FACTOR, 2)
    text = (
        f"At {n.firm}, {n.sector}, {n.products[0]} sells for {{price:0}}. Our current supplier cost is "
        "{cost:0} per unit and a new supplier quotes {quote:0} per unit with a {min_order} unit minimum order. "
        "Should we switch, and how much margin would we gain?"
    )
    fields = (*_price_cost(0, p), NumField("quote:0", "money", quote),
              NumField("min_order", "count", _firm_quantity(firm, _MIN_ORDER_GRID, 1)))
    return text, fields, 1


_BUILDERS: "MappingProxyType[str, Callable[[Firm, _Names], tuple[str, tuple[NumField, ...], int]]]" = MappingProxyType(
    {
        "quote_pricing": _quote_pricing,
        "margin_review": _margin_review,
        "break_even": _break_even,
        "sales_summary": _sales_summary,
        "supplier_comparison": _supplier_comparison,
    }
)
TASKS = tuple(_BUILDERS)


def build_prompt(task: str, firm: Firm, cues: bool) -> PromptSpec:
    builder = _BUILDERS[task]
    products = firm.products[: min(3, len(firm.products))]
    text, fields, k = builder(firm, _names(firm, products, cues))
    return PromptSpec(task, firm.firm_id, firm.sector, cues, _context(firm, cues) + text, fields, products[:k])


def _format_money(value: float) -> str:
    """Thousands separators, at least 2 decimals, and exactly the digits the value carries.

    Whole-cent originals render naturally ("12.50"); disguised values render
    with every digit the disguise produced, so the text the attacker reads
    carries exactly the disguised value (no extra rounding, no false ties).
    """
    decimals = max(2, _decimals_needed(value))
    return f"{value:,.{decimals}f}"


def _decimals_needed(value: float) -> int:
    text = repr(float(value))
    if "e" in text or "E" in text:
        mantissa, _, exp = text.lower().partition("e")
        frac = mantissa.partition(".")[2].rstrip("0")
        return max(0, len(frac) - int(exp))
    return len(text.partition(".")[2].rstrip("0"))


def format_field(field: NumField) -> str:
    if field.kind == "money":
        return _format_money(field.value)
    if field.kind == "count":
        return f"{field.value:,.0f}"
    if field.kind in ("year", "day"):
        return f"{field.value:.0f}"
    return f"{field.value:.2f}%"


MASK_TOKEN = "[number]"


def render_masked(spec: PromptSpec) -> str:
    """The prompt with every numeric field replaced by MASK_TOKEN (prose-only attackers A2 / A4 prior)."""
    return _PLACEHOLDER.sub(MASK_TOKEN, spec.template)


def render(spec: PromptSpec, fields: tuple[NumField, ...] | None = None) -> str:
    values = {f.field_id: f for f in (fields if fields is not None else spec.fields)}
    return _PLACEHOLDER.sub(lambda m: format_field(values[m.group(1)]), spec.template)
