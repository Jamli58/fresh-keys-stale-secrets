"""Prompts, JSON schemas and defense descriptions for the LLM attackers (A2, A3, A4).

PROMPT_VERSION is logged with every result. Per the plan, prompt variants are tuned
on dev only, at most 10 per attacker; bump the version on every change.
"""

from types import MappingProxyType

from numpriv.attacks.view import AttackerView
from numpriv.defenses import MetricDP, NoDefense, OrderPreserving, PerFieldScaling, SharedScaling
from numpriv.defenses.order_preserving import MAX_OFFSET, SPECIAL_RANGE

PROMPT_VERSION = "v3"
# v1 -> v2 (dev, 2026-09-24): A3 on llama3.2:3b returned margins as fractions (0.62) in the *_pct
# fields; the prompts now spell out every field and its unit (guided decoding forces the keys
# but never shows their meaning to the model).
# v2 -> v3 (dev, 2026-10-02): A4 only. On the AutoDL dev runs both Llama-3.1-8B and Qwen2.5-32B copied
# the prompt's example multipliers (1.2 / 0.8) onto the positioning cue and ignored the city, although
# both know city price levels (smoke rho 0.66 / 0.82). A4 now asks for the city price index and the
# positioning effect separately, with no example values, and multiplies them.
UNKNOWN = "unknown"

# Human-readable labels for P_pub's product classes (A4 must map products onto them).
CLASS_LABELS = MappingProxyType(
    {
        "bread": "bread loaves", "pastry": "pastries", "coffee": "coffee drinks", "cake": "whole cakes",
        "machined_part": "custom machined parts", "fixture": "jigs and fixtures", "repair_job": "machine repair jobs",
        "tshirt": "t-shirts", "jacket": "jackets", "accessory": "fashion accessories",
        "consulting_day": "consulting billed per day", "support_contract": "IT support contracts",
        "software_license": "software licenses", "sqm_install": "installation billed per square metre",
        "site_day": "crew day on site", "materials_lot": "lots of building materials",
        "small_item": "low-priced items", "mid_item": "mid-priced items", "premium_item": "high-priced items",
    }
)

FIELD_GUIDE = (
    "For each product p<i>: p<i>_margin_pct is the gross margin in PERCENT of the price, from 0 to 100 "
    "(write 35 for 35%, not 0.35); p<i>_unit_cost is the cost of one unit in US dollars; the _lo and _hi "
    "fields are the bounds of your 80% interval for the same quantity, in the same unit."
)

_ANALYST = (
    "You are an analyst at a cloud AI provider. You read messages that small businesses send to an AI "
    "assistant and try to work out their confidential figures. "
)

PROSE_ONLY_SYSTEM = _ANALYST + (
    "Every number in the message has been hidden as [number]. From the words alone (type of business, products, "
    "location, market positioning), estimate each product's gross margin (percent of its price) and unit cost "
    "(US dollars per unit), each with an 80% interval. " + FIELD_GUIDE + " Answer only with the requested JSON."
)

END_TO_END_SYSTEM = _ANALYST + (
    "The message was disguised before it reached you by the algorithm described below. You know the algorithm "
    "but not its secret key or random draws. Use the disguised numbers, what the algorithm preserves, and your "
    "knowledge of such businesses to estimate each product's TRUE gross margin (percent of the true price) and "
    "TRUE unit cost (US dollars per unit), each with an 80% interval. " + FIELD_GUIDE + " Reason briefly in the "
    "'reasoning' field first (compute from the numbers where the algorithm allows), then give the numbers."
)


def product_note(n_products: int) -> str:
    order = ", ".join(f"p{i} is the {_ORDINALS[i]} product mentioned" for i in range(n_products))
    return f"The message is about {n_products} product(s): {order}."


_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")


def quoted(text: str) -> str:
    return f'"""\n{text}\n"""'


def prose_only_user(view: AttackerView) -> str:
    return f"Message:\n{quoted(view.masked_text)}\n\n{product_note(view.n_products)}"


def end_to_end_user(view: AttackerView, defense) -> str:
    return (
        f"Disguise algorithm: {describe_defense(defense, view)}\n\n"
        f"Disguised message:\n{quoted(view.text)}\n\n{product_note(view.n_products)}"
    )


def estimate_schema(n_products: int, reasoning: bool) -> dict:
    """Flat schema: per product a point estimate and 80% interval for margin (%) and unit cost ($)."""
    props: dict[str, dict] = {"reasoning": {"type": "string"}} if reasoning else {}
    for i in range(n_products):
        for target in ("margin_pct", "unit_cost"):
            for suffix in ("", "_lo", "_hi"):
                props[f"p{i}_{target}{suffix}"] = {"type": "number"}
    return {"type": "object", "properties": props, "required": list(props)}


def describe_defense(defense, view: AttackerView) -> str:
    """Kerckhoffs: the attacker is told the algorithm and its public parameters, never the key."""
    if isinstance(defense, NoDefense):
        return "None. The numbers are shown exactly as the customer wrote them."
    if isinstance(defense, SharedScaling):
        return (
            "Every money amount was multiplied by the same secret factor k, drawn log-uniformly from "
            f"[{defense.k_low:g}, {defense.k_high:g}], and rounded to cents. Counts (units, quantities) and years are "
            "unchanged. Ratios between money amounts are therefore preserved."
        )
    if isinstance(defense, PerFieldScaling):
        return (
            "Each money amount was multiplied by its own independent secret factor, each drawn log-uniformly from "
            f"[{defense.k_low:g}, {defense.k_high:g}], and rounded to cents. Counts and years are unchanged."
        )
    if isinstance(defense, OrderPreserving):
        return (
            f"Years were shifted by one secret whole-number offset between -{MAX_OFFSET} and +{MAX_OFFSET}. Whole "
            f"numbers from {SPECIAL_RANGE[0]} to {SPECIAL_RANGE[1]} are unchanged. Every other number, money and "
            f"counts alike, was replaced through one secret, strictly increasing random map onto the range "
            f"[{defense.target_low:,.0f}, {defense.target_high:,.0f}]: the order between numbers is preserved and "
            "equal numbers stay equal, but magnitudes, differences and ratios are not."
        )
    if isinstance(defense, MetricDP):
        eps = defense.per_token_epsilon(view.shown)
        lo_m, hi_m = defense.domain["money"]
        lo_c, hi_c = defense.domain["count"]
        return (
            "Metric differential privacy (Laplace-type exponential mechanism). Each money amount and count was "
            f"replaced independently by a random value around the true one with scale {2.0 / eps:,.0f} in the number's "
            f"own unit (privacy budget {defense.epsilon:g} split over the message's numbers), truncated to "
            f"[{lo_m:,.0f}, {hi_m:,.0f}] for money and [{lo_c:,.0f}, {hi_c:,.0f}] for counts, then rounded. "
            "Years are unchanged."
        )
    raise ValueError(f"no description for defense {defense!r}")
