import re

import numpy as np
import pytest

from numpriv.config import SECTORS
from numpriv.defenses import SharedScaling
from numpriv.records import generate_firms
from numpriv.templates import TASKS, build_prompt, render

FIRMS = generate_firms(30, seed=21)


@pytest.mark.parametrize("task", TASKS)
def test_every_placeholder_has_a_field_and_vice_versa(task):
    for firm in FIRMS:
        spec = build_prompt(task, firm, cues=True)
        placeholders = set(re.findall(r"\{([^{}]+)\}", spec.template))
        assert placeholders == {f.field_id for f in spec.fields}


@pytest.mark.parametrize("task", TASKS)
def test_fields_carry_true_values(task):
    firm = FIRMS[0]
    spec = build_prompt(task, firm, cues=True)
    for f in spec.fields:
        kind, _, rest = f.field_id.partition(":")
        if kind in ("price", "cost"):
            product = spec.products[int(rest)]
            assert f.value == (product.price if kind == "price" else product.unit_cost)


@pytest.mark.parametrize("task", TASKS)
def test_cue_stripped_prompt_hides_sector_firm_and_class(task):
    for firm in FIRMS:
        text = render(build_prompt(task, firm, cues=False))
        lowered = text.lower()
        assert firm.name.lower() not in lowered
        assert SECTORS[firm.sector].display_name.lower() not in lowered
        for cls in SECTORS[firm.sector].product_classes:
            assert cls.name.replace("_", " ") not in lowered


@pytest.mark.parametrize("task", TASKS)
def test_cued_prompt_mentions_sector(task):
    firm = FIRMS[1]
    text = render(build_prompt(task, firm, cues=True)).lower()
    assert SECTORS[firm.sector].display_name.lower() in text


def test_render_uses_disguised_values():
    firm = FIRMS[2]
    spec = build_prompt("margin_review", firm, cues=True)
    rng = np.random.default_rng(0)
    d = SharedScaling()
    disguised = d.apply(spec.fields, d.sample_secret(rng, spec.fields), rng)
    original, masked = render(spec), render(spec, disguised.fields)
    assert original != masked
    assert f"{spec.products[0].price:,.2f}" in original


def test_build_prompt_is_deterministic_and_rejects_unknown_task():
    firm = FIRMS[3]
    assert build_prompt("quote_pricing", firm, cues=True) == build_prompt("quote_pricing", firm, cues=True)
    with pytest.raises(KeyError):
        build_prompt("nope", firm, cues=True)


def _rendered_numbers(spec, fields):
    """Parse each placeholder's rendered string back to a float, as an attacker reading the text would."""
    from numpriv.templates import format_field

    return {f.field_id: float(format_field(f).replace(",", "").rstrip("%")) for f in fields}


def test_rendered_text_preserves_margin_within_cent_rounding_under_shared_scaling():
    """Shown money is in whole cents, so the margin invariant holds up to |dm| <= 0.01 / shown price."""
    rng = np.random.default_rng(1)
    d = SharedScaling()
    for firm in FIRMS:
        spec = build_prompt("margin_review", firm, cues=True)
        for _ in range(20):
            shown = _rendered_numbers(spec, d.apply(spec.fields, d.sample_secret(rng, spec.fields), rng).fields)
            for i, p in enumerate(spec.products):
                margin = (shown[f"price:{i}"] - shown[f"cost:{i}"]) / shown[f"price:{i}"]
                assert abs(margin - p.margin) <= 0.01 / shown[f"price:{i}"] + 1e-12


def test_original_money_renders_naturally():
    from numpriv.fields import NumField
    from numpriv.templates import format_field

    assert format_field(NumField("price:0", "money", 12.5)) == "12.50"
    assert format_field(NumField("price:0", "money", 1234.0)) == "1,234.00"
    assert format_field(NumField("price:0", "money", 0.054713)) == "0.054713"


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("cues", (True, False))
def test_every_number_in_the_prose_is_a_field(task, cues):
    """A literal number outside a placeholder would bypass every defense (e.g. leak the year offset)."""
    for firm in FIRMS:
        spec = build_prompt(task, firm, cues=cues)
        prose = re.sub(r"\{[^{}]+\}", "", spec.template)
        assert not re.search(r"\d", prose), prose


def _count_values(task, field_id):
    return [
        next(f.value for f in build_prompt(task, firm, cues=True).fields if f.field_id == field_id)
        for firm in generate_firms(120, seed=22)
    ]


@pytest.mark.parametrize("task,field_id", [("quote_pricing", "order_qty"), ("supplier_comparison", "min_order")])
def test_template_quantities_vary_by_firm_and_avoid_d2_special_ranges(task, field_id):
    """A constant would be a known rank anchor under D2; a year-range value would leak D2's year offset."""
    values = _count_values(task, field_id)
    assert len(set(values)) >= 10
    assert not any(1990 <= v <= 2030 or 28 <= v <= 31 for v in values)
    assert values == _count_values(task, field_id)  # deterministic per firm


def test_rendered_money_reproduces_the_shown_value_exactly():
    """The text the attacker reads must carry exactly the disguised value (no extra rounding)."""
    from numpriv.fields import NumField
    from numpriv.templates import format_field

    for v in (12.5, 1234.0, 0.054713, 18474.3092871, 70163.31234, 1.00000123, 999999.123456):
        assert float(format_field(NumField("x", "money", v)).replace(",", "")) == v


POSITIONING_MARKERS = {"premium": "premium, high-quality", "standard": "mid-range prices", "discount": "low prices and high volume"}


@pytest.mark.parametrize("task", TASKS)
def test_cued_prompts_state_city_and_positioning(task):
    for firm in FIRMS:
        text = render(build_prompt(task, firm, cues=True))
        assert firm.city in text
        assert POSITIONING_MARKERS[firm.positioning] in text


@pytest.mark.parametrize("task", TASKS)
def test_cue_stripped_prompts_drop_city_and_positioning(task):
    for firm in FIRMS:
        text = render(build_prompt(task, firm, cues=False))
        assert firm.city not in text
        assert not any(marker in text for marker in POSITIONING_MARKERS.values())
