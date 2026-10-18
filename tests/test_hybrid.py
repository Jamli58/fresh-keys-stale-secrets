import numpy as np
import pytest

from numpriv.attacks import make_view
from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.attacks.hybrid import (
    EXTRACTION_SYSTEM,
    HybridAttack,
    LLMPrior,
    build_hypotheses,
    extraction_schema,
    parse_prior,
)
from numpriv.config import SECTORS
from numpriv.defenses import MetricDP, OrderPreserving, SharedScaling
from numpriv.llm import LLMError
from numpriv.priors import build_prior_table
from numpriv.records import generate_firms
from numpriv.templates import build_prompt

TABLE = build_prior_table(generate_firms(2000, seed=77))
FIRMS = generate_firms(12, seed=78)
SOLVER_KW = {"n_samples": 1500, "prior_df": 3.0, "prior_scale": 0.75}


class ScriptedLLM:
    model, seed, temperature = "fake", 0, 0.0

    def __init__(self, answer):
        self.answer = answer
        self.requests = []

    def generate_json(self, system, user, schema, max_tokens=None):
        self.requests.append((system, user, schema))
        return self.answer(schema)


def _price_fields(multiplier=1.0, positioning_pct=0.0):
    """v3 splits the price multiplier into a city index (100 = US average) and a positioning effect."""
    return {"city_price_index": 100.0 * multiplier, "positioning_price_pct": positioning_pct}


def _truthful(spec, multiplier=1.0, shift=0.0):
    def answer(schema):
        out = {"sector": spec.sector, **_price_fields(multiplier), "margin_shift_pp": shift}
        out.update({f"p{i}_class": p.product_class for i, p in enumerate(spec.products)})
        return out

    return answer


def _unknown(schema):
    out = {"sector": "unknown", **_price_fields(), "margin_shift_pp": 0.0}
    out.update({k: "unknown" for k in schema["properties"] if k.endswith("_class")})
    return out


def _views(defense, firm, task="margin_review", seed=0):
    spec = build_prompt(task, firm, cues=True)
    rng = np.random.default_rng(seed)
    shown = defense.apply(spec.fields, defense.sample_secret(rng, spec.fields), rng).fields
    return spec, make_view(spec, shown, defense.name), make_view(spec, shown, defense.name, with_oracle=True)


def _hybrid(defense, answer):
    return HybridAttack(ScriptedLLM(answer), ClassicalBayesAttack(TABLE, defense, use_oracle=False, **SOLVER_KW))


@pytest.mark.parametrize("defense", (SharedScaling(), OrderPreserving(), MetricDP(1.0)), ids=lambda d: d.name)
def test_truthful_neutral_llm_reproduces_a1_plus_exactly(defense):
    """A4 - A1+ compares priors, not solvers: with the oracle's answer and no shift they coincide."""
    a1p = ClassicalBayesAttack(TABLE, defense, use_oracle=True, **SOLVER_KW)
    for firm in FIRMS[:4]:
        spec, view, oracle_view = _views(defense, firm)
        got = _hybrid(defense, _truthful(spec)).attack(view)
        want = a1p.attack(oracle_view)
        assert [(e.margin, e.unit_cost) for e in got] == [(e.margin, e.unit_cost) for e in want]


def test_unknown_answer_falls_back_to_a1():
    d = MetricDP(1.0)
    a1 = ClassicalBayesAttack(TABLE, d, use_oracle=False, **SOLVER_KW)
    spec, view, _ = _views(d, FIRMS[0])
    got = _hybrid(d, _unknown).attack(view)
    want = a1.attack(view)
    assert [(e.margin, e.unit_cost) for e in got] == pytest.approx([(e.margin, e.unit_cost) for e in want])


def test_llm_failure_falls_back_to_a1_and_is_flagged():
    d = MetricDP(1.0)
    spec, view, _ = _views(d, FIRMS[0])

    def fail(schema):
        raise LLMError("down")

    got = _hybrid(d, fail).attack(view)
    want = ClassicalBayesAttack(TABLE, d, use_oracle=False, **SOLVER_KW).attack(view)
    assert [e.unit_cost for e in got] == pytest.approx([e.unit_cost for e in want])
    assert all(e.status == "llm_failed_fallback" for e in got)


def test_price_multiplier_and_margin_shift_move_the_prior_components():
    solver = ClassicalBayesAttack(TABLE, MetricDP(1.0), use_oracle=False, **SOLVER_KW)
    base = build_hypotheses(solver, LLMPrior("bakery_cafe", ("bread",)))["bakery_cafe"][0][0]
    moved = build_hypotheses(solver, LLMPrior("bakery_cafe", ("bread",), np.log(2.0), 0.10))["bakery_cafe"][0][0]
    assert moved.price[0] == pytest.approx(base.price[0] + np.log(2.0))
    assert moved.price[1] == base.price[1] and moved.margin[1] == base.margin[1]
    sigmoid = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    assert sigmoid(moved.margin[0]) == pytest.approx(sigmoid(base.margin[0]) + 0.10)


def test_prior_shift_moves_the_estimate_when_numbers_are_uninformative():
    d = MetricDP(0.5)
    bakery = next(f for f in FIRMS if f.sector == "bakery_cafe")  # $1-10 prices vs noise scale ~12
    spec, view, _ = _views(d, bakery, task="break_even")
    base = _hybrid(d, _truthful(spec)).attack(view)[0]
    pricier = _hybrid(d, _truthful(spec, multiplier=2.0)).attack(view)[0]
    fatter = _hybrid(d, _truthful(spec, shift=15.0)).attack(view)[0]
    assert pricier.unit_cost > base.unit_cost * 1.3
    assert fatter.margin > base.margin + 0.05


def test_class_outside_the_chosen_sector_uses_that_sectors_classes():
    d = MetricDP(1.0)
    spec, view, _ = _views(d, FIRMS[0], task="break_even")
    other = next(s for s in SECTORS if s != spec.sector)
    wrong_class = SECTORS[other].product_classes[0].name

    def answer(schema):
        return {"sector": spec.sector, **_price_fields(), "margin_shift_pp": 0.0, "p0_class": wrong_class}

    est = _hybrid(d, answer).attack(view)
    assert np.isfinite(est[0].unit_cost) and est[0].status == "ok"


def test_extreme_llm_numbers_are_clipped_not_trusted():
    d = MetricDP(0.5)
    spec, view, _ = _views(d, FIRMS[1], task="break_even")
    huge = _hybrid(d, _truthful(spec, multiplier=1e9, shift=500.0)).attack(view)[0]
    capped = _hybrid(d, _truthful(spec, multiplier=4.0, shift=40.0)).attack(view)[0]
    assert (huge.margin, huge.unit_cost) == pytest.approx((capped.margin, capped.unit_cost))


def test_extraction_reads_masked_prose_only():
    d = SharedScaling()
    spec, view, _ = _views(d, FIRMS[0])
    llm = ScriptedLLM(_truthful(spec))
    HybridAttack(llm, ClassicalBayesAttack(TABLE, d, use_oracle=False, **SOLVER_KW)).attack(view)
    _, user, schema = llm.requests[0]
    assert view.masked_text in user
    assert not any(f"{f.value:,.2f}" in user for f in view.shown if f.kind == "money")
    assert set(schema["properties"]["sector"]["enum"]) == {*SECTORS, "unknown"}


def test_extraction_schema_has_one_class_field_per_product():
    schema = extraction_schema(3)
    assert {"p0_class", "p1_class", "p2_class", "sector", "city_price_index", "positioning_price_pct",
            "margin_shift_pp"} <= set(schema["required"])
    assert "price_multiplier" not in schema["properties"]
    assert "unknown" in schema["properties"]["p2_class"]["enum"]


def test_every_product_class_has_a_label():
    from numpriv.attacks.llm_prompts import CLASS_LABELS

    assert set(CLASS_LABELS) == {c.name for s in SECTORS.values() for c in s.product_classes}


def test_city_index_and_positioning_combine_multiplicatively():
    obj = {"sector": "bakery_cafe", "p0_class": "bread", "city_price_index": 120.0,
           "positioning_price_pct": 10.0, "margin_shift_pp": 0.0}
    assert parse_prior(obj, 1).log_price_shift == pytest.approx(np.log(1.2 * 1.1))


def test_each_price_factor_is_clipped_separately():
    """A nonsense city index must not be rescued or amplified by the positioning effect."""
    wild = {"sector": "bakery_cafe", "p0_class": "bread", "city_price_index": 1e6,
            "positioning_price_pct": -99.9, "margin_shift_pp": 0.0}
    sane = {**wild, "city_price_index": 200.0, "positioning_price_pct": -60.0}
    assert parse_prior(wild, 1).log_price_shift == pytest.approx(parse_prior(sane, 1).log_price_shift)


def test_non_finite_price_answers_are_neutral():
    obj = {"sector": "bakery_cafe", "p0_class": "bread", "city_price_index": float("nan"),
           "positioning_price_pct": float("inf"), "margin_shift_pp": 0.0}
    assert parse_prior(obj, 1).log_price_shift == 0.0


def test_extraction_prompt_gives_no_example_price_values():
    """v2's '1.2 = 20% above, 0.8 = 20% below' anchored both dev models onto exactly {0.8, 1.0, 1.2}."""
    assert "1.2" not in EXTRACTION_SYSTEM and "0.8" not in EXTRACTION_SYSTEM
    assert "20%" not in EXTRACTION_SYSTEM
