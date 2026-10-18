import math
import re

import numpy as np
import pytest

from numpriv.attacks import make_view
from numpriv.attacks.llm_attacks import EndToEndAttack, ProseOnlyAttack
from numpriv.attacks.llm_prompts import describe_defense, estimate_schema
from numpriv.dataset import EvalConfig, build_eval_set
from numpriv.defenses import MetricDP, NoDefense, OrderPreserving, PerFieldScaling, SharedScaling
from numpriv.evaluate import evaluate, summarize
from numpriv.llm import LLMError
from numpriv.records import generate_firms
from numpriv.templates import TASKS, build_prompt

FIRM = generate_firms(1, seed=5)[0]
ALL_DEFENSES = (NoDefense(), SharedScaling(), PerFieldScaling(), OrderPreserving(), MetricDP(1.0))


class ScriptedLLM:
    """Answers every request with ``answer(schema)`` and records (system, user, schema, max_tokens)."""

    model, seed, temperature = "fake", 0, 0.0

    def __init__(self, answer):
        self.answer = answer
        self.requests = []

    def generate_json(self, system, user, schema, max_tokens=None):
        self.requests.append((system, user, schema, max_tokens))
        return self.answer(schema)


def _estimates_answer(margin_pct=40.0, cost=10.0):
    def answer(schema):
        out = {}
        for key in schema["properties"]:
            if key == "reasoning":
                out[key] = "thinking"
            elif "margin" in key:
                out[key] = margin_pct + (-5 if key.endswith("_lo") else 5 if key.endswith("_hi") else 0)
            elif "unit_cost" in key:
                out[key] = cost * (0.8 if key.endswith("_lo") else 1.25 if key.endswith("_hi") else 1)
        return out

    return answer


def _failing(schema):
    raise LLMError("no valid response")


def _view(task="margin_review", defense=SharedScaling(), seed=0):
    spec = build_prompt(task, FIRM, cues=True)
    rng = np.random.default_rng(seed)
    shown = defense.apply(spec.fields, defense.sample_secret(rng, spec.fields), rng).fields
    return spec, make_view(spec, shown, defense.name)


@pytest.mark.parametrize("task", TASKS)
def test_masked_text_hides_every_number_but_keeps_the_prose(task):
    spec, view = _view(task)
    assert re.search(r"\d", view.masked_text) is None
    assert "[number]" in view.masked_text
    assert FIRM.city in view.masked_text


def test_estimate_schema_has_point_and_interval_per_product():
    schema = estimate_schema(2, reasoning=False)
    for i in range(2):
        for suffix in ("", "_lo", "_hi"):
            assert f"p{i}_margin_pct{suffix}" in schema["required"]
            assert f"p{i}_unit_cost{suffix}" in schema["required"]
    assert "reasoning" not in schema["properties"]
    assert list(estimate_schema(1, reasoning=True)["properties"])[0] == "reasoning"


def test_prose_only_attack_sends_masked_text_only_and_parses_percentages():
    llm = ScriptedLLM(_estimates_answer(margin_pct=40.0, cost=10.0))
    spec, view = _view("margin_review")
    est = ProseOnlyAttack(llm).attack(view)
    _, user, _, _ = llm.requests[0]
    assert view.masked_text in user
    shown_strings = [f"{f.value:,.2f}" for f in view.shown if f.kind == "money"]
    assert not any(s in user for s in shown_strings)
    assert len(est) == view.n_products
    assert est[0].margin == pytest.approx(0.40) and est[0].unit_cost == pytest.approx(10.0)
    assert est[0].margin_interval == pytest.approx((0.35, 0.45))
    assert est[0].cost_interval == pytest.approx((8.0, 12.5))
    assert est[0].status == "ok"


def test_prose_only_attack_is_defense_independent():
    llm = ScriptedLLM(_estimates_answer())
    attack = ProseOnlyAttack(llm)
    for d in ALL_DEFENSES:
        attack.attack(_view("margin_review", d)[1])
    assert len({user for _, user, _, _ in llm.requests}) == 1


def test_llm_failure_is_a_recorded_miss_not_a_drop():
    spec, view = _view()
    est = ProseOnlyAttack(ScriptedLLM(_failing)).attack(view)
    assert len(est) == view.n_products
    assert all(math.isnan(e.margin) and math.isnan(e.unit_cost) and e.status == "llm_failed" for e in est)


def test_non_finite_or_non_positive_outputs_are_marked_invalid():
    llm = ScriptedLLM(_estimates_answer(margin_pct=40.0, cost=-3.0))
    est = ProseOnlyAttack(llm).attack(_view("break_even")[1])
    assert est[0].status == "invalid_output" and math.isnan(est[0].unit_cost)


def test_interval_bounds_are_reordered_when_the_llm_swaps_them():
    def answer(schema):
        return {"p0_margin_pct": 30, "p0_margin_pct_lo": 50, "p0_margin_pct_hi": 20,
                "p0_unit_cost": 5, "p0_unit_cost_lo": 9, "p0_unit_cost_hi": 4}

    est = ProseOnlyAttack(ScriptedLLM(answer)).attack(_view("break_even")[1])[0]
    assert est.margin_interval == pytest.approx((0.20, 0.50)) and est.cost_interval == (4, 9)


def test_end_to_end_attack_sees_disguised_numbers_and_the_defense_algorithm():
    llm = ScriptedLLM(_estimates_answer())
    spec, view = _view("margin_review", SharedScaling())
    EndToEndAttack(llm, SharedScaling()).attack(view)
    system, user, schema, max_tokens = llm.requests[0]
    assert view.text in user
    assert describe_defense(SharedScaling(), view) in user
    assert "reasoning" in schema["properties"] and max_tokens >= 512


def test_end_to_end_attack_refuses_a_view_from_another_defense():
    _, view = _view("margin_review", OrderPreserving())
    with pytest.raises(ValueError):
        EndToEndAttack(ScriptedLLM(_estimates_answer()), SharedScaling()).attack(view)


@pytest.mark.parametrize("defense", ALL_DEFENSES, ids=lambda d: d.name)
def test_every_defense_has_a_description(defense):
    _, view = _view("quote_pricing", defense)
    text = describe_defense(defense, view)
    assert len(text) > 40


def test_metric_dp_description_states_the_noise_scale_for_this_prompt():
    d = MetricDP(0.5)
    _, view = _view("quote_pricing", d)
    scale = 2.0 / d.per_token_epsilon(view.shown)
    assert f"{scale:,.0f}" in describe_defense(d, view)


def test_evaluate_counts_llm_failures_as_misses():
    ev = build_eval_set(EvalConfig(n_firms=12, n_dev=4, tasks_per_firm=1, seed=3, prior_seed=4))
    rows = evaluate(ev, "dev", NoDefense(), ProseOnlyAttack(ScriptedLLM(_failing)), (0,), with_oracle=False)
    s = summarize(rows, n_boot=10)["all"]
    assert s["margin_hit_5pp"] == 0.0 and s["unit_cost_hit_10pct"] == 0.0
    assert s["n_failed"] == len(rows)


def test_prompts_state_the_margin_unit_explicitly():
    from numpriv.attacks.llm_prompts import END_TO_END_SYSTEM, PROSE_ONLY_SYSTEM

    for system in (PROSE_ONLY_SYSTEM, END_TO_END_SYSTEM):
        assert "PERCENT" in system and "not 0.35" in system and "US dollars" in system
