import numpy as np
import pytest

from numpriv.attacks import AttackerView, PlaintextAttack, PriorOnlyAttack, make_view, truths_for
from numpriv.defenses import NoDefense, SharedScaling
from numpriv.metrics import bootstrap_ci, hit_rate, mean_abs_error
from numpriv.priors import build_prior_table
from numpriv.records import generate_firms
from numpriv.templates import build_prompt

FIRMS = generate_firms(40, seed=31)
TABLE = build_prior_table(generate_firms(400, seed=32))


def test_hit_rate_absolute_and_relative():
    est, truth = np.array([0.30, 0.36, 0.5]), np.array([0.31, 0.30, 0.5])
    assert hit_rate(est, truth, abs_tol=0.05) == pytest.approx(2 / 3)
    assert hit_rate(np.array([105.0, 150.0]), np.array([100.0, 100.0]), rel_tol=0.10) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        hit_rate(est, truth)


def test_mean_abs_error():
    assert mean_abs_error(np.array([1.0, 3.0]), np.array([2.0, 2.0])) == pytest.approx(1.0)


def test_bootstrap_ci_brackets_the_estimate_and_is_deterministic():
    values = np.random.default_rng(0).normal(0.4, 0.1, size=200)
    groups = np.repeat(np.arange(50), 4)
    lo, hi = bootstrap_ci(values, groups, n_boot=2000, seed=1)
    assert lo < values.mean() < hi
    assert (lo, hi) == bootstrap_ci(values, groups, n_boot=2000, seed=1)


def test_attacker_view_carries_no_ground_truth():
    spec = build_prompt("margin_review", FIRMS[0], cues=True)
    rng = np.random.default_rng(4)
    d = SharedScaling()
    view = make_view(spec, d.apply(spec.fields, d.sample_secret(rng, spec.fields), rng).fields, d.name)
    field_names = set(AttackerView.__dataclass_fields__)
    assert field_names == {"task", "text", "shown", "n_products", "defense_name", "oracle", "masked_text"}
    assert view.oracle is None
    assert not any(ch.isdigit() for ch in view.masked_text)
    shown = {f.field_id: f.value for f in view.shown}
    assert shown["price:0"] != spec.products[0].price


def test_oracle_is_only_attached_on_request():
    spec = build_prompt("margin_review", FIRMS[0], cues=True)
    view = make_view(spec, spec.fields, NoDefense.name, with_oracle=True)
    assert view.oracle.sector == FIRMS[0].sector
    assert view.oracle.product_classes == tuple(p.product_class for p in spec.products)


def test_plaintext_attack_recovers_everything_on_d0():
    rng = np.random.default_rng(2)
    attack, defense = PlaintextAttack(), NoDefense()
    for firm in FIRMS:
        spec = build_prompt("margin_review", firm, cues=True)
        shown = defense.apply(spec.fields, defense.sample_secret(rng, spec.fields), rng).fields
        est = attack.attack(make_view(spec, shown, defense.name))
        truth = truths_for(spec)
        assert [e.margin for e in est] == pytest.approx([t.margin for t in truth])
        assert [e.unit_cost for e in est] == pytest.approx([t.unit_cost for t in truth])


def test_plaintext_attack_refuses_disguised_input():
    rng = np.random.default_rng(3)
    spec = build_prompt("margin_review", FIRMS[0], cues=True)
    d = SharedScaling()
    shown = d.apply(spec.fields, d.sample_secret(rng, spec.fields), rng).fields
    with pytest.raises(ValueError):
        PlaintextAttack().attack(make_view(spec, shown, d.name))


def test_prior_only_attack_needs_oracle():
    spec = build_prompt("margin_review", FIRMS[1], cues=True)
    with pytest.raises(ValueError):
        PriorOnlyAttack(TABLE).attack(make_view(spec, spec.fields, NoDefense.name))


def test_prior_only_attack_ignores_numbers_and_uses_class_medians():
    attack = PriorOnlyAttack(TABLE)
    spec = build_prompt("margin_review", FIRMS[1], cues=True)
    est = attack.attack(make_view(spec, spec.fields, NoDefense.name, with_oracle=True))
    for e, p in zip(est, spec.products):
        stats = TABLE.by_class[(FIRMS[1].sector, p.product_class)]
        assert e.margin == stats.margin.median
        assert e.unit_cost == stats.unit_cost.median


def test_truths_come_from_records_not_from_any_model():
    spec = build_prompt("margin_review", FIRMS[2], cues=True)
    truth = truths_for(spec)
    assert [t.unit_cost for t in truth] == [p.unit_cost for p in spec.products]
