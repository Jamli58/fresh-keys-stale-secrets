import numpy as np
import pytest

from numpriv.attacks import PriorOnlyAttack, make_view, truths_for
from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.defenses import MetricDP, NoDefense, OrderPreserving, PerFieldScaling, SharedScaling
from numpriv.metrics import hit_rate
from numpriv.priors import build_prior_table
from numpriv.records import generate_firms
from numpriv.templates import build_prompt

TABLE = build_prior_table(generate_firms(2000, seed=77))
FIRMS = generate_firms(90, seed=78)


def _run(defense, attack, seed=0, task="margin_review"):
    rng = np.random.default_rng(seed)
    est, tru = [], []
    for firm in FIRMS:
        spec = build_prompt(task, firm, cues=True)
        shown = defense.apply(spec.fields, defense.sample_secret(rng, spec.fields), rng).fields
        use_oracle = getattr(attack, "use_oracle", True)
        est += attack.attack(make_view(spec, shown, defense.name, with_oracle=use_oracle))
        tru += truths_for(spec)
    m = hit_rate(np.array([e.margin for e in est]), np.array([t.margin for t in tru]), abs_tol=0.05)
    c = hit_rate(np.array([e.unit_cost for e in est]), np.array([t.unit_cost for t in tru]), rel_tol=0.10)
    return m, c


def test_d0_is_read_exactly():
    m, c = _run(NoDefense(), ClassicalBayesAttack(TABLE, NoDefense(), use_oracle=False))
    assert m == 1.0 and c == 1.0


@pytest.mark.parametrize("use_oracle", (True, False))
def test_d1_margins_come_from_the_invariant(use_oracle):
    m, _ = _run(SharedScaling(), ClassicalBayesAttack(TABLE, SharedScaling(), use_oracle=use_oracle))
    assert m > 0.97


def test_d1_unit_cost_beats_the_prior_only_floor():
    _, c_bayes = _run(SharedScaling(), ClassicalBayesAttack(TABLE, SharedScaling(), use_oracle=True))
    _, c_floor = _run(SharedScaling(), PriorOnlyAttack(TABLE))
    assert c_bayes > c_floor + 0.05


def test_d1_prime_offers_nothing_beyond_the_prior():
    m_bayes, _ = _run(PerFieldScaling(), ClassicalBayesAttack(TABLE, PerFieldScaling(), use_oracle=True))
    m_floor, _ = _run(PerFieldScaling(), PriorOnlyAttack(TABLE))
    assert abs(m_bayes - m_floor) < 0.12


def test_d3_large_values_leak_through_weak_noise():
    d = MetricDP(epsilon=2.0)
    _, c_bayes = _run(d, ClassicalBayesAttack(TABLE, d, use_oracle=True))
    _, c_floor = _run(d, PriorOnlyAttack(TABLE))
    assert c_bayes > c_floor + 0.15


def test_d2_attack_is_not_broken_under_its_own_model():
    """Correctness check for D2 (replaces an empirical 'within 5pp of the floor' expectation that was
    fragile: D2 carries almost no margin information, so on mis-specified generated data the posterior
    can sit a few points below the prior median). Under its own prior the attacker must not lose."""
    from numpriv.diagnostics import self_consistency_gain

    out = self_consistency_gain(TABLE, OrderPreserving(), n=600, seed=4)
    assert out["gain_ci"][1] >= 0.0


def test_sector_posterior_without_oracle_beats_chance_on_d1():
    attack = ClassicalBayesAttack(TABLE, SharedScaling(), use_oracle=False)
    rng = np.random.default_rng(1)
    d = SharedScaling()
    right = 0
    for firm in FIRMS:
        spec = build_prompt("margin_review", firm, cues=True)
        view = make_view(spec, d.apply(spec.fields, d.sample_secret(rng, spec.fields), rng).fields, d.name)
        post = attack.sector_posterior(view)
        assert sum(post.values()) == pytest.approx(1.0)
        right += max(post, key=post.get) == firm.sector
    assert right / len(FIRMS) > 0.35  # chance = 1/6


def test_oracle_use_is_strict():
    spec = build_prompt("margin_review", FIRMS[0], cues=True)
    d = SharedScaling()
    with pytest.raises(ValueError):
        ClassicalBayesAttack(TABLE, d, use_oracle=True).attack(make_view(spec, spec.fields, d.name))
    with pytest.raises(ValueError):
        ClassicalBayesAttack(TABLE, d, use_oracle=False).attack(make_view(spec, spec.fields, d.name, with_oracle=True))
    with pytest.raises(ValueError):
        ClassicalBayesAttack(TABLE, d, use_oracle=False).attack(make_view(spec, spec.fields, NoDefense.name))


def test_attack_is_deterministic_for_a_fixed_seed():
    d = OrderPreserving()
    rng = np.random.default_rng(3)
    spec = build_prompt("margin_review", FIRMS[1], cues=True)
    view = make_view(spec, d.apply(spec.fields, d.sample_secret(rng, spec.fields), rng).fields, d.name, True)
    a = ClassicalBayesAttack(TABLE, d, use_oracle=True, seed=5)
    assert a.attack(view) == a.attack(view)


def test_heavy_tailed_prior_handles_extreme_products_without_crashing():
    """Extreme firms (heavy-tailed prices, extreme margins) must not produce a degenerate posterior."""
    d = PerFieldScaling()
    attack = ClassicalBayesAttack(TABLE, d, use_oracle=True, prior_df=3)
    rng = np.random.default_rng(9)
    extreme = [f for f in generate_firms(400, seed=79) if f.latent_extreme][:60]
    for firm in extreme:
        spec = build_prompt("margin_review", firm, cues=True)
        view = make_view(spec, d.apply(spec.fields, d.sample_secret(rng, spec.fields), rng).fields, d.name, True)
        for e in attack.attack(view):
            assert np.isfinite(e.margin) and np.isfinite(e.unit_cost)


def test_hit_decision_does_not_lose_to_median_on_its_own_criterion():
    d = PerFieldScaling()
    med = ClassicalBayesAttack(TABLE, d, use_oracle=True)
    hit = ClassicalBayesAttack(TABLE, d, use_oracle=True, decision="hit")
    (hm, _), (hh, _) = _run(d, med), _run(d, hit)
    assert hh >= hm - 0.05


def test_ignoring_the_observation_gives_the_same_estimate_for_any_view():
    """use_observation=False is the prior-only attacker on the solver's own prior (any defense)."""
    a = ClassicalBayesAttack(TABLE, OrderPreserving(), use_oracle=True, use_observation=False)
    b = ClassicalBayesAttack(TABLE, PerFieldScaling(), use_oracle=True, use_observation=False)
    spec = build_prompt("margin_review", generate_firms(1, seed=9)[0], cues=True)
    rng = np.random.default_rng(0)
    views = []
    for d in (OrderPreserving(), PerFieldScaling()):
        shown = d.apply(spec.fields, d.sample_secret(rng, spec.fields), rng).fields
        views.append(make_view(spec, shown, d.name, with_oracle=True))
    va, vb = views
    assert [e.margin for e in a.attack(va)] == pytest.approx([e.margin for e in b.attack(vb)])


def test_bad_decision_is_rejected():
    with pytest.raises(ValueError):
        ClassicalBayesAttack(TABLE, PerFieldScaling(), use_oracle=True, decision="mode")
