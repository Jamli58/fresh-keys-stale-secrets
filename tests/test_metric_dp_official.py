import math

import numpy as np
import pytest

from numpriv.attacks.likelihoods import d3o_log_likelihood
from numpriv.defenses import MetricDPOfficial
from numpriv.defenses.metric_dp_official import BINS, bucket, output_grid
from numpriv.fields import NumField


def official_pmf(x, n_lower, n_upper, epsilon, discretization_size=100):
    """Verbatim probability computation of preempt/sanitizer.py::M_epsilon (commit 1032978)."""
    n_upper = int(n_upper)
    n_lower = int(n_lower)
    total_range = n_upper - n_lower
    x = (x - n_lower) * discretization_size / total_range
    p_i = [math.exp(-abs(x - s) * epsilon / 2) for s in range(discretization_size)]
    p_i = [val / sum(p_i) for val in p_i]
    outputs = [int(s * total_range / discretization_size + n_lower) for s in range(discretization_size)]
    return outputs, p_i


def official_bucket(real_money):
    """Verbatim range rule of preempt/sanitizer.py::encypt_money."""
    v = int(float(real_money))
    if v == 1:
        return 1, 2
    if v < 100:
        return 2, 1000
    if v < 1000:
        return 100, 10000
    return 1000, 10000


@pytest.mark.parametrize("value", (0.4, 1.0, 1.99, 4.5, 99.99, 100.0, 250.0, 999.0, 1400.0, 3200.5, 25000.0))
def test_bucket_matches_official_rule(value):
    assert bucket(value) == official_bucket(value)


@pytest.mark.parametrize("value", (4.5, 85.0, 420.0, 1400.0, 25000.0))
@pytest.mark.parametrize("eps", (0.5, 1.0, 2.0))
def test_likelihood_equals_official_pmf(value, eps):
    lo, hi = official_bucket(value)
    outputs, pmf = official_pmf(int(float(value)), lo, hi, eps)
    mass = {}
    for y, p in zip(outputs, pmf):
        mass[y] = mass.get(y, 0.0) + p
    for y, p in mass.items():
        got = float(np.exp(d3o_log_likelihood(float(y), np.array([value]), eps))[0])
        assert got == pytest.approx(p, rel=1e-9, abs=1e-15)
    assert sum(mass.values()) == pytest.approx(1.0)


def test_output_outside_the_values_bucket_grid_has_zero_likelihood():
    # 4.5 lives in bucket (2, 1000); 4960 = 1000 + 90 * 44 is only on the (1000, 10000) grid
    assert d3o_log_likelihood(4960.0, np.array([4.5]), 1.0)[0] == -np.inf
    assert np.isfinite(d3o_log_likelihood(4960.0, np.array([4200.0]), 1.0)[0])


def test_output_grid_has_one_point_per_bin():
    grid = output_grid(2, 1000)
    assert len(grid) == BINS and grid[0] == 2


def test_sampler_frequencies_match_official_pmf():
    d = MetricDPOfficial(1.0)
    rng = np.random.default_rng(0)
    field = (NumField("price:0", "money", 420.0),)
    draws = [d.apply(field, None, rng).fields[0].value for _ in range(20000)]
    outputs, pmf = official_pmf(420, 100, 10000, 1.0)
    expected = dict(zip(outputs, pmf))
    top = max(expected, key=expected.get)
    assert np.mean(np.array(draws) == top) == pytest.approx(expected[top], abs=0.01)
    assert set(draws) <= set(outputs)


def test_only_money_fields_are_noised_and_epsilon_is_per_value():
    d = MetricDPOfficial(1.0)
    fields = (NumField("price:0", "money", 4.5), NumField("volume:0", "count", 37), NumField("year:0", "year", 2025))
    shown = d.apply(fields, None, np.random.default_rng(1)).fields
    assert shown[1].value == 37 and shown[2].value == 2025
    assert d.sample_secret(np.random.default_rng(0), fields) is None


def test_epsilon_must_be_positive():
    with pytest.raises(ValueError):
        MetricDPOfficial(0.0)


def test_a1_plus_attacks_d3o_and_beats_the_prior_floor_on_cost():
    from numpriv.attacks import PriorOnlyAttack, make_view, truths_for
    from numpriv.attacks.bayes import ClassicalBayesAttack
    from numpriv.metrics import hit_rate
    from numpriv.priors import build_prior_table
    from numpriv.records import generate_firms
    from numpriv.templates import build_prompt

    table = build_prior_table(generate_firms(2000, seed=77))
    d = MetricDPOfficial(2.0)
    a1p = ClassicalBayesAttack(table, d, use_oracle=True, n_samples=2000, prior_df=3.0, prior_scale=0.75)
    a0 = PriorOnlyAttack(table)
    rng = np.random.default_rng(5)
    est, floor, tru = [], [], []
    for firm in generate_firms(60, seed=78):
        spec = build_prompt("margin_review", firm, cues=True)
        view = make_view(spec, d.apply(spec.fields, None, rng).fields, d.name, with_oracle=True)
        est += a1p.attack(view)
        floor += a0.attack(view)
        tru += truths_for(spec)
    c = np.array([t.unit_cost for t in tru])
    assert all(np.isfinite(e.unit_cost) for e in est)
    assert hit_rate(np.array([e.unit_cost for e in est]), c, rel_tol=0.1) > hit_rate(
        np.array([e.unit_cost for e in floor]), c, rel_tol=0.1)
