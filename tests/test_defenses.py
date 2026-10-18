import numpy as np
import pytest
from scipy import stats

from numpriv.defenses import MetricDP, NoDefense, OrderPreserving, PerFieldScaling, SharedScaling
from numpriv.fields import NumField

FIELDS = (
    NumField("price:a", "money", 12.5),
    NumField("cost:a", "money", 7.25),
    NumField("price:b", "money", 480.0),
    NumField("cost:b", "money", 300.0),
    NumField("volume:a", "count", 900.0),
    NumField("year:0", "year", 2024.0),
    NumField("day:0", "day", 30.0),
)
ALL_DEFENSES = (NoDefense(), SharedScaling(), PerFieldScaling(), OrderPreserving(), MetricDP(epsilon=1.0))


def _by_id(fields):
    return {f.field_id: f.value for f in fields}


def _margin(values, suffix):
    return (values[f"price:{suffix}"] - values[f"cost:{suffix}"]) / values[f"price:{suffix}"]


@pytest.mark.parametrize("defense", ALL_DEFENSES, ids=lambda d: d.name)
def test_client_recovers_inputs_exactly(defense):
    rng = np.random.default_rng(0)
    for _ in range(50):
        secret = defense.sample_secret(rng, FIELDS)
        disguised = defense.apply(FIELDS, secret, rng)
        assert disguised.recover() == FIELDS


@pytest.mark.parametrize("defense", ALL_DEFENSES, ids=lambda d: d.name)
def test_apply_preserves_ids_kinds_and_does_not_mutate(defense):
    rng = np.random.default_rng(1)
    before = tuple(FIELDS)
    disguised = defense.apply(FIELDS, defense.sample_secret(rng, FIELDS), rng).fields
    assert FIELDS == before
    assert [(f.field_id, f.kind) for f in disguised] == [(f.field_id, f.kind) for f in FIELDS]


def test_no_defense_is_identity():
    rng = np.random.default_rng(2)
    d = NoDefense()
    assert d.apply(FIELDS, d.sample_secret(rng, FIELDS), rng).fields == FIELDS


def test_shared_scaling_keeps_margins_within_rounding_and_counts():
    rng = np.random.default_rng(3)
    d = SharedScaling()
    for _ in range(200):
        secret = d.sample_secret(rng, FIELDS)
        assert 0.1 <= secret.k <= 10.0
        out = _by_id(d.apply(FIELDS, secret, rng).fields)
        orig = _by_id(FIELDS)
        for s in ("a", "b"):
            # each shown value is off by <= half a cent, so |dm| <= 0.01 / shown price
            bound = 0.01 / out[f"price:{s}"] + 1e-12
            assert abs(_margin(out, s) - _margin(orig, s)) <= bound
        assert out["volume:a"] == orig["volume:a"]
        assert out["year:0"] == orig["year:0"]


@pytest.mark.parametrize("defense", (SharedScaling(), PerFieldScaling()), ids=lambda d: d.name)
def test_scaled_money_is_shown_in_whole_cents(defense):
    """Showing extra digits would expose the original cent grid (a key-recovery channel)."""
    rng = np.random.default_rng(12)
    for _ in range(100):
        for f in defense.apply(FIELDS, defense.sample_secret(rng, FIELDS), rng).fields:
            if f.kind == "money":
                assert round(f.value, 2) == f.value


def test_shared_scaling_inverts_derived_money_values():
    rng = np.random.default_rng(4)
    d = SharedScaling()
    secret = d.sample_secret(rng, FIELDS)
    out = _by_id(d.apply(FIELDS, secret, rng).fields)
    revenue = out["price:b"] * 900.0
    assert d.invert_money(revenue, secret) == pytest.approx(480.0 * 900.0, rel=1e-3)


def test_per_field_scaling_breaks_margin_invariant():
    rng = np.random.default_rng(5)
    d = PerFieldScaling()
    diffs = []
    for _ in range(200):
        out = _by_id(d.apply(FIELDS, d.sample_secret(rng, FIELDS), rng).fields)
        diffs.append(abs(_margin(out, "b") - _margin(_by_id(FIELDS), "b")))
    assert np.median(diffs) > 0.1


def test_order_preserving_keeps_order_equality_and_year_differences():
    fields = FIELDS + (NumField("cost:c", "money", 12.5), NumField("year:1", "year", 2019.0))
    rng = np.random.default_rng(6)
    d = OrderPreserving()
    for _ in range(100):
        out = d.apply(fields, d.sample_secret(rng, fields), rng).fields
        mapped = [(f.value, o.value) for f, o in zip(fields, out) if f.kind not in ("year", "day")]
        for (x1, y1) in mapped:
            for (x2, y2) in mapped:
                assert (x1 < x2) == (y1 < y2)
                assert (x1 == x2) == (y1 == y2)
        values = _by_id(out)
        assert values["year:0"] - values["year:1"] == 5.0
        assert values["year:0"] != 2024.0
        assert values["day:0"] == 30.0
        assert all(1.0 <= y <= 1e6 for _, y in mapped)


def test_order_preserving_leaves_out_of_range_years_as_ordinary_numbers():
    d = OrderPreserving()
    rng = np.random.default_rng(7)
    fields = (NumField("year:old", "year", 1985.0), NumField("price:a", "money", 5.0))
    out = _by_id(d.apply(fields, d.sample_secret(rng, fields), rng).fields)
    assert out["year:old"] > out["price:a"]  # mapped with the pool, order kept


def test_metric_dp_outputs_stay_in_domain_and_keep_non_value_fields():
    d = MetricDP(epsilon=1.0)
    rng = np.random.default_rng(8)
    for _ in range(200):
        out = d.apply(FIELDS, d.sample_secret(rng, FIELDS), rng).fields
        for f in out:
            if f.kind in d.value_kinds:
                lo, hi = d.domain[f.kind]
                assert lo <= f.value <= hi
        values = _by_id(out)
        assert values["year:0"] == 2024.0 and values["day:0"] == 30.0


def test_metric_dp_splits_budget_over_value_tokens():
    d = MetricDP(epsilon=2.0)
    assert d.per_token_epsilon(FIELDS) == pytest.approx(2.0 / 5)


def test_metric_dp_density_satisfies_metric_privacy_bound():
    d = MetricDP(epsilon=1.0)
    eps, (lo, hi) = 0.3, d.domain["money"]
    rng = np.random.default_rng(9)
    for _ in range(500):
        x1, x2, y = rng.uniform(lo, hi, size=3)
        gap = d.log_density(y, x1, eps, lo, hi) - d.log_density(y, x2, eps, lo, hi)
        assert gap <= eps * abs(x1 - x2) + 1e-9


def test_metric_dp_sampler_matches_its_density():
    d = MetricDP(epsilon=1.0)
    eps, lo, hi, x = 0.05, 0.0, 200.0, 20.0
    rng = np.random.default_rng(10)
    samples = np.array([d.sample_value(x, eps, lo, hi, rng) for _ in range(4000)])
    grid = np.linspace(lo, hi, 4001)
    pdf = np.exp([d.log_density(g, x, eps, lo, hi) for g in grid])
    cdf = np.concatenate(([0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(grid))))
    result = stats.kstest(samples, lambda s: np.interp(s, grid, cdf))
    assert result.pvalue > 0.01


def test_metric_dp_rejects_bad_epsilon():
    with pytest.raises(ValueError):
        MetricDP(epsilon=0.0)


def test_order_preserving_rules_are_text_level_not_type_level():
    """Zhang et al. define the year and special-number rules on integer values, not field types."""
    d = OrderPreserving()
    rng = np.random.default_rng(11)
    fields = (
        NumField("year:0", "year", 2020.0),
        NumField("volume:a", "count", 2000.0),
        NumField("price:a", "money", 30.0),
        NumField("price:b", "money", 30.5),
    )
    for _ in range(20):
        secret = d.sample_secret(rng, fields)
        out = _by_id(d.apply(fields, secret, rng).fields)
        assert out["volume:a"] - out["year:0"] == -20.0  # same offset applied to both integers
        assert out["price:a"] == 30.0  # special number kept even though it is a price
        assert out["price:b"] != 30.5


def test_order_preserving_secret_is_a_reusable_function():
    """Reusing the secret must reuse the mapping h (key-reuse regime B2)."""
    d = OrderPreserving()
    rng = np.random.default_rng(13)
    secret = d.sample_secret(rng, FIELDS)
    first = _by_id(d.apply(FIELDS, secret, rng).fields)
    other = (NumField("price:z", "money", 12.5), NumField("cost:z", "money", 999.0))
    second = _by_id(d.apply(other, secret, rng).fields)
    assert second["price:z"] == first["price:a"]
    assert (second["cost:z"] > first["price:b"]) == (999.0 > 480.0)


def test_order_preserving_is_monotone_across_calls_and_signs():
    d = OrderPreserving()
    rng = np.random.default_rng(14)
    secret = d.sample_secret(rng, ())
    xs = np.concatenate([-np.logspace(4, -2, 40), [0.0], np.logspace(-2, 7, 200)])
    xs = xs[[not (float(x).is_integer() and (1990 <= x <= 2030 or 28 <= x <= 31)) for x in xs]]
    fields = tuple(NumField(f"n{i}", "money", float(x)) for i, x in enumerate(xs))
    ys = [f.value for f in d.apply(fields, secret, rng).fields]
    assert all(a < b for a, b in zip(ys, ys[1:]))
    assert all(1.0 <= y <= 1e6 for y in ys)


def test_order_preserving_fresh_secrets_differ():
    d = OrderPreserving()
    rng = np.random.default_rng(15)
    outs = {_by_id(d.apply(FIELDS, d.sample_secret(rng, FIELDS), rng).fields)["price:a"] for _ in range(20)}
    assert len(outs) > 15


def test_order_preserving_never_merges_distinct_values_in_display():
    """Close inputs (e.g. 48568 vs 48609, or 312011 vs 312012) must stay distinct after display rounding."""
    d = OrderPreserving()
    rng = np.random.default_rng(16)
    values = (48568.0, 48609.0, 312011.0, 312012.0, 312013.0, 5829.0, 5735.0, 0.01, 0.02, 1234567.89, 1234567.9)
    fields = tuple(NumField(f"n{i}", "money", v) for i, v in enumerate(values))
    for _ in range(2000):
        shown = [f.value for f in d.apply(fields, d.sample_secret(rng, fields), rng).fields]
        assert len(set(shown)) == len(set(values))
        order = sorted(range(len(values)), key=values.__getitem__)
        assert all(shown[a] < shown[b] for a, b in zip(order, order[1:]))


def test_order_preserving_knots_have_a_minimum_slope():
    from numpriv.defenses.order_preserving import MIN_LOG_SLOPE, N_INTERVALS, SIGNED_LOG_LIMIT

    d = OrderPreserving()
    rng = np.random.default_rng(17)
    width = 2 * SIGNED_LOG_LIMIT / N_INTERVALS
    for _ in range(500):
        knots = np.array(d.sample_secret(rng, ()).knot_log_targets)
        assert np.all(np.diff(knots) >= MIN_LOG_SLOPE * width - 1e-12)
        assert knots[0] == pytest.approx(np.log(d.target_low)) and knots[-1] == pytest.approx(np.log(d.target_high))


@pytest.mark.parametrize("law", ({"n_intervals": 16}, {"n_intervals": 256}, {"gap_concentration": 0.1},
                                 {"gap_concentration": 10.0}, {"target_high": 1e12}), ids=str)
def test_order_preserving_alternative_laws_keep_endpoints_slope_and_order(law):
    from numpriv.defenses.order_preserving import MIN_LOG_SLOPE, SIGNED_LOG_LIMIT

    d = OrderPreserving(**law)
    rng = np.random.default_rng(31)
    width = 2 * SIGNED_LOG_LIMIT / d.n_intervals
    for _ in range(200):
        secret = d.sample_secret(rng, ())
        knots = np.array(secret.knot_log_targets)
        assert knots.size == d.n_intervals + 1
        assert np.all(np.diff(knots) >= MIN_LOG_SLOPE * width - 1e-12)
        assert knots[0] == pytest.approx(np.log(d.target_low)) and knots[-1] == pytest.approx(np.log(d.target_high))
        assert secret.h(3.0) < secret.h(4.0) < secret.h(400.0)


def test_order_preserving_default_law_keeps_its_random_stream():
    """The default law must reproduce earlier results: same knots as uniform sorted cuts."""
    a = OrderPreserving().sample_secret(np.random.default_rng(5), ())
    b = OrderPreserving(n_intervals=64, gap_concentration=1.0).sample_secret(np.random.default_rng(5), ())
    assert a == b


def test_order_preserving_rejects_bad_law():
    with pytest.raises(ValueError):
        OrderPreserving(n_intervals=1)
    with pytest.raises(ValueError):
        OrderPreserving(gap_concentration=0.0)
