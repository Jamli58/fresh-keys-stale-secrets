import numpy as np
import pytest

from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.attacks.multi_query import MultiQueryAttack, key_support
from numpriv.defenses import MetricDP, OrderPreserving, PerFieldScaling, SharedScaling
from numpriv.priors import build_prior_table
from numpriv.records import generate_firms
from numpriv.sequences import build_sequence, disguise_sequence

TABLE = build_prior_table(generate_firms(2000, seed=77))
FIRMS = generate_firms(40, seed=79)
FROZEN = {"n_samples": 2000, "prior_df": 3.0, "prior_scale": 0.75}
DEFENSES = (SharedScaling(), PerFieldScaling(), OrderPreserving(), MetricDP(1.0))


def _mq(defense, regime):
    return MultiQueryAttack(ClassicalBayesAttack(TABLE, defense, use_oracle=True, **FROZEN), regime)


def _seq(defense, regime, firm, n, seed=0, start=1):
    return disguise_sequence(build_sequence(firm, n, start), defense, regime, np.random.default_rng(seed))


@pytest.mark.parametrize("defense", DEFENSES, ids=lambda d: d.name)
@pytest.mark.parametrize("regime", ("fresh", "reuse"))
def test_one_query_reproduces_single_query_a1_plus(defense, regime):
    """n = 1 is exactly M1's A1+ on that prompt (product 0)."""
    single = ClassicalBayesAttack(TABLE, defense, use_oracle=True, **FROZEN)
    for firm in FIRMS[:5]:
        seq = _seq(defense, regime, firm, 1)
        got = _mq(defense, regime).attack(seq.views)
        want = single.attack(seq.views[0])[0]
        assert (got.margin, got.unit_cost) == pytest.approx((want.margin, want.unit_cost))


@pytest.mark.parametrize("defense", (SharedScaling(), PerFieldScaling(), OrderPreserving()), ids=lambda d: d.name)
def test_reused_key_repeats_add_no_information(defense):
    """The same task asked twice under one key shows the same numbers; counting them twice would be wrong."""
    for firm in FIRMS[:5]:
        once = _seq(defense, "reuse", firm, 1, start=0)
        twice = disguise_sequence((once.specs[0], once.specs[0]), defense, "reuse", np.random.default_rng(0))
        a, b = _mq(defense, "reuse").attack(once.views), _mq(defense, "reuse").attack(twice.views)
        assert (a.margin, a.unit_cost) == pytest.approx((b.margin, b.unit_cost))


def test_key_support_intersects_the_ranges_implied_by_each_fresh_key():
    grid = np.linspace(np.log(0.1), np.log(10.0), 1001)
    full = key_support(grid, np.array([0.0]), 0.1, 10.0)
    assert full.all()
    # a second key 4x larger than the first rules out k1 > 10 / 4
    both = key_support(grid, np.array([0.0, np.log(4.0)]), 0.1, 10.0)
    assert np.exp(grid[both]).max() == pytest.approx(2.5, rel=1e-2)
    assert np.exp(grid[both]).min() == pytest.approx(0.1, rel=1e-2)


def _mean_key_error(regime, n):
    errs = []
    for i, firm in enumerate(FIRMS):
        seq = _seq(SharedScaling(), regime, firm, n, seed=i)
        k_hat = _mq(SharedScaling(), regime).attack(seq.views).key
        errs.append(abs(np.log(k_hat / seq.secrets[0].k)))
    return float(np.mean(errs))


def test_d1_fresh_keys_pin_the_key_down_as_queries_accumulate_while_a_reused_key_does_not():
    """Bounded fresh keys intersect their ranges; a reused key only repeats the same numbers
    (start=1 opens with the margin review, so every product is already seen at n = 1)."""
    fresh_1, fresh_16 = _mean_key_error("fresh", 1), _mean_key_error("fresh", 16)
    reuse_1, reuse_16 = _mean_key_error("reuse", 1), _mean_key_error("reuse", 16)
    assert fresh_16 < 0.75 * fresh_1
    assert reuse_16 == pytest.approx(reuse_1)
    assert fresh_16 < reuse_16


def test_d1_returns_a_key_estimate_and_exact_margin():
    firm = FIRMS[0]
    seq = _seq(SharedScaling(), "fresh", firm, 4)
    est = _mq(SharedScaling(), "fresh").attack(seq.views)
    assert est.key is not None
    assert est.margin == pytest.approx(firm.products[0].margin, abs=0.01)


def _mean_cost_log_error(defense, regime, n):
    errs = []
    for i, firm in enumerate(FIRMS):
        seq = _seq(defense, regime, firm, n, seed=i)
        est = _mq(defense, regime).attack(seq.views)
        errs.append(abs(np.log(est.unit_cost / firm.products[0].unit_cost)))
    return float(np.mean(errs))


def test_metric_dp_repeated_releases_average_the_noise():
    d = MetricDP(0.5)
    assert _mean_cost_log_error(d, "repeat", 16) < 0.8 * _mean_cost_log_error(d, "repeat", 1)


def test_non_metric_estimates_have_no_key_except_d1():
    seq = _seq(MetricDP(1.0), "repeat", FIRMS[0], 3)
    assert _mq(MetricDP(1.0), "repeat").attack(seq.views).key is None


def test_multi_query_requires_the_oracle_solver_and_views():
    with pytest.raises(ValueError):
        MultiQueryAttack(ClassicalBayesAttack(TABLE, SharedScaling(), use_oracle=False, **FROZEN), "fresh")
    with pytest.raises(ValueError):
        _mq(SharedScaling(), "fresh").attack(())


def test_plaintext_sequence_is_read_exactly():
    from numpriv.defenses import NoDefense

    firm = FIRMS[0]
    seq = _seq(NoDefense(), "fresh", firm, 3)
    est = _mq(NoDefense(), "fresh").attack(seq.views)
    assert est.unit_cost == firm.products[0].unit_cost
    assert est.margin == pytest.approx(firm.products[0].margin, abs=1e-3)
