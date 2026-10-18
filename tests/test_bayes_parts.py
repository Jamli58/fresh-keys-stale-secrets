import numpy as np
import pytest

from numpriv.attacks.likelihoods import d2_log_likelihood, d3_log_likelihood
from numpriv.attacks.prior_fit import fit_logit_normal, fit_log_normal, weighted_median
from numpriv.defenses import MetricDP, OrderPreserving
from numpriv.priors import Summary


def test_log_normal_fit_reproduces_median_and_iqr():
    rng = np.random.default_rng(0)
    x = np.exp(rng.normal(np.log(40.0), 0.5, size=200_000))
    mu, sigma = fit_log_normal(Summary.of(x))
    assert np.exp(mu) == pytest.approx(40.0, rel=0.01)
    assert sigma == pytest.approx(0.5, rel=0.02)


def test_logit_normal_fit_reproduces_median_and_iqr():
    rng = np.random.default_rng(1)
    m = 1 / (1 + np.exp(-rng.normal(0.4, 0.6, size=200_000)))
    mu, sigma = fit_logit_normal(Summary.of(m))
    assert mu == pytest.approx(0.4, abs=0.01)
    assert sigma == pytest.approx(0.6, rel=0.02)


def test_degenerate_summary_gets_a_positive_width():
    mu, sigma = fit_log_normal(Summary(10.0, 10.0, 10.0, 10.0, 10.0))
    assert sigma > 0


def test_weighted_median():
    assert weighted_median(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0])) == 2.0
    assert weighted_median(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 5.0])) == 3.0
    assert weighted_median(np.array([5.0, 1.0]), np.array([1.0, 3.0])) == 1.0


def test_d3_vectorized_likelihood_matches_scalar_density():
    d = MetricDP(epsilon=1.0)
    lo, hi = d.domain["money"]
    xs = np.array([3.0, 40.0, 900.0, 49_000.0])
    got = d3_log_likelihood(y=37.5, xs=xs, eps=0.2, lo=lo, hi=hi)
    want = [d.log_density(37.5, x, 0.2, lo, hi) for x in xs]
    assert got == pytest.approx(want)


def test_d2_likelihood_separates_values_more_the_further_apart_they_are():
    """D2 reveals magnitude only weakly (its targets are random): x vs 50x is near chance,
    but discrimination must grow with the gap. Measured: ~0.42 at 50x, ~0.78 at 1e5x."""
    d = OrderPreserving()
    rng = np.random.default_rng(2)
    rates = []
    for factor in (50.0, 1e3, 1e5):
        wins = 0
        for _ in range(600):
            x = float(np.exp(rng.uniform(0.0, np.log(1e5))))
            y = d.sample_secret(rng, ()).h(x)
            ll = d2_log_likelihood(y, np.array([x, x * factor, x / factor]))
            wins += ll[0] > max(ll[1], ll[2])
        rates.append(wins / 600)
    assert rates[0] < rates[1] < rates[2]
    assert rates[2] > 0.7


def test_d2_likelihood_is_calibrated():
    """Standardized residuals of log y given the true x should look roughly standard normal."""
    from numpriv.attacks.likelihoods import d2_log_y_moments

    d = OrderPreserving()
    rng = np.random.default_rng(3)
    z = []
    for _ in range(2000):
        x = float(np.exp(rng.uniform(np.log(0.5), np.log(1e6))))
        y = d.sample_secret(rng, ()).h(x)
        mean, sd = d2_log_y_moments(np.array([x]))
        z.append((np.log(y) - mean[0]) / sd[0])
    assert abs(np.mean(z)) < 0.1
    assert 0.85 < np.std(z) < 1.15


def test_d1p_likelihood_accounts_for_cent_rounding():
    from numpriv.attacks.likelihoods import d1p_log_likelihood

    xs = np.array([0.01, 0.04, 5.0, 2000.0])
    ll = d1p_log_likelihood(0.0, xs, 0.1, 10.0)  # a shown 0.00: only tiny x can round to 0
    assert np.isfinite(ll[0]) and np.isfinite(ll[1])
    assert ll[2] == -np.inf and ll[3] == -np.inf
    # exact probability: k in [0.1, 0.005/0.04] out of log-range log(100)
    assert np.exp(ll[1]) == pytest.approx(np.log(0.125 / 0.1) / np.log(100.0))
    ll_mid = d1p_log_likelihood(12.34, np.array([10.0]), 0.1, 10.0)
    assert np.exp(ll_mid[0]) == pytest.approx(np.log(12.345 / 12.335) / np.log(100.0))


def test_student_t_scale_matches_the_same_iqr():
    from scipy import stats as st

    from numpriv.attacks.prior_fit import t_scale_from_normal_sigma

    scale = t_scale_from_normal_sigma(0.5, df=3)
    iqr = 2 * st.t.ppf(0.75, 3) * scale
    assert iqr == pytest.approx(2 * st.norm.ppf(0.75) * 0.5)


def test_d2_pair_model_is_calibrated():
    """(log h(p), log h(c)) share one secret; the bivariate Gaussian model should be calibrated:
    squared Mahalanobis distances average ~2 (chi-square with 2 dof)."""
    from numpriv.attacks.likelihoods import d2_pair_moments

    d = OrderPreserving()
    rng = np.random.default_rng(20)
    m2 = []
    for _ in range(2000):
        p = float(np.exp(rng.uniform(np.log(1.0), np.log(5e3))))
        c = p * float(rng.uniform(0.05, 0.95))
        secret = d.sample_secret(rng, ())
        obs = np.log([secret.h(p), secret.h(c)])
        mean, cov = d2_pair_moments(np.array([p]), np.array([c]))
        diff = obs - mean[0]
        m2.append(float(diff @ np.linalg.solve(cov[0], diff)))
    assert 1.6 < np.mean(m2) < 2.5


def test_d2_joint_pair_identifies_margin_better_than_independent_fields():
    from numpriv.attacks.likelihoods import d2_log_likelihood, d2_pair_log_likelihood

    d = OrderPreserving()
    rng = np.random.default_rng(21)
    joint_wins = indep_wins = 0
    for _ in range(600):
        p = float(np.exp(rng.uniform(np.log(2.0), np.log(2e3))))
        m_true = float(rng.uniform(0.2, 0.7))
        m_alt = min(0.95, m_true + 0.25)
        secret = d.sample_secret(rng, ())
        yp, yc = secret.h(p), secret.h(p * (1 - m_true))
        ps = np.array([p, p])
        cs = np.array([p * (1 - m_true), p * (1 - m_alt)])
        j = d2_pair_log_likelihood(yp, yc, ps, cs)
        i = d2_log_likelihood(yp, ps) + d2_log_likelihood(yc, cs)
        joint_wins += j[0] > j[1]
        indep_wins += i[0] > i[1]
    assert joint_wins / 600 > indep_wins / 600 + 0.1


@pytest.mark.parametrize("law", ({"n_intervals": 16}, {"gap_concentration": 0.1}, {"target_high": 1e12}), ids=str)
def test_d2_pair_model_follows_the_defense_law(law):
    """The attacker knows the map law: its Gaussian model must be calibrated for each law it is given."""
    from numpriv.attacks.likelihoods import d2_pair_moments

    d = OrderPreserving(**law)
    rng = np.random.default_rng(22)
    m2 = []
    for _ in range(1500):
        p = float(np.exp(rng.uniform(np.log(1.0), np.log(5e3))))
        c = p * float(rng.uniform(0.05, 0.95))
        secret = d.sample_secret(rng, ())
        obs = np.log([secret.h(p), secret.h(c)])
        mean, cov = d2_pair_moments(np.array([p]), np.array([c]), d)
        diff = obs - mean[0]
        m2.append(float(diff @ np.linalg.solve(cov[0], diff)))
    assert 1.5 < np.mean(m2) < 2.7


def test_hit_optimal_absolute_window_maximizes_mass():
    from numpriv.attacks.prior_fit import hit_optimal_absolute

    rng = np.random.default_rng(3)
    values = np.concatenate([rng.normal(0.2, 0.01, 300), rng.uniform(0.0, 1.0, 700)])
    w = np.ones(values.size)
    g = hit_optimal_absolute(values, w, 0.05)
    mass = lambda c: w[np.abs(values - c) <= 0.05].sum()  # noqa: E731
    assert abs(g - 0.2) < 0.05
    assert all(mass(g) >= mass(c) - 1e-9 for c in np.linspace(0, 1, 201))


def test_hit_optimal_relative_window_maximizes_relative_hits():
    from numpriv.attacks.prior_fit import hit_optimal_relative

    rng = np.random.default_rng(4)
    values = np.exp(rng.normal(np.log(50.0), 0.6, 2000))
    w = rng.uniform(0.5, 1.5, values.size)
    g = hit_optimal_relative(values, w, 0.10)
    hits = lambda c: w[np.abs(c - values) <= 0.10 * values].sum()  # noqa: E731
    assert all(hits(g) >= hits(c) - 1e-9 for c in np.exp(np.linspace(np.log(10), np.log(250), 400)))


def test_hit_optimal_window_rejects_bad_weights():
    from numpriv.attacks.prior_fit import hit_optimal_absolute

    with pytest.raises(ValueError):
        hit_optimal_absolute(np.array([0.1, 0.2]), np.zeros(2), 0.05)
