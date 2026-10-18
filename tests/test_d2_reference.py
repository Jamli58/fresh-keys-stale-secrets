import numpy as np
import pytest

from numpriv.attacks.d2_reference import SimulatedMapLikelihood
from numpriv.defenses import OrderPreserving


def test_simulated_likelihood_prefers_the_true_pair():
    d = OrderPreserving()
    lik = SimulatedMapLikelihood(d, n_maps=4000, bandwidth=0.05, seed=1)
    rng = np.random.default_rng(2)
    wins = 0
    for _ in range(200):
        p = float(np.exp(rng.uniform(np.log(2.0), np.log(2e3))))
        m_true = float(rng.uniform(0.2, 0.7))
        m_alt = min(0.95, m_true + 0.25)
        secret = d.sample_secret(rng, ())
        ll = lik.pair_log_likelihood(secret.h(p), secret.h(p * (1 - m_true)),
                                     np.array([p, p]), np.array([p * (1 - m_true), p * (1 - m_alt)]))
        assert np.all(np.isfinite(ll)) or ll[0] > ll[1]
        wins += ll[0] > ll[1]
    assert wins / 200 > 0.75


def test_simulated_likelihood_is_deterministic_and_follows_the_law():
    a = SimulatedMapLikelihood(OrderPreserving(), n_maps=500, bandwidth=0.05, seed=3)
    b = SimulatedMapLikelihood(OrderPreserving(), n_maps=500, bandwidth=0.05, seed=3)
    c = SimulatedMapLikelihood(OrderPreserving(gap_concentration=0.1), n_maps=500, bandwidth=0.05, seed=3)
    args = (1500.0, 1200.0, np.array([10.0, 20.0]), np.array([5.0, 9.0]))
    assert np.allclose(a.pair_log_likelihood(*args), b.pair_log_likelihood(*args))
    assert not np.allclose(a.pair_log_likelihood(*args), c.pair_log_likelihood(*args))


def test_simulated_likelihood_density_integrates_to_about_one():
    """The kernel estimate is a density in (log y_p, log y_c): it integrates to ~1 over a wide grid."""
    lik = SimulatedMapLikelihood(OrderPreserving(), n_maps=2000, bandwidth=0.1, seed=4)
    grid = np.linspace(-1.0, 15.0, 161)
    step = grid[1] - grid[0]
    total = 0.0
    for up in grid:
        ll = np.array([lik.pair_log_likelihood(np.exp(up), np.exp(uc), np.array([40.0]), np.array([25.0]))[0]
                       for uc in grid])
        total += np.exp(ll).sum() * step * step
    assert total == pytest.approx(1.0, abs=0.05)


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        SimulatedMapLikelihood(OrderPreserving(), n_maps=0, bandwidth=0.05)
    with pytest.raises(ValueError):
        SimulatedMapLikelihood(OrderPreserving(), n_maps=10, bandwidth=0.0)


def test_reference_attack_requires_matching_law():
    from numpriv.attacks.d2_reference import SimulatedD2BayesAttack
    from numpriv.priors import build_prior_table
    from numpriv.records import generate_firms

    table = build_prior_table(generate_firms(300, seed=5))
    ref = SimulatedMapLikelihood(OrderPreserving(gap_concentration=0.1), n_maps=50, bandwidth=0.05)
    with pytest.raises(ValueError):
        SimulatedD2BayesAttack(table, OrderPreserving(), use_oracle=True, reference=ref)


def test_reference_attack_recovers_margins_like_the_gaussian_attack():
    from numpriv.attacks import make_view, truths_for
    from numpriv.attacks.bayes import ClassicalBayesAttack
    from numpriv.attacks.d2_reference import SimulatedD2BayesAttack
    from numpriv.priors import build_prior_table
    from numpriv.records import generate_firms
    from numpriv.templates import build_prompt

    d = OrderPreserving()
    table = build_prior_table(generate_firms(1000, seed=6))
    gauss = ClassicalBayesAttack(table, d, use_oracle=True, n_samples=1500)
    ref = SimulatedD2BayesAttack(table, d, use_oracle=True, n_samples=1500,
                                 reference=SimulatedMapLikelihood(d, n_maps=3000, bandwidth=0.05))
    rng = np.random.default_rng(7)
    err_g, err_r = [], []
    for firm in generate_firms(25, seed=8):
        spec = build_prompt("margin_review", firm, cues=True)
        shown = d.apply(spec.fields, d.sample_secret(rng, spec.fields), rng).fields
        view = make_view(spec, shown, d.name, with_oracle=True)
        for eg, er, t in zip(gauss.attack(view), ref.attack(view), truths_for(spec)):
            err_g.append(abs(eg.margin - t.margin))
            err_r.append(abs(er.margin - t.margin))
    assert np.mean(err_r) < np.mean(err_g) + 0.05
