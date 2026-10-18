import numpy as np
import pytest

from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.attacks.joint_map import JointMapAttack, d2_joint_moments
from numpriv.attacks.likelihoods import d2_pair_moments
from numpriv.attacks.multi_query import MultiQueryAttack
from numpriv.defenses import OrderPreserving, PerFieldScaling
from numpriv.priors import build_prior_table
from numpriv.records import generate_firms
from numpriv.sequences import build_sequence, disguise_sequence

TABLE = build_prior_table(generate_firms(1000, seed=11))
FIRM = generate_firms(60, seed=12)[7]


def test_joint_moments_reduce_to_pair_moments():
    ps, cs = np.array([12.0, 300.0]), np.array([7.0, 210.0])
    mean, cov = d2_joint_moments(np.stack([ps, cs], axis=1), OrderPreserving())
    pm, pc = d2_pair_moments(ps, cs)
    assert np.allclose(mean, pm) and np.allclose(cov, pc)


def test_joint_model_is_calibrated_for_six_values():
    d = OrderPreserving()
    rng = np.random.default_rng(13)
    m2 = []
    for _ in range(1500):
        p = np.exp(rng.uniform(np.log(1.0), np.log(5e3), 3))
        x = np.concatenate([p, p * rng.uniform(0.05, 0.95, 3)])
        secret = d.sample_secret(rng, ())
        obs = np.log([secret.h(v) for v in x])
        mean, cov = d2_joint_moments(x[None, :], d)
        diff = obs - mean[0]
        m2.append(float(diff @ np.linalg.solve(cov[0], diff)))
    assert 5.0 < np.mean(m2) < 7.5


def _views(regime, n=5, task_start=0):
    rng = np.random.default_rng([3, 4])
    seq = disguise_sequence(build_sequence(FIRM, n, start=task_start), OrderPreserving(), regime, rng)
    return seq.views


def test_single_product_query_reproduces_the_pair_attacker():
    solver = ClassicalBayesAttack(TABLE, OrderPreserving(), use_oracle=True, n_samples=1500)
    views = _views("fresh", n=1, task_start=0)  # quote_pricing: product 0 only
    a = MultiQueryAttack(solver, "fresh").attack(views)
    b = JointMapAttack(solver, "fresh").attack(views)
    assert a.margin == pytest.approx(b.margin) and a.unit_cost == pytest.approx(b.unit_cost)


def test_reuse_counts_repeated_identical_views_once():
    solver = ClassicalBayesAttack(TABLE, OrderPreserving(), use_oracle=True, n_samples=1500)
    views = _views("reuse", n=2, task_start=1)  # margin_review then break_even under one map
    att = JointMapAttack(solver, "reuse")
    a = att.attack(views[:1] * 3)
    b = att.attack(views[:1])
    assert a.margin == pytest.approx(b.margin)


def test_uses_other_products_when_shown():
    solver = ClassicalBayesAttack(TABLE, OrderPreserving(), use_oracle=True, n_samples=1500)
    views = _views("fresh", n=1, task_start=1)  # margin_review shows products 0-2
    a = MultiQueryAttack(solver, "fresh").attack(views)
    b = JointMapAttack(solver, "fresh").attack(views)
    assert np.isfinite(b.margin) and np.isfinite(b.unit_cost)
    assert (a.margin, a.unit_cost) != pytest.approx((b.margin, b.unit_cost))


def test_rejects_other_defenses_and_bad_regime():
    with pytest.raises(ValueError):
        JointMapAttack(ClassicalBayesAttack(TABLE, PerFieldScaling(), use_oracle=True), "fresh")
    with pytest.raises(ValueError):
        JointMapAttack(ClassicalBayesAttack(TABLE, OrderPreserving(), use_oracle=True), "repeat")


def test_fresh_maps_share_the_persistent_nuisance_values():
    """Other products keep their values across queries: L = E_z[prod_j q_j(Y_j | theta, z)], not prod_j E_z[q_j].
    Two identical fresh-map views must give log E[q^2] >= 2 log E[q] (Jensen), strictly for some candidates."""
    solver = ClassicalBayesAttack(TABLE, OrderPreserving(), use_oracle=True, n_samples=800)
    view = _views("fresh", n=1, task_start=1)[0]  # margin_review: products 0-2
    att = JointMapAttack(solver, "fresh", n_nuisance=100)
    one = att.log_likelihood((view,))
    two = att.log_likelihood((view, view))
    assert np.all(two >= 2 * one - 1e-9)
    assert np.any(two > 2 * one + 1e-3)


def test_reuse_log_likelihood_ignores_repeats():
    solver = ClassicalBayesAttack(TABLE, OrderPreserving(), use_oracle=True, n_samples=800)
    view = _views("reuse", n=1, task_start=1)[0]
    att = JointMapAttack(solver, "reuse", n_nuisance=100)
    assert np.allclose(att.log_likelihood((view,)), att.log_likelihood((view, view, view)))


def test_monte_carlo_integral_matches_quadrature_on_a_small_example():
    """One other product: the Monte Carlo average over its prior equals a 2-D quadrature of the same integrand."""
    from scipy.special import logsumexp
    from scipy.stats import t as student_t

    solver = ClassicalBayesAttack(TABLE, OrderPreserving(), use_oracle=True, n_samples=800)
    sector, cls = "bakery_cafe", "pastry"
    comp = solver.component(sector, cls)
    att = JointMapAttack(solver, "fresh", n_nuisance=20_000)
    nuisance = att._nuisance(sector, {1: cls}, np.random.default_rng(1))
    shown = {0: (2000.0, 1500.0), 1: (1800.0, 1100.0)}
    price, cost = np.array([4.2, 3.0]), np.array([1.9, 2.2])
    mc = logsumexp(att._group_matrix(shown, [1], nuisance, price, cost), axis=1) - np.log(att.n_nuisance)
    (mp, sp), (mm, sm) = comp.price, comp.margin
    u = np.linspace(mp - 7 * sp, mp + 7 * sp, 601)  # log price
    v = np.linspace(mm - 7 * sm, mm + 7 * sm, 601)  # logit margin
    uu, vv = np.meshgrid(u, v, indexing="ij")
    p1 = np.exp(uu.ravel())
    c1 = p1 * (1 - 1 / (1 + np.exp(-vv.ravel())))
    log_w = (student_t.logpdf(uu.ravel(), 3, mp, sp) + student_t.logpdf(vv.ravel(), 3, mm, sm)
             + np.log((u[1] - u[0]) * (v[1] - v[0])))
    grid = {1: np.stack([p1, c1], axis=1)}
    quad = []
    for k in range(price.size):
        att_k = JointMapAttack(solver, "fresh", n_nuisance=p1.size)
        m = att_k._group_matrix(shown, [1], grid, price[k:k + 1], cost[k:k + 1])[0]
        quad.append(logsumexp(m + log_w))
    assert np.allclose(mc, quad, atol=0.15)


def test_smoothing_is_added_to_every_modelled_value():
    solver = ClassicalBayesAttack(TABLE, OrderPreserving(), use_oracle=True, n_samples=200)
    with pytest.raises(ValueError):
        JointMapAttack(solver, "fresh", obs_sd=0.0)
    xs = np.array([[40.0, 25.0, 38.0, 20.0]])
    a = JointMapAttack(solver, "fresh", obs_sd=0.03)
    _, cov_raw = d2_joint_moments(xs, OrderPreserving())
    assert np.allclose(a._smoothed(cov_raw) - cov_raw, 0.03 ** 2 * np.eye(4))
