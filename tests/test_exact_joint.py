"""R046: precision-faithful joint attacker for D2 (attacks/exact_joint.py).

The small-instance tests compare the attacker's posterior with exact rejection sampling at coarse display
precision (plan V1, refine-logs/EXPERIMENT_PLAN_20261009_exact_joint.md)."""

import numpy as np
import pytest
from scipy.stats import beta as beta_dist
from scipy.stats import ks_2samp

from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.attacks.exact_joint import (
    ExactJointAttack,
    ExactJointSampler,
    MapLaw,
    Observation,
    ProductPrior,
    infer_digits,
    invert,
    log_cum_density,
    rounding_cell,
)
from numpriv.defenses import OrderPreserving
from numpriv.fields import round_sig
from numpriv.priors import build_prior_table
from numpriv.records import generate_firms
from numpriv.sequences import build_sequence, disguise_sequence

TABLE = build_prior_table(generate_firms(1000, seed=11))
FIRM = generate_firms(60, seed=12)[7]


# ---- building blocks -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("digits", [1, 2, 6])
def test_rounding_cell_is_the_preimage_of_round_sig(digits):
    rng = np.random.default_rng(0)
    ys = [round_sig(float(v), digits) for v in np.exp(rng.uniform(0, 12, 300))] + [10.0, 100.0, 1000.0, 1e5]
    for y in ys:
        lo, hi = rounding_cell(y, digits)
        assert lo < y < hi
        inner = np.linspace(lo, hi, 203)[1:-1]
        assert all(round_sig(float(x), digits) == y for x in inner)
        width = hi - lo
        assert round_sig(lo - 1e-3 * width, digits) != y and round_sig(hi + 1e-3 * width, digits) != y


def test_infer_digits_uses_the_prompt_maximum_and_the_floor():
    assert infer_digits([4509.19, 4416.86, 13026.5], 6) == 6
    assert infer_digits([4509.1931, 12.0], 6) == 8
    assert infer_digits([20.0, 300.0], 1) == 1


def test_sampled_knots_follow_the_defense_law():
    d = OrderPreserving()
    law = MapLaw.of(d)
    rng = np.random.default_rng(1)
    ours = law.sample_knots(rng, 4000)
    theirs = np.array([d.sample_secret(rng, ()).knot_log_targets for _ in range(4000)])
    assert ours.shape == theirs.shape == (4000, d.n_intervals + 1)
    for k in (1, 20, 40, 63):
        assert ks_2samp(ours[:, k], theirs[:, k]).pvalue > 1e-3
    assert np.all(np.diff(ours, axis=1) > law.floor - 1e-12)


def test_invert_recovers_values_under_a_known_map():
    d = OrderPreserving()
    law = MapLaw.of(d)
    rng = np.random.default_rng(2)
    secret = d.sample_secret(rng, ())
    xs = np.array([1.37, 12.0, 450.5, 9876.0])
    L = np.tile(np.array(secret.knot_log_targets), (xs.size, 1))
    u = np.log([secret.h(x) for x in xs])
    t, j, f = invert(law, L, u)
    assert np.allclose(np.expm1(t), xs, rtol=1e-9)
    assert np.all((0 <= f) & (f < 1))


@pytest.mark.parametrize("alpha", [1.0, 0.3])
def test_cumulative_density_matches_the_beta_marginal(alpha):
    law = MapLaw(n_intervals=8, log_low=0.0, log_high=np.log(1e6), alpha=alpha)
    c = np.linspace(0.05, 0.95, 7)[:, None]
    ours = log_cum_density(law, (3,), c)
    ref = beta_dist.logpdf(c[:, 0], alpha * 3, alpha * 5)
    assert np.allclose(ours, ref)


# ---- small-instance reference (plan V1) ----------------------------------------------------------------------
# Narrow signed-log axes so a few intervals cover the values: three or more values share an interval (exact
# constraints), neighbouring intervals propagate, and heavy-tailed draws reach the last interval (known end knot).
LAWS = {
    "m6": MapLaw(n_intervals=6, log_low=0.0, log_high=np.log(100.0), alpha=1.0, limit=float(np.log1p(1e3))),
    "m8": MapLaw(n_intervals=8, log_low=0.0, log_high=np.log(100.0), alpha=1.0, limit=float(np.log1p(1e3))),
}
PRIOR0 = ProductPrior(np.log(20.0), 0.35, 0.4, 0.5)
PRIOR1 = ProductPrior(np.log(40.0), 0.35, 0.0, 0.5)
DF = 3.0


def _h_rows(law, L, x):
    """Map each row's value through that row's knots (vectorised h)."""
    t = np.log1p(x)
    j = np.clip(np.searchsorted(law.grid, t, side="right") - 1, 0, law.n_intervals - 1)
    f = (t - law.grid[j]) / law.width
    r = np.arange(len(x))
    return np.exp((1 - f) * L[r, j] + f * L[r, j + 1])


def _draw_products(rng, priors, n):
    out = []
    for p in priors:
        price = np.exp(p.mu_price + p.sd_price * rng.standard_t(DF, n))
        margin = 1 / (1 + np.exp(-(p.mu_margin + p.sd_margin * rng.standard_t(DF, n))))
        out.append((price, price * (1 - margin)))
    return out


def _value(prods, fid):
    kind, i = fid.split(":")
    return prods[int(i)][0 if kind == "price" else 1]


def _rejection(law, queries, priors, regime, rng, want=3000, max_total=60_000_000, chunk=250_000):
    """Exact posterior draws of product 0's margin: values from the prior, maps from the law, accept on
    an exact match of every shown value at the coarse display precision."""
    cells = [{fid: rounding_cell(y, digits) for fid, y in shown.items()} for shown, digits in queries]
    accepted, total = [], 0
    while sum(a.size for a in accepted) < want and total < max_total:
        prods = _draw_products(rng, priors, chunk)
        ok = np.ones(chunk, bool)
        L_reuse = law.sample_knots(rng, chunk)
        for shown_cells in cells:
            L = L_reuse if regime == "reuse" else law.sample_knots(rng, chunk)
            for fid, (lo, hi) in shown_cells.items():
                v = _h_rows(law, L, _value(prods, fid))
                ok &= (v >= lo) & (v < hi)
        p0, c0 = prods[0]
        accepted.append(1 - c0[ok] / p0[ok])
        total += chunk
    return np.concatenate(accepted)


def _history(law, priors, min_digits, regime, layout, rng):
    """One synthetic history: shown values under the small law (true values drawn from the prior). As in D2,
    a query uses the fewest digits >= min_digits that keep distinct values distinct; histories needing more
    digits are redrawn so that exact rejection stays feasible (the comparison conditions on the history)."""
    while True:
        queries = _one_history(law, priors, min_digits, regime, layout, rng)
        if all(d == min_digits for _, d in queries):
            return queries


def _one_history(law, priors, min_digits, regime, layout, rng):
    prods = _draw_products(rng, priors, 1)
    L_reuse = law.sample_knots(rng, 1)
    queries = []
    for products in layout:
        L = L_reuse if regime == "reuse" else law.sample_knots(rng, 1)
        raw = {f"{kind}:{i}": float(_h_rows(law, L, _value(prods, f"{kind}:{i}"))[0])
               for i in products for kind in ("price", "cost")}
        digits = min_digits
        while len({round_sig(v, digits) for v in raw.values()}) < len(raw):
            digits += 1
        queries.append(({fid: round_sig(v, digits) for fid, v in raw.items()}, digits))
    return queries


def _attacker_posterior(law, queries, priors, regime, n=60_000, seed=5, mcmc_steps=0):
    s = ExactJointSampler(law, dict(enumerate(priors)), DF, regime, n, np.random.default_rng(seed),
                          mcmc_steps=mcmc_steps)
    for shown, digits in queries:
        s.update(Observation(tuple(shown), np.array(list(shown.values())), digits))
    margin, _, w = s.product0()
    order = np.argsort(margin)
    # Resampling duplicates particles, so the ESS of a later update overstates the information; the tightest
    # update bounds the effective number of independent draws.
    return margin[order], np.cumsum(w[order]), min(s.ess_history)


CASES = [  # (law, regime, layout, priors); fresh with six values is too thin for rejection under m8
    (law, regime, layout, priors)
    for law in sorted(LAWS)
    for regime, layout, priors in [
        ("fresh", [(0,)], (PRIOR0,)),
        ("fresh", [(0,), (0,)], (PRIOR0,)),
        ("reuse", [(0,), (0, 1)], (PRIOR0, PRIOR1)),
        ("fresh", [(0, 1), (0,)], (PRIOR0, PRIOR1)),
    ]
    if not (law == "m8" and layout == [(0, 1), (0,)])
]
# Resample-move must leave the posterior unchanged: rerun every multi-query case with Metropolis-Hastings steps.
CASES = [(*c, 0) for c in CASES] + [(*c, 10) for c in CASES if len(c[2]) > 1]


@pytest.mark.slow
@pytest.mark.parametrize("law_name,regime,layout,priors,mcmc_steps", CASES)
def test_small_instance_posterior_matches_rejection(law_name, regime, layout, priors, mcmc_steps):
    """Weighted posterior CDF of product 0's margin against exact rejection draws: quantiles within 0.03 and a
    Kolmogorov-Smirnov distance below the 1% critical value with the attacker's effective sample size
    (resampled particles are not independent draws, so the plain two-sample test would be miscalibrated)."""
    law = LAWS[law_name]
    queries = _history(law, priors, 1, regime, layout, np.random.default_rng(17))
    ref = np.sort(_rejection(law, queries, priors, regime, np.random.default_rng(18)))
    assert ref.size > 300, f"reference too thin ({ref.size}); widen the instance"
    margin, cdf, ess = _attacker_posterior(law, queries, priors, regime, mcmc_steps=mcmc_steps)
    assert ess > 200
    for q in (0.1, 0.5, 0.9):
        ours = margin[np.searchsorted(cdf, q)]
        assert abs(ours - np.quantile(ref, q)) < 0.03, (q, ours, np.quantile(ref, q))
    distance = np.abs(np.interp(ref, margin, cdf) - np.arange(1, ref.size + 1) / ref.size).max()
    assert distance < 1.63 * np.sqrt(1 / ref.size + 1 / ess)


# ---- behaviour on real D2 transcripts ------------------------------------------------------------------------
def test_near_deterministic_law_recovers_the_truth():
    d = OrderPreserving(gap_concentration=1e7)  # gaps almost fixed: the map is effectively public
    solver = ClassicalBayesAttack(TABLE, d, use_oracle=True, n_samples=1500)
    seq = disguise_sequence(build_sequence(FIRM, 1, start=1), d, "fresh", np.random.default_rng([3, 4]))
    est = ExactJointAttack(solver, "fresh", n_particles=4000).attack(seq.views)
    truth = FIRM.products[0]
    assert abs(est.margin - truth.margin) < 0.005
    assert abs(est.unit_cost / truth.unit_cost - 1) < 0.005


def test_reuse_repeated_identical_views_change_nothing():
    d = OrderPreserving()
    solver = ClassicalBayesAttack(TABLE, d, use_oracle=True, n_samples=1500)
    seq = disguise_sequence(build_sequence(FIRM, 1, start=1), d, "reuse", np.random.default_rng([3, 4]))
    att = ExactJointAttack(solver, "reuse", n_particles=6000, seed=9)
    a = att.attack(seq.views * 3)
    b = att.attack(seq.views)
    assert a.margin == pytest.approx(b.margin) and a.unit_cost == pytest.approx(b.unit_cost)


def test_prefix_estimates_equal_separate_runs():
    d = OrderPreserving()
    solver = ClassicalBayesAttack(TABLE, d, use_oracle=True, n_samples=1500)
    seq = disguise_sequence(build_sequence(FIRM, 3, start=0), d, "fresh", np.random.default_rng([3, 4]))
    att = ExactJointAttack(solver, "fresh", n_particles=3000, seed=9)
    pref = att.attack_prefixes(seq.views, (1, 3))
    assert pref[3].margin == pytest.approx(att.attack(seq.views).margin)
    assert pref[1].margin == pytest.approx(att.attack(seq.views[:1]).margin)


def test_year_and_special_integers_are_not_treated_as_mapped():
    d = OrderPreserving()
    solver = ClassicalBayesAttack(TABLE, d, use_oracle=True, n_samples=1500)
    seq = disguise_sequence(build_sequence(FIRM, 4, start=3), d, "fresh", np.random.default_rng([3, 4]))
    obs = ExactJointAttack(solver, "fresh").observation(seq.views[0])  # sales_summary shows a year
    assert not any(f.startswith("year") for f in obs.fields)
    assert obs.digits >= 6


def test_reuse_moves_refresh_knots_away_from_the_data():
    """Knots not adjacent to any shown value carry no data; resample-move must keep them diverse, or a value
    shown later in a so-far empty region inverts to the same few maps (a collapse seen on firm F00058)."""
    d = OrderPreserving()
    law = MapLaw.of(d)
    prior = ProductPrior(np.log(1500.0), 0.3, 0.8, 0.4)
    s = ExactJointSampler(law, {0: prior}, 3.0, "reuse", 3000, np.random.default_rng(1), mcmc_steps=5,
                          n_initial=30_000)
    s.update(Observation(("price:0", "cost:0"), np.array([9806.67, 9422.73]), 6))
    roots = s._roots()
    t = np.stack([np.log1p(s.values[f]) for f in roots], axis=1)
    occupied = np.floor((t + law.limit) / law.width).astype(int)
    far = int(occupied.min()) - 4  # a knot well below every occupied interval
    assert np.unique(s.knots[:, far]).size > 0.5 * s.n


@pytest.mark.parametrize("regime", ["fresh", "reuse"])
def test_passed_through_integers_are_known_values(regime):
    """D2 passes integers 28-31 through unchanged, so such a shown price is the true price (firm F00086)."""
    law = MapLaw.of(OrderPreserving())
    prior = ProductPrior(np.log(35.0), 0.3, 0.6, 0.4)
    s = ExactJointSampler(law, {0: prior}, 3.0, regime, 2000, np.random.default_rng(2), mcmc_steps=3,
                          n_initial=20_000)
    shown = (2210.37, 1876.52) if regime == "fresh" else (2210.37, 2210.37)  # one map repeats its output
    for y in shown:  # the cost; the price is shown as 30 both times
        s.update(Observation(("cost:0",), np.array([y]), 6, {"price:0": 30.0}))
    margin, cost, w = s.product0()
    assert np.all(s.values["price:0"] == 30.0)
    assert np.allclose(margin, 1 - cost / 30.0)
    assert np.all((cost > 0) & (cost < 30.0))


def test_observation_separates_passed_through_integers():
    d = OrderPreserving()
    solver = ClassicalBayesAttack(TABLE, d, use_oracle=True, n_samples=1500)
    seq = disguise_sequence(build_sequence(FIRM, 1, start=2), d, "fresh", np.random.default_rng([3, 4]))
    view = seq.views[0]
    shown = tuple(f if f.field_id != "price:0" else type(f)(**{**f.__dict__, "value": 30.0}) for f in view.shown)
    obs = ExactJointAttack(solver, "fresh").observation(type(view)(**{**view.__dict__, "shown": shown}))
    assert obs.known == {"price:0": 30.0} and "price:0" not in obs.fields


def test_flat_fields_get_a_proper_prior():
    """Volumes and other fields without a class prior must not pull the posterior towards x -> 0. A flat prior
    in log x is improper there, and its weight factor (1+x)/x let maps that invert a volume to below 1 take 99%
    of the weight on firm F00058's first query. The prior is uniform on the public signed-log axis instead."""
    from numpriv.dataset import EvalConfig, build_eval_set

    ev = build_eval_set(EvalConfig(seed=1001))
    idx, firm = [(i, f) for i, f in enumerate(ev.firms) if i >= ev.config.n_dev][8]
    assert firm.firm_id == "F00058"
    d = OrderPreserving()
    solver = ClassicalBayesAttack(ev.prior, d, use_oracle=True, n_samples=1500)
    seq = disguise_sequence(build_sequence(firm, 1, start=idx % 5), d, "reuse", np.random.default_rng([0, idx, 2]))
    att = ExactJointAttack(solver, "reuse")
    s = ExactJointSampler(att.law, att.priors(seq.views[0]), solver.prior_df, "reuse", 100_000,
                          np.random.default_rng(1))
    s.update(att.observation(seq.views[0]))
    w = np.exp(s.lw - s.lw.max())
    w /= w.sum()
    flat = [f for f in s.values if not f.startswith(("price:", "cost:"))]
    assert flat
    tiny = np.zeros(s.n, bool)
    for f in flat:
        tiny |= s.values[f] < 1
    # Before the fix 99% of the weight sat on such particles. A uniform prior in t still puts real mass there,
    # because the public law maps t near 0 to outputs around 10^3; it must no longer dominate.
    assert w[tiny].sum() < 0.5


def test_flat_prior_weight_is_bounded_near_zero():
    """The uniform-in-t prior turns the inversion Jacobian d log x / du into d t / du, which stays bounded as
    x -> 0 (the improper flat-in-log-x factor (1+x)/x did not)."""
    from numpriv.attacks.exact_joint import _flat_log_prior

    x = np.array([1e-9, 1e-6, 1e-3, 1.0, 1e3])
    slope = 2.0  # d t / du on the inverted interval
    dlog_x = (1 + x) / x * slope
    assert np.allclose(_flat_log_prior(x, dlog_x), np.log(slope))
