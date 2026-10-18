# Experiment Plan R046: precision-faithful joint attacker for D2

Date: 2026-10-09. Status: **planned, not run.** Endpoints and interpretation below are fixed before any result is seen.

## Why

The joint shared-map attacker (R045, `attacks/joint_map.py`) adds 0.03 log-unit observation smoothing so its
Monte Carlo integral is resolvable. Within one map interval, log h is affine in t = sign(x)·log1p|x|, and shown
values carry ≥ 6 significant digits (≈ 5e-6 relative), so three or more values in one interval satisfy an exact,
key-independent constraint: (u_3 − u_1)/(u_2 − u_1) = (t_3 − t_1)/(t_2 − t_1). The smoothing (≈ 25,000× display
precision) discards these constraints. The TIFS kill-argument (paper_tifs/KILL_ARGUMENT.md, P_1–P_4) showed that
the paper therefore cannot say the reuse curve is flat for D2, only that this attacker's is.

Feasibility probe (2026-10-09, test split, true values): of 500 prompts, 210 have ≥ 3 values in one interval
(max 3: 162, 4: 35, 5: 10, 6: 3); product 0's price and cost share an interval in 236; product 0 sits in an
interval with ≥ 3 values in 133. The constraints are common, **including within a single prompt**.

## Attacker (A-exact)

Exact posterior over the firm's values given the shown transcript, under the public map law, with display
rounding handled exactly. No Gaussian approximation, no smoothing.

Core idea: sample maps, not values. For a sampled map h, every shown value inverts uniquely (h is strictly
increasing), so values are pinned by the map and the exact constraints hold automatically.

- **Map law structure used.** Knot log-targets are affine in the cumulative gap shares. Under the default law
  the cumulative shares are uniform order statistics, which are Markov: given the knots at some indices,
  the knots between are sorted uniforms on the sub-range. Under Dirichlet(α) gaps the same holds by Dirichlet
  aggregation and neutrality (conditionals are scaled Dirichlet / Beta). The attacker therefore samples knots
  **conditionally** on knots already pinned.
- **Rounding.** The true h(x) lies in the rounding cell of the shown value. The proposal draws the position in
  the cell uniformly and inverts. The cell width is known from the shown digits, so rounding is integrated,
  not ignored.
- **Reuse regime (one map).** Sequential importance sampling over the history:
  - Particles carry the pinned knots and the pinned values.
  - At query q, values seen before must re-round to their shown digits. This is an exact check, and it is
    automatic when the map is shared.
  - New values in intervals whose knots are pinned invert deterministically.
  - New values in unpinned intervals: draw those knots from the conditional law, then invert.
  - Weight increment = prior density of newly pinned values (oracle class priors, same Student-t prior as A1+)
    × Jacobian |dx/du| × (conditional knot density ÷ proposal density) × cell factor.
- **Fresh regime (new map per query).** Particles carry the pinned values X, not the map:
  - At each query a new map is drawn, conditionally on the values in X.
  - In each interval, two pinned values determine both knots. One pinned value gives one linear equation,
    and one dimension is sampled. Further pinned values must round consistently, an exact check.
  - The weight uses the knot density of the determined coordinates with its Jacobian.
  - New products are pinned as in the reuse regime.
- **Degeneracy control.**
  - Resample when ESS < N/2 (systematic resampling).
  - N = 20,000 particles by default. Report ESS per query.
  - Rerun with 2N and an independent seed. The numerical error is the between-run difference, reported
    separately from the bootstrap CI.
- **Scope kept equal to the joint attacker.**
  - Models product prices and costs. Volumes, overhead and quotes have no prior in the attacker's table:
    pinned values are inverted but get a flat prior on the log scale, and this is stated.
  - Same oracle (true sector and classes), same decision rule (posterior median), same tolerances.

## Validation (must pass before the main runs)

- **V1 small-instance reference (the adjudicator's "certified small-instance reference").**
  - Setup: 4 and 8 intervals, 2–4 values, display at 2 and 3 significant digits. At that precision exact
    rejection sampling (sample values from the prior and a map from the law, round, accept on exact match)
    is feasible.
  - Comparison: the A-exact posterior (same coarse digits) against the rejection posterior.
  - Criterion: posterior medians within MC error, and a 2-sample KS test on product-0 margin draws that does
    not reject at 1%.
  - Coverage: both regimes, 1–3 queries.
- **V2 unit tests (written first, TDD).**
  - A degenerate law (fixed knots) recovers the truth exactly.
  - One value per interval: the posterior equals the pair model's exact 1-D density computed by quadrature.
  - Re-rounding the pinned values reproduces the shown digits.
  - Particle weights are invariant to query order under reuse.
- **V3 numerical stability.** At least two seeds × {N, 2N} on 10 firms. Margin hit rate within 1.5 pp across
  runs, else increase N.

## Runs (same firms, seeds and random streams as R017/R035/R045, so rows are paired)

| Part | Firms | Content |
|---|---|---|
| linked | eval seed 1001, first 60 test firms, 3 defense seeds | fresh and reuse, n ∈ {1, 2, 4, 8, 16}; fresh-minus-reuse contrast |
| fresh_draw | eval seed 3003, same layout | confirmation |
| single | test split, 2,100 rows | A-exact vs Gaussian pair vs joint (smoothed) vs A0, paired |
| law (secondary) | eval seed 1001, n ∈ {1, 16} | Dirichlet(0.1) gaps and 16/256 intervals (rows of tab:d2law) |

## Pre-registered endpoints

- **E1 (primary).** Fresh-minus-reuse difference in the n = 1→16 margin-hit gain under A-exact, original firms,
  firm-clustered 95% percentile bootstrap. Confirmed if the same sign holds on the fresh draw with CI above 0.
- **E2.** Reuse-regime margin gain n = 1→16 under A-exact (both draws).
- **E3.** Single-query margin-hit advantage of A-exact over A0 (test split), and over the Gaussian pair attacker.
- **E4 (secondary).** E1 and E2 under the alternative laws.
- Unit-cost versions of E1–E3 are reported alongside, not as decision endpoints.

## Interpretation fixed in advance (all outcomes reported; existing pair/joint numbers stay in the paper)

- **S1:** E1 > 0 (CI above 0, confirmed on the fresh draw) and E2 small. The paper may say that, for the D2
  instantiation, leakage growth is specific to fresh maps under an exact attacker. This restores the claim
  removed after the kill-argument, now with precision-faithful evidence.
- **S2:** E2 is also large, because new products under a reused map add exact constraints. The paper says
  history leaks under both regimes for D2. The contrast is reported as measured. Theorem 1's union view is
  the explanation: reuse bounds the history by the union view, not by the first prompt.
- **S3:** E3 is large (A-exact recovers margins well in one prompt). The single-query D2 sentence in the
  abstract, the introduction and §IV changes:
  - The current "near prior" holds for the Gaussian pair attacker only.
  - The exact attacker's single-query rate becomes the headline single-query number for D2.
  - This strengthens the attack finding and weakens the "looks safe in one prompt" framing for D2.
  - The D1′ and D3o single-query results are unaffected.
- **Not allowed:** dropping A-exact if it is weaker than expected, or tuning N, digits or the prior after
  seeing E1–E3. If V1 fails, A-exact is not used and the failure is reported to the author.

## Cost and order of work

1. Write the tests and the V1 reference harness (`tests/test_exact_joint.py`, `scripts/run_d2_exact.py --part validate`).
2. Implement `src/numpriv/attacks/exact_joint.py`.
3. Run V1–V3. Stop and report if V1 fails.
4. Run linked → fresh_draw → single → law.

Expected cost: vectorised numpy, 60 firms × 3 seeds × 2 regimes × 16 queries × 20k particles. Likely a few CPU-hours
locally; no GPU and no LLM calls. The LLM test split is not touched.

## Not in scope

- The external-population (EDGAR) experiment: deferred to rebuttal (author decision).
- Zhang et al.'s own construction: no public implementation. A-exact still evaluates our fixed-axis
  instantiation only.

## Amendment 1 (2026-10-09, before any endpoint run; approved by the author)

**What changed.** A-exact is used for the **reuse** regime and the **single-query** part only. The **fresh** arm of
E1 uses the existing joint shared-map attacker (R045, `attacks/joint_map.py`), same firms, seeds and streams, so
rows stay paired. E1 becomes: (fresh-regime gain under the joint attacker) − (reuse-regime gain under A-exact).
E2 and E3 are unchanged (A-exact). E4 (alternative laws) is limited to the reuse regime.

**Why.**
- V1 passed: 12 small-instance cases against exact rejection, with and without MCMC moves.
- V3 on real D2 transcripts showed that the fresh-regime posterior cannot be represented with particles:
  - Within-interval ratios of t are key-independent and recur in every fresh query.
  - The posterior therefore concentrates on the interval structure that every query reproduces.
  - Example: firm 0, query 7 shows six values that all lie in one interval. 0.03% of particles survive even
    with 64 inner replicates and resample-move. The run collapses at query 7–10 whatever the particle count.
- Reuse is computed exactly with map-space Metropolis-Hastings, which needs no likelihood estimate.

**Direction of the change.** Any concrete attacker's hit rate is a lower bound on what a provider can learn, so
the fresh arm is conservative: E1 can only be understated by this change. The same observation shows that a
stronger fresh-regime attack exists: identify shared intervals from the agreement of log-difference ratios across
fresh queries. This is reported as a qualitative finding, not as a number.

**What had been seen before this amendment (disclosed).**
- V3 diagnostics on the first 10 linked-query firms (original draw, defense seed 0, reuse regime) under three
  numerical settings:
  - Product-0 margin hit rates at n = 1: 0.6 / 0.7 / 0.6.
  - At n = 16: 0.5 / 0.6 / 0.5.
- No other endpoint quantity had been computed.

**Numerical settings fixed now.**
- Reuse and single query: n_particles = 20,000; n_initial = 400,000 at the first query; 20 MH steps per update;
  seed 4646.
- V3 final check (60 firms × 3 seeds, reuse, n ∈ {1, 16}): seed 4646 vs seed 4647 at the same settings.
  The margin and unit-cost hit rates must agree within 1.5 pp. Otherwise 40 MH steps and a recheck, before any
  main run.

### V3 log (Amendment 1 rule applied)
- 2026-10-09, run 1 (20 MH steps): crashed on one of 180 sequences (F00058, seed 0, query 4). Knots away from the
  data were never refreshed under reuse, so a later low-priced value inverted to t < 0 in every particle.
  Fixed by an exact Gibbs refresh of those knots from their conditional law (regression test added; all 25 tests
  pass).
- Run 2 (20 MH steps, seed 4646 vs 4647, 60 firms × 3 seeds, reuse):
  - Margin hit rate: 23.9 / 23.9 % at n = 1 and 24.4 / 24.4 % at n = 16.
  - Unit-cost hit rate: 19.4 / 18.9 % at n = 1 and 18.9 / 22.8 % at n = 16 (3.9 pp apart).
  - **FAIL** on unit cost at n = 16.
  - First-query ESS is very low for some sequences (10% quantile of the minimum ESS ≈ 5 out of 400,000 draws).
- Next, per the rule: 40 MH steps, same check.
- Run 3 (40 MH steps, same check):
  - Margin hit rate: 23.3 / 24.4 % at n = 1 (1.1 pp apart) and 24.4 / 24.4 % at n = 16.
  - Unit-cost hit rate: 18.9 / 18.9 % at n = 1 and 22.2 / 18.3 % at n = 16 (3.9 pp apart).
  - **FAIL** on unit cost at n = 16. Doubling the MH steps did not shrink the gap.
  - The bottleneck is the first-query proposal (minimum-ESS 10% quantile ≈ 3.6), which fixes the weights of
    separated modes.
  - Main runs **not started**; reported to the author for a decision.
- Disclosure: the V3 outputs are reuse-regime product-0 hit rates on the original draw (seed 4646 vs 4647), so
  they already show the E2 point estimate for margins (n = 1 → 16: about 23.9 → 24.4 %). No interval or contrast
  has been computed.

## Amendment 2 (2026-10-09, after V3 run 3, before any main run; author's decision)

- Every main part (linked, fresh_draw, single, law) runs twice, with A-exact seeds 4646 and 4647. All other
  settings are as fixed in Amendment 1, with 40 MH steps.
- **Decision endpoints are margin-based only (E1, E2, E3, E4).** They passed the V3 agreement rule
  (≤ 1.1 pp between seeds).
  - Each margin endpoint is reported per seed.
  - The between-seed difference is reported as numerical error, separately from the firm-clustered bootstrap CI.
  - Conclusions require both seeds to agree in sign and in whether the CI excludes 0.
- **Unit-cost results are reported with their between-seed spread as numerical error and support no
  conclusion.** In V3, unit cost differed by 3.9 pp at n = 16 under both 20 and 40 MH steps.
- The S1/S2/S3 interpretation rules are unchanged and apply to the margin endpoints.

### Main-run log
- 2026-10-10 00:29: fresh_draw (seed 4646) crashed. Firm F00086 (eval seed 3003) has a product-0 price of
  exactly 30.00, which D2 passes through unchanged (integers 28–31), so product 0's price was never a mapped
  value.
  - Fix: passed-through integers are known values. The attacker sees the true value, which is fixed across
    particles and excluded from maps and moves; the product prior then acts as the conditional density of the
    mapped partner. Tests added; 28 pass.
  - The fix only changes sequences that contain such a field: none in the original draw, and in the fresh draw
    only F00086 (two test-split prompts of eval seed 3003, which is not used by the single part). It consumes no
    random draws, so every other row is unchanged.
  - fresh_draw seed 4646 is rerun after the queue. No endpoint value had been read before the fix.
- 2026-10-10 03:33: law (seed 4647) failed after 22 minutes. Every particle became inconsistent in one sequence.
  - Timing points to the first law run, Dirichlet(0.1) gaps; seed 4646 took about 29 minutes per law.
  - Under Amendment 2, E4 needs both seeds. For any law without a seed-4647 result, E4 is reported as
    "numerical failure in one seed, no conclusion".
  - The settings are not changed after the fact.

## Results (2026-10-10; `pilot/scripts/summarize_r046.py`, files `pilot/results/R046_d2_exact_*_seed464{6,7}.json`)

- **E1 (primary, margin)**: fresh (joint) − reuse (A-exact), n = 1 → 16, firm-clustered 95% CI.
  - Original draw: +28.9 [+15.0, +41.7] pp (seed 4646) and +30.0 [+16.7, +42.8] pp (seed 4647).
  - Fresh draw: +22.2 [+7.2, +36.7] and +23.9 [+8.9, +38.3] pp.
  - All four CIs exclude 0; the numerical spread is ≤ 1.7 pp. **Confirmed.**
- **E2 (margin)**: reuse gain under A-exact.
  - Original draw: +1.1 [−7.2, +8.9] and +0.0 [−8.9, +8.3] pp.
  - Fresh draw: +0.6 [−9.4, +10.0] and −1.1 [−11.1, +8.9] pp.
  - All four CIs include 0 and every point estimate lies within ±1.1 pp of 0.
  - The script's literal sign rule prints "seeds disagree" because the estimates straddle 0 (one is exactly 0);
    both seeds agree that the CI includes 0. A reuse gain of up to about +9–10 pp is not excluded.
- **E3 (margin, single query, product 0 of every test prompt, 1,500 rows)**: A-exact − A0.
  - All prompts: +3.3 [−0.6, +7.3] and +3.6 [−0.3, +7.5] pp. The CI includes 0.
    - Hit rates: A0 28.4%, Gaussian pair 28.1%, joint 28.9%, A-exact 31.7–32.0%.
  - A-exact − Gaussian pair: +3.6 [+0.2, +7.1] and +3.9 [+0.3, +7.5] pp.
  - Margin-review prompts (three products, 300 rows; reported, not the pre-registered endpoint):
    +10.0 [+2.0, +18.0] pp vs A0 and +8.7 [+0.7, +17.0] vs pair; +5.0 [−1.3, +11.7] vs joint.
- **E4**: law, seed 4647 failed, so **no conclusion** (Amendment 2).
  - Seed 4646, descriptive only: reuse margin gain +4.4 [−2.2, +11.1] (Dirichlet(0.1)), +1.7 [−8.9, +12.8]
    (16 intervals), +1.1 [−5.6, +7.8] (256 intervals).
  - Under Dirichlet(0.1) gaps the single-query reuse margin hit rate at n = 1 is 45.0%.
- **Unit cost (reference only, no conclusion)**:
  - E1: original draw +13.9 / +17.8 pp, fresh draw +9.4 / +11.1 pp.
  - E2: original draw +3.3 / −0.6 pp, fresh draw −11.7 / −13.3 pp (CI excludes 0 in both seeds).

**Interpretation under the pre-registered rules.**
- **S1 applies.** E1 > 0 with CIs above 0 on both draws and in both seeds, and E2 is small. For the D2
  instantiation, margin growth under fresh maps exceeds growth under one reused map when the reuse side uses the
  precision-faithful attacker.
  - Wording constraint from Amendment 1: the fresh side is a lower bound (joint attacker).
  - The reuse statement is "not distinguishable from 0, with CI up to about +9 pp", not "flat".
- **S3 does not apply on the pre-registered endpoint.** The all-prompt single-query advantage over the prior is
  +3.3/+3.6 pp, with a CI that includes 0.
  - The three-product subset shows a significant single-query advantage (+10.0 pp vs A0). The paper's single-query
    sentence on D2 should mention it.

## Amendment 3 (2026-10-10; error found in review, fixed before any rerun result was seen; author's decision)

**Error.** A-exact gave fields without a class prior (volumes, overhead, order quantities, quotes) a flat prior in
log x. That prior is improper at x → 0, and its weight factor (1+x)/x is unbounded there. On firm F00058
(original draw, defense seed 0, first query = sales summary), 99.2% of the importance weight sat on particles
that invert a volume to below 1. The round-4 independent review flagged the impropriety; a diagnostic confirmed
the effect. Results computed with this prior (all R046 A-exact numbers) are superseded.

**Fix.** These fields now get a uniform prior on the public signed-log axis, t = log1p(x) ∈ (0, limit]. This is
proper and uses only the mechanism's own public axis. The weight factor becomes dt/du, which is bounded.
- Product fields are unchanged.
- V1 (product fields only) is unaffected; all 30 tests pass, including a regression test on F00058 and a
  bounded-weight test.
- After the fix, F00058's weight on volumes below 1 is about 0.2–0.3. This is a genuine consequence of the vague
  prior: the public law maps t ≈ 0 to outputs near 10^3.

**Rerun.** V3 (seeds 4646 vs 4647), then linked, fresh_draw, single and law with both seeds, with settings
unchanged (Amendments 1–2). The new results replace the old ones in the paper whatever their direction. The
joint-attacker and Gaussian-attacker results do not use this prior and are unchanged.

### Amendment-3 rerun log
- V3 (2026-10-10 11:54; 40 MH steps; seed 4646 vs 4647; reuse; 60 firms × 3 seeds):
  - Margin hit rate: 24.4 / 26.7 % at n = 1 (2.2 pp apart) and 30.0 / 30.0 % at n = 16.
  - Unit-cost hit rate: 21.1 / 21.7 % at n = 1 and 21.7 / 22.2 % at n = 16.
  - **FAIL** on margin at n = 1 (2.2 > 1.5 pp); unit cost now within 0.6 pp.
  - Minimum-ESS quantiles [min, 10%, median]: 3.4 / 12.1 / 144.8 (seed 4646) and 1.5 / 12.0 / 143.2 (seed 4647).
    Before the fix they were 1.2 / 3.7 / 47.7.
  - Main runs continue as already queued. Amendment 2's rule still applies: a conclusion needs both seeds to
    agree in sign and in whether the CI excludes 0.
  - Disclosure: these V3 rates imply a reuse margin gain of roughly +3 to +6 pp on the original draw (point
    values only), larger than before the fix.

## Results after Amendment 3 (2026-10-10, E1–E3; E4 pending seed 4647)

- **E1 (margin)**: fresh (joint) − reuse (A-exact).
  - Original draw: +24.4 [+9.4, +38.3] pp (seed 4646) and +26.7 [+11.7, +41.1] pp (seed 4647).
  - Fresh draw: +22.8 [+7.8, +37.2] and +23.9 [+8.9, +38.3] pp.
  - All CIs exclude 0. **Confirmed.** Before the fix: +28.9 / +30.0 (original) and +22.2 / +23.9 (fresh draw).
- **E2 (margin)**: reuse gain under A-exact.
  - Original draw: +5.6 [−3.3, +14.4] and +3.3 [−6.1, +12.8] pp.
  - Fresh draw: +0.0 [−9.4, +9.4] and −1.1 [−11.1, +8.3] pp.
  - All CIs include 0; the original-draw point estimates are larger than before the fix (+1.1 / +0.0).
- **E3 (margin, single query)**:
  - All prompts: +3.4 [−0.5, +7.2] pp in both seeds; the CI includes 0.
  - Three-product prompts: +10.0 [+2.0, +18.0] pp, unchanged.
- **Unit cost (reference only)** is now stable across seeds: E1 original +16.7 / +16.7 pp; E2 original
  +0.6 / +0.6 pp.
- **Interpretation**: S1 still applies. The paper must state the reuse gain as not distinguishable from 0, with
  a gain of up to about 14 pp not excluded. "Shows no sign of growth" is no longer used.
- **Paper updated (paper_tifs)**: §IV, §V-B, §V-D, Table III, Fig. 3 caption, Limitations, and the Appendix
  prior. The claim audit predates these numbers and must be rerun before submission.
- **E4 (margin, reuse, A-exact)**, both seeds completed (the seed-4647 law run no longer fails):
  - Dirichlet(0.1) gaps: +3.9 [−2.2, +10.6] / +2.2 [−5.0, +9.4] pp.
  - 16 intervals: +8.3 [−2.8, +19.4] / +6.7 [−3.9, +17.8] pp.
  - 256 intervals: +2.2 [−5.0, +9.4] / +2.2 [−4.4, +8.3] pp.
  - All CIs include 0 and both seeds agree. Reported in §V-D; the Limitations sentence about a failed law run
    was removed.
