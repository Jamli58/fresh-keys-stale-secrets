# R047: numerical validation of the D2 reuse-side attackers (plan, fixed before any run)

Date: 2026-10-18. Motivation: TIFS round-7 review, major 1 (thread 01a1288a-4b5b-7e51-a546-499453a15c0c):

- The joint attacker smooths shown values by 0.03 in log units. Only larger smoothing (0.1) has been tested, not smaller.
- A-exact has been checked against exact rejection sampling only on small one-digit instances. At production precision (six digits) only two sampler seeds were compared.

No language-model run is involved; the LLM test split is not touched.

## R047a: joint shared-map attacker with smaller smoothing

- **Script:** `scripts/run_d2_joint.py --part smoothing` (new part, same code path as `sensitivity`).
- **Population:** original firms (eval seed 1001), 60 test firms × 3 defense seeds, D2 default law.
- **Regimes and queries:** fresh and reuse, n ∈ {1, 16}. Same random streams as R045, so rows are paired with R045.
- **Configurations:**
  - S1: `obs_sd=0.01, n_nuisance=400` (the main estimate for this part);
  - S2: `obs_sd=0.01, n_nuisance=1600` (Monte Carlo check of S1).
  - The two run as separate processes.
- **Endpoints** (margin hits ±5 pp; unit cost reported, not used for decisions):
  - J1: fresh-minus-reuse difference in the n=1→16 margin gain, firm-clustered 95% bootstrap CI (10,000 resamples);
  - J2: reuse margin gain n=1→16, with CI;
  - J3: fresh margin gain n=1→16, with CI.
- **Numerical rule:**
  - If S1 and S2 differ by more than 5 pp in any margin hit rate at n=1 or 16, S1 is not numerically resolved, and J1–J3 are taken from S2.
  - If S2 also differs from S1 by more than 5 pp, report that smoothing 0.01 is not resolved with these Monte Carlo budgets; J1 is then reported only as such.
- **Interpretation rules** (applied to the J1 used):
  - JA: CI of J1 excludes 0. The fresh–reuse difference persists at smaller smoothing; report J1–J3 next to the 0.03 result.
  - JB: CI of J1 includes 0. Report this as weakening the joint-attacker comparison, and revise the paper text that relies on it.

## R047b: A-exact under a larger computational budget

- **Script:** `scripts/run_d2_exact.py --part budget` (new part).
- **Population:** original firms (eval seed 1001), the first 20 of the 60 test firms × 3 defense seeds = 60 reuse sequences, D2 default law. Same random streams as R046 linked.
- **Configurations** (both with sampler seed 4646):
  - base: 20,000 particles, 400,000 initial maps, 40 MH steps (as in R046);
  - high: 80,000 particles, 1,600,000 initial maps, 160 MH steps.
- **Per-sequence output:** margin and unit-cost hits and the margin estimate (posterior median) of product 0 at n = 1, 2, 4, 8, 16.
- **Endpoints:**
  - E1: paired difference high − base in the n=1→16 reuse margin gain, firm-clustered 95% bootstrap CI;
  - E2: mean absolute difference high − base of product 0's margin estimate, at n=1 and at n=16;
  - E3: margin hit-rate difference high − base at n=1 and at n=16.
- **Interpretation rules:**
  - EA (budget-stable): |E1| ≤ 3.3 pp (the largest between-seed difference already reported) and E2 ≤ 0.01 at both n. State that A-exact's reuse result is stable under a fourfold budget.
  - EB (budget-sensitive): otherwise. Report the high-budget subset numbers, state that A-exact's reuse gain depends on the budget, and narrow every sentence that uses A-exact.
- **Not done, and why:** an exact reference (rejection sampling) at six significant digits is infeasible because its acceptance probability is negligible. This is stated in the paper.

## Reporting

- All outcomes are reported, whichever rule applies.
- Results go to `results/R047a_d2_joint_smoothing.json` and `results/R047b_d2_exact_budget.json`, with an entry in this file's run log.
- Paper changes:
  - §V-D: one or two sentences with J1 and E1/E2;
  - Appendix: one small table.

## Run log

(appended during the runs)
- 2026-10-18 07:53: code added (run_d2_joint.py part "smoothing", run_d2_exact.py part "budget"; rows now also carry the margin estimate). Smoke test on one sequence passed. Started R047a S1 (obs_sd 0.01, 400 draws), R047a S2 (obs_sd 0.01, 1600 draws) and R047b (workers 8 base / 4 high) in the background; logs results/r047a_s1.log, r047a_s2.log, r047b.log.
- 2026-10-18 08:41: R047a S1 finished (2,886 s) → results/R047a_d2_joint_smoothing_obs0.01_nuis400.json.
- 2026-10-18 10:55: R047a S2 (10,899 s) and R047b (10,947 s) finished → R047a_d2_joint_smoothing_obs0.01_nuis1600.json, R047b_d2_exact_budget.json.

## Outcome (rules applied as fixed above)

- **R047a numerical rule:** S1 and S2 differ by at most 2.2 pp in any margin hit rate at n=1 or 16 (threshold 5 pp), so J1–J3 come from S1.
  - J1 (fresh − reuse margin gain): +31.7 [21.7, 41.7] pp (S2: +33.9 [23.9, 43.9]).
  - J2 (reuse gain): +5.6 [−1.1, 12.2] pp (S2: +3.9 [−1.7, 9.4]).
  - J3 (fresh gain): +37.2 [27.2, 47.2] pp.
  - **Rule JA applies:** the fresh–reuse difference persists at smaller smoothing.
- **R047b:**
  - E1 (high − base reuse margin gain): −1.7 [−5.0, 0.0] pp (|E1| ≤ 3.3).
  - E2 (mean |Δ margin estimate|): 0.003 (n=1), 0.004 (n=16), both ≤ 0.01.
  - E3: 0.0 (n=1), −1.7 [−5.0, 0.0] pp (n=16).
  - **Rule EA applies:** budget-stable.
  - Diagnostics: the P10 of the smallest per-update ESS was 10.7 (base) and 14.2 (high); the mean time per sequence was 69 s and 684 s.
- **Paper:**
  - §V-D: two sentences (J1 and J2, then E1 and E2), plus a note that no exact rejection reference exists at six significant digits;
  - new Appendix table `tab:numval`, generated by `paper_tifs/figures/gen_table_numval.py`.
