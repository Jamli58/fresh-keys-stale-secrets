# R049: prior-model mismatch, value drift and the full test set (plan, fixed before any run)

Date: 2026-10-26.

Motivation: the ccf-paper-reviewer review of 2026-10-26 (`ccfa-review-reports/fresh-keys-stale-secrets-tifs-review.md`):

- [C001] All results use one synthetic generator, and the attacker's prior comes from the same generator.
- [C002] The values are unchanged across the 16 queries; drift is untested.
- [C011] The linked-query experiments use 60 of the 250 test firms.

Scope: no language-model run is involved, and the LLM test split is not touched. The attackers stay frozen: the R008 solver configuration, and the joint attacker with the R045 defaults. Bootstrap CIs are firm-clustered, with 10,000 resamples.

## R049a: attacker prior from a different population model

The evaluated firms are unchanged: eval seed 1001, the first 60 test firms × 3 defense seeds, with the same random streams as R017 / R033 / R045. Only the attacker's prior table (P10 / Q25 / median / Q75 / P90 per sector and class) is rebuilt. It comes from 2,000 firms (prior seed 2002) drawn from an alternative model.

### Prior models

- **PM1, tails and spreads:**
  - price noise Student-t with 30 degrees of freedom instead of 3, using the same scaling, so it is lighter-tailed and narrower;
  - firm premium σ, city residual σ and firm-efficiency σ each × 1.5;
  - no extreme firms (extreme share 0 instead of 0.2).
- **PM2, locations:**
  - every product-class price median is multiplied by exp(N(0, 0.25²));
  - every sector's margin median is shifted by N(0, 0.3²) on the logit scale;
  - these factors are drawn once, with rng seed 4949;
  - everything else is unchanged.
- **PM3:** PM1 and PM2 together.

### Endpoints

1. Single query, A1+ versus A0 on the 2,100 test rows, with both attackers using the mismatched prior: margin and unit-cost advantage under D1′, D2 and D3o at ε=1.
2. Linked queries, A1+, n ∈ {1, 16}: the fresh-secret gain for the reported targets, which are D1 unit cost, D1′ unit cost, D2 margin, and D3o margin at ε=0.5, 1, 2. The reuse gains are also reported for D1, D1′ and D2.
3. Under PM3 only: the D2 joint attacker, fresh and reuse, n ∈ {1, 16}, and the fresh-minus-reuse margin contrast.

### Interpretation rules

- **MA:** under a prior model, if every reported fresh-secret gain has a CI lower endpoint above 0, the direction of the history result holds under that prior mismatch. Otherwise, name the failing targets and narrow the paper's claim accordingly.
- **MJ:** if the PM3 joint fresh-minus-reuse margin contrast has a CI excluding 0, the contrast holds under prior mismatch. Otherwise, report it as weakening the comparison.

## R049b: value drift across queries

- **Population and streams:** the original firms and prior (as in R017), with the same disguise random streams as R017. A separate rng stream ([seed, firm index, 4949, δ in per-mille]) draws the drift, so δ = 0 reproduces R017 exactly.
- **Drift model:** in query j, every product's price and unit cost are multiplied by independent factors exp(N(0, δ²)), rounded to cents, with the cost kept below the price. Hits are scored against the undrifted values.
- **Levels:** δ ∈ {0.01, 0.02, 0.05}.
- **Attacker:** the multi-query A1+ is frozen and not told about the drift.
- **Regime:** fresh secrets only (D3o draws fresh noise anyway). The reuse ceiling assumes unchanged values; reuse under drift is not evaluated, and the paper says so.
- **Endpoints:** the n = 1 → 16 gains of the reported targets (as in R049a), with CIs.
- **Interpretation rule:**
  - **DA:** for each δ, report which targets keep a CI lower endpoint above 0;
  - state in the paper the largest tested δ at which all reported targets do so;
  - if none qualifies, narrow the claim to unchanged values.

## R049c: all 250 test firms

- **Script:** `scripts/run_b2.py --n-firms 250 --n-values 1 16 --out R049c_B2_all_test.json`. All configs; same streams per firm index, so the first 60 firms reproduce R017 / R033 at n ∈ {1, 16}.
- **Endpoints:**
  - the n = 1 → 16 gains for every mechanism and regime;
  - per-sector fresh gains (about 41 firms per sector) for the reported targets.
- **Use in the paper:** a power check next to the 60-firm results. No headline number is replaced, because the stronger attackers (joint, A-exact) remain at 60 firms.
