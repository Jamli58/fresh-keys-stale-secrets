# Evidence Index

This index maps each check and result of the paper "Fresh Keys, Stale Secrets: Numeric Leakage across Linked LLM Prompts" (IEEE TIFS submission) and its Appendix to a test, script or result file in this repository.

- Numbering: Tables I–V and Figs. 1–6 are in the main paper; Tables VI–XV, Fig. 7 and Appendices A–F are in the Appendix (supplemental material).
- Locations: tests are in `tests/`, scripts in `scripts/`, results in `results/` (language-model runs in `results/autodl/`), and figure scripts in `figures/`.

## Checks (Section III)

- Pass log of the checks below: `results/R047_check_tests.txt`.
- Typed-field invariant: `tests/test_templates.py`.
- D3o mass function versus the released code: `tests/test_metric_dp_official.py`.
- Multi-query attacker equals A1+ at n=1: `tests/test_multi_query.py`.
- Neutral true-label hybrid equals A1+: `tests/test_hybrid.py`.
- D2 FinQA reconstruction: `results/R003a.json` (needs `data/raw/finqa_test.json`, see `data/README.md`).
- Self-consistency: `results/R010b.json`.
- Generator statistics of Section III and Appendix A (sector medians, atypical and extreme shares, cue correlation, residuals by positioning): `results/R001.json`.
- Population counts (distinct test products, the perturbation populations of Fig. 7): `results/R047_dataset_counts.json`, `scripts/dataset_counts.py`.
- Example prompt (Section III-A, Appendix A): `results/example_prompt.json`, `scripts/example_prompt.py`.

## D2 attackers (Section III-D)

- Reference likelihood: `src/numpriv/attacks/d2_reference.py`.
- Joint shared-map attacker: `src/numpriv/attacks/joint_map.py`.
- A-exact: `src/numpriv/attacks/exact_joint.py`, with `tests/test_exact_joint.py`.
- Display digits and branch reading: `results/R046_d2_display_check.json`, `scripts/check_d2_display.py`.
- Rejection-sampling check: `results/R046_d2_exact_v1_reference.json`, `scripts/run_d2_exact_v1.py`.

## Single-release results (Section IV)

- Table III and Fig. 3: `results/R007.json`, `results/R009.json`, `results/R033_d3o.json`; figure `figures/gen_fig_single.py`.
- Table VIII (strata): the strata of `results/R009.json` and `results/R033_d3o.json`.
- D3o with ε=10⁻⁶ (domain-only control): `results/R038_d3o_support_only.json`.
- Hit-optimal decisions: `results/R041_hit_decision.json`.
- Fig. 7: `results/R033_d3o.json`, `results/R004a.json`; figure `figures/gen_figA1_perturbation.py`.

## Linked-query results (Section V)

- Fig. 2 (headline): `results/R045_d2_joint_linked.json`, `results/R033_d3o.json`; figure `figures/gen_fig1_hero_panel.py`.
- Fig. 4: `results/R017_R019_B2.json`, `results/R033_d3o.json`; figure `figures/gen_fig2_linked_queries.py`.
- Fresh draw (Table XI): `results/R035_B2_fresh_seed3003.json`.
- Fresh-minus-reuse contrast: `results/R040_regime_contrast.json`, `scripts/regime_contrast.py`.
- Table IV:
  - reference likelihood: `results/R044_d2_reference_single.json`, `results/R044_d2_reference_linked.json`, `results/R044_d2_reference_sensitivity.json`;
  - joint attacker: `results/R045_d2_joint_linked.json`, `results/R045_d2_joint_fresh_draw.json`, `results/R045_d2_joint_single.json`, `results/R045_d2_joint_sensitivity.json`;
  - A-exact: `results/R046_d2_exact_linked_seed4646.json`, `results/R046_d2_exact_fresh_draw_seed4646.json`, `results/R046_d2_exact_single_seed4646.json`, and the seed-4647 runs.
  - Scripts: `scripts/run_d2_reference.py`, `scripts/run_d2_joint.py`, `scripts/run_d2_exact.py`, `scripts/summarize_r046.py`.
- Fig. 5: `results/R036_theory_width_montecarlo.json`, `results/R017_R019_B2.json`; figure `figures/gen_figA2_key_localization.py`.
- Map laws (Table XII): `results/R037_d2_sensitivity.json`, `scripts/run_d2_sensitivity.py`; for A-exact, `results/R046_d2_exact_law_seed4646.json` and `results/R046_d2_exact_law_seed4647.json`. The attacker's moments follow each law (`tests/test_bayes_parts.py`).
- Robustness (Section V-D):
  - Fig. 6 and Table IX: `results/R017_R019_B2.json`, `results/R033_d3o.json`, `results/R042_population_shift.json`; figure `figures/gen_fig_robust.py`.
  - Population shift (Tables IX and XIV): `results/R042_population_shift.json`, `scripts/run_population_shift.py`.
  - By sector (Table XIII): `results/R043_sector_breakdown.json`, `scripts/sector_breakdown.py`.
  - 20,000 prior draws: `results/R017_R019_B2_s20000.json`, `results/R039_d3o_linked_s20000.json`.
  - Two A-exact sampler seeds: `results/R046_d2_exact_v3.json`.
- Numerical validation (Table X):
  - joint attacker at smoothing 0.01: `results/R047a_d2_joint_smoothing_obs0.01_nuis400.json`, `results/R047a_d2_joint_smoothing_obs0.01_nuis1600.json`;
  - A-exact with a fourfold budget: `results/R047b_d2_exact_budget.json`;
  - table script: `figures/gen_table_numval.py`.

## Language-model study (Section VII, Appendix F, Table XV)

- Original runs: `results/autodl/R011_test_llama8b.json` and `results/autodl/R012_test_qwen32b.json`, with row-level outputs in `results/autodl/*_rows.jsonl` and the request cache in `results/autodl/llm_cache/`.
- Replay from cache without new model calls: `results/R034_a4_replay.json`, `scripts/run_a4_replay.py`. It reproduces every original value of Table XV; one undisplayed cell differs by one hit in 2,100 rows.
- Oracle cues: `results/R032_cue_ceiling_test.txt`, `scripts/cue_ceiling.py`.
- Development cue diagnostics: `results/R031b_a4_v3_cue_use_dev.txt`, `scripts/a4_cue_diagnostic.py`.
- Serving the two models: `cloud/` (vLLM on a rented GPU; see `cloud/README.md`).

## Shared code and constants

- Bootstrap: `src/numpriv/metrics.py`.
- Mechanisms: `src/numpriv/defenses/`.
- Likelihoods (16,000 simulated maps): `src/numpriv/attacks/likelihoods.py`.
- Constants: `src/numpriv/config.py`.
- Margin anchors: `scripts/sector_margins.py`.
- Cities: `scripts/build_city_table.py`.

## Prespecified plans

The three plans in `plans/` were committed to the authors' research repository before the runs they govern:

| Plan | Governs | Committed before |
|---|---|---|
| `plans/EXPERIMENT_PLAN_20260924_112737.md` | language-model endpoints C1a–C1c and the success criterion (row C1) | the test run |
| `plans/EXPERIMENT_PLAN_20261009_exact_joint.md` | A-exact endpoints and numerical checks | the main R046 runs |
| `plans/EXPERIMENT_PLAN_20261018_numerical_validation.md` | R047 | the R047 runs |

Each plan's run log records the runs and outcomes.
