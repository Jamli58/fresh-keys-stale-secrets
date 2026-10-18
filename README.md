# Fresh Keys, Stale Secrets

Code, results and figure scripts for the paper **"Fresh Keys, Stale Secrets: Numeric Leakage across Linked LLM Prompts"** (submitted to IEEE Transactions on Information Forensics and Security).

The paper asks how much a cloud provider recovers of the confidential prices, unit costs and margins that small firms put into prompts, when the numbers are disguised and the provider links a firm's repeated queries about the same values. The measurement uses synthetic firms with ground truth and four numeric disguises:
- shared and per-field scaling;
- a fixed-axis order-preserving substitution;
- the released money sanitizer of Prεεmpt.

The attacks are Bayesian attackers that know each mechanism and hold sector-level price and margin statistics.

The central finding: per-prompt evaluation understates leakage. With fresh secret randomness per query, recovery grows with the number of linked queries; one reused secret caps what the queries reveal.

## Layout

| Path | Contents |
|---|---|
| `src/numpriv/` | generator, prompt templates, disguises (`defenses/`), attackers (`attacks/`), metrics |
| `tests/` | unit tests, including every implementation check cited in the paper |
| `scripts/` | experiment scripts; each writes one result file to `results/` |
| `results/` | all result files used in the paper; language-model runs in `results/autodl/` |
| `figures/` | scripts for the paper's figures and the numerical-validation table |
| `cloud/` | scripts that serve the two language models with vLLM on a rented GPU (notes in Chinese) |
| `plans/` | prespecified experiment plans, committed before the runs they govern |
| `data/README.md` | sources of the third-party raw data (not redistributed) |
| `EVIDENCE.md` | **maps every table, figure and check of the paper to its result file and script** |

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest                 # all unit tests
uv run pytest -m "not slow"   # skip the rejection-sampling reference checks (minutes)
```

## Reproducing the results

Every number in the paper comes from a JSON file in `results/`. `EVIDENCE.md` gives the file and field for each table and figure.

**Re-running the classical experiments** (CPU only, from the repository root). For example:

```bash
uv run python scripts/run_m0.py                                  # generator statistics, implementation checks
uv run python scripts/run_b2.py                                  # linked queries, D1 / D1' / D2 / D3 (Fig. 4)
uv run python scripts/run_d3o.py                                 # released Prεεmpt sanitizer (D3o)
uv run python scripts/run_d2_joint.py --part linked              # joint shared-map attacker on D2
uv run python scripts/run_d2_exact.py --part linked --workers 8  # particle attacker A-exact (hours)
uv run python scripts/run_population_shift.py                    # population shifts
```

Each script's docstring lists its parts and options. The A-exact runs take about one to two minutes per 16-query sequence on an Apple M4 with eight worker processes.

**Figures:**

```bash
cd figures
uv run --with matplotlib python gen_fig1_hero_panel.py
```

The other `gen_*.py` scripts work the same way. They read `results/` and write PDFs into `figures/`.

**Language-model study.** The test run used Llama-3.1-8B-Instruct and Qwen2.5-32B-Instruct (AWQ), served with vLLM; `cloud/README.md` describes the setup. The recorded answers are in `results/autodl/` (row-level outputs and the request cache `llm_cache/`), so the hybrid attacker can be replayed without any model calls:

```bash
uv run python scripts/run_a4_replay.py
```

**Raw data.** The experiments use the derived city table and sector constants shipped in `src/numpriv/`. The raw files are needed only to rebuild them or to rerun the FinQA reconstruction check; see `data/README.md`.

## Citation

If you use this code, please cite the paper (bibliographic details to be added on publication):

```
Fresh Keys, Stale Secrets: Numeric Leakage across Linked LLM Prompts.
Submitted to IEEE Transactions on Information Forensics and Security, 2026.
```

## License

MIT; see `LICENSE`.
