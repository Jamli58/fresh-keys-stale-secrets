"""Table XII (supplementary material): LLM hybrids vs the oracle classical attacker A1+ (test split, 2,100 rows per cell).

Sources: R034_a4_replay.json (A4 as recorded; A4 with true sector/class, cue shifts kept) and
R032_cue_ceiling_test.txt (oracle-cue diagnostic: true city level; + true brand premium; no LLM).
"""

import json

from paper_plot_style import FIG_DIR, RESULTS, load, signed, signed_ci

R34 = load("R034_a4_replay.json")["result"]
ORACLE_ORDER = ("D1", "D3@0.5", "D3@1", "D2")  # line order of R032_cue_ceiling_test.txt
oracle = {}
for key, line in zip(ORACLE_ORDER, (RESULTS / "R032_cue_ceiling_test.txt").read_text().splitlines()):
    name, payload = line.split(" ", 1)
    oracle[key] = json.loads(payload)
CELLS = [("D1", "unit_cost_hit_10pct"), ("D2", "margin_hit_5pp"), ("D2", "unit_cost_hit_10pct"),
         ("D3@0.5", "unit_cost_hit_10pct"), ("D3@0.5", "margin_hit_5pp"),
         ("D3o@1", "unit_cost_hit_10pct"), ("D3o@1", "margin_hit_5pp")]
NAMES = {"D1": "D1", "D2": "D2 (ours)", "D3@0.5": "D3, $\\varepsilon$=0.5", "D3o@1": "D3o, $\\varepsilon$=1"}
METRIC = {"unit_cost_hit_10pct": "cost", "margin_hit_5pp": "margin"}


def ci(t):
    v, bounds = t
    return signed_ci(v, bounds)


def oc(d, m, mode):
    if d not in oracle:
        return "--"
    _, adv, bounds = oracle[d][mode][m]
    return signed_ci(adv, bounds)


rows = []
for d, m in CELLS:
    cells = [ci(R34[f"{mdl}/{d}"][k][m]) for mdl in ("llama8b", "qwen32b")
             for k in ("A4_minus_A1plus", "A4truelab_minus_A1plus")]
    rows.append(f"{NAMES[d]} & {METRIC[m]} & " + " & ".join(cells) + f" & {oc(d, m, 'city')} & {oc(d, m, 'latent')} \\\\")
lines = [
    "\\begin{table*}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
    "\\caption{Do LLM priors help? Paired differences to the oracle classical attacker A1+ in hit rate (pp; cost $\\pm$10\\%, "
    "margin $\\pm$5\\,pp) with firm-clustered 95\\% bootstrap CIs (10,000 resamples; 2,000 for oracle cues), test split (250 counterfactual firms; 700 product observations $\\times$ 3 seeds = 2,100 rows per cell). A4: hybrid that reads the "
    "masked prose and shifts the prior (the recorded test-run answers, replayed from cache). A4-TL: same cue shifts, sector and class "
    "replaced by the truth. Oracle cues: the generator's true city level (and brand premium) fed to the same solver, no LLM; "
    "a diagnostic, not a bound. Pre-registered criterion (positive with CI above zero on D1 cost or D3 $\\varepsilon$=0.5 for "
    "$\\geq$2 models): unmet.}",
    "\\label{tab:llm}",
    "\\begin{tabular}{llrrrrrr}", "\\toprule",
    " & & \\multicolumn{2}{c}{Llama-3.1-8B} & \\multicolumn{2}{c}{Qwen2.5-32B} & \\multicolumn{2}{c}{Oracle cues (no LLM)} \\\\",
    "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
    "Mechanism & Metric & A4 & A4-TL & A4 & A4-TL & city & city + premium \\\\",
    "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table*}",
]
out = FIG_DIR / "TABLE_2_llm.tex"
out.write_text("\n".join(lines) + "\n")
print(f"Saved: {out}")
