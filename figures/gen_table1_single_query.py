"""Table IV (main paper): single-query leakage on the test split (250 firms, 700 products x 3 seeds = 2,100 rows).

Sources: R007.json (A0), R009.json (A1+: D0, D1, D1', D2, D3 paper-reading variant), R033_d3o.json (A1+: D3o).
"""

from paper_plot_style import FIG_DIR, load, pct, signed_ci

R7, R9 = load("R007.json")["result"], load("R009.json")["result"]
D3O = load("R033_d3o.json")["result"]
ROWS = [
    ("No disguise (D0)", R9["D0_none"]["all"]),
    ("Shared scaling (D1)", R9["D1_shared_scaling"]["all"]),
    ("Per-field scaling (D1$'$)", R9["D1p_per_field_scaling"]["all"]),
    ("Zhang et al., our impl.\\ (D2)", R9["D2_zhang_order_preserving"]["all"]),
    *[(f"Pr$\\varepsilon\\varepsilon$mpt released (D3o), $\\varepsilon$={e}",
       D3O[f"D3o@eps={e}"]["single_query_test"]["A1plus_bayes_oracle"]["all"]) for e in ("0.5", "1.0", "2.0")],
    *[(f"Pr$\\varepsilon\\varepsilon$mpt paper reading (D3), $\\varepsilon$={e}",
       R9[f"D3_prempt_metric_dp@eps={e}"]["all"]) for e in ("0.5", "1.0", "2.0")],
]
a0 = R7["D0_none"]["all"]


def row(name, s, floor=False):
    if floor:
        return f"{name} & {pct(s['margin_hit_5pp'])} & {pct(s['unit_cost_hit_10pct'])} & {s['margin_mae_pp']:.1f} & -- & -- \\\\"
    mae = s["margin_mae_pp"]
    mae_txt = f"{mae:.3f}" if 0 < mae < 0.1 else f"{mae:.1f}"
    return (f"{name} & {pct(s['margin_hit_5pp'])} & {pct(s['unit_cost_hit_10pct'])} & {mae_txt} & "
            f"{signed_ci(s['advantage_margin_hit_5pp'], s['advantage_margin_hit_5pp_ci'])} & "
            f"{signed_ci(s['advantage_unit_cost_hit_10pct'], s['advantage_unit_cost_hit_10pct_ci'])} \\\\")


lines = [
    "\\begin{table*}[t]", "\\centering", "\\small",
    "\\caption{Single-query leakage on the test split (250 firms; 700 product observations, 450 distinct products, "
    "$\\times$ 3 defense seeds = 2,100 rows). Margin hit: absolute error $\\leq$5\\,pp; cost hit: relative error "
    "$\\leq$10\\%. Attacker: the oracle classical Bayesian attacker A1+ (told the true sector and product class); MAE of "
    "its margin estimate; advantage = paired difference to the prior-only attacker A0 with firm-clustered 95\\% "
    "bootstrap CIs (10,000 resamples). D2 is our implementation of Zhang et al.'s numeric substitution without topic "
    "shift; D3o is our implementation of Pr$\\varepsilon\\varepsilon$mpt's released money-sanitizer functions, with "
    "the full $\\varepsilon$ per money value; D3 follows the paper's description with $\\varepsilon/t$ per value "
    "token (variant), so equal nominal $\\varepsilon$ is not an equal budget allocation.}",
    "\\label{tab:single}",
    "\\begin{tabular}{lrrrrr}", "\\toprule",
    "Mechanism & Margin hit (\\%) & Cost hit (\\%) & Margin MAE (pp) & $\\Delta$ margin vs A0 (pp) & $\\Delta$ cost vs A0 (pp) \\\\",
    "\\midrule",
    row("Prior only (A0)", a0, floor=True),
    "\\midrule",
    *[row(n, s) for n, s in ROWS[:7]],
    "\\midrule",
    *[row(n, s) for n, s in ROWS[7:]],
    "\\bottomrule", "\\end{tabular}", "\\end{table*}",
]
out = FIG_DIR / "TABLE_1_single_query.tex"
out.write_text("\n".join(lines) + "\n")
print(f"Saved: {out}")
