"""Table V (main paper): D2 attack validation — simulation-based reference likelihood and joint shared-map attacker.

Sources: R044_d2_reference_single.json, R044_d2_reference_linked.json (reference likelihood, attacks/d2_reference.py;
20,000 simulated maps, bandwidth 0.03), R045_d2_joint_linked.json and R045_d2_joint_fresh_draw.json (joint attacker,
attacks/joint_map.py), with the Gaussian pair-only attacker from R009 / R017_R019_B2.json / R037 / R035.
"""

from paper_plot_style import FIG_DIR, load, signed_ci

REF_S = load("R044_d2_reference_single.json")["result"]
REF_L = load("R044_d2_reference_linked.json")["result"]
J_O = load("R045_d2_joint_linked.json")["result"]
J_F = load("R045_d2_joint_fresh_draw.json")["result"]
J_S = load("R045_d2_joint_sensitivity.json")["result"]
B2 = load("R017_R019_B2.json")["result"]
B2F = load("R035_B2_fresh_seed3003.json")["result"]
LAW = load("R037_d2_sensitivity.json")["result"]
IRREG = "irregular gaps (Dirichlet 0.1)"
M, C = "margin_hit_5pp", "unit_cost_hit_10pct"


def adv(x, m=M):
    return signed_ci(x[f"advantage_{m}"], x[f"advantage_{m}_ci"])


def gain(x, m=M):
    return signed_ci(x[f"gain_{m}_n16_vs_n1"], x[f"gain_{m}_n16_vs_n1_ci"])


def rate(x, n):
    return f"{100 * x[n][M]:.1f}"


rows = [
    ("Gaussian pair (paper), default law", "fresh", adv(REF_S["default"]["gaussian_vs_A0"]), rate(B2["D2/fresh"], "1"),
     rate(B2["D2/fresh"], "16"), gain(B2["D2/fresh"]), gain(B2["D2/fresh"], C)),
    ("Reference likelihood, default law", "fresh", adv(REF_S["default"]["reference_vs_A0"]), rate(REF_L["default"], "1"),
     rate(REF_L["default"], "16"), gain(REF_L["default"]), gain(REF_L["default"], C)),
    ("Gaussian pair, Dirichlet(0.1) gaps", "fresh", adv(REF_S[IRREG]["gaussian_vs_A0"]), rate(LAW[IRREG]["linked_fresh"], "1"),
     rate(LAW[IRREG]["linked_fresh"], "16"), gain(LAW[IRREG]["linked_fresh"]), gain(LAW[IRREG]["linked_fresh"], C)),
    ("Reference likelihood, Dirichlet(0.1) gaps", "fresh", adv(REF_S[IRREG]["reference_vs_A0"]), rate(REF_L[IRREG], "1"),
     rate(REF_L[IRREG], "16"), gain(REF_L[IRREG]), gain(REF_L[IRREG], C)),
    ("Joint shared-map attacker", "fresh", "--", rate(J_O["D2/fresh"], "1"), rate(J_O["D2/fresh"], "16"),
     gain(J_O["D2/fresh"]), gain(J_O["D2/fresh"], C)),
    ("Joint shared-map attacker", "reuse", "--", rate(J_O["D2/reuse"], "1"), rate(J_O["D2/reuse"], "16"),
     gain(J_O["D2/reuse"]), gain(J_O["D2/reuse"], C)),
    ("Joint, fresh draw", "fresh", "--", rate(J_F["D2/fresh"], "1"), rate(J_F["D2/fresh"], "16"),
     gain(J_F["D2/fresh"]), gain(J_F["D2/fresh"], C)),
    ("Joint, fresh draw", "reuse", "--", rate(J_F["D2/reuse"], "1"), rate(J_F["D2/reuse"], "16"),
     gain(J_F["D2/reuse"]), gain(J_F["D2/reuse"], C)),
]
co, cf = J_O["fresh_minus_reuse"][M], J_F["fresh_minus_reuse"][M]
lines = ["\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
         "\\caption{Validation of the \\Dtwo attacker. \\emph{Reference likelihood}: the Gaussian pair likelihood is "
         "replaced by a kernel density over 20{,}000 simulated maps (bandwidth 0.03 in log units); same prior, oracle and "
         "decision rule. \\emph{Joint shared-map attacker}: models the prices and costs of all products shown under the "
         "same map (observation smoothing 0.03 in log units), integrating the other products' unchanged values once over "
         "their class priors (400 shared draws). Single query: margin advantage over \\Azero "
         "(test split). Linked: product~0 margin hit rate (\\%) at $n{=}1$ and $16$ and the gains (pp), 60 test firms "
         "$\\times$ 3 seeds, original firms unless noted. Fresh-minus-reuse margin gain under the joint attacker: "
         f"{signed_ci(co['contrast'], co['contrast_ci'])} (original), {signed_ci(cf['contrast'], cf['contrast_ci'])} "
         "(fresh draw); with 1{,}600 draws or smoothing 0.1 instead: "
         + ", ".join(signed_ci(v["fresh_minus_reuse"][M]["contrast"], v["fresh_minus_reuse"][M]["contrast_ci"]) for v in J_S.values())
         + ". Firm-clustered 95\\% bootstrap CIs.}",
         "\\label{tab:d2valid}", "\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
         "Attacker & Map & Single $\\Delta$ margin & $n{=}1$ & $n{=}16$ & Margin gain & Cost gain \\\\", "\\midrule"]
lines += [" & ".join(r) + " \\\\" for r in rows]
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
out = FIG_DIR / "TABLE_8_d2_validation.tex"
out.write_text("\n".join(lines) + "\n")
print(f"Saved: {out}")
