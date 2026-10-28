"""Table XI (supplementary material): D2 leakage under alternative map laws (the attacker is told each law).

Source: R037_d2_sensitivity.json (scripts/run_d2_sensitivity.py). Single query: A1+ vs A0 on the test split
(2,100 rows); linked queries: fresh maps, product 0, 60 test firms x 3 seeds, n = 1 -> 16. The default law
reproduces R009.json (single query) and R017_R019_B2.json D2/fresh (linked).
"""

from paper_plot_style import FIG_DIR, load, signed_ci

R = load("R037_d2_sensitivity.json")["result"]
NAMES = {
    "default (64 intervals, uniform cuts, targets [1, 1e6])": "default: 64 intervals, uniform cuts, $[1,10^6]$",
    "16 intervals": "16 intervals",
    "256 intervals": "256 intervals",
    "irregular gaps (Dirichlet 0.1)": "irregular gaps, Dirichlet(0.1)",
    "regular gaps (Dirichlet 10)": "regular gaps, Dirichlet(10)",
    "targets [1, 1e12]": "targets $[1,10^{12}]$",
}


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


lines = ["\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
         "\\caption{\\Dtwo under alternative laws of the secret map, with the attacker told each law (it simulates "
         "its Gaussian moments for that law; solver otherwise frozen). Single query: advantage of \\Aplus over \\Azero "
         "on the test split (2{,}100 rows). Linked queries: fresh maps, product~0, 60 test firms $\\times$ 3 seeds, "
         "margin hit rate at $n{=}1$ and $n{=}16$ and the paired gain. Firm-clustered 95\\% bootstrap CIs (pp). "
         "The default law reproduces \\cref{tab:single} and \\cref{fig:linked}.}",
         "\\label{tab:d2law}", "\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
         " & \\multicolumn{2}{c}{Single query, $\\Delta$ vs \\Azero} & \\multicolumn{3}{c}{Linked, margin hit (\\%)} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}",
         "Map law & margin & cost & $n{=}1$ & $n{=}16$ & gain \\\\", "\\midrule"]
for key, name in NAMES.items():
    s, link = R[key]["single_query_test"], R[key]["linked_fresh"]
    lines.append(f"{name} & {signed_ci(s['advantage_margin_hit_5pp'], s['advantage_margin_hit_5pp_ci'])} & "
                 f"{signed_ci(s['advantage_unit_cost_hit_10pct'], s['advantage_unit_cost_hit_10pct_ci'])} & "
                 f"{pct(link['1']['margin_hit_5pp'])} & {pct(link['16']['margin_hit_5pp'])} & "
                 f"{signed_ci(link['gain_margin_hit_5pp_n16_vs_n1'], link['gain_margin_hit_5pp_n16_vs_n1_ci'])} \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
out = FIG_DIR / "TABLE_4_d2_law.tex"
out.write_text("\n".join(lines) + "\n")
print(f"Saved: {out}")
