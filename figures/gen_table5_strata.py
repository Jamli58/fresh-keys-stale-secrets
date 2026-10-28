"""Table VII (supplementary material): single-query recovery for typical vs atypical firms (median product margin outside the
sector's P10-P90 band of the prior table).

Sources: R009.json (A0 baseline rows inside each mechanism's advantage; D1, D1', D2) and R033_d3o.json
(D3o single_query_test, A0 and A1+), strata 'typical' (1,602 rows) and 'atypical' (498 rows).
"""

from paper_plot_style import FIG_DIR, load, signed_ci

R009 = load("R009.json")["result"]
D3O = load("R033_d3o.json")["result"]
ROWS = [("\\Done", R009["D1_shared_scaling"]), ("\\Donep", R009["D1p_per_field_scaling"]),
        ("\\Dtwo (ours)", R009["D2_zhang_order_preserving"])]
ROWS += [(f"\\Dthreeo, $\\varepsilon$={e}", D3O[f"D3o@eps={e}"]["single_query_test"]["A1plus_bayes_oracle"])
         for e in ("0.5", "1.0", "2.0")]
A0 = D3O["D3o@eps=1.0"]["single_query_test"]["A0_prior_only"]


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


lines = ["\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
         "\\caption{Single-query recovery by firm type, test split. A firm is atypical when its median product margin "
         "lies outside its sector's P10--P90 band in the prior table (1{,}602 typical and 498 atypical rows). Hit rate "
         "of \\Aplus (\\%) and advantage over \\Azero (pp, firm-clustered 95\\% bootstrap CI). \\Azero hits "
         f"{pct(A0['typical']['margin_hit_5pp'])}\\% / {pct(A0['atypical']['margin_hit_5pp'])}\\% of margins and "
         f"{pct(A0['typical']['unit_cost_hit_10pct'])}\\% / {pct(A0['atypical']['unit_cost_hit_10pct'])}\\% of unit "
         "costs (typical / atypical).}",
         "\\label{tab:strata}", "\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule",
         " & \\multicolumn{4}{c}{Typical firms} & \\multicolumn{4}{c}{Atypical firms} \\\\",
         "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
         "Mechanism & margin & $\\Delta$ margin & cost & $\\Delta$ cost & margin & $\\Delta$ margin & cost & $\\Delta$ cost \\\\",
         "\\midrule"]
for name, r in ROWS:
    cells = []
    for st in ("typical", "atypical"):
        v = r[st]
        cells += [pct(v["margin_hit_5pp"]), signed_ci(v["advantage_margin_hit_5pp"], v["advantage_margin_hit_5pp_ci"]),
                  pct(v["unit_cost_hit_10pct"]), signed_ci(v["advantage_unit_cost_hit_10pct"], v["advantage_unit_cost_hit_10pct_ci"])]
    lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
out = FIG_DIR / "TABLE_5_strata.tex"
out.write_text("\n".join(lines) + "\n")
print(f"Saved: {out}")
