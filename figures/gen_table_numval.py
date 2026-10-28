"""Table X (supplementary material): numerical validation of the D2 reuse-side attackers (R045 and R047).

Sources: R045_d2_joint_linked.json (default smoothing 0.03, 400 draws), R045_d2_joint_sensitivity.json,
R047a_d2_joint_smoothing_obs0.01_nuis{400,1600}.json, R047b_d2_exact_budget.json. Margin hits (+-5 pp),
original firms, firm-clustered 95% bootstrap CIs. Prints the LaTeX rows.
"""

from paper_plot_style import load

M = "margin_hit_5pp"


def ci(est: float, lo_hi) -> str:
    return f"{100 * est:+.1f} [{100 * lo_hi[0]:+.1f}, {100 * lo_hi[1]:+.1f}]"


def gain(d: dict) -> str:
    return ci(d[f"gain_{M}_n16_vs_n1"], d[f"gain_{M}_n16_vs_n1_ci"])


sens = load("R045_d2_joint_sensitivity.json")["result"]
JOINT = [("0.03", "400 (main)", load("R045_d2_joint_linked.json")["result"]),
         ("0.03", "1,600", sens["obs_sd=0.03, n_nuisance=1600"]),
         ("0.1", "400", sens["obs_sd=0.1, n_nuisance=400"]),
         ("0.01", "400", load("R047a_d2_joint_smoothing_obs0.01_nuis400.json")["result"]),
         ("0.01", "1,600", load("R047a_d2_joint_smoothing_obs0.01_nuis1600.json")["result"])]
for sd, draws, r in JOINT:
    c = r["fresh_minus_reuse"][M]
    print(f"{sd} & {draws} & {gain(r['D2/fresh'])} & {gain(r['D2/reuse'])} & {ci(c['contrast'], c['contrast_ci'])} \\\\")
b = load("R047b_d2_exact_budget.json")["result"]
for name, label in (("base", "20{,}000 / 400{,}000 / 40 (main)"), ("high", "80{,}000 / 1{,}600{,}000 / 160")):
    s = b["summaries"][name]
    print(f"{label} & {100 * s['1'][M]:.1f} & {100 * s['16'][M]:.1f} & {gain(s)} \\\\")
print(f"% E1 {ci(b['E1_gain_diff_high_minus_base'], b['E1_ci'])}; E2 {b['E2_mean_abs_margin_est_diff_n1']:.3f} / "
      f"{b['E2_mean_abs_margin_est_diff_n16']:.3f}; sequences {b['n_sequences']}")
