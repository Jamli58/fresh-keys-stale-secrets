"""Supplementary-material tables from R049 (plan: EXPERIMENT_PLAN_20261026_r049.md).

Table A (prior-mismatch and drift rows of Table VIII): n = 1 -> 16 fresh-secret gains of the reported targets for
the original setting (R017_R019_B2.json, R033_d3o.json), the three prior models (R049a_prior_mismatch.json) and the
three drift levels (R049b_drift.json).
Table B (Table IX): the same gains by sector on all 250 test firms (R049c_power_sector.json).
Prints the LaTeX rows.
"""

from paper_plot_style import load

COLS = (("D1/fresh", "unit_cost_hit_10pct"), ("D1p/fresh", "unit_cost_hit_10pct"), ("D2/fresh", "margin_hit_5pp"),
        ("D3o@0.5/repeat", "margin_hit_5pp"), ("D3o@1/repeat", "margin_hit_5pp"), ("D3o@2/repeat", "margin_hit_5pp"))
SECTORS = (("bakery_cafe", r"Bakery/caf\'e"), ("machine_shop", "Machine shop"), ("apparel_ecommerce", "Apparel e-com."),
           ("it_consulting", "IT consulting"), ("construction_sub", "Construction sub."),
           ("specialty_retail", "Specialty retail"))


def ci(est: float, lo_hi) -> str:
    return f"{100 * est:+.1f} [{100 * lo_hi[0]:+.1f}, {100 * lo_hi[1]:+.1f}]"


def gain(entry: dict, metric: str) -> str:
    return ci(entry[f"gain_{metric}_n16_vs_n1"], entry[f"gain_{metric}_n16_vs_n1_ci"])


def original() -> dict:
    b2, d3o = load("R017_R019_B2.json")["result"], load("R033_d3o.json")["result"]
    out = {k: b2[k] for k in ("D1/fresh", "D1p/fresh", "D2/fresh")}
    for eps in ("0.5", "1", "2"):
        out[f"D3o@{eps}/repeat"] = d3o[f"D3o@eps={float(eps)}"]["linked_queries"]
    return out


def row(label: str, cells: dict) -> str:
    return f"{label} & " + " & ".join(gain(cells[k], m) for k, m in COLS) + r" \\"


mismatch, drift = load("R049a_prior_mismatch.json")["result"], load("R049b_drift.json")["result"]
print("% Table A")
print(row("original", original()))
for pm, label in (("PM1", "prior PM1 (tails, spreads)"), ("PM2", "prior PM2 (locations)"), ("PM3", "prior PM3 (both)")):
    print(row(label, mismatch[pm]["linked"]))
for d in ("0.01", "0.02", "0.05"):
    print(row(f"drift $\\delta={100 * float(d):.0f}\\%$", drift[f"delta={d}"]))
print("% fallbacks (sequences reverting to the prior, of 180): "
      + "; ".join(f"{d}: " + ", ".join(f"{k}={v['n_fallback_warnings']}" for k, v in drift[f"delta={d}"].items())
                  for d in ("0.01", "0.02", "0.05")))
print("% single-query margin advantages under prior models: "
      + "; ".join(f"{pm} " + ", ".join(f"{lab} {ci(e['advantage_margin_hit_5pp'], e['advantage_margin_hit_5pp_ci'])}"
                                         for lab, e in mismatch[pm]["single_query_test"].items() if lab != "A0")
                  for pm in ("PM1", "PM2", "PM3")))
print("% Table B")
sector = load("R049c_power_sector.json")["result"]
for key, label in SECTORS:
    print(f"{label} & " + " & ".join(ci(sector[k][key]["gain"], sector[k][key]["gain_ci"]) for k, _ in COLS) + r" \\")
print(f"% firms per sector: {[sector['D2/fresh'][k]['n_firms'] for k, _ in SECTORS]}")
