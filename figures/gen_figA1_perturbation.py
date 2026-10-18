"""Fig. A1: median relative price perturbation by sector at eps=1, released Prεεmpt (D3o) vs paper reading (D3).

Sources: R033_d3o.json (relative_price_noise_median_by_sector, D3o eps=1) and R004a.json (D3 eps=1).
Perturbation of the shown price, not recovery. The two summaries use different populations, so the markers are
not connected.
"""

import numpy as np

from paper_plot_style import COLOR, COLUMN_IN, load, plt, save

d3o = load("R033_d3o.json")["result"]["D3o@eps=1.0"]["relative_price_noise_median_by_sector"]
d3 = load("R004a.json")["result"]["median_relative_price_error_by_sector"]["eps=1.0"]
names = {"bakery_cafe": "Bakery/café", "specialty_retail": "Specialty retail", "apparel_ecommerce": "Apparel e-com.",
         "machine_shop": "Machine shop", "it_consulting": "IT consulting", "construction_sub": "Construction sub."}
order = sorted(names, key=lambda s: -d3o[s])
y = np.arange(len(order))[::-1]


def fmt(v: float) -> str:
    return f"{v:.1f}" if v >= 1 else f"{v:.2f}"


fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.2))
for yi, s in zip(y, order):
    a, b = 100 * d3o[s], 100 * d3[s]
    ax.plot(a, yi, "D", ms=5, color=COLOR["D3o"], zorder=3)
    ax.plot(b, yi, "o", ms=5, mfc="white", mec=COLOR["D3o"], mew=1.2, zorder=3)
    ax.annotate(fmt(a), (a, yi), xytext=(5, 0), textcoords="offset points", va="center", fontsize=6)
    ax.annotate(fmt(b), (b, yi), xytext=(-5, 0), textcoords="offset points", va="center", ha="right", fontsize=6)
ax.set_xscale("log")
ax.set_xlim(0.2, 300)
ax.set_yticks(y, [names[s] for s in order])
ax.set_xlabel("Median relative price perturbation at \u03b5=1, %")
ax.grid(axis="y", visible=False)
ax.plot([], [], "D", color=COLOR["D3o"], label="D3o (released code)")
ax.plot([], [], "o", mfc="white", mec=COLOR["D3o"], label="D3 (paper reading)")
ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "figA1_perturbation")
