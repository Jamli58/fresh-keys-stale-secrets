"""Fresh-secret gains (n = 1 -> 16) on the original firms, a fresh draw and five population shifts (forest plot).

Sources: R017_R019_B2.json and R033_d3o.json (original firms), R042_population_shift.json (fresh draw, eval seed
3003, scenario "none", and the five shifted scenarios; same prior table and frozen solver). 60 test firms x 3 seeds
per estimate; firm-clustered 95% bootstrap CIs.
"""

from paper_plot_style import COLOR, MARKER, TEXT_IN, load, plt, save

ORIG = load("R017_R019_B2.json")["result"]
D3O = load("R033_d3o.json")["result"]["D3o@eps=2.0"]["linked_queries"]
SHIFT = load("R042_population_shift.json")["result"]

PANELS = (("D1", "D1/fresh", "unit_cost_hit_10pct", "D1 unit cost"),
          ("D1p", "D1p/fresh", "unit_cost_hit_10pct", "D1′ unit cost"),
          ("D2", "D2/fresh", "margin_hit_5pp", "D2 margin"),
          ("D3o", "D3o@2/repeat", "margin_hit_5pp", "D3o margin, ε=2"))
ROWS = (("original", None), ("fresh draw", "none"), ("margins +0.5 logit", "margins +0.5 logit"),
        ("margins −0.5 logit", "margins -0.5 logit"), ("price level ×1.5", "price level x1.5"),
        ("price spread ×1.5", "price spread x1.5"), ("efficiency sd ×2", "firm efficiency sd x2"))


def gain(ent: str, key: str, metric: str, scenario: str | None) -> tuple[float, float, float]:
    if scenario is None:
        cur = D3O if ent == "D3o" else ORIG[key]
    else:
        cur = SHIFT[scenario]["linked"][key]
    lo, hi = cur[f"gain_{metric}_n16_vs_n1_ci"]
    return 100 * cur[f"gain_{metric}_n16_vs_n1"], 100 * lo, 100 * hi


ys = [len(ROWS) - 1 - i + (0.4 if i < 2 else 0.0) for i in range(len(ROWS))]  # gap below the two baseline rows
fig, axes = plt.subplots(1, len(PANELS), figsize=(TEXT_IN, 1.95), sharey=True)
lows = []
for ax, (ent, key, metric, title) in zip(axes, PANELS):
    for (label, scenario), y in zip(ROWS, ys):
        g, lo, hi = gain(ent, key, metric, scenario)
        lows.append(lo)
        filled = scenario is None
        ax.plot([lo, hi], [y, y], color=COLOR[ent], lw=1.2, solid_capstyle="butt")
        ax.plot(g, y, MARKER[ent], ms=4.5, color=COLOR[ent], mfc=COLOR[ent] if filled else "white", mew=1.1)
    ax.axvline(0, color="#6b6a64", lw=0.7, ls=":")
    ax.axhline((ys[1] + ys[2]) / 2, color="#d6d5cc", lw=0.6)
    ax.set_xlim(-4, 58)
    ax.set_xticks([0, 20, 40])
    ax.set_title(title, fontsize=7.5, pad=3)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(ys, [label for label, _ in ROWS])
axes[0].tick_params(axis="y", length=0)
fig.supxlabel("Gain in hit rate from n=1 to n=16 (pp), 95% CI", fontsize=7.5, y=-0.04)
fig.subplots_adjust(wspace=0.08)
assert min(lows) > 0, min(lows)
save(fig, "fig_robust")
print(f"smallest lower endpoint: {min(lows):.1f} pp")
