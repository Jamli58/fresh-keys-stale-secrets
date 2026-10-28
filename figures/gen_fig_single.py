"""Single-query advantage of A1+ over the prior-only attacker A0 (dot plot), test split, 2,100 rows per mechanism.

Sources: R009.json (D1, D1', D2, D3 paper-reading variant), R033_d3o.json (D3o, released Prεεmpt). Same fields as
Table III: advantage_{margin_hit_5pp, unit_cost_hit_10pct} and their firm-clustered 95% bootstrap CIs.
"""

from paper_plot_style import COLOR, COLUMN_IN, MARKER, load, plt, save

R9 = load("R009.json")["result"]
D3O = load("R033_d3o.json")["result"]
D3_COLOR, D3_MARKER = "#8a7fb8", "v"
ROWS = [("D1", "D1", R9["D1_shared_scaling"]["all"]),
        ("D1′", "D1p", R9["D1p_per_field_scaling"]["all"]),
        ("D2", "D2", R9["D2_zhang_order_preserving"]["all"]),
        *[(f"D3o, ε={e}", "D3o", D3O[f"D3o@eps={f}"]["single_query_test"]["A1plus_bayes_oracle"]["all"])
          for e, f in (("0.5", "0.5"), ("1", "1.0"), ("2", "2.0"))],
        *[(f"D3, ε={e}", "D3", R9[f"D3_prempt_metric_dp@eps={f}"]["all"]) for e, f in (("0.5", "0.5"), ("1", "1.0"), ("2", "2.0"))]]
PANELS = (("margin_hit_5pp", "Margin advantage (pp)"), ("unit_cost_hit_10pct", "Unit-cost advantage (pp)"))
XMAX = 56

ys = [len(ROWS) - 1 - i + (0.5 if i < 3 else 0.25 if i < 6 else 0.0) for i in range(len(ROWS))]
fig, axes = plt.subplots(1, 2, figsize=(COLUMN_IN, 2.35), sharey=True)
for ax, (metric, title) in zip(axes, PANELS):
    ax.axvline(0, color="#6b6a64", lw=0.7, ls=":")
    for (label, ent, s), y in zip(ROWS, ys):
        g, (lo, hi) = 100 * s[f"advantage_{metric}"], (100 * c for c in s[f"advantage_{metric}_ci"])
        color, marker = (D3_COLOR, D3_MARKER) if ent == "D3" else (COLOR[ent], MARKER[ent])
        if g > XMAX:  # D1 margins: off scale, marked at the edge
            ax.annotate(f"+{g:.1f} →", (XMAX, y), ha="right", va="center", fontsize=6.5, color=color)
            continue
        ax.plot([lo, hi], [y, y], color=color, lw=1.1, solid_capstyle="butt")
        ax.plot(g, y, marker, ms=4, color=color)
    ax.set_xlim(-8, XMAX)
    ax.set_xticks([0, 20, 40])
    ax.set_title(title, fontsize=7.5, pad=3)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(ys, [label for label, _, _ in ROWS])
axes[0].tick_params(axis="y", length=0)
fig.subplots_adjust(wspace=0.1)
save(fig, "fig3_single_prompt_advantage")
