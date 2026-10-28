"""Fig. 2: recovery of product 0 vs number of linked queries (60 test firms x 3 seeds = 180 rows per n).

Sources: R017_R019_B2.json (D1, D1', D2; fresh and reuse) and R033_d3o.json (D3o eps=1, fresh noise per
release). Same firms in both files, so one prior-only floor (A0 ignores the shown numbers).
"""

from matplotlib.lines import Line2D

from paper_plot_style import COLOR, COLUMN_IN, FLOOR_COLOR, LABEL, MARKER, N_VALUES, TEXT_IN, load, plt, save

B2 = load("R017_R019_B2.json")["result"]
D3O = load("R033_d3o.json")["result"]["D3o@eps=1.0"]["linked_queries"]
SERIES = [  # (entity, regime, curve dict)
    ("D1", "fresh", B2["D1/fresh"]), ("D1", "reuse", B2["D1/reuse"]),
    ("D1p", "fresh", B2["D1p/fresh"]), ("D1p", "reuse", B2["D1p/reuse"]),
    ("D2", "fresh", B2["D2/fresh"]), ("D2", "reuse", B2["D2/reuse"]),
    ("D3o", "fresh", D3O),
]
PANELS = [("unit_cost_hit_10pct", "Unit-cost hit rate (±10%), %"), ("margin_hit_5pp", "Margin hit rate (±5 pp), %")]
ns = [int(n) for n in N_VALUES]

fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.35), sharex=True)
for ax, (metric, ylabel) in zip(axes, PANELS):
    floor = 100 * B2["D1/fresh"]["1"]["a0_" + metric]
    ax.axhline(floor, color=FLOOR_COLOR, ls=":", lw=1.0, zorder=1)
    for ent, regime, cur in SERIES:
        y = [100 * cur[n][metric] for n in N_VALUES]
        lo = [100 * cur[n][metric + "_ci"][0] for n in N_VALUES]
        hi = [100 * cur[n][metric + "_ci"][1] for n in N_VALUES]
        solid = regime == "fresh"
        ax.fill_between(ns, lo, hi, color=COLOR[ent], alpha=0.12 if solid else 0.06, lw=0, zorder=2)
        ax.plot(ns, y, color=COLOR[ent], ls="-" if solid else (0, (4, 2)), marker=MARKER[ent], ms=4,
                mfc=COLOR[ent] if solid else "white", mec=COLOR[ent], mew=1.0, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("Linked queries about the same product, n")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 105 if metric == "margin_hit_5pp" else 80)

handles = [Line2D([], [], color=COLOR[e], marker=MARKER[e], ms=4, lw=1.4, label=LABEL[e] + (", ε=1" if e == "D3o" else "")) for e in ("D1", "D1p", "D2", "D3o")]
handles += [Line2D([], [], color="#4a4944", ls="-", lw=1.4, label="fresh secret per query (D3o: ε=1, fresh noise)"),
            Line2D([], [], color="#4a4944", ls=(0, (4, 2)), marker="o", mfc="white", ms=4, lw=1.4,
                   label="one reused secret (D2: pair-only attacker)"),
            Line2D([], [], color=FLOOR_COLOR, ls=":", lw=1.0, label="prior-only baseline (A0)")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.17))
fig.tight_layout(w_pad=2.0)
save(fig, "fig4_linked_queries_recovery")
