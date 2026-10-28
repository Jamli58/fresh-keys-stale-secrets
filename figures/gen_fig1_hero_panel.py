"""Headline figure: margin recovery vs linked queries, D2 under the joint shared-map attacker (fresh vs one reused
map, the matched comparison) and D3o eps=2.

Sources: R045_d2_joint_linked.json (D2, joint attacker), R033_d3o.json (D3o eps=2); 60 test firms x 3 seeds per n;
prior-only baseline from R017_R019_B2.json (same firms).
"""

from paper_plot_style import COLOR, COLUMN_IN, FLOOR_COLOR, MARKER, N_VALUES, load, plt, save

B2 = load("R017_R019_B2.json")["result"]
JOINT = load("R045_d2_joint_linked.json")["result"]
D3O = load("R033_d3o.json")["result"]["D3o@eps=2.0"]["linked_queries"]
M = "margin_hit_5pp"
ns = [int(n) for n in N_VALUES]
series = [
    ("D2", JOINT["D2/fresh"], "-", "D2 fixed-axis, joint attacker, fresh maps"),
    ("D2", JOINT["D2/reuse"], (0, (4, 2)), "D2 fixed-axis, joint attacker, one reused map"),
    ("D3o", D3O, "-", "Prεεmpt released, ε=2, fresh noise (default attacker)"),
]
fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.15))
floor = 100 * B2["D2/fresh"]["1"]["a0_" + M]
ax.axhline(floor, color=FLOOR_COLOR, ls=":", lw=1.0)
ax.text(16, floor - 1.5, f"prior-only baseline {floor:.1f}%", ha="right", va="top", fontsize=6.5, color=FLOOR_COLOR)
for ent, cur, ls, label in series:
    y = [100 * cur[n][M] for n in N_VALUES]
    lo = [100 * cur[n][M + "_ci"][0] for n in N_VALUES]
    hi = [100 * cur[n][M + "_ci"][1] for n in N_VALUES]
    filled = ls == "-"
    ax.fill_between(ns, lo, hi, color=COLOR[ent], alpha=0.12 if filled else 0.06, lw=0)
    ax.plot(ns, y, color=COLOR[ent], ls=ls, marker=MARKER[ent] if ent == "D2" else "D", ms=4,
            mfc=COLOR[ent] if filled else "white", mec=COLOR[ent], label=label)
    dy = {"D3o": -4}.get(ent, 3 if ent == "D2" and filled else 0)  # separate the two close endpoints
    ax.annotate(f"{y[-1]:.1f}", (16, y[-1]), xytext=(4, dy), textcoords="offset points", va="center",
                fontsize=8, color=COLOR[ent])
ax.annotate("", xy=(24.4, 100 * JOINT["D2/fresh"]["16"][M]), xytext=(24.4, 100 * JOINT["D2/reuse"]["16"][M]),
            arrowprops=dict(arrowstyle="<->", color="#2b2a27", lw=0.8))
ax.text(13.5, 38, "fresh − reuse\ngain difference\n+29.4 pp", ha="right", va="center", fontsize=6.5, color="#2b2a27")
ax.set_xscale("log", base=2)
ax.set_xticks(ns, [str(n) for n in ns])
ax.set_xlim(0.8, 26.5)
ax.set_ylim(0, 75)
ax.set_xlabel("Linked queries about the same product, n")
ax.set_ylabel("Margin hit rate (±5 pp), %")
ax.legend(frameon=False, loc="upper left", fontsize=6.5)
ax.text(0.99, 0.02, "60 firms × 3 seeds", transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=FLOOR_COLOR)
save(fig, "fig2_intro_fresh_vs_reused")
