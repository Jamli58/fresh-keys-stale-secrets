"""Fig. A2: D1 key localization. Left: expected feasible log-interval width (Theorem 3, exact observations)
vs Monte Carlo. Right: observed median |log k_hat - log k| of the multi-query attacker, fresh vs reuse.
Different quantities, shown in separate panels.

Writes results/R036_theory_width_montecarlo.json. Sources: formula 2 log(b/a)/(n+1) with a=0.1, b=10 (Theorem 3 of the paper); Monte Carlo below
(seeded); R017_R019_B2.json key_median_abs_log_error (60 test firms x 3 seeds per n).
"""

import numpy as np

import json

from paper_plot_style import COLOR, COLUMN_IN, MARKER, N_VALUES, RESULTS, load, plt, save

A, B, DRAWS = 0.1, 10.0, 20000
L = np.log(B / A)
ns = np.array([int(n) for n in N_VALUES])
rng = np.random.default_rng(2026)
mc = [float(np.mean([L - np.ptp(rng.uniform(np.log(A), np.log(B), n)) for _ in range(DRAWS)])) for n in ns]
b2 = load("R017_R019_B2.json")["result"]
(RESULTS / "R036_theory_width_montecarlo.json").write_text(json.dumps({
    "run": "R036_theory_width_montecarlo",
    "config": {"a": A, "b": B, "draws_per_n": DRAWS, "seed": 2026, "generator": "figures/gen_figA2_key_localization.py"},
    "result": {str(n): {"theory": float(2 * L / (n + 1)), "monte_carlo": m} for n, m in zip(ns, mc)}}, indent=2))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COLUMN_IN, 3.6))
ax1.plot(ns, 2 * L / (ns + 1), color="#4a4944", lw=1.4, label="theory 2 log(b/a)/(n+1)")
ax1.plot(ns, mc, ls="none", marker="o", ms=4, mfc="white", mec=COLOR["D1"], label=f"Monte Carlo ({DRAWS:,} draws)")
ax1.set_ylabel("Expected feasible width")
ax2.plot(ns, [b2["D1/fresh"][n]["key_median_abs_log_error"] for n in N_VALUES], color=COLOR["D1"],
         marker=MARKER["D1"], ms=4, label="fresh key per query")
ax2.plot(ns, [b2["D1/reuse"][n]["key_median_abs_log_error"] for n in N_VALUES], color=COLOR["D1"], ls=(0, (4, 2)),
         marker=MARKER["D1"], mfc="white", ms=4, label="one reused key")
ax2.set_ylabel(r"Median $|\log\hat{k}_1-\log k_1|$")
for ax in (ax1, ax2):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("Linked queries, n")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=6.5)
fig.tight_layout(h_pad=1.2)
save(fig, "figA2_key_localization")
