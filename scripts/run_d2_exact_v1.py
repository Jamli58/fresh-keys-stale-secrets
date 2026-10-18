"""R046 V1 record: A-exact against exact rejection sampling on the small instances of tests/test_exact_joint.py.

Writes, per case, the reference size, the posterior quantiles (0.1, 0.5, 0.9) of product 0's margin from both
methods, the Kolmogorov-Smirnov distance between the weighted posterior CDF and the rejection draws, and the 1%
critical value 1.63 * sqrt(1/n_ref + 1/ESS) with ESS the attacker's smallest per-update effective sample size.
    uv run python scripts/run_d2_exact_v1.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_exact_joint as t  # noqa: E402  (the same instances, seeds and helpers as the slow tests)


def main() -> None:
    rows = []
    for law_name, regime, layout, priors, steps in t.CASES:
        law = t.LAWS[law_name]
        queries = t._history(law, priors, 1, regime, layout, np.random.default_rng(17))
        ref = np.sort(t._rejection(law, queries, priors, regime, np.random.default_rng(18)))
        margin, cdf, ess = t._attacker_posterior(law, queries, priors, regime, mcmc_steps=steps)
        qs = (0.1, 0.5, 0.9)
        ours = [float(margin[np.searchsorted(cdf, q)]) for q in qs]
        theirs = [float(np.quantile(ref, q)) for q in qs]
        dist = float(np.abs(np.interp(ref, margin, cdf) - np.arange(1, ref.size + 1) / ref.size).max())
        crit = float(1.63 * np.sqrt(1 / ref.size + 1 / ess))
        rows.append({"law": law_name, "n_intervals": law.n_intervals, "regime": regime, "layout": layout,
                     "mcmc_steps": steps, "display_digits": [d for _, d in queries], "n_reference": int(ref.size),
                     "min_ess": float(ess), "quantiles": list(qs), "attacker": ours, "rejection": theirs,
                     "max_abs_quantile_diff": max(abs(a - b) for a, b in zip(ours, theirs)),
                     "ks_distance": dist, "ks_critical_1pct": crit, "pass": max(abs(a - b) for a, b in zip(ours, theirs)) < 0.03 and dist < crit})
    out = ROOT / "results" / "R046_d2_exact_v1_reference.json"
    out.write_text(json.dumps({"run": "R046_d2_exact_v1_reference", "n_cases": len(rows),
                               "all_pass": all(r["pass"] for r in rows), "cases": rows}, indent=2))
    print(out, "all_pass", all(r["pass"] for r in rows), "max quantile diff",
          round(max(r["max_abs_quantile_diff"] for r in rows), 4))


if __name__ == "__main__":
    main()
