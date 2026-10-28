"""Multiplicity check for the headline linked-query claim: with fresh secrets, every mechanism (and D3o / D3
setting) has a target whose n=1->16 gain is significant. Family: 9 mechanism settings x 2 targets = 18 tests.

Holm correction on p-values recovered from the reported firm-clustered bootstrap CIs (normal approximation), plus
an exact firm-clustered bootstrap check at the Bonferroni level for the settings with row-level outputs.
    uv run python scripts/holm_check.py   # writes results/R048_holm_check.json
"""

import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
METRICS = ("margin_hit_5pp", "unit_cost_hit_10pct")
ALPHA = 0.05


def _sources() -> dict:
    r = json.loads((RESULTS / "R017_R019_B2.json").read_text())["result"]
    d = json.loads((RESULTS / "R033_d3o.json").read_text())["result"]
    return {"D1": r["D1/fresh"], "D1p": r["D1p/fresh"], "D2": r["D2/fresh"],
            "D3o@0.5": d["D3o@eps=0.5"]["linked_queries"], "D3o@1": d["D3o@eps=1.0"]["linked_queries"],
            "D3o@2": d["D3o@eps=2.0"]["linked_queries"],
            "D3@0.5": r["D3@0.5/repeat"], "D3@1": r["D3@1/repeat"], "D3@2": r["D3@2/repeat"]}


def holm() -> tuple[list[dict], dict]:
    z = NormalDist()
    tests = []
    for name, s in _sources().items():
        for m in METRICS:
            g = s[f"gain_{m}_n16_vs_n1"]
            lo, hi = s[f"gain_{m}_n16_vs_n1_ci"]
            se = (hi - lo) / (2 * z.inv_cdf(0.975))
            p = 1.0 if se == 0 else 2 * (1 - z.cdf(abs(g) / se))
            tests.append({"setting": name, "metric": m, "gain": g, "ci": [lo, hi], "p": p})
    tests.sort(key=lambda t: t["p"])
    running = 0.0
    for i, t in enumerate(tests):
        running = max(running, min(1.0, (len(tests) - i) * t["p"]))
        t["p_holm"] = running
    best = {}
    for t in tests:
        best[t["setting"]] = min(best.get(t["setting"], 1.0), t["p_holm"])
    return tests, best


def exact_bonferroni(n_boot: int = 10_000) -> dict:
    """Firm-clustered bootstrap CI at level 1 - ALPHA/18 for the row-level settings (fresh / repeat regimes)."""
    rows = [json.loads(line) for line in (RESULTS / "R017_R019_B2_rows.jsonl").read_text().splitlines()]
    configs = {"D1": "D1/fresh", "D1p": "D1p/fresh", "D2": "D2/fresh",
               "D3@0.5": "D3@0.5/repeat", "D3@1": "D3@1/repeat", "D3@2": "D3@2/repeat"}
    rng = np.random.default_rng(0)
    q = ALPHA / 18 / 2
    out = {}
    for name, cfg in configs.items():
        by = {(r["firm_id"], r["seed"], r["n"]): r for r in rows if r["config"] == cfg}
        pairs = sorted({(f, s) for f, s, _ in by})
        firms = sorted({f for f, _ in pairs})
        idx = {f: [i for i, (g, _) in enumerate(pairs) if g == f] for f in firms}
        res = {}
        for m in METRICS:
            diff = np.array([float(by[(f, s, 16)][m]) - float(by[(f, s, 1)][m]) for f, s in pairs])
            boot = [diff[np.concatenate([idx[f] for f in rng.choice(firms, len(firms))])].mean()
                    for _ in range(n_boot)]
            res[m] = {"gain": float(diff.mean()), "ci_bonferroni": [float(np.quantile(boot, q)),
                                                                   float(np.quantile(boot, 1 - q))]}
        out[name] = res
    return out


def main() -> None:
    tests, best = holm()
    exact = exact_bonferroni()
    result = {"family_size": len(tests), "alpha": ALPHA, "tests": tests,
              "best_holm_p_per_setting": best,
              "every_setting_significant_after_holm": all(v < ALPHA for v in best.values()),
              "exact_bonferroni_rowlevel": exact}
    (RESULTS / "R048_holm_check.json").write_text(json.dumps({"result": result}, indent=1))
    print("max over settings of best Holm p:", max(best.values()))
    for name, res in exact.items():
        print(name, {m[:6]: [round(100 * x, 1) for x in v["ci_bonferroni"]] for m, v in res.items()})


if __name__ == "__main__":
    main()
