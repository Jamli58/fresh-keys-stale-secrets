"""R040: direct test of the fresh-vs-reuse difference in linked-query gains (no new attacks).

From the saved B2 rows (original R017_R019_B2 and fresh-draw R035), for D1, D1' and D2:
contrast = (fresh hit at n=16 - fresh hit at n=1) - (reuse hit at n=16 - reuse hit at n=1),
paired by (firm, defense seed): both regimes disguise the same firm's sequence from the same
rng stream. Firm-clustered paired bootstrap 95% CIs (10,000 resamples). For D2 the reuse attacker
is pair-only, so the D2 contrast compares two attackers' gains, not total leakage.
    uv run python scripts/regime_contrast.py
"""

import json
from pathlib import Path

import numpy as np

from numpriv.metrics import bootstrap_ci

RESULTS = Path(__file__).resolve().parents[1] / "results"
SOURCES = {"original": "R017_R019_B2_rows.jsonl", "fresh_draw": "R035_B2_fresh_seed3003_rows.jsonl"}
MECHANISMS = ("D1", "D1p", "D2")
METRICS = ("margin_hit_5pp", "unit_cost_hit_10pct")
N_FIRST, N_LAST = 1, 16
N_BOOT = 10_000


def _gains(rows: list[dict], config: str, metric: str) -> dict[tuple[str, int], float]:
    by_key = {(r["firm_id"], r["seed"], r["n"]): float(r[metric]) for r in rows if r["config"] == config}
    keys = sorted({(f, s) for f, s, _ in by_key})
    return {k: by_key[(*k, N_LAST)] - by_key[(*k, N_FIRST)] for k in keys}


def contrast(rows: list[dict], mechanism: str, metric: str) -> dict:
    fresh, reuse = _gains(rows, f"{mechanism}/fresh", metric), _gains(rows, f"{mechanism}/reuse", metric)
    if fresh.keys() != reuse.keys():
        raise ValueError(f"{mechanism}: fresh and reuse rows are not paired")
    keys = sorted(fresh)
    diff = np.array([fresh[k] - reuse[k] for k in keys])
    groups = np.array([k[0] for k in keys])
    return {"n_pairs": len(keys), "fresh_gain": float(np.mean([fresh[k] for k in keys])),
            "reuse_gain": float(np.mean([reuse[k] for k in keys])), "contrast": float(diff.mean()),
            "contrast_ci": bootstrap_ci(diff, groups, n_boot=N_BOOT)}


def main() -> None:
    result = {}
    for name, fname in SOURCES.items():
        rows = [json.loads(line) for line in (RESULTS / fname).read_text().splitlines() if line.strip()]
        result[name] = {f"{m}/{metric}": contrast(rows, m, metric) for m in MECHANISMS for metric in METRICS}
    config = {"sources": SOURCES, "n_first": N_FIRST, "n_last": N_LAST, "n_boot": N_BOOT,
              "pairing": "(firm_id, defense seed); same rng stream for both regimes (run_b2.py)",
              "note": "D2 reuse attacker is pair-only"}
    out = RESULTS / "R040_regime_contrast.json"
    out.write_text(json.dumps({"run": "R040_regime_contrast", "config": config, "result": result}, indent=2))
    for name, cells in result.items():
        for k, v in cells.items():
            print(name, k, round(100 * v["fresh_gain"], 1), round(100 * v["reuse_gain"], 1),
                  round(100 * v["contrast"], 1), [round(100 * c, 1) for c in v["contrast_ci"]])


if __name__ == "__main__":
    main()
