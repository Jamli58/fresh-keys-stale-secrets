"""R043: linked-query gains (n = 1 -> 16) by sector, from saved rows (no new attacks).

Original firms (R017_R019_B2_rows.jsonl, eval seed 1001) and the fresh draw (R035 rows, eval seed 3003; the only
row file with D3o). Each sector has 10 of the 60 firms (round-robin sectors), i.e. 30 paired rows per cell;
firm-clustered 95% bootstrap CIs (10,000 resamples).
    uv run python scripts/sector_breakdown.py
"""

import json
from pathlib import Path

import numpy as np

from numpriv.dataset import EvalConfig, build_eval_set
from numpriv.metrics import bootstrap_ci

RESULTS = Path(__file__).resolve().parents[1] / "results"
SOURCES = {"original": ("R017_R019_B2_rows.jsonl", 1001), "fresh_draw": ("R035_B2_fresh_seed3003_rows.jsonl", 3003)}
ENDPOINTS = (("D1/fresh", "unit_cost_hit_10pct"), ("D1p/fresh", "unit_cost_hit_10pct"), ("D2/fresh", "margin_hit_5pp"),
             ("D3o@2/repeat", "margin_hit_5pp"))
N_BOOT = 10_000


def _cell(rows: list[dict], cfg: str, metric: str, firms: set[str]) -> dict | None:
    hit = {(r["firm_id"], r["seed"], r["n"]): float(r[metric]) for r in rows if r["config"] == cfg and r["firm_id"] in firms}
    keys = sorted({(f, s) for f, s, _ in hit})
    if not keys:
        return None
    diff = np.array([hit[(*k, 16)] - hit[(*k, 1)] for k in keys])
    return {"n_rows": len(keys), "n1": float(np.mean([hit[(*k, 1)] for k in keys])),
            "n16": float(np.mean([hit[(*k, 16)] for k in keys])), "gain": float(diff.mean()),
            "gain_ci": bootstrap_ci(diff, np.array([k[0] for k in keys]), n_boot=N_BOOT)}


def main() -> None:
    result = {}
    for name, (fname, seed) in SOURCES.items():
        sector_of = {f.firm_id: f.sector for f in build_eval_set(EvalConfig(seed=seed)).firms}
        rows = [json.loads(line) for line in (RESULTS / fname).read_text().splitlines() if line.strip()]
        result[name] = {}
        for sector in sorted(set(sector_of.values())):
            firms = {f for f, s in sector_of.items() if s == sector}
            cells = {f"{cfg}/{metric}": _cell(rows, cfg, metric, firms) for cfg, metric in ENDPOINTS}
            result[name][sector] = {k: v for k, v in cells.items() if v is not None}
    out = RESULTS / "R043_sector_breakdown.json"
    out.write_text(json.dumps({"run": "R043_sector_breakdown", "config": {"sources": SOURCES, "n_boot": N_BOOT},
                               "result": result}, indent=2))
    for name, sectors in result.items():
        for sector, cells in sectors.items():
            print(name, sector, {k.split("/")[0]: (round(100 * v["gain"], 1), [round(100 * c, 1) for c in v["gain_ci"]], v["n_rows"])
                                 for k, v in cells.items()})


if __name__ == "__main__":
    main()
