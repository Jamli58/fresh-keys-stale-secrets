"""B2 (R017-R019): leakage of product 0 as a firm's linked queries accumulate, per key regime.

60 test firms, each asking N_MAX questions that cycle through the task templates (start
task = firm index mod 5); 3 defense seeds; the frozen A1+ (R008_tuning.json) extended to
sequences (attacks/multi_query.py). Curves are prefixes of one disguised sequence, so
n = 1, 2, 4, 8, 16 are paired. A0 (P_pub class median) is the floor.
LLM attackers are not run: M2 test showed A4 - A1+ ~ 0, and an LLM prior shifts every n alike.
    uv run python scripts/run_b2.py
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from numpriv.attacks import PriorOnlyAttack
from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.attacks.multi_query import MultiQueryAttack
from numpriv.config import SECTOR_PARAMETER_PROVENANCE
from numpriv.dataset import EvalConfig, build_eval_set
from numpriv.defenses import MetricDP, MetricDPOfficial, OrderPreserving, PerFieldScaling, SharedScaling
from numpriv.evaluate import COST_REL_TOL, MARGIN_TOL_5PP
from numpriv.metrics import bootstrap_ci
from numpriv.sequences import build_sequence, disguise_sequence

LOG = logging.getLogger("b2")
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FROZEN_CONFIG = RESULTS / "R008_tuning.json"
N_VALUES = (1, 2, 4, 8, 16)
N_FIRMS = 60
SEEDS = (0, 1, 2)
CONFIGS = (
    ("D1", SharedScaling(), ("fresh", "reuse")),
    ("D1p", PerFieldScaling(), ("fresh", "reuse")),
    ("D2", OrderPreserving(), ("fresh", "reuse")),
    ("D3@0.5", MetricDP(0.5), ("repeat",)),
    ("D3@1", MetricDP(1.0), ("repeat",)),
    ("D3@2", MetricDP(2.0), ("repeat",)),
    # appended (R033): earlier configs keep their rng streams, so the default run reproduces R017-R019
    ("D3o@0.5", MetricDPOfficial(0.5), ("repeat",)),
    ("D3o@1", MetricDPOfficial(1.0), ("repeat",)),
    ("D3o@2", MetricDPOfficial(2.0), ("repeat",)),
)


def _rows(ev, firms, label, defense, regime, solver_kw, defense_idx, n_values, solver=None, attack=None):
    """solver: a prebuilt single-query solver (default: the frozen ClassicalBayesAttack from solver_kw);
    attack: a prebuilt multi-query attack (default: MultiQueryAttack over that solver)."""
    solver = solver or ClassicalBayesAttack(ev.prior, defense, use_oracle=True, **solver_kw)
    attack = attack or MultiQueryAttack(solver, regime)
    floor = PriorOnlyAttack(ev.prior)
    LOG.info("%s %s: start", label, regime)  # before the loop, so solver warnings follow their config
    rows = []
    for seed in SEEDS:
        for idx, firm in firms:
            specs = build_sequence(firm, max(n_values), start=idx % 5)
            rng = np.random.default_rng([seed, idx, defense_idx])
            seq = disguise_sequence(specs, defense, regime, rng)
            truth = firm.products[0]
            a0 = floor.attack(seq.views[0])[0]
            for n in n_values:
                est = attack.attack(seq.views[:n])
                key_err = abs(float(np.log(est.key / seq.secrets[0].k))) if est.key is not None else None
                rows.append({
                    "firm_id": firm.firm_id, "seed": seed, "n": n,
                    "margin_hit_5pp": abs(est.margin - truth.margin) <= MARGIN_TOL_5PP,
                    "unit_cost_hit_10pct": abs(est.unit_cost - truth.unit_cost) <= COST_REL_TOL * truth.unit_cost,
                    "key_log_error": key_err,
                    "a0_margin_hit_5pp": abs(a0.margin - truth.margin) <= MARGIN_TOL_5PP,
                    "a0_unit_cost_hit_10pct": abs(a0.unit_cost - truth.unit_cost) <= COST_REL_TOL * truth.unit_cost,
                })
    LOG.info("%s %s: %d rows", label, regime, len(rows))
    return rows


def _summary(rows, n_boot, n_values):
    by_n = {n: [r for r in rows if r["n"] == n] for n in n_values}
    out = {}
    for n, rs in by_n.items():
        groups = np.array([r["firm_id"] for r in rs])
        entry = {"n_rows": len(rs)}
        for m in ("margin_hit_5pp", "unit_cost_hit_10pct"):
            v = np.array([r[m] for r in rs], float)
            entry[m] = float(v.mean())
            entry[m + "_ci"] = bootstrap_ci(v, groups, n_boot=n_boot)
            entry["a0_" + m] = float(np.mean([r["a0_" + m] for r in rs]))
        keys = [r["key_log_error"] for r in rs if r["key_log_error"] is not None]
        if keys:
            entry["key_median_abs_log_error"] = float(np.median(keys))
            entry["key_within_5pct"] = float(np.mean(np.array(keys) <= np.log(1.05)))
        out[str(n)] = entry
    # paired gain from the first to the last query count, same firms/seeds/sequences
    first, last = by_n[n_values[0]], by_n[n_values[-1]]
    groups = np.array([r["firm_id"] for r in first])
    for m in ("margin_hit_5pp", "unit_cost_hit_10pct"):
        diff = np.array([b[m] for b in last], float) - np.array([a[m] for a in first], float)
        out[f"gain_{m}_n{n_values[-1]}_vs_n{n_values[0]}"] = float(diff.mean())
        out[f"gain_{m}_n{n_values[-1]}_vs_n{n_values[0]}_ci"] = bootstrap_ci(diff, groups, n_boot=n_boot)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-firms", type=int, default=N_FIRMS)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--out", default="R017_R019_B2.json")
    parser.add_argument("--eval-seed", type=int, default=EvalConfig().seed,
                        help="firm-generation seed; a new value gives fresh firms (P_pub stays prior_seed's)")
    parser.add_argument("--n-values", type=int, nargs="+", default=list(N_VALUES))
    parser.add_argument("--only", nargs="+", default=None, help="config labels to run (default: all)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = time.time()
    cfg = EvalConfig(seed=args.eval_seed)
    ev = build_eval_set(cfg)
    n_values = tuple(sorted(args.n_values))
    firms = [(i, f) for i, f in enumerate(ev.firms) if i >= cfg.n_dev][: args.n_firms]
    solver_kw = {**json.loads(FROZEN_CONFIG.read_text())["result"]["frozen"], "n_samples": args.n_samples}
    result, raw = {}, {}
    for d_idx, (label, defense, regimes) in enumerate(CONFIGS):
        if args.only and label not in args.only:
            continue
        for regime in regimes:
            rows = _rows(ev, firms, label, defense, regime, solver_kw, d_idx, n_values)
            result[f"{label}/{regime}"] = _summary(rows, args.n_boot, n_values)
            raw[f"{label}/{regime}"] = rows
    config = {"eval_config": cfg.__dict__, "n_firms": len(firms), "seeds": SEEDS, "n_values": n_values,
              "frozen_solver": solver_kw, "target": "product 0 (shown in every query)",
              "sector_parameters": SECTOR_PARAMETER_PROVENANCE, "total_seconds": time.time() - t0}
    out = RESULTS / args.out
    out.write_text(json.dumps({"run": "R017_R019_B2", "config": config, "result": result}, indent=2))
    (RESULTS / args.out.replace(".json", "_rows.jsonl")).write_text(
        "\n".join(json.dumps({"config": k, **r}) for k, rows in raw.items() for r in rows))
    LOG.info("wrote %s (%.0f s)", out, time.time() - t0)


if __name__ == "__main__":
    main()
