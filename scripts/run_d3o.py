"""R033: Prεεmpt's money sanitizer as released (D3o) — the cells the paper-based D3 stood in for.

Same eval set, defense seeds and frozen A1+ (R008_tuning.json) as M1, and the same 60-firm
linked-query protocol as B2 (n in 1, 2, 4, 8, 16). Also the per-sector median relative price
perturbation (as R004a). No retuning: the attacker config is frozen across defenses.
    uv run python scripts/run_d3o.py
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
from numpriv.config import SECTOR_PARAMETER_PROVENANCE, SECTORS
from numpriv.dataset import EvalConfig, build_eval_set
from numpriv.defenses import MetricDPOfficial
from numpriv.evaluate import COST_REL_TOL, MARGIN_TOL_5PP, evaluate, summarize
from numpriv.metrics import bootstrap_ci
from numpriv.sequences import build_sequence, disguise_sequence

LOG = logging.getLogger("d3o")
RESULTS = Path(__file__).resolve().parents[1] / "results"
EPSILONS = (0.5, 1.0, 2.0)
SEEDS = (0, 1, 2)
N_VALUES = (1, 2, 4, 8, 16)
N_SEQ_FIRMS = 60
DEFENSE_IDX_OFFSET = 100  # rng stream distinct from run_b2.py's defense indices


def _relative_noise(ev, d, rng) -> dict:
    by_sector = {s: [] for s in SECTORS}
    for f in ev.firms:
        for p in f.products:
            for _ in range(5):
                y = d.apply((type(ev.prompts[0].spec.fields[0])("price:0", "money", p.price),), None, rng).fields[0].value
                by_sector[f.sector].append(abs(y - p.price) / p.price)
    return {s: float(np.median(v)) for s, v in by_sector.items()}


def _single_query(ev, d, solver_kw, n_boot) -> dict:
    floor = evaluate(ev, "test", d, PriorOnlyAttack(ev.prior), SEEDS, with_oracle=True)
    a1p = evaluate(ev, "test", d, ClassicalBayesAttack(ev.prior, d, use_oracle=True, **solver_kw), SEEDS, True)
    return {"A0_prior_only": summarize(floor, floor, n_boot), "A1plus_bayes_oracle": summarize(a1p, floor, n_boot)}


def _linked(ev, d, solver_kw, n_boot, d_idx) -> dict:
    cfg = ev.config
    firms = [(i, f) for i, f in enumerate(ev.firms) if i >= cfg.n_dev][:N_SEQ_FIRMS]
    attack = MultiQueryAttack(ClassicalBayesAttack(ev.prior, d, use_oracle=True, **solver_kw), "repeat")
    rows = []
    for seed in SEEDS:
        for idx, firm in firms:
            seq = disguise_sequence(build_sequence(firm, max(N_VALUES), idx % 5), d, "repeat",
                                    np.random.default_rng([seed, idx, DEFENSE_IDX_OFFSET + d_idx]))
            t = firm.products[0]
            for n in N_VALUES:
                e = attack.attack(seq.views[:n])
                rows.append({"firm_id": firm.firm_id, "n": n,
                             "margin_hit_5pp": abs(e.margin - t.margin) <= MARGIN_TOL_5PP,
                             "unit_cost_hit_10pct": abs(e.unit_cost - t.unit_cost) <= COST_REL_TOL * t.unit_cost})
    out = {}
    for n in N_VALUES:
        rs = [r for r in rows if r["n"] == n]
        g = np.array([r["firm_id"] for r in rs])
        out[str(n)] = {m: float(np.mean([r[m] for r in rs])) for m in ("margin_hit_5pp", "unit_cost_hit_10pct")}
        out[str(n)].update({m + "_ci": bootstrap_ci(np.array([r[m] for r in rs], float), g, n_boot=n_boot)
                            for m in ("margin_hit_5pp", "unit_cost_hit_10pct")})
    first = [r for r in rows if r["n"] == 1]
    last = [r for r in rows if r["n"] == N_VALUES[-1]]
    g = np.array([r["firm_id"] for r in first])
    for m in ("margin_hit_5pp", "unit_cost_hit_10pct"):
        diff = np.array([b[m] for b in last], float) - np.array([a[m] for a in first], float)
        out[f"gain_{m}_n16_vs_n1"] = float(diff.mean())
        out[f"gain_{m}_n16_vs_n1_ci"] = bootstrap_ci(diff, g, n_boot=n_boot)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(EPSILONS),
                        help="e.g. 1e-6: the eps -> 0 limit, where a release reveals only its value's domain")
    parser.add_argument("--out", default="R033_d3o.json")
    parser.add_argument("--skip-single", action="store_true", help="linked queries only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = time.time()
    ev = build_eval_set(EvalConfig())
    solver_kw = {**json.loads((RESULTS / "R008_tuning.json").read_text())["result"]["frozen"],
                 "n_samples": args.n_samples}
    result = {}
    for d_idx, eps in enumerate(args.epsilons):
        d = MetricDPOfficial(eps)
        key = f"D3o@eps={eps}"
        LOG.info("%s: start", key)
        result[key] = {"linked_queries": _linked(ev, d, solver_kw, args.n_boot, d_idx)}
        if not args.skip_single:
            result[key]["relative_price_noise_median_by_sector"] = _relative_noise(ev, d, np.random.default_rng([7, d_idx]))
            result[key]["single_query_test"] = _single_query(ev, d, solver_kw, args.n_boot)
    config = {"eval_config": ev.config.__dict__, "seeds": SEEDS, "frozen_solver": solver_kw,
              "mechanism": "preempt/sanitizer.py encypt_money + M_epsilon, commit 1032978",
              "sector_parameters": SECTOR_PARAMETER_PROVENANCE, "total_seconds": time.time() - t0}
    out = RESULTS / args.out
    out.write_text(json.dumps({"run": out.stem, "config": config, "result": result}, indent=2))
    LOG.info("wrote %s (%.0f s)", out, time.time() - t0)


if __name__ == "__main__":
    main()
