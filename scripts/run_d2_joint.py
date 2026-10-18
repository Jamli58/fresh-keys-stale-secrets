"""R045: D2 linked queries with the joint shared-map attacker (attacks/joint_map.py), fresh vs reused map.

The pair-only attacker of R017-R019 scores product 0's (price, cost) pair alone; this attacker also uses the
other products' prices and costs shown under the same map, integrated over their class priors. Same firms,
defense seeds, random streams and frozen prior as R017 (original firms) / R035 (fresh draw), so rows are
paired with the pair-only results. Parts:
  linked       original firms (eval seed 1001), fresh and reuse, n = 1..16, plus the fresh-minus-reuse contrast
  fresh_draw   the same on the fresh draw (eval seed 3003)
  single       product 0 of every test prompt (single query): joint vs pair-only vs A0, paired
  sensitivity  original firms, n = 1 and 16, fresh and reuse, other smoothing / Monte Carlo settings
  smoothing    R047a: one smaller-smoothing configuration (--obs-sd, --n-nuisance), as sensitivity
    uv run python scripts/run_d2_joint.py --part linked
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from run_b2 import FROZEN_CONFIG, N_FIRMS, N_VALUES, SEEDS, _rows, _summary

from numpriv.attacks import PriorOnlyAttack, make_view
from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.attacks.joint_map import JointMapAttack
from numpriv.attacks.multi_query import MultiQueryAttack
from numpriv.dataset import EvalConfig, build_eval_set
from numpriv.defenses import OrderPreserving
from numpriv.evaluate import COST_REL_TOL, MARGIN_TOL_5PP
from numpriv.metrics import bootstrap_ci

LOG = logging.getLogger("d2_joint")
RESULTS = Path(__file__).resolve().parents[1] / "results"
B2_D2_DEFENSE_IDX = 2
METRICS = ("margin_hit_5pp", "unit_cost_hit_10pct")


def _contrast(fresh: list[dict], reuse: list[dict], n_boot: int, n_values=N_VALUES) -> dict:
    """(fresh gain n16 - n1) - (reuse gain n16 - n1), paired by (firm, seed)."""
    def gains(rows, m):
        by = {(r["firm_id"], r["seed"], r["n"]): float(r[m]) for r in rows}
        return {(f, s): by[(f, s, n_values[-1])] - by[(f, s, n_values[0])] for f, s, _ in by}
    out = {}
    for m in METRICS:
        gf, gr = gains(fresh, m), gains(reuse, m)
        keys = sorted(gf)
        diff = np.array([gf[k] - gr[k] for k in keys])
        out[m] = {"contrast": float(diff.mean()),
                  "contrast_ci": bootstrap_ci(diff, np.array([k[0] for k in keys]), n_boot=n_boot)}
    return out


def _linked(seed: int, kw: dict, n_boot: int, n_values=N_VALUES, joint_kw=None) -> dict:
    ev = build_eval_set(EvalConfig(seed=seed))
    firms = [(i, f) for i, f in enumerate(ev.firms) if i >= ev.config.n_dev][:N_FIRMS]
    d = OrderPreserving()
    solver = ClassicalBayesAttack(ev.prior, d, use_oracle=True, **kw)
    rows = {}
    for regime in ("fresh", "reuse"):
        LOG.info("linked seed %d %s %s: start", seed, regime, joint_kw or {})
        rows[regime] = _rows(ev, firms, "D2", d, regime, kw, B2_D2_DEFENSE_IDX, n_values,
                             attack=JointMapAttack(solver, regime, **(joint_kw or {})))
    out = {f"D2/{r}": _summary(v, n_boot, n_values) for r, v in rows.items()}
    out["fresh_minus_reuse"] = _contrast(rows["fresh"], rows["reuse"], n_boot, n_values)
    return out


SENSITIVITY = ({"obs_sd": 0.03, "n_nuisance": 1600}, {"obs_sd": 0.1, "n_nuisance": 400})


def _sensitivity(kw: dict, n_boot: int) -> dict:
    return {f"obs_sd={c['obs_sd']}, n_nuisance={c['n_nuisance']}": _linked(1001, kw, n_boot, (1, 16), c)
            for c in SENSITIVITY}


def _single(kw: dict, n_boot: int) -> dict:
    ev = build_eval_set(EvalConfig())
    d = OrderPreserving()
    solver = ClassicalBayesAttack(ev.prior, d, use_oracle=True, **kw)
    attacks = {"joint": JointMapAttack(solver, "fresh"), "pair": MultiQueryAttack(solver, "fresh")}
    floor = PriorOnlyAttack(ev.prior)
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng([seed, 4242])
        for prompt in (p for p in ev.prompts if p.split == "test"):
            spec = prompt.spec
            view = make_view(spec, d.apply(spec.fields, d.sample_secret(rng, spec.fields), rng).fields, d.name, True)
            truth = spec.products[0]
            row = {"firm_id": spec.firm_id, "seed": seed, "task": spec.task}
            a0 = floor.attack(view)[0]
            ests = {"A0": (a0.margin, a0.unit_cost)}
            ests.update({k: (lambda e: (e.margin, e.unit_cost))(a.attack((view,))) for k, a in attacks.items()})
            for k, (m, c) in ests.items():
                row[f"{k}_margin_hit_5pp"] = abs(m - truth.margin) <= MARGIN_TOL_5PP
                row[f"{k}_unit_cost_hit_10pct"] = abs(c - truth.unit_cost) <= COST_REL_TOL * truth.unit_cost
            rows.append(row)
    out = {"n_rows": len(rows), "n_multi_product_rows": sum(r["task"] == "margin_review" for r in rows)}
    for subset, rs in (("all", rows), ("margin_review", [r for r in rows if r["task"] == "margin_review"])):
        g = np.array([r["firm_id"] for r in rs])
        cell = {}
        for m in METRICS:
            for k in ("A0", "pair", "joint"):
                cell[f"{k}_{m}"] = float(np.mean([r[f"{k}_{m}"] for r in rs]))
            for a, b in (("joint", "pair"), ("joint", "A0"), ("pair", "A0")):
                diff = np.array([float(r[f"{a}_{m}"]) - float(r[f"{b}_{m}"]) for r in rs])
                cell[f"{a}_minus_{b}_{m}"] = float(diff.mean())
                cell[f"{a}_minus_{b}_{m}_ci"] = bootstrap_ci(diff, g, n_boot=n_boot)
        out[subset] = cell
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("linked", "fresh_draw", "single", "sensitivity", "smoothing"), required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--obs-sd", type=float, default=0.01, help="smoothing part only")
    parser.add_argument("--n-nuisance", type=int, default=400, help="smoothing part only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = time.time()
    kw = {**json.loads(FROZEN_CONFIG.read_text())["result"]["frozen"], "n_samples": 4000}
    result = {"linked": lambda: _linked(1001, kw, args.n_boot), "fresh_draw": lambda: _linked(3003, kw, args.n_boot),
              "single": lambda: _single(kw, args.n_boot),
              "sensitivity": lambda: _sensitivity(kw, args.n_boot),
              "smoothing": lambda: _linked(1001, kw, args.n_boot, (1, 16),
                                           {"obs_sd": args.obs_sd, "n_nuisance": args.n_nuisance})}[args.part]()
    config = {"part": args.part, "frozen_solver": kw, "n_nuisance": JointMapAttack.n_nuisance,
              "obs_sd": JointMapAttack.obs_sd, "nuisance": "shared across queries (integrated once over the history)",
              "modelled_fields": "prices and costs of all products shown (other products integrated over class priors)",
              "seeds": SEEDS, "n_values": N_VALUES, "total_seconds": time.time() - t0}
    run = f"R045_d2_joint_{args.part}"
    if args.part == "smoothing":  # R047a
        config.update(obs_sd=args.obs_sd, n_nuisance=args.n_nuisance,
                      plan="refine-logs/EXPERIMENT_PLAN_20261018_numerical_validation.md")
        run = f"R047a_d2_joint_smoothing_obs{args.obs_sd}_nuis{args.n_nuisance}"
    out = RESULTS / f"{run}.json"
    out.write_text(json.dumps({"run": run, "config": config, "result": result}, indent=2))
    LOG.info("wrote %s (%.0f s)", out, time.time() - t0)


if __name__ == "__main__":
    main()
