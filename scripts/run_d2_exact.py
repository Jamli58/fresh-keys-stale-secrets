"""R046: D2 with the precision-faithful attacker A-exact (attacks/exact_joint.py); plan and Amendment 1 in
refine-logs/EXPERIMENT_PLAN_20261009_exact_joint.md.

A-exact is used for the reuse regime and single queries; the fresh arm uses the joint shared-map attacker of R045
(its posterior cannot be represented with particles, Amendment 1). Same firms, defense seeds and random streams as
R017/R035/R045, so rows are paired across attackers and regimes. Parts:
  v3          numerical check: reuse, n = 1 and 16, A-exact seed 4646 vs 4647 (hit rates must agree within 1.5 pp)
  linked      original firms (eval seed 1001): reuse under A-exact, fresh under the joint attacker, contrast (E1, E2)
  fresh_draw  the same on the fresh draw (eval seed 3003)
  single      product 0 of every test prompt: A-exact vs Gaussian pair vs joint vs A0 (E3)
  law         reuse under A-exact with alternative map laws, n = 1 and 16 (E4)
    uv run python scripts/run_d2_exact.py --part v3
"""

import argparse
import json
import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from run_b2 import FROZEN_CONFIG, N_FIRMS, N_VALUES, SEEDS, _summary

from numpriv.attacks import PriorOnlyAttack, make_view
from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.attacks.exact_joint import ExactJointAttack
from numpriv.attacks.joint_map import JointMapAttack
from numpriv.attacks.multi_query import MultiQueryAttack
from numpriv.dataset import EvalConfig, build_eval_set
from numpriv.defenses import OrderPreserving
from numpriv.evaluate import COST_REL_TOL, MARGIN_TOL_5PP
from numpriv.metrics import bootstrap_ci
from numpriv.sequences import build_sequence, disguise_sequence

LOG = logging.getLogger("d2_exact")
RESULTS = Path(__file__).resolve().parents[1] / "results"
B2_D2_DEFENSE_IDX = 2
METRICS = ("margin_hit_5pp", "unit_cost_hit_10pct")
EXACT = {"n_particles": 20_000, "n_initial": 400_000, "mcmc_steps": 40, "seed": 4646}  # Amendment 1; 40 after the V3 rule
LAWS = {"dirichlet0.1": {"gap_concentration": 0.1}, "intervals16": {"n_intervals": 16},
        "intervals256": {"n_intervals": 256}}
_CACHE: dict = {}


def _setup(eval_seed: int, defense: OrderPreserving):
    key = (eval_seed, defense)
    if key not in _CACHE:
        kw = {**json.loads(FROZEN_CONFIG.read_text())["result"]["frozen"], "n_samples": 4000}
        ev = build_eval_set(EvalConfig(seed=eval_seed))
        firms = [(i, f) for i, f in enumerate(ev.firms) if i >= ev.config.n_dev][:N_FIRMS]
        _CACHE[key] = (ev, firms, ClassicalBayesAttack(ev.prior, defense, use_oracle=True, **kw))
    return _CACHE[key]


def _hits(est, truth) -> dict:
    return {"margin_hit_5pp": bool(abs(est.margin - truth.margin) <= MARGIN_TOL_5PP),
            "unit_cost_hit_10pct": bool(abs(est.unit_cost - truth.unit_cost) <= COST_REL_TOL * truth.unit_cost)}


def _sequence_rows(job: tuple) -> list[dict]:
    """Rows for one (firm, defense seed): the requested attackers over n_values, as run_b2._rows does."""
    eval_seed, k, seed, regimes, n_values, exact_kw, law = job
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    d = OrderPreserving(**law)
    ev, firms, solver = _setup(eval_seed, d)
    idx, firm = firms[k]
    specs = build_sequence(firm, max(n_values), start=idx % 5)
    truth = firm.products[0]
    floor = PriorOnlyAttack(ev.prior)
    rows = []
    for label, regime in regimes:
        seq = disguise_sequence(specs, d, regime, np.random.default_rng([seed, idx, B2_D2_DEFENSE_IDX]))
        a0 = floor.attack(seq.views[0])[0]
        t0 = time.time()
        if label == "exact":
            ests, ess = ExactJointAttack(solver, regime, **exact_kw).run(seq.views, n_values)
            diag = {"min_ess": float(min(ess))}
        else:
            att = JointMapAttack(solver, regime)
            ests, diag = {n: att.attack(seq.views[:n]) for n in n_values}, {}
        for n in n_values:
            rows.append({"attacker": label, "regime": regime, "firm_id": firm.firm_id, "seed": seed, "n": n,
                         "margin_est": float(ests[n].margin),
                         **_hits(ests[n], truth), "key_log_error": None,
                         "a0_margin_hit_5pp": bool(abs(a0.margin - truth.margin) <= MARGIN_TOL_5PP),
                         "a0_unit_cost_hit_10pct": bool(abs(a0.unit_cost - truth.unit_cost)
                                                        <= COST_REL_TOL * truth.unit_cost),
                         "seconds": time.time() - t0, **diag})
    return rows


def _run(jobs: list, workers: int) -> list[dict]:
    with ProcessPoolExecutor(workers) as ex:
        return [r for rows in ex.map(_sequence_rows, jobs) for r in rows]


def _jobs(eval_seed, regimes, n_values, exact_kw=None, law=None):
    exact_kw = dict(exact_kw or EXACT)  # read at call time, after --exact-seed
    return [(eval_seed, k, s, regimes, n_values, exact_kw, law or {}) for s in SEEDS for k in range(N_FIRMS)]


def _contrast(fresh: list[dict], reuse: list[dict], n_boot: int, n_values) -> dict:
    """(fresh gain n_last - n_first) - (reuse gain), paired by (firm, seed)."""
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


def _pick(rows, attacker, regime):
    return [r for r in rows if r["attacker"] == attacker and r["regime"] == regime]


def _diag(rows) -> dict:
    ess = [r["min_ess"] for r in rows if "min_ess" in r]
    return {"min_ess_quantiles": np.quantile(ess, [0, 0.1, 0.5]).tolist() if ess else None,
            "seconds_mean": float(np.mean([r["seconds"] for r in rows]))}


def part_v3(n_boot: int, workers: int) -> dict:
    nv = (1, 16)
    runs = {s: _run(_jobs(1001, (("exact", "reuse"),), nv, {**EXACT, "seed": s}), workers) for s in (4646, 4647)}
    out = {}
    for n in nv:
        a, b = ([r for r in runs[s] if r["n"] == n] for s in (4646, 4647))
        out[str(n)] = {m: {"seed4646": float(np.mean([r[m] for r in a])), "seed4647": float(np.mean([r[m] for r in b])),
                           "abs_diff_pp": 100 * abs(np.mean([r[m] for r in a]) - np.mean([r[m] for r in b]))}
                       for m in METRICS}
    out["pass"] = all(out[str(n)][m]["abs_diff_pp"] <= 1.5 for n in nv for m in METRICS)
    out["diagnostics"] = {s: _diag(v) for s, v in runs.items()}
    return out


def part_linked(eval_seed: int, n_boot: int, workers: int) -> dict:
    rows = _run(_jobs(eval_seed, (("exact", "reuse"), ("joint", "fresh")), N_VALUES), workers)
    reuse, fresh = _pick(rows, "exact", "reuse"), _pick(rows, "joint", "fresh")
    return {"D2/reuse/A-exact": _summary(reuse, n_boot, N_VALUES),
            "D2/fresh/joint": _summary(fresh, n_boot, N_VALUES),
            "fresh_joint_minus_reuse_exact": _contrast(fresh, reuse, n_boot, N_VALUES),
            "diagnostics": _diag(reuse)}


def _single_views() -> list[tuple]:
    """(seed, spec, view) for product 0 of every test prompt; one stream per seed, advanced in prompt order as in
    R045 single, so rows are paired with it."""
    d = OrderPreserving()
    ev, _, _ = _setup(1001, d)
    out = []
    for seed in SEEDS:
        rng = np.random.default_rng([seed, 4242])
        for prompt in (p for p in ev.prompts if p.split == "test"):
            spec = prompt.spec
            view = make_view(spec, d.apply(spec.fields, d.sample_secret(rng, spec.fields), rng).fields, d.name, True)
            out.append((seed, spec, view))
    return out


def _single_row(job: tuple) -> dict:
    seed, spec, view, exact_kw = job
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ev, _, solver = _setup(1001, OrderPreserving())
    attacks = {"pair": MultiQueryAttack(solver, "fresh"), "joint": JointMapAttack(solver, "fresh"),
               "exact": ExactJointAttack(solver, "reuse", **exact_kw)}
    truth = spec.products[0]
    row = {"firm_id": spec.firm_id, "seed": seed, "task": spec.task}
    ests = {"A0": PriorOnlyAttack(ev.prior).attack(view)[0], **{k: a.attack((view,)) for k, a in attacks.items()}}
    for k, e in ests.items():
        for m, v in _hits(e, truth).items():
            row[f"{k}_{m}"] = v
    return row


def part_single(n_boot: int, workers: int) -> dict:
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_single_row, [(*v, dict(EXACT)) for v in _single_views()], chunksize=8))
    out = {"n_rows": len(rows)}
    for subset, rs in (("all", rows), ("margin_review", [r for r in rows if r["task"] == "margin_review"])):
        g = np.array([r["firm_id"] for r in rs])
        cell = {}
        for m in METRICS:
            for k in ("A0", "pair", "joint", "exact"):
                cell[f"{k}_{m}"] = float(np.mean([r[f"{k}_{m}"] for r in rs]))
            for a, b in (("exact", "A0"), ("exact", "pair"), ("exact", "joint")):
                diff = np.array([float(r[f"{a}_{m}"]) - float(r[f"{b}_{m}"]) for r in rs])
                cell[f"{a}_minus_{b}_{m}"] = float(diff.mean())
                cell[f"{a}_minus_{b}_{m}_ci"] = bootstrap_ci(diff, g, n_boot=n_boot)
        out[subset] = cell
    return out


def part_law(n_boot: int, workers: int) -> dict:
    out = {}
    for name, law in LAWS.items():
        rows = _run(_jobs(1001, (("exact", "reuse"),), (1, 16), law=law), workers)
        out[name] = {"D2/reuse/A-exact": _summary(rows, n_boot, (1, 16)), "diagnostics": _diag(rows)}
    return out


BUDGET = {"base": {"n_particles": 20_000, "n_initial": 400_000, "mcmc_steps": 40},
          "high": {"n_particles": 80_000, "n_initial": 1_600_000, "mcmc_steps": 160}}
BUDGET_FIRMS = 20


def part_budget(n_boot: int, workers: int) -> dict:
    """R047b: A-exact on reused maps for the first 20 test firms x 3 seeds, base vs fourfold budget, paired."""
    rows = {}
    for name, cfg in BUDGET.items():
        jobs = [(1001, k, s, (("exact", "reuse"),), N_VALUES, {**cfg, "seed": 4646}, {})
                for s in SEEDS for k in range(BUDGET_FIRMS)]
        rows[name] = _run(jobs, workers if name == "base" else max(1, workers // 2))
    key = lambda r: (r["firm_id"], r["seed"], r["n"])  # noqa: E731
    by = {name: {key(r): r for r in rs} for name, rs in rows.items()}
    pairs = sorted({(f, s) for f, s, _ in by["base"]})
    groups = np.array([f for f, _ in pairs])
    m = "margin_hit_5pp"
    gain = {name: np.array([float(by[name][(f, s, 16)][m]) - float(by[name][(f, s, 1)][m]) for f, s in pairs])
            for name in BUDGET}
    diff = gain["high"] - gain["base"]
    out = {"n_sequences": len(pairs),
           "E1_gain_diff_high_minus_base": float(diff.mean()), "E1_ci": bootstrap_ci(diff, groups, n_boot=n_boot),
           "gain_base": float(gain["base"].mean()), "gain_high": float(gain["high"].mean())}
    for n in (1, 16):
        est = np.array([by["high"][(f, s, n)]["margin_est"] - by["base"][(f, s, n)]["margin_est"] for f, s in pairs])
        hits = np.array([float(by["high"][(f, s, n)][m]) - float(by["base"][(f, s, n)][m]) for f, s in pairs])
        out[f"E2_mean_abs_margin_est_diff_n{n}"] = float(np.abs(est).mean())
        out[f"E3_margin_hit_diff_n{n}"] = float(hits.mean())
        out[f"E3_margin_hit_diff_n{n}_ci"] = bootstrap_ci(hits, groups, n_boot=n_boot)
    out["summaries"] = {name: _summary(rs, n_boot, N_VALUES) for name, rs in rows.items()}
    out["diagnostics"] = {name: _diag(rs) for name, rs in rows.items()}
    out["stable"] = bool(abs(out["E1_gain_diff_high_minus_base"]) <= 0.033
                         and max(out["E2_mean_abs_margin_est_diff_n1"], out["E2_mean_abs_margin_est_diff_n16"]) <= 0.01)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("v3", "linked", "fresh_draw", "single", "law", "budget"), required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--exact-seed", type=int, default=EXACT["seed"], help="A-exact seed (Amendment 2: 4646, 4647)")
    args = parser.parse_args()
    EXACT["seed"] = args.exact_seed  # workers are spawned: every part passes EXACT to them explicitly
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = time.time()
    result = {"v3": lambda: part_v3(args.n_boot, args.workers),
              "linked": lambda: part_linked(1001, args.n_boot, args.workers),
              "fresh_draw": lambda: part_linked(3003, args.n_boot, args.workers),
              "single": lambda: part_single(args.n_boot, args.workers),
              "law": lambda: part_law(args.n_boot, args.workers),
              "budget": lambda: part_budget(args.n_boot, args.workers)}[args.part]()
    config = {"part": args.part, "exact": EXACT, "seeds": SEEDS, "n_values": N_VALUES, "n_firms": N_FIRMS,
              "fresh_arm": "joint shared-map attacker (R045 settings), Amendment 1",
              "plan": "refine-logs/EXPERIMENT_PLAN_20261009_exact_joint.md", "total_seconds": time.time() - t0}
    suffix = "" if args.part == "v3" else f"_seed{args.exact_seed}"
    run = f"R046_d2_exact_{args.part}{suffix}"
    if args.part == "budget":  # R047b
        config.update(budget=BUDGET, budget_firms=BUDGET_FIRMS,
                      plan="refine-logs/EXPERIMENT_PLAN_20261018_numerical_validation.md")
        run = "R047b_d2_exact_budget"
    out = RESULTS / f"{run}.json"
    out.write_text(json.dumps({"run": run, "config": config, "result": result}, indent=2))
    LOG.info("wrote %s (%.0f s)", out, time.time() - t0)


if __name__ == "__main__":
    main()
