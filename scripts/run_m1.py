"""Milestone M1 (classical baselines) R007-R010. Writes results/<RUN>.json.

R007 A0 prior-only floor (dev + test)
R008 A1 classical Bayes, sector unknown: tuned on DEV only, then frozen and run on test
R009 A1+ classical Bayes with oracle sector/class (same frozen config)
R010 gate: A1+ must beat A0 on D1 and D2 (advantage CI excludes 0), else the classical attacker is broken
"""

import argparse
import itertools
import json
import logging
from pathlib import Path

from numpriv.attacks import PriorOnlyAttack
from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.config import SECTOR_PARAMETER_PROVENANCE
from numpriv.dataset import PRIOR_DRAW_SIZE, EvalConfig, build_eval_set
from numpriv.defenses import MetricDP, NoDefense, OrderPreserving, PerFieldScaling, SharedScaling
from numpriv.evaluate import evaluate, summarize

LOG = logging.getLogger("m1")
RESULTS = Path(__file__).resolve().parents[1] / "results"
TUNING_GRID = {"prior_df": (None, 3.0), "prior_scale": (0.75, 1.0, 1.5)}


def _defenses(epsilons: tuple[float, ...]) -> tuple:
    return (NoDefense(), SharedScaling(), PerFieldScaling(), OrderPreserving(), *(MetricDP(e) for e in epsilons))


def _label(d) -> str:
    return f"{d.name}@eps={d.epsilon}" if isinstance(d, MetricDP) else d.name


def _save(run: str, payload: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{run}.json").write_text(json.dumps(payload, indent=2, default=float))
    LOG.info("wrote %s", RESULTS / f"{run}.json")


def _objective(summary: dict) -> float:
    s = summary["all"]
    return 0.5 * (s["margin_hit_5pp"] + s["unit_cost_hit_10pct"])


def tune_on_dev(ev, defenses, n_samples: int) -> dict:
    """Pick one (prior_df, prior_scale) for both A1 and A1+, by mean objective over disguised defenses on dev."""
    scores = []
    for df, scale in itertools.product(TUNING_GRID["prior_df"], TUNING_GRID["prior_scale"]):
        vals = []
        for d in defenses:
            if isinstance(d, NoDefense):
                continue
            for oracle in (False, True):
                a = ClassicalBayesAttack(ev.prior, d, use_oracle=oracle, prior_df=df, prior_scale=scale,
                                         n_samples=n_samples)
                vals.append(_objective(summarize(evaluate(ev, "dev", d, a, (0,), oracle), n_boot=10)))
        scores.append({"prior_df": df, "prior_scale": scale, "dev_objective": sum(vals) / len(vals)})
        LOG.info("dev tuning %s", scores[-1])
    best = max(scores, key=lambda s: s["dev_objective"])
    return {"grid": scores, "frozen": {"prior_df": best["prior_df"], "prior_scale": best["prior_scale"]}}


def run_test(ev, defenses, seeds, frozen: dict, n_samples: int, n_boot: int) -> dict:
    out = {}
    for d in defenses:
        floor = evaluate(ev, "test", d, PriorOnlyAttack(ev.prior), seeds, with_oracle=True)
        entry = {"A0_prior_only": summarize(floor, floor, n_boot)}
        for oracle in (False, True):
            a = ClassicalBayesAttack(ev.prior, d, use_oracle=oracle, n_samples=n_samples, **frozen)
            entry[a.name] = summarize(evaluate(ev, "test", d, a, seeds, oracle), floor, n_boot)
        out[_label(d)] = entry
        LOG.info("test %s done", _label(d))
    return out


def gate(test: dict) -> dict:
    checks = {}
    for label in (SharedScaling.name, OrderPreserving.name):
        s = test[label]["A1plus_bayes_oracle"]["all"]
        lo_m, _ = s["advantage_margin_hit_5pp_ci"]
        lo_c, _ = s["advantage_unit_cost_hit_10pct_ci"]
        checks[label] = {"margin_ci_low": lo_m, "cost_ci_low": lo_c, "passes": lo_m > 0 or lo_c > 0}
    return {"checks": checks, "pass": all(c["passes"] for c in checks.values())}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--prior-seed", type=int, default=2002)
    ap.add_argument("--defense-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epsilons", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()
    cfg = EvalConfig(seed=args.seed, prior_seed=args.prior_seed)
    ev, defenses, seeds = build_eval_set(cfg), _defenses(tuple(args.epsilons)), tuple(args.defense_seeds)
    settings = {**cfg.__dict__, "defense_seeds": seeds, "epsilons": args.epsilons, "n_samples": args.n_samples,
                "n_boot": args.n_boot, "prior_draw_size": PRIOR_DRAW_SIZE, "tuning_grid": TUNING_GRID,
                "sector_parameters": SECTOR_PARAMETER_PROVENANCE}

    tuning = tune_on_dev(ev, defenses, args.n_samples)
    _save("R008_tuning", {"run": "R008_tuning", "config": settings, "result": tuning})
    test = run_test(ev, defenses, seeds, tuning["frozen"], args.n_samples, args.n_boot)
    settings = {**settings, "frozen_attacker_config": tuning["frozen"]}
    _save("R007", {"run": "R007", "config": settings,
                   "result": {k: v["A0_prior_only"] for k, v in test.items()}})
    _save("R008", {"run": "R008", "config": settings, "result": {k: v["A1_bayes"] for k, v in test.items()}})
    _save("R009", {"run": "R009", "config": settings,
                   "result": {k: v["A1plus_bayes_oracle"] for k, v in test.items()}})
    _save("R010", {"run": "R010", "config": settings, "result": gate(test)})


if __name__ == "__main__":
    main()
