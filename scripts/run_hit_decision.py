"""R041: single-query recovery with hit-optimal decisions (attack validation for the small-advantage cells).

The paper's attackers report posterior (A1+) or prior (A0) medians, which minimize absolute error. The hit
metric rewards landing within +-5 pp (margin) or +-10% (cost), so for each mechanism with a small single-query
advantage we also evaluate the decision that maximizes the posterior probability of a hit, against the prior-only
control with the same rule on the solver's own prior (use_observation=False). Same test split, defense seeds and
frozen solver as R009/R033.
    uv run python scripts/run_hit_decision.py
"""

import argparse
import json
import logging
import time
from pathlib import Path

from numpriv.attacks import PriorOnlyAttack
from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.config import SECTOR_PARAMETER_PROVENANCE
from numpriv.dataset import EvalConfig, build_eval_set
from numpriv.defenses import MetricDPOfficial, OrderPreserving, PerFieldScaling
from numpriv.evaluate import evaluate, summarize

LOG = logging.getLogger("hit_decision")
RESULTS = Path(__file__).resolve().parents[1] / "results"
SEEDS = (0, 1, 2)
MECHANISMS = (("D1p", PerFieldScaling()), ("D2", OrderPreserving()), ("D3o@0.5", MetricDPOfficial(0.5)),
              ("D3o@1", MetricDPOfficial(1.0)), ("D3o@2", MetricDPOfficial(2.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--n-samples", type=int, default=4000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = time.time()
    ev = build_eval_set(EvalConfig())
    kw = {**json.loads((RESULTS / "R008_tuning.json").read_text())["result"]["frozen"], "n_samples": args.n_samples}
    table_a0 = None
    prior_rows = {}
    result = {}
    for label, d in MECHANISMS:
        LOG.info("%s: start", label)
        if table_a0 is None:  # the prior-only attackers ignore the shown numbers: one run serves every mechanism
            table_a0 = evaluate(ev, "test", d, PriorOnlyAttack(ev.prior), SEEDS, with_oracle=True)
            for dec in ("median", "hit"):
                a = ClassicalBayesAttack(ev.prior, d, use_oracle=True, decision=dec, use_observation=False, **kw)
                prior_rows[dec] = evaluate(ev, "test", d, a, SEEDS, with_oracle=True)
            result["A0_variants_vs_table_median"] = {
                dec: summarize(rows, table_a0, args.n_boot)["all"] for dec, rows in prior_rows.items()}
        med = evaluate(ev, "test", d, ClassicalBayesAttack(ev.prior, d, use_oracle=True, **kw), SEEDS, True)
        hit = evaluate(ev, "test", d, ClassicalBayesAttack(ev.prior, d, use_oracle=True, decision="hit", **kw),
                       SEEDS, True)
        result[label] = {
            "median_vs_A0_table": summarize(med, table_a0, args.n_boot)["all"],
            "hit_vs_A0_hit": summarize(hit, prior_rows["hit"], args.n_boot)["all"],
            "hit_vs_median": summarize(hit, med, args.n_boot)["all"],
        }
    config = {"eval_config": ev.config.__dict__, "seeds": SEEDS, "frozen_solver": kw,
              "A0_hit": "ClassicalBayesAttack(use_observation=False, decision='hit'): the solver's own prior",
              "sector_parameters": SECTOR_PARAMETER_PROVENANCE, "total_seconds": time.time() - t0}
    out = RESULTS / "R041_hit_decision.json"
    out.write_text(json.dumps({"run": "R041_hit_decision", "config": config, "result": result}, indent=2))
    LOG.info("wrote %s (%.0f s)", out, time.time() - t0)


if __name__ == "__main__":
    main()
