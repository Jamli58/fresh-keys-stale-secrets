"""R037: sensitivity of D2 leakage to its map law (our assumption A-D2-1/A-D2-8).

For each law the attacker knows the law (Kerckhoffs) and simulates its own Gaussian moments for it;
the solver configuration stays frozen (R008_tuning.json). Per law:
  * single query: A1+ vs A0 on the test split (as R009; 2,100 rows);
  * linked queries: fresh maps, product 0, 60 test firms x 3 seeds, n = 1..16 (as R017-R019).
The linked rows reuse run_b2.py's random stream for D2 (defense index 2), so the default law
reproduces R017_R019_B2.json's D2/fresh cell exactly.
    uv run python scripts/run_d2_sensitivity.py
"""

import argparse
import json
import logging
import time
from pathlib import Path

from run_b2 import FROZEN_CONFIG, N_FIRMS, N_VALUES, _rows, _summary

from numpriv.attacks import PriorOnlyAttack
from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.config import SECTOR_PARAMETER_PROVENANCE
from numpriv.dataset import EvalConfig, build_eval_set
from numpriv.defenses import OrderPreserving
from numpriv.evaluate import evaluate, summarize

LOG = logging.getLogger("d2_sensitivity")
RESULTS = Path(__file__).resolve().parents[1] / "results"
SEEDS = (0, 1, 2)
B2_D2_DEFENSE_IDX = 2  # position of D2 in run_b2.CONFIGS
LAWS = {
    "default (64 intervals, uniform cuts, targets [1, 1e6])": {},
    "16 intervals": {"n_intervals": 16},
    "256 intervals": {"n_intervals": 256},
    "irregular gaps (Dirichlet 0.1)": {"gap_concentration": 0.1},
    "regular gaps (Dirichlet 10)": {"gap_concentration": 10.0},
    "targets [1, 1e12]": {"target_high": 1e12},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--out", default="R037_d2_sensitivity.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = time.time()
    ev = build_eval_set(EvalConfig())
    solver_kw = {**json.loads(FROZEN_CONFIG.read_text())["result"]["frozen"], "n_samples": args.n_samples}
    firms = [(i, f) for i, f in enumerate(ev.firms) if i >= ev.config.n_dev][:N_FIRMS]
    floor = None
    result = {}
    for label, law in LAWS.items():
        d = OrderPreserving(**law)
        LOG.info("%s: start", label)
        if floor is None:
            floor = evaluate(ev, "test", d, PriorOnlyAttack(ev.prior), SEEDS, with_oracle=True)
        a1p = evaluate(ev, "test", d, ClassicalBayesAttack(ev.prior, d, use_oracle=True, **solver_kw), SEEDS, True)
        rows = _rows(ev, firms, "D2", d, "fresh", solver_kw, B2_D2_DEFENSE_IDX, N_VALUES)
        result[label] = {"law": dict(zip(("target_low", "target_high", "n_intervals", "gap_concentration"), d.law)),
                         "single_query_test": summarize(a1p, floor, args.n_boot)["all"],
                         "linked_fresh": _summary(rows, args.n_boot, N_VALUES)}
    config = {"eval_config": ev.config.__dict__, "seeds": SEEDS, "n_firms_linked": len(firms),
              "frozen_solver": solver_kw, "a0_floor": "P_pub class median (law-independent)",
              "sector_parameters": SECTOR_PARAMETER_PROVENANCE, "total_seconds": time.time() - t0}
    out = RESULTS / args.out
    out.write_text(json.dumps({"run": "R037_d2_sensitivity", "config": config, "result": result}, indent=2))
    LOG.info("wrote %s (%.0f s)", out, time.time() - t0)


if __name__ == "__main__":
    main()
