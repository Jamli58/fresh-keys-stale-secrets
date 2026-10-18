"""R044: D2 attack with a simulation-based reference likelihood instead of the Gaussian approximation.

Same prior draws, oracle, decision rule (posterior median), test split, defense seeds and linked-query
random streams as R009 / R017 / R037; only the pair likelihood changes (kernel density over simulated maps,
attacks/d2_reference.py). Parts (run as separate processes):
  single    single-query test split, A1+ Gaussian vs reference, paired, for the default law and Dirichlet(0.1)
  linked    fresh-map linked queries (60 test firms x 3 seeds, n = 1..16), reference likelihood, both laws
  sensitivity  bandwidth / number-of-maps sensitivity, default law, first 100 test firms (single query)
    uv run python scripts/run_d2_reference.py --part single
"""

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path

from run_b2 import FROZEN_CONFIG, N_FIRMS, N_VALUES, _rows, _summary

from numpriv.attacks import PriorOnlyAttack
from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.attacks.d2_reference import SimulatedD2BayesAttack, SimulatedMapLikelihood
from numpriv.dataset import EvalConfig, build_eval_set
from numpriv.defenses import OrderPreserving
from numpriv.evaluate import evaluate, summarize

LOG = logging.getLogger("d2_reference")
RESULTS = Path(__file__).resolve().parents[1] / "results"
SEEDS = (0, 1, 2)
B2_D2_DEFENSE_IDX = 2
N_MAPS, BANDWIDTH = 20_000, 0.03
LAWS = {"default": {}, "irregular gaps (Dirichlet 0.1)": {"gap_concentration": 0.1}}
SENSITIVITY = ((20_000, 0.015), (20_000, 0.06), (50_000, 0.03))
SENSITIVITY_FIRMS = 100


def _reference(ev, d, kw, n_maps=N_MAPS, bandwidth=BANDWIDTH):
    return SimulatedD2BayesAttack(ev.prior, d, use_oracle=True, **kw,
                                  reference=SimulatedMapLikelihood(d, n_maps=n_maps, bandwidth=bandwidth))


def _single(ev, kw, n_boot) -> dict:
    out = {}
    for label, law in LAWS.items():
        d = OrderPreserving(**law)
        LOG.info("single %s: start", label)
        a0 = evaluate(ev, "test", d, PriorOnlyAttack(ev.prior), SEEDS, with_oracle=True)
        gauss = evaluate(ev, "test", d, ClassicalBayesAttack(ev.prior, d, use_oracle=True, **kw), SEEDS, True)
        ref = evaluate(ev, "test", d, _reference(ev, d, kw), SEEDS, True)
        out[label] = {"gaussian_vs_A0": summarize(gauss, a0, n_boot)["all"],
                      "reference_vs_A0": summarize(ref, a0, n_boot)["all"],
                      "reference_vs_gaussian": summarize(ref, gauss, n_boot)["all"]}
    return out


def _linked(ev, kw, n_boot) -> dict:
    firms = [(i, f) for i, f in enumerate(ev.firms) if i >= ev.config.n_dev][:N_FIRMS]
    out = {}
    for label, law in LAWS.items():
        d = OrderPreserving(**law)
        LOG.info("linked %s: start", label)
        rows = _rows(ev, firms, "D2", d, "fresh", kw, B2_D2_DEFENSE_IDX, N_VALUES, solver=_reference(ev, d, kw))
        out[label] = _summary(rows, n_boot, N_VALUES)
    return out


def _sensitivity(ev, kw, n_boot) -> dict:
    keep = {f.firm_id for f in ev.firms[ev.config.n_dev:ev.config.n_dev + SENSITIVITY_FIRMS]}
    sub = dataclasses.replace(ev, prompts=tuple(p for p in ev.prompts if p.spec.firm_id in keep))
    d = OrderPreserving()
    base = evaluate(sub, "test", d, _reference(sub, d, kw), SEEDS, True)
    out = {f"n_maps={N_MAPS}, bandwidth={BANDWIDTH}": summarize(base, None, n_boot)["all"]}
    for n_maps, bw in SENSITIVITY:
        LOG.info("sensitivity n_maps=%d bandwidth=%.3f: start", n_maps, bw)
        rows = evaluate(sub, "test", d, _reference(sub, d, kw, n_maps, bw), SEEDS, True)
        out[f"n_maps={n_maps}, bandwidth={bw}"] = summarize(rows, base, n_boot)["all"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("single", "linked", "sensitivity"), required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = time.time()
    ev = build_eval_set(EvalConfig())
    kw = {**json.loads(FROZEN_CONFIG.read_text())["result"]["frozen"], "n_samples": 4000}
    result = {"single": _single, "linked": _linked, "sensitivity": _sensitivity}[args.part](ev, kw, args.n_boot)
    config = {"eval_config": ev.config.__dict__, "seeds": SEEDS, "frozen_solver": kw, "n_maps": N_MAPS,
              "bandwidth": BANDWIDTH, "laws": LAWS, "part": args.part, "total_seconds": time.time() - t0}
    out = RESULTS / f"R044_d2_reference_{args.part}.json"
    out.write_text(json.dumps({"run": f"R044_d2_reference_{args.part}", "config": config, "result": result}, indent=2))
    LOG.info("wrote %s (%.0f s)", out, time.time() - t0)


if __name__ == "__main__":
    main()
