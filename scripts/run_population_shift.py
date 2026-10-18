"""R042: frozen attacker under population shift (and a fresh held-out single-query check).

The attacker's prior table and solver stay those of the paper (prior_seed 2002, R008_tuning.json). The evaluated
firms are a fresh draw (eval seed 3003, never used for development) from the generator with one constant changed
per scenario; the same random stream is used in every scenario, so scenarios differ only by the shift.
Scenario "none" is the unshifted fresh draw, i.e. a held-out repetition of the single-query table.
Endpoints: single-query A1+ vs A0 on the test split; linked-query n = 1 -> 16 gains (60 test firms x 3 seeds).
    uv run python scripts/run_population_shift.py
"""

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path
from unittest import mock

import numpy as np
from run_b2 import FROZEN_CONFIG, N_FIRMS, _rows, _summary

from numpriv import config
from numpriv.attacks import PriorOnlyAttack
from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.dataset import EvalConfig, EvalPrompt, _tasks_for, build_eval_set
from numpriv.defenses import MetricDPOfficial, OrderPreserving, PerFieldScaling, SharedScaling
from numpriv.evaluate import evaluate, summarize
from numpriv.records import generate_firms
from numpriv.templates import build_prompt

LOG = logging.getLogger("population_shift")
RESULTS = Path(__file__).resolve().parents[1] / "results"
EVAL_SEED = 3003
SEEDS = (0, 1, 2)
N_VALUES = (1, 16)
SINGLE = (("D1", SharedScaling()), ("D1p", PerFieldScaling()), ("D2", OrderPreserving()), ("D3o@1", MetricDPOfficial(1.0)))
# (label, defense, regime, run_b2 defense index); indices as in run_b2.CONFIGS
LINKED = (("D1", SharedScaling(), "fresh", 0), ("D1p", PerFieldScaling(), "fresh", 1),
          ("D2", OrderPreserving(), "fresh", 2), ("D3o@2", MetricDPOfficial(2.0), "repeat", 8))


def _logit(m: float) -> float:
    return float(np.log(m / (1 - m)))


def _sectors(margin_shift: float = 0.0, price_factor: float = 1.0, spread_factor: float = 1.0):
    out = {}
    for key, s in config.SECTORS.items():
        classes = tuple(dataclasses.replace(c, price_median=c.price_median * price_factor,
                                            price_log_sigma=c.price_log_sigma * spread_factor)
                        for c in s.product_classes)
        margin = float(1 / (1 + np.exp(-(_logit(s.margin_median) + margin_shift))))
        out[key] = dataclasses.replace(s, product_classes=classes, margin_median=margin)
    return out


SCENARIOS = {
    "none": {},
    "margins +0.5 logit": {"SECTORS": _sectors(margin_shift=0.5)},
    "margins -0.5 logit": {"SECTORS": _sectors(margin_shift=-0.5)},
    "price level x1.5": {"SECTORS": _sectors(price_factor=1.5)},
    "price spread x1.5": {"SECTORS": _sectors(spread_factor=1.5)},
    "firm efficiency sd x2": {"FIRM_EFFICIENCY_SIGMA": 2 * config.FIRM_EFFICIENCY_SIGMA},
}


def _shifted_eval_set(base, patches: dict):
    with mock.patch.multiple(config, **patches) if patches else mock.patch.dict({}):
        firms = generate_firms(base.config.n_firms, seed=EVAL_SEED)
        prompts = tuple(EvalPrompt("dev" if i < base.config.n_dev else "test", build_prompt(t, f, cues=base.config.cues))
                        for i, f in enumerate(firms) for t in _tasks_for(i, base.config.tasks_per_firm))
    return dataclasses.replace(base, firms=firms, prompts=prompts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = time.time()
    base = build_eval_set(EvalConfig(seed=EVAL_SEED))  # prior table from prior_seed: unchanged by any scenario
    kw = {**json.loads(FROZEN_CONFIG.read_text())["result"]["frozen"], "n_samples": 4000}
    result = {}
    for name, patches in SCENARIOS.items():
        LOG.info("scenario %s: start", name)
        ev = _shifted_eval_set(base, patches)
        a0 = evaluate(ev, "test", SINGLE[0][1], PriorOnlyAttack(ev.prior), SEEDS, with_oracle=True)
        single = {"A0": summarize(a0, None, args.n_boot)["all"]}
        for label, d in SINGLE:
            rows = evaluate(ev, "test", d, ClassicalBayesAttack(ev.prior, d, use_oracle=True, **kw), SEEDS, True)
            single[label] = summarize(rows, a0, args.n_boot)["all"]
        firms = [(i, f) for i, f in enumerate(ev.firms) if i >= ev.config.n_dev][:N_FIRMS]
        linked = {f"{label}/{regime}": _summary(_rows(ev, firms, label, d, regime, kw, idx, N_VALUES), args.n_boot, N_VALUES)
                  for label, d, regime, idx in LINKED}
        result[name] = {"single_query_test": single, "linked": linked}
    cfg = {"eval_seed": EVAL_SEED, "prior_seed": base.config.prior_seed, "seeds": SEEDS, "n_values": N_VALUES,
           "frozen_solver": kw, "scenarios": {k: sorted(v) for k, v in SCENARIOS.items()},
           "total_seconds": time.time() - t0}
    out = RESULTS / "R042_population_shift.json"
    out.write_text(json.dumps({"run": "R042_population_shift", "config": cfg, "result": result}, indent=2))
    LOG.info("wrote %s (%.0f s)", out, time.time() - t0)


if __name__ == "__main__":
    main()
