"""R049 (plan: EXPERIMENT_PLAN_20261026_r049.md): prior-model mismatch, value drift, all test firms.

Parts:
  mismatch      R049a: A1+ single query and linked queries with the prior table built from another population model
  joint         R049a: D2 joint attacker, fresh vs reuse, under prior model PM3
  drift         R049b: fresh-secret linked queries when prices and costs drift between queries
  power_sector  R049c: per-sector gains from R049c_B2_all_test_rows.jsonl (run run_b2.py with --n-firms 250 first)
The evaluated firms, defense seeds and disguise streams are those of R017 / R033 / R045, so rows are paired.
    uv run python scripts/run_r049.py --part mismatch
"""

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path
from unittest import mock

import numpy as np
from run_b2 import CONFIGS, FROZEN_CONFIG, SEEDS, _rows, _summary
from run_d2_joint import B2_D2_DEFENSE_IDX, _contrast

from numpriv import config
from numpriv.attacks import PriorOnlyAttack
from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.attacks.joint_map import JointMapAttack
from numpriv.attacks.multi_query import MultiQueryAttack
from numpriv.dataset import PRIOR_DRAW_SIZE, EvalConfig, build_eval_set
from numpriv.defenses import MetricDPOfficial, OrderPreserving, PerFieldScaling
from numpriv.evaluate import COST_REL_TOL, MARGIN_TOL_5PP, evaluate, summarize
from numpriv.metrics import bootstrap_ci
from numpriv.priors import build_prior_table
from numpriv.records import generate_firms
from numpriv.sequences import disguise_sequence
from numpriv.templates import TASKS, build_prompt

LOG = logging.getLogger("r049")


class _FallbackCounter(logging.Handler):
    """Counts the attackers' fallback warnings (no consistent key or prior sample: the estimate reverts to the prior)."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "consistent" in record.getMessage():
            self.count += 1


def _counted(fn, *args, **kwargs):
    """Run fn and return (result, number of fallback warnings it logged)."""
    handler = _FallbackCounter()
    logging.getLogger().addHandler(handler)
    try:
        return fn(*args, **kwargs), handler.count
    finally:
        logging.getLogger().removeHandler(handler)
RESULTS = Path(__file__).resolve().parents[1] / "results"
N_FIRMS = 60
N_VALUES = (1, 16)
LOCATION_SEED = 4949
DRIFT_TAG = 4949
DRIFTS = (0.01, 0.02, 0.05)
# (label, run_b2 config index, regime): the reported targets plus the reuse regimes of the keyed mechanisms
LINKED = (("D1", 0, "fresh"), ("D1p", 1, "fresh"), ("D2", 2, "fresh"),
          ("D3o@0.5", 6, "repeat"), ("D3o@1", 7, "repeat"), ("D3o@2", 8, "repeat"),
          ("D1", 0, "reuse"), ("D1p", 1, "reuse"), ("D2", 2, "reuse"))
DRIFT_LINKED = LINKED[:6]
SINGLE = (("D1p", PerFieldScaling()), ("D2", OrderPreserving()), ("D3o@1", MetricDPOfficial(1.0)))
REPORTED = {"D1/fresh": "unit_cost_hit_10pct", "D1p/fresh": "unit_cost_hit_10pct", "D2/fresh": "margin_hit_5pp",
            "D3o@0.5/repeat": "margin_hit_5pp", "D3o@1/repeat": "margin_hit_5pp", "D3o@2/repeat": "margin_hit_5pp"}


def _location_sectors() -> dict:
    """PM2: every class price median x exp(N(0, 0.25^2)), every sector margin median +N(0, 0.3^2) on the logit scale."""
    rng = np.random.default_rng(LOCATION_SEED)
    out = {}
    for key, s in config.SECTORS.items():
        classes = tuple(dataclasses.replace(c, price_median=c.price_median * float(np.exp(rng.normal(0.0, 0.25))))
                        for c in s.product_classes)
        logit = float(np.log(s.margin_median / (1 - s.margin_median))) + float(rng.normal(0.0, 0.3))
        out[key] = dataclasses.replace(s, product_classes=classes, margin_median=float(1 / (1 + np.exp(-logit))))
    return out


PM1 = {"patches": {"PRICE_NOISE_DF": 30, "FIRM_PREMIUM_SIGMA": 1.5 * config.FIRM_PREMIUM_SIGMA,
                   "CITY_RESIDUAL_SIGMA": 1.5 * config.CITY_RESIDUAL_SIGMA,
                   "FIRM_EFFICIENCY_SIGMA": 1.5 * config.FIRM_EFFICIENCY_SIGMA}, "extreme_share": 0.0}
PM2 = {"patches": {"SECTORS": _location_sectors()}, "extreme_share": 0.2}
PRIOR_MODELS = {"PM1": PM1, "PM2": PM2,
                "PM3": {"patches": {**PM1["patches"], **PM2["patches"]}, "extreme_share": 0.0}}


def _prior_table(model: dict, prior_seed: int):
    with mock.patch.multiple(config, **model["patches"]):
        firms = generate_firms(PRIOR_DRAW_SIZE, seed=prior_seed, extreme_share=model["extreme_share"])
    return build_prior_table(firms)


def _mismatched_eval_set(name: str):
    base = build_eval_set(EvalConfig())
    return dataclasses.replace(base, prior=_prior_table(PRIOR_MODELS[name], base.config.prior_seed))


def _test_firms(ev, n_firms: int = N_FIRMS):
    return [(i, f) for i, f in enumerate(ev.firms) if i >= ev.config.n_dev][:n_firms]


def _solver_kw() -> dict:
    return {**json.loads(FROZEN_CONFIG.read_text())["result"]["frozen"], "n_samples": 4000}


def _mismatch(n_boot: int) -> dict:
    kw, out = _solver_kw(), {}
    for name in PRIOR_MODELS:
        LOG.info("prior model %s: start", name)
        ev = _mismatched_eval_set(name)
        a0 = evaluate(ev, "test", SINGLE[0][1], PriorOnlyAttack(ev.prior), SEEDS, with_oracle=True)
        single = {"A0": summarize(a0, None, n_boot)["all"]}
        for label, d in SINGLE:
            rows = evaluate(ev, "test", d, ClassicalBayesAttack(ev.prior, d, use_oracle=True, **kw), SEEDS, True)
            single[label] = summarize(rows, a0, n_boot)["all"]
        firms = _test_firms(ev)
        linked = {}
        for label, idx, regime in LINKED:
            rows, fallbacks = _counted(_rows, ev, firms, label, CONFIGS[idx][1], regime, kw, idx, N_VALUES)
            linked[f"{label}/{regime}"] = {**_summary(rows, n_boot, N_VALUES), "n_fallback_warnings": fallbacks}
        out[name] = {"single_query_test": single, "linked": linked}
    return out


def _joint(n_boot: int) -> dict:
    kw = _solver_kw()
    ev = _mismatched_eval_set("PM3")
    d = OrderPreserving()
    solver = ClassicalBayesAttack(ev.prior, d, use_oracle=True, **kw)
    rows = {regime: _rows(ev, _test_firms(ev), "D2", d, regime, kw, B2_D2_DEFENSE_IDX, N_VALUES,
                          attack=JointMapAttack(solver, regime)) for regime in ("fresh", "reuse")}
    out = {f"D2/{r}": _summary(v, n_boot, N_VALUES) for r, v in rows.items()}
    out["fresh_minus_reuse"] = _contrast(rows["fresh"], rows["reuse"], n_boot, N_VALUES)
    return {"PM3": out}


def _drifted(firm, rng: np.random.Generator, delta: float):
    products = []
    for p in firm.products:
        price = round(max(0.5, p.price * float(np.exp(rng.normal(0.0, delta)))), 2)
        cost = round(p.unit_cost * float(np.exp(rng.normal(0.0, delta))), 2)
        products.append(dataclasses.replace(p, price=price, unit_cost=min(max(cost, 0.01), round(price - 0.01, 2))))
    return dataclasses.replace(firm, products=tuple(products))


def _drift_rows(ev, firms, label, defense, regime, kw, defense_idx, delta):
    """As run_b2._rows, but query j shows drifted values; hits are scored against the undrifted values."""
    attack = MultiQueryAttack(ClassicalBayesAttack(ev.prior, defense, use_oracle=True, **kw), regime)
    floor = PriorOnlyAttack(ev.prior)
    rows = []
    for seed in SEEDS:
        for idx, firm in firms:
            drng = np.random.default_rng([seed, idx, DRIFT_TAG, int(round(1000 * delta))])
            specs = tuple(build_prompt(TASKS[(idx % 5 + j) % len(TASKS)],
                                       _drifted(firm, drng, delta) if delta > 0 else firm, cues=True)
                          for j in range(max(N_VALUES)))
            seq = disguise_sequence(specs, defense, regime, np.random.default_rng([seed, idx, defense_idx]))
            truth = firm.products[0]
            a0 = floor.attack(seq.views[0])[0]
            for n in N_VALUES:
                est = attack.attack(seq.views[:n])
                rows.append({
                    "firm_id": firm.firm_id, "seed": seed, "n": n,
                    "margin_hit_5pp": abs(est.margin - truth.margin) <= MARGIN_TOL_5PP,
                    "unit_cost_hit_10pct": abs(est.unit_cost - truth.unit_cost) <= COST_REL_TOL * truth.unit_cost,
                    "key_log_error": None,
                    "a0_margin_hit_5pp": abs(a0.margin - truth.margin) <= MARGIN_TOL_5PP,
                    "a0_unit_cost_hit_10pct": abs(a0.unit_cost - truth.unit_cost) <= COST_REL_TOL * truth.unit_cost,
                })
    LOG.info("drift %.2f %s %s: %d rows", delta, label, regime, len(rows))
    return rows


def _drift(n_boot: int, drifts=DRIFTS, n_firms: int = N_FIRMS) -> dict:
    kw = _solver_kw()
    ev = build_eval_set(EvalConfig())
    firms = _test_firms(ev, n_firms)
    out = {}
    for delta in drifts:
        for label, idx, regime in DRIFT_LINKED:
            rows, fallbacks = _counted(_drift_rows, ev, firms, label, CONFIGS[idx][1], regime, kw, idx, delta)
            out.setdefault(f"delta={delta}", {})[f"{label}/{regime}"] = {
                **_summary(rows, n_boot, N_VALUES), "n_fallback_warnings": fallbacks}
    return out


def _power_sector(n_boot: int) -> dict:
    ev = build_eval_set(EvalConfig())
    sector = {f.firm_id: f.sector for f in ev.firms}
    rows = [json.loads(line) for line in (RESULTS / "R049c_B2_all_test_rows.jsonl").read_text().splitlines()]
    out = {}
    for cfg, metric in REPORTED.items():
        rs = [r for r in rows if r["config"] == cfg]
        for sec in sorted(set(sector.values())):
            by = {(r["firm_id"], r["seed"], r["n"]): float(r[metric]) for r in rs if sector[r["firm_id"]] == sec}
            keys = sorted({(f, s) for f, s, _ in by})
            diff = np.array([by[(f, s, 16)] - by[(f, s, 1)] for f, s in keys])
            out.setdefault(cfg, {})[sec] = {"metric": metric, "n_firms": len({k[0] for k in keys}),
                                            "gain": float(diff.mean()),
                                            "gain_ci": bootstrap_ci(diff, np.array([k[0] for k in keys]), n_boot=n_boot)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("mismatch", "joint", "drift", "power_sector"), required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true", help="drift part: delta 0 and 0.05 on 5 firms (check only)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = time.time()
    if args.smoke:
        result = _drift(args.n_boot, (0.0, 0.05), 5)
    else:
        result = {"mismatch": lambda: _mismatch(args.n_boot), "joint": lambda: _joint(args.n_boot),
                  "drift": lambda: _drift(args.n_boot), "power_sector": lambda: _power_sector(args.n_boot)}[args.part]()
    cfg = {"part": args.part, "plan": "refine-logs/EXPERIMENT_PLAN_20261026_r049.md", "eval_seed": EvalConfig().seed,
           "prior_seed": EvalConfig().prior_seed, "n_firms": N_FIRMS, "seeds": SEEDS, "n_values": N_VALUES,
           "prior_models": {k: {"patches": sorted(v["patches"]), "extreme_share": v["extreme_share"]}
                            for k, v in PRIOR_MODELS.items()},
           "location_seed": LOCATION_SEED, "drifts": DRIFTS, "smoke": args.smoke, "total_seconds": time.time() - t0}
    name = "R049_smoke" if args.smoke else {"mismatch": "R049a_prior_mismatch", "joint": "R049a_joint_pm3",
                                            "drift": "R049b_drift", "power_sector": "R049c_power_sector"}[args.part]
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps({"run": name, "config": cfg, "result": result}, indent=2))
    LOG.info("wrote %s (%.0f s)", out, time.time() - t0)


if __name__ == "__main__":
    main()
