"""Ceiling for C1: A4 given *perfect* cue knowledge instead of an LLM's reading of the prose.

Same frozen solver and the true sector/classes (as A1+), plus a firm price-level shift from
the generator's own latents:
  city  : log(city RPP / 100)                       (what a perfect city reader could know)
  latent: log(city RPP / 100) + brand premium       (everything but the city residual)
Delta vs A1+ on the same views, paired firm bootstrap (summarize with A1+ rows as the floor).
    uv run python scripts/cue_ceiling.py [dev|test]
"""

import json
import math
import sys
from dataclasses import dataclass

from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.attacks.hybrid import LLMPrior, build_hypotheses
from numpriv.attacks.view import AttackerView, Estimate
from numpriv.dataset import EvalConfig, build_eval_set
from numpriv.defenses import MetricDP, OrderPreserving, SharedScaling
from numpriv.evaluate import evaluate, summarize
from numpriv.templates import render_masked

FROZEN = {"prior_df": 3.0, "prior_scale": 0.75, "n_samples": 4000}
METRICS = ("margin_hit_5pp", "unit_cost_hit_10pct")


@dataclass(frozen=True)
class OracleCueAttack:
    solver: ClassicalBayesAttack
    priors: dict  # masked text -> LLMPrior
    name: str = "A4_oracle_cues"

    def attack(self, view: AttackerView) -> tuple[Estimate, ...]:
        return self.solver.solve_with(view, build_hypotheses(self.solver, self.priors[view.masked_text]))


def priors_for(ev, mode: str) -> dict:
    firms = {f.firm_id: f for f in ev.firms}
    out = {}
    for ep in ev.prompts:
        spec, firm = ep.spec, firms[ep.spec.firm_id]
        shift = math.log(firm.city_rpp / 100.0) + (firm.premium if mode == "latent" else 0.0)
        out[render_masked(spec)] = LLMPrior(spec.sector, tuple(p.product_class for p in spec.products), shift, 0.0)
    return out


def main(split: str) -> None:
    ev = build_eval_set(EvalConfig())
    seeds = (0,) if split == "dev" else (0, 1, 2)
    report = {}
    for d in (SharedScaling(), MetricDP(0.5), MetricDP(1.0), OrderPreserving()):
        a1p = ClassicalBayesAttack(ev.prior, d, use_oracle=True, **FROZEN)
        base = evaluate(ev, split, d, a1p, seeds, with_oracle=True)
        report[d.name] = {"A1+": {m: summarize(base, None, 10)["all"][m] for m in METRICS}}
        for mode in ("city", "latent"):
            atk = OracleCueAttack(ClassicalBayesAttack(ev.prior, d, use_oracle=False, **FROZEN), priors_for(ev, mode))
            rows = evaluate(ev, split, d, atk, seeds, with_oracle=False)
            s = summarize(rows, base, 2000)["all"]
            report[d.name][mode] = {m: (round(s[m], 3), round(s["advantage_" + m], 3),
                                        [round(x, 3) for x in s["advantage_" + m + "_ci"]]) for m in METRICS}
        print(d.name, json.dumps(report[d.name]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
