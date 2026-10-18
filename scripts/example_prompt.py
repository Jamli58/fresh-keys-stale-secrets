"""Example prompt for the paper appendix: one test firm's margin-review prompt, plain and under each disguise.

Writes results/example_prompt.json (eval seed 1001, first test firm, defense rng seed 0).
    uv run python scripts/example_prompt.py
"""

import json
from pathlib import Path

import numpy as np

from numpriv.dataset import EvalConfig, build_eval_set
from numpriv.defenses import MetricDPOfficial, OrderPreserving, PerFieldScaling, SharedScaling
from numpriv.templates import build_prompt, render

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ev = build_eval_set(EvalConfig())
    firm = ev.firms[ev.config.n_dev]
    spec = build_prompt("margin_review", firm, cues=True)
    out = {"firm_id": firm.firm_id, "sector": firm.sector, "task": spec.task, "plain": render(spec),
           "truth": [{"price": p.price, "unit_cost": p.unit_cost, "margin": p.margin} for p in spec.products]}
    for label, d in (("D1", SharedScaling()), ("D1p", PerFieldScaling()), ("D2", OrderPreserving()),
                     ("D3o@1", MetricDPOfficial(1.0))):
        rng = np.random.default_rng(0)
        out[label] = render(spec, d.apply(spec.fields, d.sample_secret(rng, spec.fields), rng).fields)
    (RESULTS / "example_prompt.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
