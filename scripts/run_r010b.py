"""R010b: self-consistency of the frozen classical attacker on every disguised defense."""

import json
import logging
from pathlib import Path

from numpriv.dataset import EvalConfig, build_eval_set
from numpriv.defenses import MetricDP, OrderPreserving, PerFieldScaling, SharedScaling
from numpriv.diagnostics import self_consistency_gain

LOG = logging.getLogger("r010b")
RESULTS = Path(__file__).resolve().parents[1] / "results"
N_DRAWS = 2000


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frozen = json.loads((RESULTS / "R009.json").read_text())["config"]["frozen_attacker_config"]
    ev = build_eval_set(EvalConfig())
    out = {}
    for d in (SharedScaling(), PerFieldScaling(), OrderPreserving(), MetricDP(0.5), MetricDP(1.0), MetricDP(2.0)):
        label = f"{d.name}@eps={d.epsilon}" if isinstance(d, MetricDP) else d.name
        out[label] = self_consistency_gain(ev.prior, d, n=N_DRAWS, seed=0, **frozen)
        LOG.info("%s %s", label, out[label])
    broken = [k for k, v in out.items() if v["gain_ci"][1] < 0]
    payload = {"run": "R010b", "config": {"n_draws": N_DRAWS, "n_boot": 10_000, "frozen_attacker_config": frozen},
               "result": {"per_defense": out, "attacker_broken_on": broken, "pass": not broken}}
    (RESULTS / "R010b.json").write_text(json.dumps(payload, indent=2, default=float))


if __name__ == "__main__":
    main()
