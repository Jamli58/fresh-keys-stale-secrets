"""Display checks behind the A-exact appendix: the significant digits of every D2 display the R045/R046 runs
evaluate, and how often A-exact's text-level branch rule (mapped / passed through / year) misreads a shown value.

    uv run python scripts/check_d2_display.py   # writes results/R046_d2_display_check.json
"""

import collections
import json
from pathlib import Path

import numpy as np
from run_b2 import N_FIRMS, N_VALUES, SEEDS

import run_d2_exact as exact
from numpriv.attacks.exact_joint import MIN_DISPLAY_DIGITS, _mapped, _passed_through, infer_digits
from numpriv.defenses import OrderPreserving
from numpriv.defenses.order_preserving import _is_special, _is_year
from numpriv.sequences import build_sequence, disguise_sequence

OUT = Path(__file__).resolve().parents[1] / "results" / "R046_d2_display_check.json"


def _true_branch(value: float) -> str:
    return "year" if _is_year(value) else "special" if _is_special(value) else "mapped"


def _read_branch(value: float) -> str:
    return "mapped" if _mapped(value) else "special" if _passed_through(value) else "year"


def _tally(context: str, fields, view, branches: collections.Counter, digits: collections.Counter,
           misread: list) -> None:
    truth = {f.field_id: float(f.value) for f in fields}
    ys = np.array([float(g.value) for g in view.shown if _mapped(float(g.value))])
    digits[f"{context}|{infer_digits(ys, MIN_DISPLAY_DIGITS) if ys.size else 'none'}"] += 1
    for f in view.shown:
        if f.field_id in truth:
            t, r = _true_branch(truth[f.field_id]), _read_branch(float(f.value))
            branches[f"{t}->{r}"] += 1
            if t != r:
                misread.append({"context": context, "field": f.field_id, "true": truth[f.field_id],
                                "shown": float(f.value)})


def main() -> None:
    branches, digits, misread = collections.Counter(), collections.Counter(), []
    laws = {"default": {}, **exact.LAWS}
    for eval_seed, names in ((1001, list(laws)), (3003, ["default"])):
        for name in names:
            d = OrderPreserving(**laws[name])
            _, firms, _ = exact._setup(eval_seed, d)
            for seed in SEEDS:
                for idx, firm in firms[:N_FIRMS]:
                    specs = build_sequence(firm, max(N_VALUES), start=idx % 5)
                    for regime in ("reuse", "fresh"):
                        rng = np.random.default_rng([seed, idx, exact.B2_D2_DEFENSE_IDX])
                        seq = disguise_sequence(specs, d, regime, rng)
                        for spec, view in zip(specs, seq.views):
                            _tally(f"{eval_seed}/{name}/{regime}", spec.fields, view, branches, digits, misread)
    for _, spec, view in exact._single_views():
        _tally("single", spec.fields, view, branches, digits, misread)
    result = {"branches": dict(branches), "digits": dict(digits), "misread": misread,
              "shown_values": sum(branches.values())}
    OUT.write_text(json.dumps({"result": result}, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "misread"}, indent=1), len(misread))


if __name__ == "__main__":
    main()
