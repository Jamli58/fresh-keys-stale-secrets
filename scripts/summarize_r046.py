"""Summarise R046 under Amendment 2 (refine-logs/EXPERIMENT_PLAN_20261009_exact_joint.md).

Decision endpoints are margin-based and reported per A-exact seed; a conclusion needs both seeds to agree in sign
and in whether the firm-clustered 95% CI excludes 0. Unit-cost results are listed with their between-seed spread
as numerical error and carry no conclusion.
    uv run python scripts/summarize_r046.py
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEEDS = (4646, 4647)
M, C = "margin_hit_5pp", "unit_cost_hit_10pct"


def _load(part: str) -> dict[int, dict]:
    return {s: json.loads((RESULTS / f"R046_d2_exact_{part}_seed{s}.json").read_text())["result"] for s in SEEDS}


def _fmt(est: float, ci) -> str:
    return f"{100 * est:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] pp"


def _verdict(cells: dict[int, tuple[float, tuple[float, float]]]) -> str:
    signs = {s: (e > 0) - (e < 0) for s, (e, _) in cells.items()}
    excl = {s: ci[0] > 0 or ci[1] < 0 for s, (_, ci) in cells.items()}
    if len(set(signs.values())) > 1 or len(set(excl.values())) > 1:
        return "seeds disagree: no conclusion"
    return ("CI excludes 0" if all(excl.values()) else "CI includes 0") + " in both seeds"


def _endpoint(name: str, cells: dict[int, tuple[float, tuple[float, float]]]) -> list[str]:
    spread = 100 * abs(cells[SEEDS[0]][0] - cells[SEEDS[1]][0])
    lines = [f"{name}: " + " | ".join(f"seed {s} {_fmt(*cells[s])}" for s in SEEDS)
             + f" | numerical spread {spread:.1f} pp | {_verdict(cells)}"]
    return lines


def _gain(summary: dict, metric: str) -> tuple[float, tuple[float, float]]:
    return summary[f"gain_{metric}_n16_vs_n1"], tuple(summary[f"gain_{metric}_n16_vs_n1_ci"])


def linked(part: str) -> list[str]:
    r = _load(part)
    out = [f"== {part} =="]
    for metric, tag in ((M, "margin (decision)"), (C, "unit cost (reference only)")):
        out += _endpoint(f"E1 fresh(joint) - reuse(A-exact), {tag}",
                         {s: (r[s]["fresh_joint_minus_reuse_exact"][metric]["contrast"],
                              tuple(r[s]["fresh_joint_minus_reuse_exact"][metric]["contrast_ci"])) for s in SEEDS})
        out += _endpoint(f"E2 reuse gain n1->16 under A-exact, {tag}",
                         {s: _gain(r[s]["D2/reuse/A-exact"], metric) for s in SEEDS})
    for s in SEEDS:
        ru = r[s]["D2/reuse/A-exact"]
        out.append(f"   seed {s}: reuse A-exact margin hit n1 {100 * ru['1'][M]:.1f}% -> n16 {100 * ru['16'][M]:.1f}%;"
                   f" min-ESS quantiles {r[s]['diagnostics']['min_ess_quantiles']}")
    fr = r[SEEDS[0]]["D2/fresh/joint"]
    out.append(f"   fresh joint (seed-independent) margin hit n1 {100 * fr['1'][M]:.1f}% -> n16 {100 * fr['16'][M]:.1f}%,"
               f" gain {_fmt(*_gain(fr, M))}")
    return out


def single() -> list[str]:
    r = _load("single")
    out = ["== single (product 0 of every test prompt) =="]
    for subset in ("all", "margin_review"):
        for metric, tag in ((M, "margin (decision)"), (C, "unit cost (reference only)")):
            for other in ("A0", "pair", "joint"):
                key = f"exact_minus_{other}_{metric}"
                out += _endpoint(f"E3 [{subset}] A-exact - {other}, {tag}",
                                 {s: (r[s][subset][key], tuple(r[s][subset][key + "_ci"])) for s in SEEDS})
        rates = {s: {k: 100 * r[s][subset][f"{k}_{M}"] for k in ("A0", "pair", "joint", "exact")} for s in SEEDS}
        out.append(f"   [{subset}] margin hit rates: " + "; ".join(
            f"seed {s} " + ", ".join(f"{k} {v:.1f}%" for k, v in rates[s].items()) for s in SEEDS))
    return out


def law() -> list[str]:
    r = _load("law")
    out = ["== law (reuse only, E4) =="]
    for name in r[SEEDS[0]]:
        out += _endpoint(f"E4 {name}: reuse gain n1->16, margin",
                         {s: _gain(r[s][name]["D2/reuse/A-exact"], M) for s in SEEDS})
    return out


def main() -> None:
    lines = []
    for fn in (lambda: linked("linked"), lambda: linked("fresh_draw"), single, law):
        try:
            lines += fn()
        except FileNotFoundError as e:
            lines.append(f"(missing: {e.filename})")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
