"""Milestone M0 (sanity & fidelity) runs R001-R006. Writes results/<RUN>.json.

LLM-dependent parts (D2 topic shift + remote accuracy, D3 ConvFinQA utility,
paraphrases) are out of scope here and reported as BLOCKED by the caller.
"""

import argparse
import json
import logging
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from numpriv import config as gen_config
from numpriv.attacks import PlaintextAttack, PriorOnlyAttack, make_view, truths_for
from numpriv.config import SECTOR_PARAMETER_PROVENANCE, SECTORS
from numpriv.dataset import PRIOR_DRAW_SIZE, EvalConfig, build_eval_set
from numpriv.defenses import MetricDP, NoDefense, OrderPreserving, PerFieldScaling, SharedScaling
from numpriv.defenses.order_preserving import N_INTERVALS
from numpriv.fields import DISPLAY_SIG_DIGITS, NumField
from numpriv.finqa import answers_match, execute_program, is_arithmetic_only, program_literals, substitute_literals
from numpriv.metrics import hit_rate, mean_abs_error
from numpriv.priors import label_stratum
from numpriv.templates import format_field, render

LOG = logging.getLogger("m0")
RESULTS = Path(__file__).resolve().parents[1] / "results"
FINQA = Path(__file__).resolve().parents[1] / "data" / "raw" / "finqa_test.json"


def _save(run: str, payload: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{run}.json").write_text(json.dumps(payload, indent=2, default=float))
    LOG.info("wrote %s", RESULTS / f"{run}.json")


def r001(cfg: EvalConfig) -> dict:
    ev = build_eval_set(cfg)
    by_sector = defaultdict(list)
    for f in ev.firms:
        by_sector[f.sector].extend(p.margin for p in f.products)
    strata = Counter(label_stratum(f, ev.prior) for f in ev.firms)
    share = strata["atypical"] / len(ev.firms)
    return {
        "sector_parameters": SECTOR_PARAMETER_PROVENANCE,
        "n_firms": len(ev.firms),
        "firms_per_sector": dict(Counter(f.sector for f in ev.firms)),
        "margin_median_generated_vs_spec": {
            s: {"generated": statistics.median(m), "spec": SECTORS[s].margin_median} for s, m in by_sector.items()
        },
        "atypical_share": share,
        "atypical_share_within_20pct_pm3pp": abs(share - 0.20) <= 0.03,
        "latent_extreme_share": sum(f.latent_extreme for f in ev.firms) / len(ev.firms),
        "cues": _cue_report(ev.firms),
    }


def _cue_report(firms) -> dict:
    """R028: how strongly the shown cues (city, positioning) signal the hidden price level."""
    resid = np.array([
        np.mean([np.log(p.price / next(c.price_median for c in SECTORS[f.sector].product_classes
                                         if c.name == p.product_class)) for p in f.products])
        for f in firms
    ])
    rpp = np.log([f.city_rpp for f in firms])
    positions = ("discount", "standard", "premium")
    return {
        "corr_log_city_rpp_vs_log_price_residual": float(np.corrcoef(rpp, resid)[0, 1]),
        "positioning_share": {p: sum(f.positioning == p for f in firms) / len(firms) for p in positions},
        "mean_log_price_residual_by_positioning": {
            p: float(resid[[f.positioning == p for f in firms]].mean()) for p in positions
        },
        "n_distinct_cities": len({f.city for f in firms}),
    }


def _shown_numbers(fields: tuple[NumField, ...]) -> dict[str, float]:
    return {f.field_id: float(format_field(f).replace(",", "").rstrip("%")) for f in fields}


def _inverted(defense, secret, field_id: str, shown: float) -> float:
    k = secret.k if isinstance(defense, SharedScaling) else secret.k_by_field[field_id]
    return shown / k


def r002(cfg: EvalConfig, seeds: tuple[int, ...]) -> dict:
    """Invert what the attacker sees (rendered, whole-cent money) with the client's key."""
    ev = build_eval_set(cfg)
    out = {}
    for defense in (SharedScaling(), PerFieldScaling()):
        exact_cents, rel_err, margin_err = [], [], []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for ep in ev.prompts:
                secret = defense.sample_secret(rng, ep.spec.fields)
                d = defense.apply(ep.spec.fields, secret, rng)
                shown = _shown_numbers(d.fields)
                for f in ep.spec.fields:
                    if f.kind == "money":
                        back = _inverted(defense, secret, f.field_id, shown[f.field_id])
                        exact_cents.append(round(back, 2) == f.value)
                        rel_err.append(abs(back - f.value) / f.value)
                for i, p in enumerate(ep.spec.products):
                    m = (shown[f"price:{i}"] - shown[f"cost:{i}"]) / shown[f"price:{i}"]
                    margin_err.append(abs(m - p.margin))
        out[defense.name] = {
            "money_inverted_exact_to_cent_rate": float(np.mean(exact_cents)),
            "money_inverted_rel_err_median": float(np.median(rel_err)),
            "money_inverted_rel_err_p99": float(np.quantile(rel_err, 0.99)),
            "rendered_margin_abs_err_median": float(np.median(margin_err)),
            "rendered_margin_abs_err_p99": float(np.quantile(margin_err, 0.99)),
            "n_money_values": len(exact_cents),
        }
    return out


# Thousands separators only in 3-digit groups, so "31,2011" (a date) is two numbers, not 312011.
_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?!\d)(?:\.\d+)?%?|-?\d+(?:\.\d+)?%?")


def _literal_field(i: int, token: str) -> NumField:
    raw = token.replace(",", "")
    return NumField(f"n{i}", "percent" if raw.endswith("%") else "money", float(raw.rstrip("%")))


def _display(field: NumField) -> str:
    """The disguised value exactly as the defense produced it (no re-rounding)."""
    return f"{field.value!r}" + ("%" if field.kind == "percent" else "")


def r003a(seed: int, sample: int) -> dict:
    data = json.loads(FINQA.read_text())
    arithmetic = [x for x in data if is_arithmetic_only(x["qa"]["program"])]
    rng = np.random.default_rng(seed)
    picked = [arithmetic[i] for i in rng.choice(len(arithmetic), size=min(sample, len(arithmetic)), replace=False)]
    defense = OrderPreserving()
    stats = Counter()
    for item in picked:
        program, gold = item["qa"]["program"], item["qa"]["exe_ans"]
        stats["executor_matches_gold"] += answers_match(execute_program(program), gold)
        evidence = " ".join(item["qa"].get("gold_inds", {}).values())
        tokens = list(dict.fromkeys(program_literals(program) + tuple(_NUM_RE.findall(evidence))))
        fields = tuple(_literal_field(i, t) for i, t in enumerate(tokens))
        d = defense.apply(fields, defense.sample_secret(rng, fields), rng)
        to_disguised = {t: _display(f) for t, f in zip(tokens, d.fields)}
        to_original = {v: t for t, v in to_disguised.items()}
        # A collision is two *distinct values* sharing one display (same value, different format, is fine).
        collided = len({_display(f) for f in d.fields}) < len({(f.value, f.kind) for f in fields})
        stats["display_collision"] += collided
        remote_program = substitute_literals(program, to_disguised)  # stand-in for a perfect remote model
        restored = substitute_literals(remote_program, to_original)
        stats["reconstruction_matches_gold"] += answers_match(execute_program(restored), gold)
        stats["reconstruction_equals_original_execution"] += (
            not collided and execute_program(restored) == execute_program(program)
        )
        stats["disguised_program_matches_gold"] += answers_match(execute_program(remote_program), gold)
        stats["has_year_range_integer"] += any(f.value.is_integer() and 1990 <= f.value <= 2030 for f in fields)
        stats["has_special_28_31"] += any(f.value.is_integer() and 28 <= f.value <= 31 for f in fields)
    n = len(picked)
    return {
        "finqa_test_items": len(data),
        "arithmetic_only_items": len(arithmetic),
        "sampled": n,
        **{k: v / n for k, v in stats.items()},
        "note": "LLM-free: gold program stands in for remote code. Zhang et al. accuracy needs an LLM (BLOCKED).",
    }


def r004a(cfg: EvalConfig, seeds: tuple[int, ...], epsilons: tuple[float, ...]) -> dict:
    ev = build_eval_set(cfg)
    out = {}
    for eps in epsilons:
        defense = MetricDP(epsilon=eps)
        rel_err = defaultdict(list)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for ep in ev.prompts:
                shown = {f.field_id: f.value for f in defense.apply(ep.spec.fields, None, rng).fields}
                for i, p in enumerate(ep.spec.products):
                    rel_err[ep.spec.sector].append(abs(shown[f"price:{i}"] - p.price) / p.price)
        out[f"eps={eps}"] = {s: float(np.median(v)) for s, v in rel_err.items()}
    return {"median_relative_price_error_by_sector": out, "note": "ConvFinQA utility trend needs an LLM (BLOCKED)."}


def r005(cfg: EvalConfig, seed: int, n: int) -> dict:
    ev = build_eval_set(cfg)
    rng = np.random.default_rng(seed)
    lines = []
    for ep in ev.prompts[:n]:
        lines.append(f"## {ep.spec.firm_id} / {ep.spec.task}\n\n**Original:** {render(ep.spec)}\n")
        for defense in (SharedScaling(), OrderPreserving(), MetricDP(1.0)):
            d = defense.apply(ep.spec.fields, defense.sample_secret(rng, ep.spec.fields), rng)
            lines.append(f"**{defense.name}:** {render(ep.spec, d.fields)}\n")
    stripped = build_eval_set(EvalConfig(**{**cfg.__dict__, "cues": False}))
    lines.append("## Cue-stripped examples\n")
    lines += [f"- {render(ep.spec)}" for ep in stripped.prompts[:5]]
    (RESULTS / "R005_samples.md").write_text("\n".join(lines))
    return {"samples_file": "results/R005_samples.md", "note": "Paraphrases need a local LLM (BLOCKED); manual check pending."}


def r006(cfg: EvalConfig) -> dict:
    ev = build_eval_set(cfg)
    rng = np.random.default_rng(0)
    out = {}
    for attack in (PriorOnlyAttack(ev.prior), PlaintextAttack()):
        est, tru = [], []
        for ep in (p for p in ev.prompts if p.split == "test"):
            shown = NoDefense().apply(ep.spec.fields, None, rng).fields
            est += attack.attack(make_view(ep.spec, shown, NoDefense.name, with_oracle=attack.name.startswith("A0")))
            tru += truths_for(ep.spec)
        em, tm = np.array([e.margin for e in est]), np.array([t.margin for t in tru])
        ec, tc = np.array([e.unit_cost for e in est]), np.array([t.unit_cost for t in tru])
        out[attack.name] = {
            "margin_hit_2pp": hit_rate(em, tm, abs_tol=0.02),
            "margin_hit_5pp": hit_rate(em, tm, abs_tol=0.05),
            "margin_mae_pp": 100 * mean_abs_error(em, tm),
            "unit_cost_hit_10pct": hit_rate(ec, tc, rel_tol=0.10),
            "n_products": len(est),
        }
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["R001", "R002", "R003a", "R004a", "R005", "R006"])
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--prior-seed", type=int, default=2002)
    ap.add_argument("--defense-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--finqa-sample", type=int, default=100)
    ap.add_argument("--epsilons", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    args = ap.parse_args()
    cfg = EvalConfig(seed=args.seed, prior_seed=args.prior_seed)
    seeds = tuple(args.defense_seeds)
    runners = {
        "R001": lambda: r001(cfg),
        "R002": lambda: r002(cfg, seeds),
        "R003a": lambda: r003a(seeds[0], args.finqa_sample),
        "R004a": lambda: r004a(cfg, seeds, tuple(args.epsilons)),
        "R005": lambda: r005(cfg, seeds[0], 6),
        "R006": lambda: r006(cfg),
    }
    settings = {
        **cfg.__dict__,
        "defense_seeds": seeds,
        "finqa_sample": args.finqa_sample,
        "epsilons": args.epsilons,
        "prior_draw_size": PRIOR_DRAW_SIZE,
        "extreme_share": 0.2,
        "d1_k_range": [SharedScaling().k_low, SharedScaling().k_high],
        "d2_target_range": [OrderPreserving().target_low, OrderPreserving().target_high],
        "d2_n_intervals": N_INTERVALS,
        "d3_domain": {k: list(v) for k, v in MetricDP(1.0).domain.items()},
        "display_sig_digits": DISPLAY_SIG_DIGITS,
        "sector_parameters": SECTOR_PARAMETER_PROVENANCE,
        "generator_extreme_efficiency_range": list(gen_config.EXTREME_EFFICIENCY_RANGE),
    }
    for run in args.runs:
        _save(run, {"run": run, "config": settings, "result": runners[run]()})


if __name__ == "__main__":
    main()
