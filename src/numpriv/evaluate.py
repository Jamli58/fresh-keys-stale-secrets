"""Run an attacker against a defense over an eval split and summarize leakage.

Defense randomness is seeded per (seed, prompt index), so every attacker sees the
*same* disguised prompts: comparisons between attackers are paired. LLM attackers
can run with ``workers`` > 1 (concurrent requests); rows keep the serial order.
A failed estimate (NaN) counts as a miss in every hit rate; error averages are
over the estimates that exist, and ``n_failed`` reports the rest.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from numpriv.attacks import make_view, truths_for
from numpriv.dataset import EvalSet
from numpriv.metrics import bootstrap_ci
from numpriv.priors import label_stratum

MARGIN_TOL_2PP, MARGIN_TOL_5PP, COST_REL_TOL = 0.02, 0.05, 0.10


@dataclass(frozen=True)
class Row:
    firm_id: str
    task: str
    seed: int
    product_index: int
    stratum: str
    margin_est: float
    margin_true: float
    cost_est: float
    cost_true: float
    status: str = "ok"
    margin_interval: tuple[float, float] | None = None
    cost_interval: tuple[float, float] | None = None

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (self.firm_id, self.task, self.seed, self.product_index)


def _views(ev: EvalSet, split: str, defense, seeds: tuple[int, ...], with_oracle: bool) -> list[tuple]:
    out = []
    for seed in seeds:
        for idx, ep in enumerate(ev.prompts):
            if ep.split != split:
                continue
            rng = np.random.default_rng([seed, idx])
            shown = defense.apply(ep.spec.fields, defense.sample_secret(rng, ep.spec.fields), rng).fields
            out.append((seed, ep.spec, make_view(ep.spec, shown, defense.name, with_oracle=with_oracle)))
    return out


def evaluate(ev: EvalSet, split: str, defense, attack, seeds: tuple[int, ...], with_oracle: bool,
             workers: int = 1) -> tuple[Row, ...]:
    strata = {f.firm_id: label_stratum(f, ev.prior) for f in ev.firms}
    views = _views(ev, split, defense, seeds, with_oracle)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            estimates = list(pool.map(lambda v: attack.attack(v[2]), views))
    else:
        estimates = [attack.attack(v[2]) for v in views]
    rows: list[Row] = []
    for (seed, spec, _), est in zip(views, estimates):
        for e, t in zip(est, truths_for(spec)):
            rows.append(Row(spec.firm_id, spec.task, seed, e.product_index, strata[spec.firm_id],
                            e.margin, t.margin, e.unit_cost, t.unit_cost, e.status, e.margin_interval,
                            e.cost_interval))
    return tuple(rows)


def _hits(rows: tuple[Row, ...]) -> dict[str, np.ndarray]:
    m_err = np.array([abs(r.margin_est - r.margin_true) for r in rows])
    c_rel = np.array([abs(r.cost_est - r.cost_true) / r.cost_true for r in rows])
    return {
        "margin_hit_2pp": (m_err <= MARGIN_TOL_2PP).astype(float),
        "margin_hit_5pp": (m_err <= MARGIN_TOL_5PP).astype(float),
        "unit_cost_hit_10pct": (c_rel <= COST_REL_TOL).astype(float),
        "_m_err": m_err,
        "_c_rel": c_rel,
    }


def _nan_stat(fn, values: np.ndarray) -> float:
    return float(fn(values)) if np.isfinite(values).any() else float("nan")


def _covered(interval: tuple[float, float] | None, truth: float) -> float:
    return float(interval is not None and interval[0] <= truth <= interval[1])


def _coverage(rows: tuple[Row, ...]) -> dict[str, float]:
    """Share of truths inside the attacker's 80% interval (only for attackers that give intervals)."""
    if all(r.margin_interval is None and r.cost_interval is None for r in rows):
        return {}
    return {
        "margin_80_coverage": float(np.mean([_covered(r.margin_interval, r.margin_true) for r in rows])),
        "unit_cost_80_coverage": float(np.mean([_covered(r.cost_interval, r.cost_true) for r in rows])),
    }


def _summary(rows: tuple[Row, ...], floor: dict[tuple, Row] | None, n_boot: int) -> dict:
    if not rows:
        return {"n": 0}
    h = _hits(rows)
    groups = np.array([r.firm_id for r in rows])
    out = {
        "n": len(rows),
        "margin_hit_2pp": float(h["margin_hit_2pp"].mean()),
        "margin_hit_5pp": float(h["margin_hit_5pp"].mean()),
        "margin_mae_pp": _nan_stat(np.nanmean, 100 * h["_m_err"]),
        "unit_cost_hit_10pct": float(h["unit_cost_hit_10pct"].mean()),
        "unit_cost_median_rel_err": _nan_stat(np.nanmedian, h["_c_rel"]),
        "n_failed": int(sum(r.status != "ok" or np.isnan(r.margin_est) for r in rows)),
        **_coverage(rows),
    }
    if floor is not None:
        fh = _hits(tuple(floor[r.key] for r in rows))
        for metric in ("margin_hit_5pp", "unit_cost_hit_10pct"):
            diff = h[metric] - fh[metric]
            out[f"advantage_{metric}"] = float(diff.mean())
            out[f"advantage_{metric}_ci"] = bootstrap_ci(diff, groups, n_boot=n_boot, seed=0)
    return out


def summarize(rows: tuple[Row, ...], floor_rows: tuple[Row, ...] | None = None, n_boot: int = 10_000) -> dict:
    floor = {r.key: r for r in floor_rows} if floor_rows is not None else None
    return {
        name: _summary(tuple(r for r in rows if name == "all" or r.stratum == name), floor, n_boot)
        for name in ("all", "typical", "atypical")
    }
