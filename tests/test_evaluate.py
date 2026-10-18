import numpy as np
import pytest

from numpriv.attacks import PlaintextAttack, PriorOnlyAttack
from numpriv.dataset import EvalConfig, build_eval_set
from numpriv.defenses import NoDefense, SharedScaling
from numpriv.evaluate import evaluate, summarize

EV = build_eval_set(EvalConfig(n_firms=30, n_dev=10, tasks_per_firm=2, seed=3, prior_seed=4))


def test_evaluate_records_one_row_per_product_with_truth_and_stratum():
    rows = evaluate(EV, "test", NoDefense(), PlaintextAttack(), seeds=(0,), with_oracle=False)
    n_products = sum(len(p.spec.products) for p in EV.prompts if p.split == "test")
    assert len(rows) == n_products
    assert {r.stratum for r in rows} <= {"typical", "atypical"}
    assert all(r.margin_est == pytest.approx(r.margin_true) for r in rows)


def test_evaluate_repeats_per_seed_and_only_uses_requested_split():
    rows = evaluate(EV, "dev", SharedScaling(), PriorOnlyAttack(EV.prior), seeds=(0, 1), with_oracle=True)
    dev_firms = {p.spec.firm_id for p in EV.prompts if p.split == "dev"}
    assert {r.firm_id for r in rows} <= dev_firms
    assert {r.seed for r in rows} == {0, 1}


def test_summarize_reports_hit_rates_advantage_and_firm_clustered_ci():
    rows = evaluate(EV, "test", NoDefense(), PlaintextAttack(), seeds=(0,), with_oracle=False)
    floor = evaluate(EV, "test", NoDefense(), PriorOnlyAttack(EV.prior), seeds=(0,), with_oracle=True)
    s = summarize(rows, floor_rows=floor, n_boot=200)
    assert s["all"]["margin_hit_5pp"] == 1.0
    assert s["all"]["unit_cost_hit_10pct"] == 1.0
    lo, hi = s["all"]["advantage_margin_hit_5pp_ci"]
    assert lo <= s["all"]["advantage_margin_hit_5pp"] <= hi
    assert set(s) >= {"all", "typical", "atypical"}


def test_parallel_workers_give_identical_rows():
    serial = evaluate(EV, "test", SharedScaling(), PriorOnlyAttack(EV.prior), seeds=(0,), with_oracle=True)
    parallel = evaluate(EV, "test", SharedScaling(), PriorOnlyAttack(EV.prior), seeds=(0,), with_oracle=True,
                        workers=4)
    assert serial == parallel


def test_summary_reports_interval_coverage_when_intervals_exist():
    from dataclasses import replace

    rows = evaluate(EV, "test", NoDefense(), PlaintextAttack(), seeds=(0,), with_oracle=False)
    wide = tuple(replace(r, margin_interval=(r.margin_true - 0.1, r.margin_true + 0.1),
                         cost_interval=(0.0, r.cost_true / 2)) for r in rows)
    s = summarize(wide, n_boot=10)["all"]
    assert s["margin_80_coverage"] == 1.0 and s["unit_cost_80_coverage"] == 0.0
    assert "margin_80_coverage" not in summarize(rows, n_boot=10)["all"]
