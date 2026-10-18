from numpriv.dataset import EvalConfig, build_eval_set


def test_eval_set_splits_and_sizes_follow_plan():
    ev = build_eval_set(EvalConfig(n_firms=300, n_dev=50, tasks_per_firm=2, seed=1001, prior_seed=2002))
    dev = [p for p in ev.prompts if p.split == "dev"]
    test = [p for p in ev.prompts if p.split == "test"]
    assert len(dev) == 100 and len(test) == 500
    assert {p.spec.firm_id for p in dev}.isdisjoint({p.spec.firm_id for p in test})


def test_each_firm_gets_distinct_tasks_and_all_tasks_are_used():
    ev = build_eval_set(EvalConfig(n_firms=60, n_dev=10, tasks_per_firm=2, seed=5, prior_seed=6))
    by_firm: dict[str, list[str]] = {}
    for p in ev.prompts:
        by_firm.setdefault(p.spec.firm_id, []).append(p.spec.task)
    assert all(len(set(t)) == 2 for t in by_firm.values())
    assert {t for ts in by_firm.values() for t in ts} == {
        "quote_pricing", "margin_review", "break_even", "sales_summary", "supplier_comparison"
    }


def test_prior_table_is_built_from_a_separate_draw():
    ev = build_eval_set(EvalConfig(n_firms=60, n_dev=10, tasks_per_firm=2, seed=5, prior_seed=6))
    eval_names = {f.name for f in ev.firms}
    assert ev.prior_source_names.isdisjoint(eval_names) or len(ev.prior_source_names & eval_names) < 3
    assert ev.config.prior_seed != ev.config.seed
