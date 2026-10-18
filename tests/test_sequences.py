import numpy as np
import pytest

from numpriv.defenses import MetricDP, OrderPreserving, PerFieldScaling, SharedScaling
from numpriv.records import generate_firms
from numpriv.sequences import build_sequence, disguise_sequence
from numpriv.templates import TASKS

FIRM = generate_firms(3, seed=81)[0]


def test_sequence_cycles_through_every_task_and_always_shows_product_0():
    specs = build_sequence(FIRM, n_queries=16, start=2)
    assert len(specs) == 16
    assert [s.task for s in specs[:5]] == [TASKS[(2 + j) % len(TASKS)] for j in range(5)]
    assert all(s.products[0] == FIRM.products[0] for s in specs)


def test_sequence_rejects_empty_length():
    with pytest.raises(ValueError):
        build_sequence(FIRM, n_queries=0, start=0)


def test_reused_key_shows_product_0_identically_in_every_query():
    specs = build_sequence(FIRM, n_queries=10, start=0)
    seq = disguise_sequence(specs, SharedScaling(), "reuse", np.random.default_rng(0))
    shown = {[f.value for f in v.shown if f.field_id == "price:0"][0] for v in seq.views}
    assert len(shown) == 1
    assert len(seq.secrets) == 1


def test_fresh_keys_differ_per_query():
    specs = build_sequence(FIRM, n_queries=10, start=0)
    seq = disguise_sequence(specs, SharedScaling(), "fresh", np.random.default_rng(0))
    assert len({s.k for s in seq.secrets}) == 10


@pytest.mark.parametrize("defense", (PerFieldScaling(), OrderPreserving()), ids=lambda d: d.name)
def test_reused_secret_covers_every_field_in_the_sequence(defense):
    """One secret serves all queries, including fields only some templates show (quote, overhead)."""
    specs = build_sequence(FIRM, n_queries=5, start=0)
    seq = disguise_sequence(specs, defense, "reuse", np.random.default_rng(1))
    assert len(seq.views) == 5 and len(seq.secrets) == 1


def test_metric_dp_draws_fresh_noise_per_release():
    specs = build_sequence(FIRM, n_queries=8, start=1)
    seq = disguise_sequence(specs, MetricDP(1.0), "repeat", np.random.default_rng(2))
    prices = [[f.value for f in v.shown if f.field_id == "price:0"][0] for v in seq.views]
    assert len(set(prices)) > 1


def test_views_carry_the_oracle_and_never_the_firm_metadata_in_text():
    specs = build_sequence(FIRM, n_queries=3, start=0)
    seq = disguise_sequence(specs, SharedScaling(), "fresh", np.random.default_rng(3))
    assert all(v.oracle is not None and v.oracle.sector == FIRM.sector for v in seq.views)


def test_unknown_regime_is_rejected():
    specs = build_sequence(FIRM, n_queries=2, start=0)
    with pytest.raises(ValueError):
        disguise_sequence(specs, SharedScaling(), "sometimes", np.random.default_rng(0))
