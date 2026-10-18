import dataclasses

import numpy as np
import pytest

from numpriv.config import SECTORS
from numpriv.records import generate_firms


def test_same_seed_is_deterministic():
    assert generate_firms(20, seed=7) == generate_firms(20, seed=7)


def test_different_seed_differs():
    assert generate_firms(20, seed=7) != generate_firms(20, seed=8)


def test_product_invariants_hold():
    for firm in generate_firms(200, seed=1):
        assert 3 <= len(firm.products) <= 8
        for p in firm.products:
            assert p.price > p.unit_cost > 0
            assert 0 < p.margin < 1
            assert len(p.monthly_volume) == 12
            assert all(v >= 0 for v in p.monthly_volume)
            assert p.product_class in {c.name for c in SECTORS[firm.sector].product_classes}


def test_prices_and_costs_are_whole_cents():
    for firm in generate_firms(50, seed=2):
        for p in firm.products:
            assert round(p.price, 2) == p.price
            assert round(p.unit_cost, 2) == p.unit_cost


def test_all_sectors_appear_and_ids_are_unique():
    firms = generate_firms(300, seed=3)
    assert {f.sector for f in firms} == set(SECTORS)
    assert len({f.firm_id for f in firms}) == len(firms)
    assert len({f.name for f in firms}) == len(firms)


def test_records_are_immutable():
    firm = generate_firms(1, seed=4)[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        firm.sector = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        firm.products[0].price = 1.0  # type: ignore[misc]


def test_extreme_efficiency_share_tracks_parameter():
    firms = generate_firms(2000, seed=5, extreme_share=0.2)
    share = sum(f.latent_extreme for f in firms) / len(firms)
    assert 0.17 < share < 0.23


def test_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        generate_firms(0, seed=1)
    with pytest.raises(ValueError):
        generate_firms(10, seed=1, extreme_share=1.5)


def test_firms_get_a_real_city_with_its_bea_price_parity():
    from numpriv.cities import load_cities

    cities = {c.name: c.rpp for c in load_cities()}
    assert len(cities) == 60
    for firm in generate_firms(100, seed=9):
        assert cities[firm.city] == firm.city_rpp


def test_positioning_shares_are_roughly_25_50_25():
    firms = generate_firms(3000, seed=10)
    share = {p: sum(f.positioning == p for f in firms) / len(firms) for p in ("premium", "standard", "discount")}
    assert 0.19 < share["premium"] < 0.33
    assert 0.19 < share["discount"] < 0.33
    assert 0.38 < share["standard"] < 0.62


def _log_price_residual(firm):
    from numpriv.config import SECTORS

    medians = {c.name: c.price_median for c in SECTORS[firm.sector].product_classes}
    return float(np.mean([np.log(p.price / medians[p.product_class]) for p in firm.products]))


def test_city_price_level_and_positioning_are_real_signals_of_prices():
    firms = generate_firms(3000, seed=11)
    resid = np.array([_log_price_residual(f) for f in firms])
    rpp = np.log([f.city_rpp for f in firms])
    assert np.corrcoef(rpp, resid)[0, 1] > 0.1
    by_pos = {p: resid[[f.positioning == p for f in firms]].mean() for p in ("premium", "standard", "discount")}
    assert by_pos["premium"] > by_pos["standard"] > by_pos["discount"]
