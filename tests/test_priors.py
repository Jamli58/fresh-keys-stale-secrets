import pytest

from numpriv.priors import build_prior_table, label_stratum
from numpriv.records import Firm, Product, generate_firms


def _product(price: float, cost: float, cls: str = "bread") -> Product:
    return Product(
        name="p", product_class=cls, price=price, unit_cost=cost, monthly_volume=(1,) * 12
    )


def _firm(margins: list[float], firm_id: str = "f") -> Firm:
    products = tuple(_product(100.0, round(100.0 * (1 - m), 2)) for m in margins)
    return Firm(
        firm_id=firm_id,
        name=firm_id,
        sector="bakery_cafe",
        products=products,
        monthly_overhead=1000.0,
        latent_extreme=False,
    )


def test_prior_table_summaries_are_ordered():
    table = build_prior_table(generate_firms(500, seed=11))
    for stats in table.by_class.values():
        for s in (stats.price, stats.unit_cost, stats.margin):
            assert s.p10 <= s.q25 <= s.median <= s.q75 <= s.p90
    for s in table.sector_margin.values():
        assert s.q25 <= s.median <= s.q75


def test_prior_table_covers_every_sector_class_seen():
    firms = generate_firms(500, seed=12)
    table = build_prior_table(firms)
    seen = {(f.sector, p.product_class) for f in firms for p in f.products}
    assert seen == set(table.by_class)


def test_prior_table_quantiles_match_hand_computation():
    firms = [_firm([0.1, 0.2, 0.3], "a"), _firm([0.4, 0.5], "b")]
    table = build_prior_table(firms)
    margin = table.by_class[("bakery_cafe", "bread")].margin
    assert margin.median == pytest.approx(0.3)
    assert margin.q25 == pytest.approx(0.2)
    assert margin.q75 == pytest.approx(0.4)


def test_label_stratum_uses_sector_p10_p90_band():
    """Atypical = firm median margin outside the sector's P10-P90 band (~20% if distributions match).

    An IQR rule would label ~50% of firms atypical by construction.
    """
    reference = [_firm([m / 20]) for m in range(1, 20)]  # firm margins 0.05 ... 0.95
    table = build_prior_table(reference)
    band = table.sector_margin["bakery_cafe"]
    assert band.p10 == pytest.approx(0.14) and band.p90 == pytest.approx(0.86)
    assert label_stratum(_firm([0.5]), table) == "typical"
    assert label_stratum(_firm([0.2, 0.25]), table) == "typical"  # outside IQR, inside P10-P90
    assert label_stratum(_firm([0.9]), table) == "atypical"
    assert label_stratum(_firm([0.1]), table) == "atypical"


def test_atypical_share_is_near_twenty_percent_on_generated_data():
    table = build_prior_table(generate_firms(2000, seed=40))
    firms = generate_firms(1000, seed=41)
    share = sum(label_stratum(f, table) == "atypical" for f in firms) / len(firms)
    assert 0.15 < share < 0.25


def test_label_stratum_rejects_unknown_sector():
    table = build_prior_table([_firm([0.2, 0.3])])
    other = Firm("x", "x", "machine_shop", _firm([0.3]).products, 1.0, False)
    with pytest.raises(KeyError):
        label_stratum(other, table)
