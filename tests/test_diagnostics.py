from numpriv.defenses import NoDefense, PerFieldScaling, SharedScaling
from numpriv.diagnostics import self_consistency_gain
from numpriv.priors import build_prior_table
from numpriv.records import generate_firms

TABLE = build_prior_table(generate_firms(2000, seed=90))


def test_gain_is_large_when_the_defense_leaks_the_margin():
    out = self_consistency_gain(TABLE, SharedScaling(), n=200, seed=1)
    assert out["gain"] > 0.5
    assert out["gain_ci"][0] > 0


def test_gain_is_maximal_without_defense():
    out = self_consistency_gain(TABLE, NoDefense(), n=200, seed=2)
    assert out["posterior_hit_5pp"] == 1.0


def test_gain_is_small_when_each_field_has_its_own_key():
    out = self_consistency_gain(TABLE, PerFieldScaling(), n=300, seed=3)
    assert abs(out["gain"]) < 0.1
    assert out["n"] == 300
