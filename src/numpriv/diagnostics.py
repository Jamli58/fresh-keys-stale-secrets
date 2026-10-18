"""Self-consistency check for the classical Bayesian attacker (R010b).

Draw products from the attacker's *own* prior, disguise them, and compare the
posterior estimate with the prior median. Under its own model a correct Bayesian
attacker cannot lose, and it gains wherever the defense leaks information. A
near-zero gain means the defense hides that quantity from numeric attacks; a
negative gain means the attacker is broken.
"""

import numpy as np

from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.attacks.view import AttackerView, Oracle
from numpriv.fields import NumField
from numpriv.metrics import bootstrap_ci
from numpriv.priors import PriorTable

MARGIN_TOL = 0.05


def _draw_product(attack: ClassicalBayesAttack, sector: str, cls: str, rng: np.random.Generator):
    comp = attack._component(sector, cls)
    z = rng.standard_t(attack.prior_df, 2) if attack.prior_df is not None else rng.standard_normal(2)
    price = round(max(float(np.exp(comp.price[0] + comp.price[1] * z[0])), 0.5), 2)
    margin = float(1 / (1 + np.exp(-(comp.margin[0] + comp.margin[1] * z[1]))))
    cost = min(max(round(price * (1 - margin), 2), 0.01), round(price - 0.01, 2))
    return price, cost, 1 / (1 + np.exp(-comp.margin[0]))


def self_consistency_gain(table: PriorTable, defense, n: int = 1000, seed: int = 0, n_boot: int = 10_000,
                          **attack_kwargs) -> dict:
    rng = np.random.default_rng(seed)
    attack = ClassicalBayesAttack(table, defense, use_oracle=True, **attack_kwargs)
    keys = list(table.by_class)
    post_hits, prior_hits = np.zeros(n), np.zeros(n)
    for i in range(n):
        sector, cls = keys[rng.integers(len(keys))]
        price, cost, prior_median = _draw_product(attack, sector, cls, rng)
        truth = (price - cost) / price
        fields = (NumField("price:0", "money", price), NumField("cost:0", "money", cost))
        shown = defense.apply(fields, defense.sample_secret(rng, fields), rng).fields
        view = AttackerView("margin_review", "", shown, 1, defense.name, Oracle(sector, (cls,)))
        post_hits[i] = abs(attack.attack(view)[0].margin - truth) <= MARGIN_TOL
        prior_hits[i] = abs(prior_median - truth) <= MARGIN_TOL
    diff = post_hits - prior_hits
    return {
        "n": n,
        "posterior_hit_5pp": float(post_hits.mean()),
        "prior_median_hit_5pp": float(prior_hits.mean()),
        "gain": float(diff.mean()),
        "gain_ci": bootstrap_ci(diff, np.arange(n), n_boot=n_boot, seed=seed),
    }
