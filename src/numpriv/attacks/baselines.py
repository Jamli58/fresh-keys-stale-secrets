"""A0 prior-only floor and A1's plaintext branch (D0 sanity check)."""

from dataclasses import dataclass

from numpriv.attacks.view import AttackerView, Estimate
from numpriv.defenses import NoDefense
from numpriv.priors import PriorTable


@dataclass(frozen=True)
class PriorOnlyAttack:
    """A0: P_pub class medians. Uses the oracle sector *and* product class (logged assumption);
    the class is visible in cued product names anyway. Ignores the numbers."""

    table: PriorTable
    name: str = "A0_prior_only"

    def attack(self, view: AttackerView) -> tuple[Estimate, ...]:
        if view.oracle is None:
            raise ValueError("A0 needs the oracle sector and product classes")
        out = []
        for i, cls in enumerate(view.oracle.product_classes):
            stats = self.table.by_class[(view.oracle.sector, cls)]
            out.append(Estimate(i, stats.margin.median, stats.unit_cost.median))
        return tuple(out)


@dataclass(frozen=True)
class PlaintextAttack:
    """A1 on D0: reads price and cost directly. Refuses disguised input (M1 handles those)."""

    name: str = "A1_plaintext"

    def attack(self, view: AttackerView) -> tuple[Estimate, ...]:
        if view.defense_name != NoDefense.name:
            raise ValueError(f"PlaintextAttack only handles {NoDefense.name}, got {view.defense_name}")
        values = {f.field_id: f.value for f in view.shown}
        out = []
        for i in range(view.n_products):
            price, cost = values[f"price:{i}"], values[f"cost:{i}"]
            out.append(Estimate(i, (price - cost) / price, cost))
        return tuple(out)
