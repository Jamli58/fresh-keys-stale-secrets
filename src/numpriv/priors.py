"""Public prior table P_pub: aggregate summaries the attacker is allowed to know.

Built from a *separate* generator draw than the evaluation firms, mimicking
published sector statistics (P10/Q25/median/Q75/P90 only, no firm-level records).
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Literal, Mapping

import numpy as np

from numpriv.records import Firm

Stratum = Literal["typical", "atypical"]


@dataclass(frozen=True)
class Summary:
    p10: float
    q25: float
    median: float
    q75: float
    p90: float

    @classmethod
    def of(cls, values: Iterable[float]) -> "Summary":
        arr = np.asarray(list(values), dtype=float)
        if arr.size == 0:
            raise ValueError("cannot summarize an empty sample")
        return cls(*(float(q) for q in np.quantile(arr, (0.10, 0.25, 0.5, 0.75, 0.90))))

    def in_central_80(self, value: float) -> bool:
        return self.p10 <= value <= self.p90


@dataclass(frozen=True)
class ClassStats:
    price: Summary
    unit_cost: Summary
    margin: Summary


@dataclass(frozen=True)
class PriorTable:
    by_class: Mapping[tuple[str, str], ClassStats]
    sector_margin: Mapping[str, Summary]


def build_prior_table(firms: Iterable[Firm]) -> PriorTable:
    firms = tuple(firms)
    groups: dict[tuple[str, str], list] = {}
    sector_margins: dict[str, list[float]] = {}
    for firm in firms:
        for p in firm.products:
            groups.setdefault((firm.sector, p.product_class), []).append(p)
        sector_margins.setdefault(firm.sector, []).append(_firm_margin(firm))
    by_class = {
        key: ClassStats(
            price=Summary.of(p.price for p in products),
            unit_cost=Summary.of(p.unit_cost for p in products),
            margin=Summary.of(p.margin for p in products),
        )
        for key, products in groups.items()
    }
    sector = {s: Summary.of(m) for s, m in sector_margins.items()}
    return PriorTable(MappingProxyType(by_class), MappingProxyType(sector))


def _firm_margin(firm: Firm) -> float:
    return float(np.median([p.margin for p in firm.products]))


def label_stratum(firm: Firm, table: PriorTable) -> Stratum:
    """A firm is atypical when its median product margin lies outside the sector P10-P90 band.

    (An IQR rule would label ~50% of firms atypical by construction.)
    """
    band = table.sector_margin[firm.sector]
    return "typical" if band.in_central_80(_firm_margin(firm)) else "atypical"
