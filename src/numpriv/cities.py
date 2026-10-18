"""Real US metro areas with BEA Regional Price Parities (2024, all items; US = 100).

Derived by scripts/build_city_table.py from the public-domain BEA MARPP table.
"""

import csv
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources


@dataclass(frozen=True)
class City:
    name: str
    rpp: float


@lru_cache(maxsize=1)
def load_cities() -> tuple[City, ...]:
    text = resources.files("numpriv").joinpath("data/cities_rpp2024.csv").read_text(encoding="utf-8")
    return tuple(City(r["city"], float(r["rpp_all_items_2024"])) for r in csv.DictReader(text.splitlines()))
