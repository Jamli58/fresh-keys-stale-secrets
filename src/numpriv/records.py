"""Counterfactual SME business records drawn from the latent firm model P_gen."""

from dataclasses import dataclass

import numpy as np

from numpriv import config
from numpriv.cities import load_cities
from numpriv.names import pseudo_word, unique_firm_names

_FIRM_SUFFIXES = ("& Co", "Works", "Studio", "Supply", "Group", "Partners", "House")


@dataclass(frozen=True)
class Product:
    name: str
    product_class: str
    price: float
    unit_cost: float
    monthly_volume: tuple[int, ...]

    @property
    def margin(self) -> float:
        return (self.price - self.unit_cost) / self.price


@dataclass(frozen=True)
class Firm:
    firm_id: str
    name: str
    sector: str
    products: tuple[Product, ...]
    monthly_overhead: float
    latent_extreme: bool
    city: str = ""
    city_rpp: float = 100.0
    premium: float = 0.0  # latent log-price shift (never shown directly)
    positioning: str = "standard"  # shown cue: premium / standard / discount


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _firm_efficiency(rng: np.random.Generator, extreme: bool) -> float:
    if not extreme:
        return float(rng.normal(0.0, config.FIRM_EFFICIENCY_SIGMA))
    low, high = config.EXTREME_EFFICIENCY_RANGE
    return float(rng.choice((-1.0, 1.0)) * rng.uniform(low, high))


def _monthly_volumes(rng: np.random.Generator, spec: config.SectorSpec) -> tuple[int, ...]:
    base = spec.volume_median * np.exp(rng.normal(0.0, spec.volume_log_sigma))
    phase = rng.uniform(0.0, 2.0 * np.pi)
    months = np.arange(12)
    season = 1.0 + config.SEASONALITY_AMPLITUDE * np.sin(2.0 * np.pi * months / 12.0 + phase)
    noise = np.exp(rng.normal(0.0, 0.08, size=12))
    return tuple(int(v) for v in np.maximum(0, np.round(base * season * noise)))


def _product(
    rng: np.random.Generator,
    spec: config.SectorSpec,
    efficiency: float,
    location: float,
) -> Product:
    cls = spec.product_classes[rng.integers(len(spec.product_classes))]
    price_noise = cls.price_log_sigma * rng.standard_t(config.PRICE_NOISE_DF) / np.sqrt(3.0)
    price = round(max(0.5, cls.price_median * location * float(np.exp(price_noise))), 2)
    margin_logit = _logit(spec.margin_median) + efficiency + rng.normal(0.0, spec.margin_logit_sigma)
    margin = min(max(_sigmoid(margin_logit), 0.01), 0.97)
    unit_cost = round(price * (1.0 - margin), 2)
    unit_cost = min(max(unit_cost, 0.01), round(price - 0.01, 2))
    return Product(
        name=f"{pseudo_word(rng)} {cls.name.replace('_', ' ')}",
        product_class=cls.name,
        price=price,
        unit_cost=unit_cost,
        monthly_volume=_monthly_volumes(rng, spec),
    )


def _positioning(rng: np.random.Generator, premium: float, efficiency: float) -> str:
    score = (
        premium / config.FIRM_PREMIUM_SIGMA
        + config.POSITIONING_EFFICIENCY_WEIGHT * efficiency / config.FIRM_EFFICIENCY_SIGMA
        + rng.normal(0.0, config.POSITIONING_NOISE)
    )
    if score > config.POSITIONING_CUT:
        return "premium"
    return "discount" if score < -config.POSITIONING_CUT else "standard"


def _firm(rng: np.random.Generator, index: int, name: str, sector: str, extreme: bool) -> Firm:
    spec = config.SECTORS[sector]
    efficiency = _firm_efficiency(rng, extreme)
    cities = load_cities()
    city = cities[rng.integers(len(cities))]
    premium = float(rng.normal(0.0, config.FIRM_PREMIUM_SIGMA))
    location = city.rpp / 100.0 * float(np.exp(rng.normal(0.0, config.CITY_RESIDUAL_SIGMA) + premium))
    n_products = int(rng.integers(config.MIN_PRODUCTS, config.MAX_PRODUCTS + 1))
    products = tuple(_product(rng, spec, efficiency, location) for _ in range(n_products))
    overhead = round(spec.overhead_median * float(np.exp(rng.normal(0.0, 0.4))), 2)
    return Firm(
        firm_id=f"F{index:05d}",
        name=name,
        sector=sector,
        products=products,
        monthly_overhead=overhead,
        latent_extreme=extreme,
        city=city.name,
        city_rpp=city.rpp,
        premium=premium,
        positioning=_positioning(rng, premium, efficiency),
    )


def generate_firms(n: int, seed: int, extreme_share: float = 0.2) -> tuple[Firm, ...]:
    """Generate ``n`` counterfactual firms; sectors are assigned round-robin for balance."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0.0 <= extreme_share <= 1.0:
        raise ValueError(f"extreme_share must be in [0, 1], got {extreme_share}")
    rng = np.random.default_rng(seed)
    sectors = tuple(config.SECTORS)
    names = unique_firm_names(rng, n, _FIRM_SUFFIXES)
    return tuple(
        _firm(rng, i, names[i], sectors[i % len(sectors)], bool(rng.random() < extreme_share))
        for i in range(n)
    )
