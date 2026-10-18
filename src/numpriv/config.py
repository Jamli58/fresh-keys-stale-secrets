"""Sector specifications for the counterfactual SME generator (P_gen).

PROVISIONAL: the numbers below are placeholder assumptions chosen to be
plausible, not values sourced from public statistics. Per the experiment plan,
R001 is not complete until they are replaced by cited public sector
aggregates. Prices are in a generic currency unit.
"""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ProductClassSpec:
    name: str
    price_median: float
    price_log_sigma: float


@dataclass(frozen=True)
class SectorSpec:
    display_name: str
    product_classes: tuple[ProductClassSpec, ...]
    margin_median: float
    margin_logit_sigma: float
    volume_median: float
    volume_log_sigma: float
    overhead_median: float


SECTORS = MappingProxyType(
    {
        "bakery_cafe": SectorSpec(
            "Bakery & Café",
            (
                ProductClassSpec("bread", 4.5, 0.35),
                ProductClassSpec("pastry", 3.8, 0.30),
                ProductClassSpec("coffee", 4.2, 0.25),
                ProductClassSpec("cake", 32.0, 0.40),
            ),
            margin_median=0.58,  # SOI 2022: Food services and drinking places
            margin_logit_sigma=0.35,
            volume_median=900.0,
            volume_log_sigma=0.6,
            overhead_median=18_000.0,
        ),
        "machine_shop": SectorSpec(
            "Machine Shop",
            (
                ProductClassSpec("machined_part", 85.0, 0.8),
                ProductClassSpec("fixture", 420.0, 0.6),
                ProductClassSpec("repair_job", 260.0, 0.5),
            ),
            margin_median=0.36,  # SOI 2022: ...machine shops; screw, nut, and bolt
            margin_logit_sigma=0.35,
            volume_median=60.0,
            volume_log_sigma=0.8,
            overhead_median=45_000.0,
        ),
        "apparel_ecommerce": SectorSpec(
            "Apparel E-commerce",
            (
                ProductClassSpec("tshirt", 24.0, 0.35),
                ProductClassSpec("jacket", 120.0, 0.45),
                ProductClassSpec("accessory", 18.0, 0.5),
            ),
            margin_median=0.45,  # SOI 2022: Clothing and clothing accessories retailers
            margin_logit_sigma=0.35,
            volume_median=300.0,
            volume_log_sigma=0.9,
            overhead_median=25_000.0,
        ),
        "it_consulting": SectorSpec(
            "IT Consulting",
            (
                ProductClassSpec("consulting_day", 950.0, 0.35),
                ProductClassSpec("support_contract", 1500.0, 0.5),
                ProductClassSpec("software_license", 400.0, 0.6),
            ),
            margin_median=0.24,  # Damodaran 2026: Computer Services
            margin_logit_sigma=0.4,
            volume_median=25.0,
            volume_log_sigma=0.7,
            overhead_median=60_000.0,
        ),
        "construction_sub": SectorSpec(
            "Construction Subcontractor",
            (
                ProductClassSpec("sqm_install", 55.0, 0.4),
                ProductClassSpec("site_day", 1400.0, 0.35),
                ProductClassSpec("materials_lot", 3200.0, 0.6),
            ),
            margin_median=0.34,  # SOI 2022: Specialty trade contractors
            margin_logit_sigma=0.35,
            volume_median=40.0,
            volume_log_sigma=0.8,
            overhead_median=70_000.0,
        ),
        "specialty_retail": SectorSpec(
            "Specialty Retail",
            (
                ProductClassSpec("small_item", 15.0, 0.5),
                ProductClassSpec("mid_item", 65.0, 0.45),
                ProductClassSpec("premium_item", 240.0, 0.5),
            ),
            margin_median=0.385,  # SOI 2022: Sporting goods, hobby, book, music, and misc. retailers
            margin_logit_sigma=0.35,
            volume_median=150.0,
            volume_log_sigma=0.8,
            overhead_median=20_000.0,
        ),
    }
)

SECTOR_PARAMETER_PROVENANCE = (
    "margins: IRS SOI corporations TY2022 Table 1 (5 sectors), Damodaran Computer Services Jan 2026 (IT); "
    "prices, volumes, overhead and the firm-level model: documented assumptions"
)

# Latent firm-level model (P_gen). Documented assumptions, not fitted to data.
FIRM_EFFICIENCY_SIGMA = 0.30  # logit-margin shift for ordinary firms
EXTREME_EFFICIENCY_RANGE = (1.2, 2.0)  # |logit shift| for extreme (atypical-prone) firms
# Location (M0.5): firm price level = city BEA RPP / 100 x exp(N(0, CITY_RESIDUAL_SIGMA)).
CITY_RESIDUAL_SIGMA = 0.08
# Brand premium (M0.5): firm-level log-price shift shared by all its products.
FIRM_PREMIUM_SIGMA = 0.15
# Positioning cue (M0.5), fixed before M2: ordered score
#   s = premium / FIRM_PREMIUM_SIGMA + POSITIONING_EFFICIENCY_WEIGHT * efficiency / FIRM_EFFICIENCY_SIGMA + N(0, POSITIONING_NOISE)
#   discount if s < -POSITIONING_CUT, premium if s > POSITIONING_CUT, else standard  (~25 / 50 / 25).
POSITIONING_EFFICIENCY_WEIGHT = 0.5
POSITIONING_NOISE = 0.7
POSITIONING_CUT = 0.8
PRICE_NOISE_DF = 3  # Student-t degrees of freedom for heavy-tailed price noise
SEASONALITY_AMPLITUDE = 0.10
MIN_PRODUCTS, MAX_PRODUCTS = 3, 8
