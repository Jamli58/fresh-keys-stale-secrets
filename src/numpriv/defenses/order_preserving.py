"""D2: numeric part of Zhang et al. (AAAI'26, arXiv 2504.00299v2) substitution.

Paper rules (text-level, applied to number values regardless of field type):
  * integers in the year range get one shared offset (differences preserved);
  * special numbers (28-31, month-end days) stay unchanged;
  * all other numbers are "sorted into intervals and mapped to randomly sampled
    values from a target range while preserving their original order".
Unspecified in the paper and fixed here as logged assumptions (research contract):
  A-D2-1 the secret is a keyed monotone map h: the signed-log axis
         t = sign(x)*log1p(|x|) over [-T, T] is cut into N_INTERVALS equal
         intervals whose endpoints get sorted log-uniform targets in
         [1, 1e6]; h interpolates linearly in log-target space. Equal inputs
         map to equal outputs and order is preserved across *all* prompts that
         share the secret. Within one interval h acts like a power law
         (log y is affine in t), which an attacker may exploit.
  A-D2-2 year offset uniform integer in [-10, 10] minus {0}, year range [1990, 2030];
  A-D2-3 special = integers 28-31; year/special rules are text-level;
  A-D2-4 a fresh secret per prompt in the fresh regime; reused per firm in B2.
  A-D2-8 the law is configurable for sensitivity runs (R037): n_intervals, and gap_concentration
         alpha (interval gaps ~ Dirichlet(alpha); alpha = 1 is the default sorted-uniform cuts and
         keeps the default random stream).
  A-D2-7 h has a minimum log-slope (MIN_LOG_SLOPE, ~4% of the log range) so it is
         strictly increasing; outputs are shown with the fewest significant
         digits (>= 6) that keep every distinct input in the prompt distinct.
         Merging distinct values would create false ties and break reconstruction.
Targets are random, so there is no cent grid to expose. The topic-shift step
(local LLM) is not in this module.
"""

from dataclasses import dataclass

import numpy as np

from numpriv.fields import Disguised, NumField, disguised, round_sig

YEAR_RANGE = (1990, 2030)
SPECIAL_RANGE = (28, 31)
MAX_OFFSET = 10
N_INTERVALS = 64
SIGNED_LOG_LIMIT = float(np.log1p(1e12))  # |x| up to 1e12 before clipping
MIN_LOG_SLOPE = 0.01  # d log(y) / d t floor; keeps h strictly increasing
MIN_DISPLAY_DIGITS, MAX_DISPLAY_DIGITS = 6, 15


def _is_int(x: float) -> bool:
    return float(x).is_integer()


def _is_year(x: float) -> bool:
    return _is_int(x) and YEAR_RANGE[0] <= x <= YEAR_RANGE[1]


def _is_special(x: float) -> bool:
    return _is_int(x) and SPECIAL_RANGE[0] <= x <= SPECIAL_RANGE[1]


def _signed_log(x: float) -> float:
    return float(np.clip(np.sign(x) * np.log1p(abs(x)), -SIGNED_LOG_LIMIT, SIGNED_LOG_LIMIT))


@dataclass(frozen=True)
class OrderSecret:
    year_offset: int
    knot_log_targets: tuple[float, ...]  # N_INTERVALS + 1 strictly increasing values of log(target)

    def h(self, x: float) -> float:
        grid = np.linspace(-SIGNED_LOG_LIMIT, SIGNED_LOG_LIMIT, len(self.knot_log_targets))
        return float(np.exp(np.interp(_signed_log(x), grid, self.knot_log_targets)))


@dataclass(frozen=True)
class OrderPreserving:
    target_low: float = 1.0
    target_high: float = 1e6
    n_intervals: int = N_INTERVALS
    gap_concentration: float = 1.0
    name: str = "D2_zhang_order_preserving"

    def __post_init__(self) -> None:
        if self.n_intervals < 2:
            raise ValueError(f"n_intervals must be >= 2, got {self.n_intervals}")
        if not self.gap_concentration > 0:
            raise ValueError(f"gap_concentration must be > 0, got {self.gap_concentration}")
        if not 0 < self.target_low < self.target_high:
            raise ValueError("need 0 < target_low < target_high")

    @property
    def law(self) -> tuple[float, float, int, float]:
        """Public parameters of the map distribution (the attacker knows them)."""
        return self.target_low, self.target_high, self.n_intervals, self.gap_concentration

    def sample_secret(self, rng: np.random.Generator, fields: tuple[NumField, ...]) -> OrderSecret:
        offsets = [o for o in range(-MAX_OFFSET, MAX_OFFSET + 1) if o != 0]
        low, high = float(np.log(self.target_low)), float(np.log(self.target_high))
        n = self.n_intervals
        floor = MIN_LOG_SLOPE * 2.0 * SIGNED_LOG_LIMIT / n
        spare = (high - low) - n * floor
        if spare <= 0:
            raise ValueError("target range too narrow for MIN_LOG_SLOPE")
        if self.gap_concentration == 1.0:  # sorted uniform cuts: the default stream
            cuts = np.sort(rng.uniform(0.0, 1.0, size=n - 1))
            shares = np.diff(np.concatenate(([0.0], cuts, [1.0])))
        else:
            shares = rng.dirichlet(np.full(n, self.gap_concentration))
        gaps = floor + spare * shares
        knots = tuple(float(v) for v in low + np.concatenate(([0.0], np.cumsum(gaps))))
        return OrderSecret(int(offsets[rng.integers(len(offsets))]), (*knots[:-1], high))

    def apply(self, fields: tuple[NumField, ...], secret: OrderSecret, rng: np.random.Generator) -> Disguised:
        new_values: dict[str, float] = {}
        pooled = [f for f in fields if not _is_year(f.value) and not _is_special(f.value)]
        for f in fields:
            if _is_year(f.value):
                new_values[f.field_id] = f.value + secret.year_offset
        raw = {f.value: secret.h(f.value) for f in pooled}
        digits = _display_digits(raw)
        new_values.update({f.field_id: round_sig(raw[f.value], digits) for f in pooled})
        return disguised(fields, new_values)


def _display_digits(raw: dict[float, float]) -> int:
    """Fewest significant digits (>= MIN_DISPLAY_DIGITS) under which distinct inputs stay distinct."""
    for digits in range(MIN_DISPLAY_DIGITS, MAX_DISPLAY_DIGITS + 1):
        if len({round_sig(y, digits) for y in raw.values()}) == len(raw):
            return digits
    return MAX_DISPLAY_DIGITS
