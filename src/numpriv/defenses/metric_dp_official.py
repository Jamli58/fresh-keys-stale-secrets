"""D3o: Prεεmpt's money sanitizer exactly as released (github.com/danshumaan/preempt, commit 1032978,
preempt/sanitizer.py: ``encypt_money`` with use_mdp=True and ``M_epsilon``).

Differs from our paper-based D3 (metric_dp.py) in four ways, all taken from the code:
  * the value is truncated to an integer first: v = int(x);
  * the domain depends on v: v == 1 -> [1, 2]; v < 100 -> [2, 1000]; v < 1000 -> [100, 10000];
    otherwise [1000, 10000] (values above 10,000 fall outside their domain and are pulled into it);
  * the domain is cut into 100 bins and distance is counted in bins:
    P(s) ∝ exp(-|v' - s| ε / 2), v' = (v - lo) * 100 / (hi - lo), s = 0..99;
    shown value = int(s * (hi - lo) / 100 + lo);
  * every money value gets the full ε (no split across the prompt's value tokens).
Only money fields are sanitized (counts, years and days pass through). The experiments repository
(preempt-experiments, commit d40dac6) calls a different variant (ranges [0, 2·10^k], 1,000 bins) whose
``M_epsilon`` signature does not match its call sites; we follow the released library.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from numpriv.fields import Disguised, NumField, disguised

BINS = 100


def bucket(value: float) -> tuple[int, int]:
    v = int(float(value))
    if v == 1:
        return 1, 2
    if v < 100:
        return 2, 1000
    if v < 1000:
        return 100, 10000
    return 1000, 10000


@lru_cache(maxsize=None)
def output_grid(lo: int, hi: int) -> tuple[int, ...]:
    """Shown value for each bin s, with the official arithmetic order."""
    return tuple(int(s * (hi - lo) / BINS + lo) for s in range(BINS))


def bin_log_probs(v: np.ndarray, lo: int, hi: int, eps: float) -> np.ndarray:
    """(n, BINS) log P(s | v) for integer-truncated values v in one bucket."""
    scaled = (np.asarray(v, float) - lo) * BINS / (hi - lo)
    logits = -np.abs(scaled[:, None] - np.arange(BINS)[None, :]) * eps / 2.0
    return logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)


@dataclass(frozen=True)
class MetricDPOfficial:
    epsilon: float
    name: str = "D3o_prempt_official"

    def __post_init__(self) -> None:
        if not self.epsilon > 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")

    def sample_secret(self, rng: np.random.Generator, fields: tuple[NumField, ...]) -> None:
        return None  # fresh noise per release; nothing to reuse

    def apply(self, fields: tuple[NumField, ...], secret: None, rng: np.random.Generator) -> Disguised:
        new_values = {}
        for f in fields:
            if f.kind == "money":
                v = int(float(f.value))
                lo, hi = bucket(f.value)
                p = np.exp(bin_log_probs(np.array([v]), lo, hi, self.epsilon)[0])
                s = int(rng.choice(BINS, p=p / p.sum()))
                new_values[f.field_id] = float(output_grid(lo, hi)[s])
        return disguised(fields, new_values)

