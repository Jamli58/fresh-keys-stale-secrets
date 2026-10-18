"""A-exact (R046): precision-faithful joint attacker for D2, without Gaussian approximation or smoothing.

The joint shared-map attacker (joint_map.py) approximates log h by a Gaussian and smooths every shown value by
0.03 log units. Within one map interval log h is affine in t = sign(x) log1p|x|, and D2 shows >= 6 significant
digits, so three or more values in one interval satisfy an exact, key-independent constraint that the smoothing
discards. This attacker targets the exact posterior instead (plan: refine-logs/EXPERIMENT_PLAN_20261009_exact_joint.md).

Idea: sample maps, not values. Every knot vector of the public law makes h strictly increasing, so a shown value
inverts uniquely; the exact constraints then hold by construction. Display rounding is integrated exactly: the
latent log h(x) of a shown value y is drawn uniformly in the rounding cell of y.

  * First query (both regimes): maps from the law, every shown value inverted (sequential importance sampling,
    weight = prior density of the inverted values times |d log x / d log h|).
  * Reuse: each particle keeps its map; values shown before must fall in their new rounding cells (exact check),
    new values are inverted under the particle's map.
  * Fresh: each particle keeps its values X, not a map. For a new query the probability that a fresh map rounds
    the pinned values to their shown digits is estimated without bias: knots touched by pinned values are solved
    from uniformly drawn in-cell positions (two values in an interval give both knots; neighbouring intervals
    propagate through one value each; a run of single-value intervals has one free knot drawn uniformly over its
    feasible range), remaining pinned values are checked against their cells, and the knot density follows from
    Dirichlet aggregation. The other knots are drawn from their exact conditional law to invert new values.

Fields: product prices and costs carry the oracle class prior (Student-t in log price and logit margin, as A1+);
other mapped fields (volumes, overhead, quotes) get a uniform prior on the public signed-log axis t = log1p(x),
0 < t <= limit. (A flat prior in log x is improper at x -> 0: its weight (1+x)/x let maps that invert a volume
to below 1 dominate some posteriors, e.g. firm F00058.) Shown integers in the year range
(year-offset rule) or in 28-31 (passed through) are not mapped and are skipped.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaln
from scipy.stats import t as student_t

from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.attacks.multi_query import MultiQueryEstimate
from numpriv.attacks.prior_fit import weighted_median
from numpriv.attacks.view import AttackerView
from numpriv.defenses import OrderPreserving
from numpriv.defenses.order_preserving import (
    MAX_OFFSET,
    MIN_DISPLAY_DIGITS,
    MIN_LOG_SLOPE,
    SIGNED_LOG_LIMIT,
    SPECIAL_RANGE,
    YEAR_RANGE,
)
from numpriv.fields import round_sig

LOG = logging.getLogger(__name__)
REGIMES = ("fresh", "reuse")
SHOWN_YEAR_RANGE = (YEAR_RANGE[0] - MAX_OFFSET, YEAR_RANGE[1] + MAX_OFFSET)
MAX_DIGITS = 17


# ---- the public map law ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MapLaw:
    """Knot law of OrderPreserving: L_k = log_low + k * floor + spare * C_k, C the cumulative Dirichlet(alpha)
    shares of n_intervals gaps (alpha = 1: sorted uniform cuts)."""

    n_intervals: int
    log_low: float
    log_high: float
    alpha: float = 1.0
    limit: float = SIGNED_LOG_LIMIT

    def __post_init__(self) -> None:
        if self.n_intervals < 4:
            raise ValueError(f"n_intervals must be >= 4 (positive values must avoid the end knots), got {self.n_intervals}")
        if not self.alpha > 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if not self.spare > 0:
            raise ValueError("target range too narrow for the minimum slope")

    @classmethod
    def of(cls, defense: OrderPreserving) -> "MapLaw":
        return cls(defense.n_intervals, float(np.log(defense.target_low)), float(np.log(defense.target_high)),
                   defense.gap_concentration)

    @property
    def width(self) -> float:
        return 2.0 * self.limit / self.n_intervals

    @property
    def floor(self) -> float:
        return MIN_LOG_SLOPE * self.width

    @property
    def spare(self) -> float:
        return (self.log_high - self.log_low) - self.n_intervals * self.floor

    @property
    def grid(self) -> np.ndarray:
        return np.linspace(-self.limit, self.limit, self.n_intervals + 1)

    def cum(self, knots: np.ndarray, k) -> np.ndarray:
        return (knots - self.log_low - np.asarray(k) * self.floor) / self.spare

    def knot(self, cum: np.ndarray, k) -> np.ndarray:
        return self.log_low + np.asarray(k) * self.floor + self.spare * cum

    def sample_knots(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """(n, n_intervals + 1) knot log-targets; normalised gammas = Dirichlet(alpha) shares."""
        g = rng.gamma(self.alpha, size=(n, self.n_intervals))
        gaps = self.floor + self.spare * g / g.sum(axis=1, keepdims=True)
        knots = self.log_low + np.concatenate([np.zeros((n, 1)), np.cumsum(gaps, axis=1)], axis=1)
        knots[:, -1] = self.log_high
        return knots


def log_cum_density(law: MapLaw, idx: tuple[int, ...], cum: np.ndarray) -> np.ndarray:
    """Log density of the cumulative shares at interior knot indices idx (sorted): by Dirichlet aggregation the
    spacings between them (and the ends 0 and 1) are Dirichlet(alpha * index gaps)."""
    cum = np.atleast_2d(cum)
    full = np.concatenate(([0], np.asarray(idx), [law.n_intervals]))
    a = law.alpha * np.diff(full)
    ends = np.ones((cum.shape[0], 1))
    d = np.diff(np.concatenate([0 * ends, cum, ends], axis=1), axis=1)
    bad = ~np.all(d > 0, axis=1) | ~np.all(np.isfinite(d), axis=1)
    safe = np.where(d > 0, d, 1.0)
    out = gammaln(law.alpha * law.n_intervals) - gammaln(a).sum() + ((a - 1) * np.log(safe)).sum(axis=1)
    out[bad] = -np.inf
    return out


# ---- display rounding --------------------------------------------------------------------------------------
def rounding_cell(y: float, digits: int) -> tuple[float, float]:
    """[lo, hi): every x > 0 with round_sig(x, digits) == y. Below a power of ten the step is ten times finer."""
    if not y > 0:
        raise ValueError(f"need y > 0, got {y}")
    step = 10.0 ** (int(np.floor(np.log10(y))) - digits + 1)
    lo, hi = y - step / 2, y + step / 2
    if round(y / step) == 10 ** (digits - 1):
        lo = y - step / 20
    return lo, hi


def _min_digits(y: float) -> int:
    for k in range(1, MAX_DIGITS):
        if round_sig(y, k) == y:
            return k
    return MAX_DIGITS


def infer_digits(values, min_digits: int) -> int:
    """Significant digits of a prompt: D2 shows every mapped value with one count, the fewest >= the floor."""
    return max([min_digits] + [_min_digits(float(v)) for v in values])


def invert(law: MapLaw, knots: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise inverse of h: t, interval index j and fraction f with log h = (1 - f) L_j + f L_{j+1}."""
    knots, u = np.asarray(knots, float), np.asarray(u, float)
    j = np.sum(knots[:, 1:-1] <= u[:, None], axis=1)
    r = np.arange(u.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (u - knots[r, j]) / (knots[r, j + 1] - knots[r, j])
    return law.grid[j] + f * law.width, j, f


# ---- observations and priors -------------------------------------------------------------------------------
@dataclass(frozen=True)
class ProductPrior:
    mu_price: float  # log price
    sd_price: float
    mu_margin: float  # logit margin
    sd_margin: float


@dataclass(frozen=True)
class Observation:
    """Mapped fields of one query, their shown values and the prompt's significant digits; `known` holds fields
    whose shown integer D2 passes through unchanged (28-31), i.e. their true values."""

    fields: tuple[str, ...]
    y: np.ndarray
    digits: int
    known: dict[str, float] = field(default_factory=dict)


def _product_index(field: str) -> int | None:
    kind, _, rest = field.partition(":")
    return int(rest) if kind in ("price", "cost") and rest.isdigit() else None


def _log_std(z: np.ndarray, df: float | None) -> np.ndarray:
    return student_t.logpdf(z, df) if df is not None else -0.5 * z * z - 0.5 * np.log(2 * np.pi)


def product_log_density(prior: ProductPrior, df: float | None, price: np.ndarray, cost: np.ndarray,
                        jac_price: np.ndarray, jac_cost: np.ndarray) -> np.ndarray:
    """Log density of the latent (log h(price), log h(cost)): the prior in (log p, logit m), times
    |d(log p, logit m) / d(log p, log c)| = 1/m, times the inversion Jacobians d log x / d log h."""
    with np.errstate(divide="ignore", invalid="ignore"):
        margin = 1.0 - cost / price
        ok = (price > 0) & (cost > 0) & (margin > 0) & (margin < 1) & (jac_price > 0) & (jac_cost > 0)
        m = np.where(ok, margin, 0.5)
        out = (_log_std((np.log(np.where(ok, price, 1.0)) - prior.mu_price) / prior.sd_price, df)
               - np.log(prior.sd_price)
               + _log_std((np.log(m / (1 - m)) - prior.mu_margin) / prior.sd_margin, df) - np.log(prior.sd_margin)
               - np.log(m) + np.log(np.where(ok, jac_price, 1.0)) + np.log(np.where(ok, jac_cost, 1.0)))
    return np.where(ok, out, -np.inf)


def _flat_log_prior(x: np.ndarray, dlog_x: np.ndarray) -> np.ndarray:
    """Uniform prior on t = log1p(x): from a Jacobian d log x / d(.) to d t / d(.) = d log x / d(.) * x / (1 + x)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(dlog_x) + np.log(x) - np.log1p(x)


def _ess(lw: np.ndarray) -> float:
    if not np.any(np.isfinite(lw)):
        return 0.0
    w = np.exp(lw - lw[np.isfinite(lw)].max())
    return float(w.sum() ** 2 / (w * w).sum())


# ---- the sampler -------------------------------------------------------------------------------------------
class ExactJointSampler:
    """Sequential importance sampler over the firm's values given D2 transcripts (one update per query)."""

    def __init__(self, law: MapLaw, priors: dict[int, ProductPrior], df: float | None, regime: str, n: int,
                 rng: np.random.Generator, resample_frac: float = 0.5, mcmc_steps: int = 0, inner: int = 8,
                 n_initial: int | None = None) -> None:
        if regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
        if n < 2:
            raise ValueError(f"need at least 2 particles, got {n}")
        self.law, self.priors, self.df, self.regime, self.n = law, priors, df, regime, n
        self.rng, self.resample_frac = rng, resample_frac
        self.values: dict[str, np.ndarray] = {}  # field -> x per particle
        self.latent: dict[str, np.ndarray] = {}  # reuse: field -> log h(x) under the particle's map
        self.knots: np.ndarray | None = None  # reuse: the particle's map
        self.lw = np.zeros(n)
        self.ess = float(n)  # before resampling, latest update
        self.ess_history: list[float] = []
        # Resample-move: after each update the particles are resampled and moved by mcmc_steps Metropolis-Hastings
        # steps that leave the current posterior invariant (0 = plain sequential importance sampling).
        self.mcmc_steps, self.inner = mcmc_steps, inner
        # The first query proposes maps from the law, which is cheap but inefficient: draw n_initial particles
        # there and resample down to n, so the weights of separated posterior modes rest on many draws.
        self.n_target = n
        if n_initial is not None and n_initial > n:
            self.n = n_initial
            self.lw = np.zeros(n_initial)
        self.alias: dict[str, str] = {}  # field -> canonical field holding the same value
        self.queries: list[list[dict]] = []  # fresh: the nodes of every query (likelihood of the history)
        self.cells: dict[str, tuple[float, float]] = {}  # reuse: canonical field -> intersected cell of log h
        self.known: set[str] = set()  # fields with a passed-through (exactly known) value
        self.acceptance: list[float] = []

    # -- public ---------------------------------------------------------------------------------------------
    def update(self, obs: Observation) -> None:
        for f, v in obs.known.items():  # passed through unchanged: the value itself, not part of any map
            if f not in self.values:
                self.values[f] = np.full(self.n, float(v))
                self.known.add(f)
        nodes = self._nodes(obs)
        pinned = [nd for nd in nodes if nd["pinned"]]
        new = [nd for nd in nodes if not nd["pinned"]]
        inc = np.zeros(self.n)
        for nd in pinned:
            inc += self._consistency(nd)
            root = self._canon(nd["pinned"][0])
            for f in nd["fields"]:
                if f not in self.values:  # first shown here, with the value of a pinned field
                    self.values[f] = self.values[root]
                    self.alias[f] = root
                    if self.regime == "reuse":
                        self.latent[f] = self.latent[root]
        if self.regime == "reuse":
            if self.knots is None:
                self.knots = self.law.sample_knots(self.rng, self.n)
            for nd in pinned:
                lat = self.latent[nd["pinned"][0]]
                inc[~((lat >= nd["cell"][0]) & (lat < nd["cell"][1]))] = -np.inf
            knots = self.knots
        elif pinned:
            ll, knots = self._fresh_pinned(pinned, need_map=bool(new))
            inc += ll
        else:
            knots = self.law.sample_knots(self.rng, self.n)
        if new:
            inc += self._invert_new(new, knots, obs)
        self.lw = np.where(np.isfinite(inc), self.lw + inc, -np.inf)
        self.ess = _ess(self.lw)
        self.ess_history.append(self.ess)
        if not np.any(np.isfinite(self.lw)):
            raise RuntimeError("every particle is inconsistent with the transcript; increase the particle count")
        for nd in nodes:
            root = self._canon(nd["fields"][0])
            nd["root"] = root
            if self.regime == "reuse":
                lo, hi = self.cells.get(root, (-np.inf, np.inf))
                self.cells[root] = (max(lo, nd["cell"][0]), min(hi, nd["cell"][1]))
        self.queries.append(nodes)
        if self.n != self.n_target:
            self._resample(self.n_target)
        if self.mcmc_steps > 0:
            self._resample()
            if self.regime == "fresh":
                self._move_fresh()
            else:
                self._move_reuse()
        elif self.ess < self.resample_frac * self.n:
            self._resample()

    def product0(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Margin and unit cost of product 0 per particle, and normalised weights."""
        if "price:0" not in self.values or "cost:0" not in self.values:
            raise KeyError("product 0 was never shown as a mapped value")
        p, c = self.values["price:0"], self.values["cost:0"]
        w = np.exp(self.lw - self.lw.max())
        return 1.0 - c / p, c, w / w.sum()

    # -- structure of one observation -----------------------------------------------------------------------
    def _nodes(self, obs: Observation) -> list[dict]:
        """Distinct shown values (equal inputs map to equal outputs), sorted by shown value."""
        groups: dict[float, list[str]] = {}
        for fid, y in zip(obs.fields, obs.y):
            groups.setdefault(float(y), []).append(fid)
        nodes = []
        for y in sorted(groups):
            lo, hi = rounding_cell(y, obs.digits)
            fids = groups[y]
            nodes.append({"y": y, "fields": fids, "cell": (float(np.log(lo)), float(np.log(hi))),
                          "pinned": [f for f in fids if f in self.values]})
        return nodes

    def _consistency(self, nd: dict) -> np.ndarray:
        """Fields shown with one value must hold one value."""
        out = np.zeros(self.n)
        ref = self.values[nd["pinned"][0]]
        for f in nd["pinned"][1:]:
            out[~np.isclose(self.values[f], ref, rtol=1e-12, atol=0.0)] = -np.inf
        return out

    # -- new values -----------------------------------------------------------------------------------------
    def _invert_new(self, new: list[dict], knots: np.ndarray, obs: Observation) -> np.ndarray:
        law, inc = self.law, np.zeros(self.n)
        r = np.arange(self.n)
        jac: dict[str, np.ndarray] = {}
        for nd in new:
            lo, hi = nd["cell"]
            u = lo + (hi - lo) * self.rng.random(self.n)
            t, j, _ = invert(law, knots, u)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                x = np.expm1(t)
                dlog = (1 + x) / x * law.width / (knots[r, j + 1] - knots[r, j])
            ok = (u >= knots[:, 0]) & (u <= knots[:, -1]) & (t > 0) & np.isfinite(dlog) & (dlog > 0)
            inc[~ok] = -np.inf
            x = np.where(ok, x, 1.0)
            for f in nd["fields"]:
                self.values[f] = x
                jac[f] = np.where(ok, dlog, 1.0)
                if f != nd["fields"][0]:
                    self.alias[f] = nd["fields"][0]
                if self.regime == "reuse":
                    self.latent[f] = u
        inc += self._prior_terms(new, jac, obs)
        return inc

    def _prior_terms(self, new: list[dict], jac: dict[str, np.ndarray], obs: Observation) -> np.ndarray:
        """Prior of the newly inverted values: one product term when a price and its cost are both new and
        shown together, otherwise a flat prior in log x per distinct value."""
        out = np.zeros(self.n)
        shown = set(obs.fields)
        done: set[int] = set()
        for nd in new:
            idx = [_product_index(f) for f in nd["fields"]]
            for f in self.known:
                jac.setdefault(f, np.ones(self.n))
            paired = [i for i in idx if i is not None and i in self.priors
                      and all((f"{k}:{i}" in shown or f"{k}:{i}" in obs.known) and f"{k}:{i}" in jac
                              for k in ("price", "cost"))]
            if paired:
                for i in paired:
                    if i not in done:
                        done.add(i)
                        out += product_log_density(self.priors[i], self.df, self.values[f"price:{i}"],
                                                   self.values[f"cost:{i}"], jac[f"price:{i}"], jac[f"cost:{i}"])
            else:
                f = nd["fields"][0]
                out += _flat_log_prior(self.values[f], jac[f])
        return out

    # -- fresh map given pinned values ----------------------------------------------------------------------
    def _fresh_pinned(self, pinned: list[dict], need_map: bool) -> tuple[np.ndarray, np.ndarray | None]:
        law = self.law
        xs = np.stack([self.values[nd["pinned"][0]] for nd in pinned], axis=1)  # sorted by shown value
        ll = np.zeros(self.n)
        ll[~np.all(np.diff(xs, axis=1) > 0, axis=1)] = -np.inf  # h is increasing: order must match
        t = np.log1p(xs)
        J = np.clip(np.floor((t + law.limit) / law.width).astype(int), 0, law.n_intervals - 1)
        F = (t - law.grid[J]) / law.width
        cells = [nd["cell"] for nd in pinned]
        knots = np.empty((self.n, law.n_intervals + 1)) if need_map else None
        structures, inverse = np.unique(J, axis=0, return_inverse=True)
        for g, struct in enumerate(structures):
            idx = np.flatnonzero(inverse.ravel() == g)
            lg, fixed = self._pinned_knots(tuple(int(v) for v in struct), F[idx], cells)
            ll[idx] += lg
            if need_map:
                knots[idx] = self._fill_map(fixed, idx.size)
        return np.where(np.isfinite(ll), ll, -np.inf), knots

    def _pinned_knots(self, struct: tuple[int, ...], F: np.ndarray, cells) -> tuple[np.ndarray, dict]:
        """Unbiased estimate of P(pinned values round to their cells | X) for particles sharing an interval
        structure, and the solved knots."""
        law, rng = self.law, self.rng
        ng = F.shape[0]
        members: dict[int, list[int]] = {}
        for p, j in enumerate(struct):
            members.setdefault(j, []).append(p)
        M = law.n_intervals
        ends = {0: np.full(ng, law.log_low), M: np.full(ng, law.log_high)}  # known end knots
        runs, run = [], []
        for j in sorted(members):
            if run and j != run[-1] + 1:
                runs.append(run)
                run = []
            run.append(j)
        runs.append(run)
        logq, ok, fixed, used = np.zeros(ng), np.ones(ng, bool), {}, set()

        def knot(k):
            return fixed[k] if k in fixed else ends[k]

        def draw(p):
            lo, hi = cells[p]
            return lo + (hi - lo) * rng.random(ng), np.log(hi - lo)

        free = []
        for run in runs:
            if run[-1] == M - 1:  # anchored at the known last knot: propagate left (members of interval 0,
                for j in reversed(run):  # whose left knot is known too, are checked below)
                    if j == 0:
                        break
                    m = members[j][0]
                    (u, lc), f = draw(m), F[:, m]
                    fixed[j] = (u - f * knot(j + 1)) / (1 - f)
                    logq += -lc + np.log1p(-f)
                    used.add(m)
                continue
            if run[0] == 0:  # anchored at the known first knot: propagate right
                for j in run:
                    m = members[j][-1]
                    (u, lc), f = draw(m), F[:, m]
                    fixed[j + 1] = (u - (1 - f) * knot(j)) / f
                    logq += -lc + np.log(f)
                    used.add(m)
                continue
            multi = [j for j in run if len(members[j]) >= 2]
            if not multi:
                free.append(run)
                continue
            piv = max(multi, key=lambda j: len(members[j]))
            a, b = members[piv][0], members[piv][-1]
            (ua, la), (ub, lb) = draw(a), draw(b)
            fa, fb = F[:, a], F[:, b]
            den = fb - fa
            ok &= den > 0
            den = np.where(den > 0, den, 1.0)
            fixed[piv] = (ua * fb - fa * ub) / den
            fixed[piv + 1] = ((1 - fa) * ub - (1 - fb) * ua) / den
            logq += -la - lb + np.log(den)
            used |= {a, b}
            for j in (j for j in run if j > piv):  # left knot known: use the member with the largest f
                m = members[j][-1]
                (u, lc), f = draw(m), F[:, m]
                fixed[j + 1] = (u - (1 - f) * fixed[j]) / f
                logq += -lc + np.log(f)
                used.add(m)
            for j in sorted((j for j in run if j < piv), reverse=True):  # right knot known: smallest f
                m = members[j][0]
                (u, lc), f = draw(m), F[:, m]
                fixed[j] = (u - f * fixed[j + 1]) / (1 - f)
                logq += -lc + np.log1p(-f)
                used.add(m)
        for run in free:  # one value per interval: one free knot, drawn uniformly over its feasible range
            a_coef, b_coef = {run[0]: np.zeros(ng)}, {run[0]: np.ones(ng)}
            for j in run:
                m = members[j][0]
                (u, lc), f = draw(m), F[:, m]
                a_coef[j + 1] = (u - (1 - f) * a_coef[j]) / f
                b_coef[j + 1] = -(1 - f) * b_coef[j] / f
                logq += -lc + np.log(f)
                used.add(m)
            start, end = run[0], run[-1] + 1
            cons = [(a_coef[k + 1] - a_coef[k] - law.floor, b_coef[k + 1] - b_coef[k]) for k in range(start, end)]
            kl = max([k for k in fixed if k < start], default=0)
            left = knot(kl)
            cons.append((a_coef[start] - left - (start - kl) * law.floor, b_coef[start]))
            kr = min([k for k in fixed if k > end], default=M)
            right = knot(kr)
            cons.append((right - a_coef[end] - (kr - end) * law.floor, -b_coef[end]))
            zlo, zhi = np.full(ng, -np.inf), np.full(ng, np.inf)
            for c0, c1 in cons:
                with np.errstate(divide="ignore", invalid="ignore"):
                    bound = -c0 / c1
                zlo = np.where(c1 > 0, np.maximum(zlo, bound), zlo)
                zhi = np.where(c1 < 0, np.minimum(zhi, bound), zhi)
                ok &= (c1 != 0) | (c0 > 0)
            span = zhi - zlo
            good = np.isfinite(span) & (span > 0)
            ok &= good
            span = np.where(good, span, 1.0)
            z = np.where(good, zlo, 0.0) + span * rng.random(ng)
            logq += -np.log(span)
            for k in a_coef:
                fixed[k] = a_coef[k] + b_coef[k] * z
        for p, j in enumerate(struct):  # the remaining pinned values must round to their shown digits
            if p not in used:
                pred = (1 - F[:, p]) * knot(j) + F[:, p] * knot(j + 1)
                ok &= (pred >= cells[p][0]) & (pred < cells[p][1])
        idx = tuple(sorted(fixed))
        cum = np.stack([law.cum(fixed[k], k) for k in idx], axis=1)
        logp = log_cum_density(law, idx, cum) - len(idx) * np.log(law.spare)
        with np.errstate(invalid="ignore"):
            out = logp - logq
        out[~ok | ~np.isfinite(out)] = -np.inf
        return out, fixed

    def _fill_map(self, fixed: dict, ng: int) -> np.ndarray:
        """The other knots from their exact conditional law (scaled Dirichlet between fixed knots)."""
        law = self.law
        M = law.n_intervals
        knots = np.empty((ng, M + 1))
        knots[:, 0], knots[:, M] = law.log_low, law.log_high
        anchors = [0] + sorted(fixed) + [M]
        cum = {0: np.zeros(ng), M: np.ones(ng)}
        for k, v in fixed.items():
            knots[:, k] = v
            cum[k] = law.cum(v, k)
        for a, b in zip(anchors[:-1], anchors[1:]):
            if b - a >= 2:
                g = self.rng.gamma(law.alpha, size=(ng, b - a))
                s = np.cumsum(g / g.sum(axis=1, keepdims=True), axis=1)[:, :-1]
                ks = np.arange(a + 1, b)
                knots[:, a + 1:b] = law.knot(cum[a][:, None] + (cum[b] - cum[a])[:, None] * s, ks)
        return knots

    # -- resample-move --------------------------------------------------------------------------------------
    def _canon(self, field: str) -> str:
        while field in self.alias:
            field = self.alias[field]
        return field

    def _roots(self) -> list[str]:
        """Distinct values that are mapped (passed-through values are fixed)."""
        return sorted(f for f in self.values if f not in self.alias and f not in self.known)

    def _prior_split(self) -> tuple[list[int], list[str]]:
        """Products with a prior and both values pinned; the other distinct values get a uniform prior in t."""
        products = [i for i in self.priors if f"price:{i}" in self.values and f"cost:{i}" in self.values]
        covered = {self._canon(f"{k}:{i}") for i in products for k in ("price", "cost")}
        return products, [f for f in self._roots() if f not in covered]

    def _log_prior(self, values: dict[str, np.ndarray], jac: dict[str, np.ndarray]) -> np.ndarray:
        """Log prior of the distinct values, in the coordinates whose Jacobians d log x / d(.) are given."""
        products, flat = self._prior_split()
        out = np.zeros(next(iter(values.values())).shape)
        for f in self.known:  # fixed values: their factor is constant, so the product term is a conditional
            values = {**values, f: np.full(out.shape, self.values[f][0])}
            jac = {**jac, f: np.ones(out.shape)}
        for i in products:
            pf, cf = self._canon(f"price:{i}"), self._canon(f"cost:{i}")
            out += product_log_density(self.priors[i], self.df, values[pf], values[cf], jac[pf], jac[cf])
        for f in flat:
            out += _flat_log_prior(values[f], jac[f])
        return out

    def _query_loglik(self, nodes: list[dict], values: dict[str, np.ndarray]) -> np.ndarray:
        """log of an unbiased estimate (mean of `inner` replicates) of P(query rounds as shown | values) under
        a fresh map."""
        xs = np.stack([values[nd["root"]] for nd in nodes], axis=1)
        n = xs.shape[0]
        out = np.full(n, -np.inf)
        good = np.all(np.diff(xs, axis=1) > 0, axis=1) & np.all(xs > 0, axis=1)
        if not np.any(good):
            return out
        law, k = self.law, self.inner
        t = np.log1p(xs[good])
        J = np.clip(np.floor((t + law.limit) / law.width).astype(int), 0, law.n_intervals - 1)
        F = (t - law.grid[J]) / law.width
        cells = [nd["cell"] for nd in nodes]
        res = np.empty(t.shape[0])
        structures, inverse = np.unique(J, axis=0, return_inverse=True)
        for g, struct in enumerate(structures):
            idx = np.flatnonzero(inverse.ravel() == g)
            lg, _ = self._pinned_knots(tuple(int(v) for v in struct), np.repeat(F[idx], k, axis=0), cells)
            lg = lg.reshape(idx.size, k)
            top = lg.max(axis=1)
            safe = np.where(np.isfinite(top), top, 0.0)
            with np.errstate(invalid="ignore"):
                res[idx] = np.where(np.isfinite(top),
                                    safe + np.log(np.exp(lg - safe[:, None]).mean(axis=1)), -np.inf)
        out[good] = res
        return out

    def _fresh_log_target(self, values: dict[str, np.ndarray]) -> np.ndarray:
        jac = {f: (1 + v) / v for f, v in values.items()}  # d log x / d t
        lp = self._log_prior(values, jac)
        for nodes in self.queries:
            lp = lp + self._query_loglik(nodes, values)
        return lp

    def _move_fresh(self) -> None:
        """Pseudo-marginal Metropolis-Hastings on the values (in t = log1p x). A move shifts, or scales about
        its mean, every value in one map interval together: within-interval ratios of t, which the shown digits
        pin exactly, are unchanged, and moves that would carry a value out of its interval are rejected (so the
        proposal is symmetric)."""
        law, rng, n = self.law, self.rng, self.n
        roots = self._roots()
        T = np.stack([np.log1p(self.values[f]) for f in roots], axis=1)
        J = np.floor((T + law.limit) / law.width).astype(int)
        sd = np.maximum(T.std(axis=0), 1e-9)
        lp = self._fresh_log_target({f: self.values[f] for f in roots})
        r = np.arange(n)
        accepted = 0
        for _ in range(self.mcmc_steps):
            i = rng.integers(len(roots), size=n)
            mask = J == J[r, i][:, None]
            size = mask.sum(axis=1)
            scale_move = (rng.random(n) < 0.5) & (size >= 2)
            step = sd[i] * rng.choice([0.5, 0.1, 0.02], size=n) * rng.standard_normal(n)
            logb = rng.choice([0.1, 0.01], size=n) * rng.standard_normal(n)
            centre = (T * mask).sum(axis=1) / size
            T_new = np.where(scale_move[:, None] & mask, centre[:, None] + np.exp(logb)[:, None] * (T - centre[:, None]),
                             np.where(mask, T + step[:, None], T))
            log_jac = np.where(scale_move, size * logb, 0.0)
            valid = np.all(np.floor((T_new + law.limit) / law.width).astype(int) == J, axis=1) & np.all(T_new > 0, axis=1)
            sub = np.flatnonzero(valid)
            if sub.size == 0:
                continue
            prop = {f: np.expm1(T_new[sub, c]) for c, f in enumerate(roots)}
            lp_new = self._fresh_log_target(prop)
            acc = np.log(rng.random(sub.size)) < lp_new - lp[sub] + log_jac[sub]
            take = sub[acc]
            T[take] = T_new[take]
            lp[take] = lp_new[acc]
            accepted += take.size
        self.acceptance.append(accepted / (n * self.mcmc_steps))
        for c, f in enumerate(roots):
            self.values[f] = np.expm1(T[:, c])
        for f in self.alias:
            self.values[f] = self.values[self._canon(f)]

    def _reuse_state(self, knots: np.ndarray, latent: dict[str, np.ndarray]):
        """Values, Jacobians d log x / d log h and log target of maps and in-cell positions (exact, no estimate)."""
        law = self.law
        n = knots.shape[0]
        r = np.arange(n)
        values, jac, J = {}, {}, {}
        ok = np.ones(n, bool)
        for f, u in latent.items():
            t, j, _ = invert(law, knots, u)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                x = np.expm1(t)
                d = (1 + x) / x * law.width / (knots[r, j + 1] - knots[r, j])
            good = (t > 0) & np.isfinite(d) & (d > 0)
            ok &= good
            values[f], jac[f], J[f] = np.where(good, x, 1.0), np.where(good, d, 1.0), j
        lp = self._log_prior(values, jac)
        if law.alpha != 1.0:
            g = np.diff(law.cum(knots, np.arange(law.n_intervals + 1)), axis=1)
            with np.errstate(divide="ignore"):
                lp = lp + (law.alpha - 1) * np.log(np.where(g > 0, g, 0.0)).sum(axis=1)
        return values, J, np.where(ok & np.isfinite(lp), lp, -np.inf)

    def _refresh_free_knots(self, knots: np.ndarray, J: dict[str, np.ndarray]) -> np.ndarray:
        """Exact Gibbs step: knots not adjacent to an occupied interval are redrawn from their conditional law
        given the adjacent ones. No value lies between two consecutive adjacent knots, so no inverted value
        changes, and the data do not depend on these knots."""
        occ = np.sort(np.stack([J[f] for f in sorted(J)], axis=1), axis=1)
        structures, inverse = np.unique(occ, axis=0, return_inverse=True)
        M = self.law.n_intervals
        for g, struct in enumerate(structures):
            idx = np.flatnonzero(inverse.ravel() == g)
            anchors = sorted(({int(j) for j in struct} | {int(j) + 1 for j in struct}) - {0, M})
            knots[idx] = self._fill_map({k: knots[idx, k] for k in anchors}, idx.size)
        return knots

    def _move_reuse(self) -> None:
        """Exact Metropolis-Hastings on the shared map and the in-cell positions. A knot next to a value is
        redrawn uniformly between its neighbours (the selection probability is corrected), or one value's
        position is redrawn uniformly in its cell; values always follow by exact inversion."""
        law, rng, n = self.law, self.rng, self.n
        roots = self._roots()
        latent = {f: self.latent[f] for f in roots}
        _, J, _ = self._reuse_state(self.knots, latent)
        knots = self._refresh_free_knots(self.knots.copy(), J)
        _, J, lp = self._reuse_state(knots, latent)
        r = np.arange(n)
        M = law.n_intervals
        accepted = 0

        def knot_count(Jd, k):
            return sum(((Jd[f] == k) | (Jd[f] + 1 == k)).astype(int) for f in roots)

        for _ in range(self.mcmc_steps):
            i = rng.integers(len(roots), size=n)
            ji = np.stack([J[f] for f in roots], axis=1)[r, i]
            k = ji + (rng.random(n) < 0.5)
            knot_move = (rng.random(n) < 0.5) & (k > 0) & (k < M)
            new_knots, new_latent = knots.copy(), dict(latent)
            kk = np.where(knot_move, k, 1)
            lo_c, hi_c = law.cum(knots[r, kk - 1], kk - 1), law.cum(knots[r, kk + 1], kk + 1)
            c_new = lo_c + (hi_c - lo_c) * rng.random(n)
            new_knots[r, kk] = np.where(knot_move, law.knot(c_new, kk), knots[r, kk])
            pick = rng.integers(len(roots), size=n)
            for c, f in enumerate(roots):
                lo, hi = self.cells[f]
                redraw = ~knot_move & (pick == c)
                new_latent[f] = np.where(redraw, lo + (hi - lo) * rng.random(n), latent[f])
            _, J_new, lp_new = self._reuse_state(new_knots, new_latent)
            log_sel = np.where(knot_move, np.log(np.maximum(knot_count(J_new, kk), 1e-300))
                               - np.log(knot_count(J, kk)), 0.0)
            acc = np.log(rng.random(n)) < lp_new - lp + log_sel
            acc &= np.isfinite(lp_new) & ~(knot_move & (knot_count(J_new, kk) == 0))
            knots[acc] = new_knots[acc]
            for f in roots:
                latent[f] = np.where(acc, new_latent[f], latent[f])
                J[f] = np.where(acc, J_new[f], J[f])
            lp = np.where(acc, lp_new, lp)
            accepted += int(acc.sum())
        self.acceptance.append(accepted / (n * self.mcmc_steps))
        self.knots = self._refresh_free_knots(knots, J)
        values, _, _ = self._reuse_state(self.knots, latent)
        for f in roots:
            self.values[f], self.latent[f] = values[f], latent[f]
        for f in self.alias:
            self.values[f] = self.values[self._canon(f)]
            self.latent[f] = self.latent[self._canon(f)]

    # -- resampling -----------------------------------------------------------------------------------------
    def _resample(self, size: int | None = None) -> None:
        """Systematic resampling to `size` particles (default: keep the count)."""
        w = np.exp(self.lw - self.lw.max())
        w /= w.sum()
        size = size or self.n
        positions = (self.rng.random() + np.arange(size)) / size
        idx = np.minimum(np.searchsorted(np.cumsum(w), positions), self.n - 1)
        roots = {k: v[idx] for k, v in self.values.items() if k not in self.alias}
        self.values = {k: roots[self._canon(k)] for k in self.values}
        if self.latent:
            lat = {k: v[idx] for k, v in self.latent.items() if k not in self.alias}
            self.latent = {k: lat[self._canon(k)] for k in self.latent}
        if self.knots is not None:
            self.knots = self.knots[idx]
        self.n = size
        self.lw = np.zeros(size)


# ---- attacker on D2 views ----------------------------------------------------------------------------------
def _mapped(value: float) -> bool:
    if not value > 0:
        return False
    if float(value).is_integer():
        v = int(value)
        if SPECIAL_RANGE[0] <= v <= SPECIAL_RANGE[1] or SHOWN_YEAR_RANGE[0] <= v <= SHOWN_YEAR_RANGE[1]:
            return False
    return True


def _passed_through(value: float) -> bool:
    return float(value).is_integer() and SPECIAL_RANGE[0] <= int(value) <= SPECIAL_RANGE[1]


@dataclass(frozen=True)
class ExactJointAttack:
    solver: ClassicalBayesAttack  # frozen A1+ (use_oracle=True) with an OrderPreserving defense: prior only
    regime: str
    n_particles: int = 20_000
    seed: int = 4646
    resample_frac: float = 0.5
    mcmc_steps: int = 0
    inner: int = 8
    n_initial: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.solver.defense, OrderPreserving):
            raise ValueError("ExactJointAttack is for D2 (OrderPreserving)")
        if not self.solver.use_oracle:
            raise ValueError("build the solver with use_oracle=True")
        if self.regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}, got {self.regime!r}")

    @property
    def law(self) -> MapLaw:
        return MapLaw.of(self.solver.defense)

    def observation(self, view: AttackerView) -> Observation:
        shown = [(f.field_id, float(f.value)) for f in view.shown if _mapped(float(f.value))]
        known = {f.field_id: float(f.value) for f in view.shown if _passed_through(float(f.value))}
        ys = np.array([v for _, v in shown])
        return Observation(tuple(f for f, _ in shown), ys, infer_digits(ys, MIN_DISPLAY_DIGITS), known)

    def priors(self, view: AttackerView) -> dict[int, ProductPrior]:
        out = {}
        for i, cls in enumerate(view.oracle.product_classes):
            comp = self.solver.component(view.oracle.sector, cls)
            out[i] = ProductPrior(comp.price[0], comp.price[1], comp.margin[0], comp.margin[1])
        return out

    def run(self, views: tuple[AttackerView, ...], ns) -> tuple[dict[int, MultiQueryEstimate], list[float]]:
        """Estimates after the first n queries for every n in ns from one sequential run, and the effective
        sample size of every update (before resampling)."""
        ns = sorted(set(int(n) for n in ns))
        if not views or ns[0] < 1 or ns[-1] > len(views):
            raise ValueError(f"need 1 <= n <= {len(views)}, got {ns}")
        priors = {}
        for v in views[: ns[-1]]:
            priors.update(self.priors(v))
        sampler = ExactJointSampler(self.law, priors, self.solver.prior_df, self.regime, self.n_particles,
                                    np.random.default_rng(self.seed), self.resample_frac, self.mcmc_steps, self.inner,
                                    self.n_initial)
        out = {}
        for q, view in enumerate(views[: ns[-1]], start=1):
            sampler.update(self.observation(view))
            if q in ns:
                margin, cost, w = sampler.product0()
                out[q] = MultiQueryEstimate(weighted_median(margin, w), weighted_median(cost, w))
        return out, list(sampler.ess_history)

    def attack_prefixes(self, views: tuple[AttackerView, ...], ns) -> dict[int, MultiQueryEstimate]:
        return self.run(views, ns)[0]

    def attack(self, views: tuple[AttackerView, ...]) -> MultiQueryEstimate:
        return self.attack_prefixes(views, (len(views),))[len(views)]
