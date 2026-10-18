"""A1 / A1+: classical Bayesian attack on disguised price/cost pairs.

Knows each defense's algorithm and public parameters (Kerckhoffs) and the public
prior table P_pub, never the key. A1+ is told the true sector and product
classes (same oracle as the A0 floor); A1 uses a uniform mixture over sectors
and, within a sector, over product classes.

  D0  read values directly.
  D1  margin from the shared-key invariant; k from a grid posterior that
      combines the log-uniform key prior with every product's price prior.
  D1' / D3 / D3o  importance sampling from the (price, margin) prior with the
      defense's per-field likelihood (see likelihoods.py).
  D2  same sampler, but each (price, cost) pair is scored jointly: both share one
      secret map h, and log h(p) - log h(c) carries the margin (M1 gate fix;
      the first field-by-field version could not beat the prior-only floor).
A4 (M2) reuses this solver through ``solve_with`` with an LLM-derived prior, so
the M2 contrast A4 - A1+ compares priors, not solvers. Known conservative simplifications:
non-product money fields (overhead, quotes) are not used; D2's shared map is
used within each price/cost pair but not across products; D2 special numbers
(28-31, passed through) are not exploited.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from numpriv.attacks.likelihoods import (
    d1p_log_likelihood,
    d2_pair_log_likelihood,
    d3_log_likelihood,
    d3o_log_likelihood,
)
from numpriv.attacks.prior_fit import (
    fit_log_normal,
    fit_logit_normal,
    hit_optimal_absolute,
    hit_optimal_relative,
    t_scale_from_normal_sigma,
    weighted_median,
)
from numpriv.attacks.view import AttackerView, Estimate
from numpriv.defenses import MetricDP, MetricDPOfficial, NoDefense, OrderPreserving, PerFieldScaling, SharedScaling
from numpriv.priors import PriorTable

_M_CLIP = 1e-4
DECISIONS = ("median", "hit")
MARGIN_HIT_TOL, COST_HIT_REL_TOL = 0.05, 0.10  # the evaluation's hit rules (evaluate.py)
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class PriorComponent:
    sector: str
    price: tuple[float, float]  # (mu, sigma) of log price
    margin: tuple[float, float]  # (mu, sigma) of logit margin


# sector -> per product -> candidate components (uniform weights within a product)
Hypotheses = dict[str, list[list[PriorComponent]]]


def _log_normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma)


def _logit(m: np.ndarray) -> np.ndarray:
    m = np.clip(m, _M_CLIP, 1 - _M_CLIP)
    return np.log(m / (1 - m))


@dataclass(frozen=True)
class ClassicalBayesAttack:
    table: PriorTable
    defense: object
    use_oracle: bool
    n_samples: int = 4000
    seed: int = 0
    prior_scale: float = 1.0
    prior_df: float | None = 3.0  # Student-t prior (in log / logit space) with P_pub's IQR; None = normal
    grid_points: int = 1201
    # "median": posterior medians (minimize absolute error); "hit": maximize the posterior probability of a hit
    # (margin within MARGIN_HIT_TOL, cost within COST_HIT_REL_TOL). Sampling path only (not D0/D1).
    decision: str = "median"
    use_observation: bool = True  # False: the solver's own prior, ignoring the shown numbers (prior-only control)

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}, got {self.decision!r}")

    @property
    def name(self) -> str:
        return "A1plus_bayes_oracle" if self.use_oracle else "A1_bayes"

    # ---- public API -------------------------------------------------------
    def attack(self, view: AttackerView) -> tuple[Estimate, ...]:
        return self._solve(view)[0]

    def sector_posterior(self, view: AttackerView) -> dict[str, float]:
        return self._solve(view)[1]

    def solve_with(self, view: AttackerView, hypotheses: "Hypotheses") -> tuple[Estimate, ...]:
        """Same solver with an externally supplied prior (A4: LLM-derived sector, classes and shifts)."""
        self._check(view)
        return self._solve(view, hypotheses)[0]

    def component(self, sector: str, cls: str) -> PriorComponent:
        """P_pub prior for one (sector, class), with the frozen shape and width."""
        return self._component(sector, cls)

    # B2 (multi-query) reuses the same prior, draws and likelihoods.
    def log_prior(self, x: np.ndarray, mu: float, scale: float) -> np.ndarray:
        return self._log_prior(x, mu, scale)

    def prior_draws(self) -> np.ndarray:
        """The (2, n_samples) standard draws the sampler maps onto log price and logit margin."""
        rng = np.random.default_rng(self.seed)
        if self.prior_df is None:
            return rng.standard_normal((2, self.n_samples))
        return rng.standard_t(self.prior_df, size=(2, self.n_samples))

    def pair_log_likelihood(self, view: AttackerView, p_shown: float, c_shown: float,
                            price: np.ndarray, cost: np.ndarray) -> np.ndarray:
        return self._pair_log_lik(view, p_shown, c_shown, price, cost)

    # ---- setup ------------------------------------------------------------
    def _check(self, view: AttackerView) -> None:
        if view.defense_name != self.defense.name:
            raise ValueError(f"attack built for {self.defense.name}, view is {view.defense_name}")
        if self.use_oracle and view.oracle is None:
            raise ValueError("A1+ needs the oracle sector and classes")
        if not self.use_oracle and view.oracle is not None:
            raise ValueError("A1 must not receive oracle information")

    def _component(self, sector: str, cls: str) -> PriorComponent:
        stats = self.table.by_class[(sector, cls)]
        mu_p, s_p = fit_log_normal(stats.price)
        mu_m, s_m = fit_logit_normal(stats.margin)
        if self.prior_df is not None:
            s_p, s_m = t_scale_from_normal_sigma(s_p, self.prior_df), t_scale_from_normal_sigma(s_m, self.prior_df)
        return PriorComponent(sector, (mu_p, s_p * self.prior_scale), (mu_m, s_m * self.prior_scale))

    def _log_prior(self, x: np.ndarray, mu: float, scale: float) -> np.ndarray:
        if self.prior_df is None:
            return _log_normal_pdf(x, mu, scale)
        from scipy.stats import t as student_t

        return student_t.logpdf(x, self.prior_df, loc=mu, scale=scale)

    def _hypotheses(self, view: AttackerView) -> "Hypotheses":
        """sector -> per-product list of candidate components (uniform weights)."""
        if self.use_oracle:
            s = view.oracle.sector
            return {s: [[self._component(s, c)] for c in view.oracle.product_classes]}
        by_sector: dict[str, list[str]] = {}
        for sector, cls in self.table.by_class:
            by_sector.setdefault(sector, []).append(cls)
        return {
            s: [[self._component(s, c) for c in sorted(classes)] for _ in range(view.n_products)]
            for s, classes in by_sector.items()
        }

    def _pairs(self, view: AttackerView) -> list[tuple[float, float]]:
        values = {f.field_id: f.value for f in view.shown}
        return [(values[f"price:{i}"], values[f"cost:{i}"]) for i in range(view.n_products)]

    # ---- solvers ----------------------------------------------------------
    def _solve(self, view: AttackerView, hypotheses: "Hypotheses | None" = None
               ) -> tuple[tuple[Estimate, ...], dict[str, float]]:
        self._check(view)
        hyps = hypotheses if hypotheses is not None else self._hypotheses(view)
        pairs = self._pairs(view)
        if isinstance(self.defense, NoDefense):
            est = tuple(Estimate(i, (p - c) / p, c) for i, (p, c) in enumerate(pairs))
            return est, {}
        if isinstance(self.defense, SharedScaling):
            return self._solve_shared_scaling(hyps, pairs)
        return self._solve_sampling(view, hyps, pairs)

    def _solve_shared_scaling(self, hyps: "Hypotheses", pairs) -> tuple[tuple[Estimate, ...], dict[str, float]]:
        grid = np.linspace(np.log(self.defense.k_low), np.log(self.defense.k_high), self.grid_points)
        margins = np.array([np.clip(1 - c / p, _M_CLIP, 1 - _M_CLIP) for p, c in pairs])
        log_post = {}
        for sector, per_product in hyps.items():
            total = np.zeros_like(grid)
            for (p_shown, _), m_obs, comps in zip(pairs, margins, per_product):
                terms = [
                    self._log_prior(np.log(p_shown) - grid, *comp.price)
                    + self._log_prior(_logit(np.array(m_obs)), *comp.margin)
                    for comp in comps
                ]
                total += logsumexp(np.vstack(terms), axis=0) - np.log(len(comps))
            log_post[sector] = total
        stacked = np.vstack(list(log_post.values()))
        norm = logsumexp(stacked)
        k_weights = np.exp(logsumexp(stacked, axis=0) - norm)
        k_hat = float(np.exp(weighted_median(grid, k_weights)))
        sectors = {s: float(np.exp(logsumexp(v) - norm)) for s, v in log_post.items()}
        est = tuple(Estimate(i, float(m), c / k_hat) for i, ((_, c), m) in enumerate(zip(pairs, margins)))
        return est, sectors

    def _field_log_lik(self, view: AttackerView, y: float, xs: np.ndarray) -> np.ndarray:
        d = self.defense
        if isinstance(d, PerFieldScaling):
            return d1p_log_likelihood(y, xs, d.k_low, d.k_high)
        if isinstance(d, MetricDP):
            eps = d.per_token_epsilon(view.shown)
            lo, hi = d.domain["money"]
            return d3_log_likelihood(y, xs, eps, lo, hi)
        if isinstance(d, MetricDPOfficial):
            return d3o_log_likelihood(y, xs, d.epsilon)
        raise ValueError(f"unsupported defense {d!r}")

    def _pair_log_lik(self, view, p_shown: float, c_shown: float, price: np.ndarray, cost: np.ndarray) -> np.ndarray:
        if isinstance(self.defense, OrderPreserving):
            return d2_pair_log_likelihood(p_shown, c_shown, price, cost, self.defense)
        return self._field_log_lik(view, p_shown, price) + self._field_log_lik(view, c_shown, cost)

    def _solve_sampling(self, view: AttackerView, hyps: "Hypotheses", pairs) -> tuple[tuple[Estimate, ...], dict[str, float]]:
        z = self.prior_draws()
        # per sector, per product: list of (log marginal, price samples, margin samples, log weights)
        cache: dict[str, list[list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]]] = {}
        log_sector = {}
        for sector, per_product in hyps.items():
            rows, total = [], 0.0
            for (p_shown, c_shown), comps in zip(pairs, per_product):
                row = []
                for comp in comps:
                    price = np.exp(comp.price[0] + comp.price[1] * z[0])
                    margin = 1 / (1 + np.exp(-(comp.margin[0] + comp.margin[1] * z[1])))
                    ll = self._pair_log_lik(view, p_shown, c_shown, price, price * (1 - margin))
                    row.append((float(logsumexp(ll) - np.log(ll.size)), price, margin, ll))
                total += float(logsumexp([r[0] for r in row]) - np.log(len(row)))
                rows.append(row)
            cache[sector], log_sector[sector] = rows, total
        norm = logsumexp(list(log_sector.values()))
        if not np.isfinite(norm):  # no sector explains the data: fall back to the uniform sector prior
            LOG.warning("no sector hypothesis is consistent with %s; using uniform sector weights", view.task)
            sectors = {s: 1.0 / len(log_sector) for s in log_sector}
        else:
            sectors = {s: float(np.exp(v - norm)) for s, v in log_sector.items()}
        return tuple(self._pooled_estimate(i, cache, sectors) for i in range(len(pairs))), sectors

    def _pooled_estimate(self, i: int, cache, sectors: dict[str, float]) -> Estimate:
        """Pool samples over sectors and classes. If no sample is consistent with the observation
        (prior tails too thin), fall back to the prior for that product and log it."""
        margins, costs, weights, prior_w = [], [], [], []
        for sector, rows in cache.items():
            row = rows[i]
            log_margs = np.array([r[0] for r in row])
            class_w = np.exp(log_margs - logsumexp(log_margs)) if np.isfinite(logsumexp(log_margs)) else None
            for j, (log_marg, price, margin, ll) in enumerate(row):
                margins.append(margin)
                costs.append(price * (1 - margin))
                prior_w.append(np.full(margin.size, sectors[sector] / len(row)))
                if class_w is None or not np.isfinite(log_marg):
                    weights.append(np.zeros(margin.size))
                else:
                    weights.append(np.exp(ll - logsumexp(ll)) * class_w[j] * sectors[sector])
        w = np.concatenate(weights) if self.use_observation else np.concatenate(prior_w)
        if not (np.isfinite(w).all() and w.sum() > 0):
            LOG.warning("product %d: no prior sample consistent with the observation; using the prior", i)
            w = np.concatenate(prior_w)
        m, c = np.concatenate(margins), np.concatenate(costs)
        if self.decision == "hit":
            return Estimate(i, hit_optimal_absolute(m, w, MARGIN_HIT_TOL), hit_optimal_relative(c, w, COST_HIT_REL_TOL))
        return Estimate(i, weighted_median(m, w), weighted_median(c, w))
