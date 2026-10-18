"""R034: replay cached A4 answers (no new LLM calls) to isolate the semantic-cue gain and cover D3o.

A4's extraction reads only the masked prose, which is identical under every defense, so the
test-split answers cached by the M2 runs serve any defense. Two hybrids, both on the frozen solver:
  A4          the LLM's answer as is (sector, classes, city index, positioning, margin shift);
  A4_truelab  the same cue shifts, but sector and classes replaced by the truth — the information
              A1+ receives — so A4_truelab - A1+ measures the cues alone.
Any cache miss is a hard error: the replay must not differ from the recorded run.
    uv run python scripts/run_a4_replay.py
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path

from numpriv.attacks import PriorOnlyAttack
from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.attacks.hybrid import HybridAttack, build_hypotheses
from numpriv.attacks.view import AttackerView
from numpriv.dataset import EvalConfig, build_eval_set
from numpriv.defenses import MetricDP, MetricDPOfficial, OrderPreserving, PerFieldScaling, SharedScaling
from numpriv.evaluate import evaluate, summarize
from numpriv.llm import CachedClient, LLMError
from numpriv.templates import render_masked

LOG = logging.getLogger("a4_replay")
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CACHE_DIR = RESULTS / "autodl" / "llm_cache"
MODELS = {"llama8b": "llama3.1-8b-instruct-bf16", "qwen32b": "qwen2.5-32b-instruct-awq"}
SEEDS = (0, 1, 2)
DEFENSES = {
    "D1": SharedScaling, "D1p": PerFieldScaling, "D2": OrderPreserving,
    "D3@0.5": lambda: MetricDP(0.5), "D3@1": lambda: MetricDP(1.0),
    "D3o@0.5": lambda: MetricDPOfficial(0.5), "D3o@1": lambda: MetricDPOfficial(1.0),
    "D3o@2": lambda: MetricDPOfficial(2.0),
}


@dataclass(frozen=True)
class CacheOnlyClient:
    """Same identity fields as OpenAICompatClient (they enter the cache key); never calls a server."""
    model: str
    seed: int = 0
    temperature: float = 0.0

    def generate_json(self, system, user, schema, max_tokens=None):
        raise LLMError(f"cache miss for {self.model}: replay must not issue new calls")


@dataclass(frozen=True)
class TrueLabelHybrid:
    hybrid: HybridAttack
    labels: dict  # masked text -> (sector, classes)
    name: str = "A4_truelab"

    def attack(self, view: AttackerView):
        prior = self.hybrid.extract(view)
        if prior is None:
            raise LLMError("A4 extraction missing from the cache")
        sector, classes = self.labels[view.masked_text]
        prior = replace(prior, sector=sector, classes=classes)
        return self.hybrid.solver.solve_with(view, build_hypotheses(self.hybrid.solver, prior))


def _delta(rows, base, n_boot):
    s = summarize(rows, base, n_boot)["all"]
    return {m: (s[f"advantage_{m}"], s[f"advantage_{m}_ci"]) for m in ("margin_hit_5pp", "unit_cost_hit_10pct")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = time.time()
    ev = build_eval_set(EvalConfig(seed=1001, prior_seed=2002))
    solver_kw = json.loads((RESULTS / "R008_tuning.json").read_text())["result"]["frozen"]
    labels = {render_masked(p.spec): (p.spec.sector, tuple(q.product_class for q in p.spec.products))
              for p in ev.prompts}
    result = {}
    for key, model in MODELS.items():
        llm = CachedClient(CacheOnlyClient(model), CACHE_DIR / f"127.0.0.1_8000__{model}.jsonl")
        for label, make in DEFENSES.items():
            d = make()
            a1p = evaluate(ev, "test", d, ClassicalBayesAttack(ev.prior, d, use_oracle=True, **solver_kw), SEEDS, True)
            floor = evaluate(ev, "test", d, PriorOnlyAttack(ev.prior), SEEDS, True)
            hybrid = HybridAttack(llm, ClassicalBayesAttack(ev.prior, d, use_oracle=False, **solver_kw))
            a4 = evaluate(ev, "test", d, hybrid, SEEDS, False)
            if any(r.status != "ok" for r in a4):
                raise SystemExit(f"{key} {label}: A4 fell back to A1 on some prompts (cache miss)")
            a4t = evaluate(ev, "test", d, TrueLabelHybrid(hybrid, labels), SEEDS, False)
            result[f"{key}/{label}"] = {
                "A1plus": {m: summarize(a1p, floor, 10)["all"][m] for m in ("margin_hit_5pp", "unit_cost_hit_10pct")},
                "A4_minus_A1plus": _delta(a4, a1p, args.n_boot),
                "A4truelab_minus_A1plus": _delta(a4t, a1p, args.n_boot),
            }
            LOG.info("%s %s done (cache %s)", key, label, llm.stats())
    config = {"seeds": SEEDS, "frozen_solver": solver_kw, "split": "test", "cache_dir": str(CACHE_DIR.relative_to(ROOT)),
              "total_seconds": time.time() - t0}
    (RESULTS / "R034_a4_replay.json").write_text(json.dumps({"run": "R034_a4_replay", "config": config,
                                                             "result": result}, indent=2, default=float))
    LOG.info("wrote R034_a4_replay.json (%.0f s)", time.time() - t0)


if __name__ == "__main__":
    main()
