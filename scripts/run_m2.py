"""Milestone M2 (R011-R014): LLM attackers A2 / A3 / A4 against D0-D3.

Same eval set, defense seeds and frozen A1 config as M1, so A0 / A1+ / A2-A4 are
paired on identical disguised prompts. Talks to any OpenAI-compatible server:
vLLM on the cloud GPU for paper numbers, Ollama's /v1 locally for debugging.

  # local debug (dev split, a few firms)
  python scripts/run_m2.py --base-url http://localhost:11434/v1 --model llama3.2:3b --max-firms 5
  # cloud (vLLM)
  python scripts/run_m2.py --model <served-name> --run R011_llama8b --split test --confirm-test

Writes results/<run>.json (summaries, provenance, throughput) and results/<run>_rows.jsonl.
Responses are cached in results/llm_cache/<server>__<model>.jsonl, so an interrupted run resumes.
"""

import argparse
import json
import logging
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, replace
from pathlib import Path

from numpriv.attacks import PriorOnlyAttack
from numpriv.attacks.bayes import ClassicalBayesAttack
from numpriv.attacks.hybrid import HybridAttack
from numpriv.attacks.llm_attacks import EndToEndAttack, ProseOnlyAttack
from numpriv.attacks.llm_prompts import PROMPT_VERSION
from numpriv.dataset import EvalConfig, build_eval_set
from numpriv.defenses import MetricDP, NoDefense, OrderPreserving, PerFieldScaling, SharedScaling
from numpriv.evaluate import evaluate, summarize
from numpriv.llm import CachedClient, OpenAICompatClient

LOG = logging.getLogger("m2")
RESULTS = Path(__file__).resolve().parents[1] / "results"
FROZEN_CONFIG = RESULTS / "R008_tuning.json"
DEFENSES = {
    "D0": NoDefense, "D1": SharedScaling, "D1p": PerFieldScaling, "D2": OrderPreserving,
    "D3@0.5": lambda: MetricDP(0.5), "D3@1": lambda: MetricDP(1.0), "D3@2": lambda: MetricDP(2.0),
}
ATTACKERS = ("A2", "A3", "A4")


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model name as served (vLLM --served-model-name / Ollama tag)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--run", default=None, help="output name (default R011_<split>_<model>)")
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--confirm-test", action="store_true", help="required for --split test (touched once)")
    ap.add_argument("--max-firms", type=int, default=None, help="debug: only the first N firms of the split")
    ap.add_argument("--attackers", nargs="+", choices=ATTACKERS, default=list(ATTACKERS))
    ap.add_argument("--defenses", nargs="+", choices=list(DEFENSES), default=list(DEFENSES))
    ap.add_argument("--defense-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--workers", type=int, default=8, help="concurrent LLM requests")
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()
    if args.split == "test" and not args.confirm_test:
        ap.error("the test split is touched once: pass --confirm-test deliberately")
    if args.workers < 1 or (args.max_firms is not None and args.max_firms < 1):
        ap.error("--workers and --max-firms must be >= 1")
    return args


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _frozen_solver_config() -> dict:
    if not FROZEN_CONFIG.exists():
        raise SystemExit(f"{FROZEN_CONFIG} not found: run scripts/run_m1.py first (it freezes the A1 config)")
    return json.loads(FROZEN_CONFIG.read_text())["result"]["frozen"]


def _provenance(base_url: str) -> dict:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=30) as resp:
            return json.loads(resp.read().decode())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot reach the model server at {base_url}: {exc}") from exc


def _subset(ev, split: str, max_firms: int | None):
    if max_firms is None:
        return ev
    firm_ids = list(dict.fromkeys(p.spec.firm_id for p in ev.prompts if p.split == split))[:max_firms]
    keep = set(firm_ids)
    return replace(ev, prompts=tuple(p for p in ev.prompts if p.spec.firm_id in keep))


def _llm_attack(code: str, llm, defense, solver_kw: dict, prior):
    if code == "A2":
        return ProseOnlyAttack(llm)
    if code == "A3":
        return EndToEndAttack(llm, defense)
    return HybridAttack(llm, ClassicalBayesAttack(prior, defense, use_oracle=False, **solver_kw))


def _delta(summary: dict) -> dict:
    """Keep only the paired differences (summarize's 'advantage_*' against the A1+ rows), renamed 'delta_*'."""
    return {stratum: {("delta_" + k[len("advantage_"):] if k.startswith("advantage_") else k): v
                      for k, v in s.items() if k == "n" or k.startswith("advantage_")}
            for stratum, s in summary.items()}


def _run_defense(ev, args, label: str, llm, solver_kw: dict, rows_out) -> dict:
    defense, seeds = DEFENSES[label](), tuple(args.defense_seeds)
    floor = evaluate(ev, args.split, defense, PriorOnlyAttack(ev.prior), seeds, with_oracle=True)
    a1p_attack = ClassicalBayesAttack(ev.prior, defense, use_oracle=True, **solver_kw)
    a1p = evaluate(ev, args.split, defense, a1p_attack, seeds, with_oracle=True, workers=args.workers)
    entry = {"A0_prior_only": summarize(floor, floor, args.n_boot), a1p_attack.name: summarize(a1p, floor, args.n_boot)}
    for code in args.attackers:
        attack = _llm_attack(code, llm, defense, solver_kw, ev.prior)
        before, t0 = llm.stats()["misses"], time.time()
        rows = evaluate(ev, args.split, defense, attack, seeds, with_oracle=False, workers=args.workers)
        seconds, calls = time.time() - t0, llm.stats()["misses"] - before
        entry[attack.name] = {**summarize(rows, floor, args.n_boot), "seconds": seconds, "new_llm_calls": calls}
        if code == "A4":
            entry["delta_A4_minus_A1plus"] = _delta(summarize(rows, a1p, args.n_boot))
        for r in rows:
            rows_out.write(json.dumps({"defense": label, "attacker": attack.name, **asdict(r)}) + "\n")
        LOG.info("%s %s: %.0fs, %d new calls, margin@5pp %.3f, cost@10%% %.3f", label, attack.name, seconds, calls,
                 entry[attack.name]["all"]["margin_hit_5pp"], entry[attack.name]["all"]["unit_cost_hit_10pct"])
    return entry


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _args()
    run = args.run or f"R011_{args.split}_{_safe(args.model)}"
    solver_kw = {**_frozen_solver_config(), "n_samples": args.n_samples}
    provenance = _provenance(args.base_url)
    cfg = EvalConfig(seed=1001, prior_seed=2002)  # same eval set as M1
    ev = _subset(build_eval_set(cfg), args.split, args.max_firms)
    # cache per (server, model): local Ollama answers must never be reused for a vLLM run
    server = urllib.parse.urlparse(args.base_url).netloc
    llm = CachedClient(OpenAICompatClient(args.model, base_url=args.base_url),
                       RESULTS / "llm_cache" / f"{_safe(server)}__{_safe(args.model)}.jsonl")
    RESULTS.mkdir(exist_ok=True)
    result, t0 = {}, time.time()
    with (RESULTS / f"{run}_rows.jsonl").open("w") as rows_out:
        for label in args.defenses:
            result[label] = _run_defense(ev, args, label, llm, solver_kw, rows_out)
    config = {**vars(args), "eval_config": cfg.__dict__, "frozen_solver": solver_kw, "prompt_version": PROMPT_VERSION,
              "server_models": provenance, "cache": llm.stats(), "total_seconds": time.time() - t0,
              "n_prompts": sum(p.split == args.split for p in ev.prompts)}
    (RESULTS / f"{run}.json").write_text(json.dumps({"run": run, "config": config, "result": result}, indent=2,
                                                    default=float))
    LOG.info("wrote %s (%.0fs, cache %s)", RESULTS / f"{run}.json", config["total_seconds"], llm.stats())


if __name__ == "__main__":
    main()
