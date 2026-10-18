"""Smoke test of the attacker models before M2 (R031; writes results/<out>).

Backends: ``ollama`` (native API, local Mac) or ``openai`` (any OpenAI-compatible
server: vLLM on AutoDL, or Ollama's /v1).

1. Provenance: model name, digest, parameter size and quantization (from /api/show).
2. Throughput and schema adherence on one structured-extraction call.
3. World-knowledge probe: each model estimates the price level (US = 100) of the 60
   generator cities; Spearman correlation with the BEA 2024 RPP. If the models do not
   know city price levels, the city cue cannot help the M2 attackers (C1a/C1b).
"""

import argparse
import json
import logging
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

from numpriv.cities import load_cities
from numpriv.llm import OllamaClient, OpenAICompatClient

LOG = logging.getLogger("smoke")
RESULTS = Path(__file__).resolve().parents[1] / "results"
BASE = "http://localhost:11434"
PROBE_SCHEMA = {
    "type": "object",
    "properties": {"price_level": {"type": "number"}},
    "required": ["price_level"],
}
PROBE_SYSTEM = "You are an economist. Answer only with the requested JSON."
PROBE_USER = (
    "Estimate the overall price level of consumer goods and services in the metropolitan area "
    "'{city}' relative to the United States average, where the US average is 100. "
    'Return {{"price_level": <number>}}.'
)


def _post(path: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def provenance(model: str) -> dict:
    info = _post("/api/show", {"model": model})
    details = info.get("details", {})
    tags = json.loads(urllib.request.urlopen(f"{BASE}/api/tags", timeout=30).read().decode())
    digest = next((m["digest"] for m in tags.get("models", []) if m["name"] == model), None)
    return {"model": model, "digest": digest, "parameter_size": details.get("parameter_size"),
            "quantization": details.get("quantization_level"), "family": details.get("family")}


def throughput(model: str) -> dict:
    payload = {"model": model, "stream": False, "format": PROBE_SCHEMA, "options": {"temperature": 0, "seed": 0},
               "messages": [{"role": "user", "content": PROBE_USER.format(city="Chicago-Naperville-Elgin, IL-IN")}]}
    t0 = time.time()
    reply = _post("/api/chat", payload)
    wall = time.time() - t0
    tok_s = reply.get("eval_count", 0) / max(reply.get("eval_duration", 1) / 1e9, 1e-9)
    return {"wall_seconds_incl_load": wall, "decode_tokens_per_s": tok_s, "content": reply["message"]["content"]}


def openai_provenance(base_url: str, model: str) -> dict:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=30) as resp:
        served = json.loads(resp.read().decode()).get("data", [])
    entry = next((m for m in served if m.get("id") == model), None)
    if entry is None:
        raise SystemExit(f"{model} is not served at {base_url}; served: {[m.get('id') for m in served]}")
    return {"model": model, "server_entry": entry}


def openai_throughput(base_url: str, model: str) -> dict:
    """Single-request latency; aggregate throughput under concurrency is in the city probe's wall time."""
    payload = {"model": model, "temperature": 0, "seed": 0, "max_tokens": 64,
               "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": PROBE_SCHEMA}},
               "messages": [{"role": "user", "content": PROBE_USER.format(city="Chicago-Naperville-Elgin, IL-IN")}]}
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        reply = json.loads(resp.read().decode())
    wall = time.time() - t0
    tokens = reply.get("usage", {}).get("completion_tokens", 0)
    return {"wall_seconds": wall, "completion_tokens": tokens, "tokens_per_s_incl_prefill": tokens / max(wall, 1e-9),
            "content": reply["choices"][0]["message"]["content"]}


def _guess(client, city: str) -> float:
    try:
        return float(client.generate_json(PROBE_SYSTEM, PROBE_USER.format(city=city), PROBE_SCHEMA)["price_level"])
    except Exception as exc:  # recorded, not swallowed: counted and logged
        LOG.warning("%s failed on %s: %s", client.model, city, exc)
        return float("nan")


def city_probe(client, workers: int = 1) -> dict:
    cities = load_cities()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        guesses = list(pool.map(lambda c: _guess(client, c.name), cities))
    wall = time.time() - t0
    failures = int(sum(np.isnan(guesses)))
    truth, guess = np.array([c.rpp for c in cities]), np.array(guesses)
    ok = ~np.isnan(guess)
    rho, p = stats.spearmanr(truth[ok], guess[ok])
    return {"n": int(ok.sum()), "failures": failures, "spearman_rho": float(rho), "p_value": float(p),
            "mae_points": float(np.mean(np.abs(truth[ok] - guess[ok]))), "wall_seconds": wall, "workers": workers,
            "guesses": {c.name: g for c, g in zip(cities, guesses)}}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("ollama", "openai"), default="ollama")
    ap.add_argument("--base-url", default=None, help="openai backend: e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--models", nargs="+", default=["llama3.1:8b", "gemma3:27b", "llama3.2:3b"])
    ap.add_argument("--workers", type=int, default=1, help="concurrent city-probe requests")
    ap.add_argument("--out", default="LLM_SMOKE.json")
    args = ap.parse_args()
    if args.backend == "openai" and not args.base_url:
        ap.error("--base-url is required for the openai backend")
    out = {}
    for m in args.models:
        if args.backend == "ollama":
            prov, thr, client = provenance(m), throughput(m), OllamaClient(m, seed=0)
        else:
            client = OpenAICompatClient(m, base_url=args.base_url, max_tokens=64)
            prov, thr = openai_provenance(args.base_url, m), openai_throughput(args.base_url, m)
        out[m] = {"backend": args.backend, "provenance": prov, "throughput": thr,
                  "city_probe": city_probe(client, args.workers)}
        LOG.info("%s: city rho %.2f, %d failures", m, out[m]["city_probe"]["spearman_rho"],
                 out[m]["city_probe"]["failures"])
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / args.out).write_text(json.dumps(out, indent=2))
    LOG.info("wrote %s", RESULTS / args.out)


if __name__ == "__main__":
    main()
