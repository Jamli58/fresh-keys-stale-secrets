"""Diagnostic: do A4's v3 price answers track the firm's true city price level and positioning?

Rebuilds each dev prompt's A4 request key (masked prose is identical under every defense),
joins the cached LLM answer to the generated firm, and reports Spearman correlations.
    uv run python scripts/a4_cue_diagnostic.py <cache.jsonl> <served model name>
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from numpriv.attacks import make_view
from numpriv.attacks.hybrid import EXTRACTION_SYSTEM, extraction_schema
from numpriv.attacks.llm_prompts import product_note, quoted
from numpriv.cities import load_cities
from numpriv.dataset import EvalConfig, build_eval_set
from numpriv.defenses import NoDefense
from numpriv.llm.cache import request_key


@dataclass(frozen=True)
class KeyClient:
    model: str
    seed: int = 0
    temperature: float = 0.0


def main(cache_path: str, model: str) -> None:
    cache = {}
    for line in open(cache_path):
        rec = json.loads(line)
        cache[rec["key"]] = rec["response"]
    rpp = {c.name: c.rpp for c in load_cities()}
    es = build_eval_set(EvalConfig())
    firms = {f.firm_id: f for f in es.firms}
    client, d = KeyClient(model), NoDefense()
    rows = []
    for p in es.prompts:
        if p.split != "dev":
            continue
        spec = p.spec
        view = make_view(spec, spec.fields, d.name)
        user = f"Message:\n{quoted(view.masked_text)}\n\n{product_note(view.n_products)}"
        key = request_key(client, EXTRACTION_SYSTEM, user, extraction_schema(view.n_products), None)
        if key not in cache:
            continue
        firm = firms[spec.firm_id]
        rows.append((cache[key], firm))
    print(f"{model}: matched {len(rows)} dev A4 answers")
    if not rows:
        return
    city_ans = np.array([r["city_price_index"] for r, _ in rows], float)
    city_true = np.array([rpp[f.city] for _, f in rows], float)
    pos_ans = np.array([r["positioning_price_pct"] for r, _ in rows], float)
    rho, p = spearmanr(city_ans, city_true)
    print(f"city_price_index vs BEA RPP: Spearman {rho:.2f} (p {p:.1g}); answers {np.unique(city_ans).size} distinct, "
          f"range {city_ans.min():.0f}-{city_ans.max():.0f}")
    by_pos = defaultdict(list)
    for (r, f) in rows:
        by_pos[f.positioning].append(r["positioning_price_pct"])
    print("positioning_price_pct by true positioning:",
          {k: (round(float(np.mean(v)), 1), len(v)) for k, v in sorted(by_pos.items())})
    print("margin_shift_pp distinct:", sorted({r["margin_shift_pp"] for r, _ in rows}))
    mult = (city_ans / 100.0) * (1 + pos_ans / 100.0)
    print(f"combined multiplier: {np.unique(np.round(mult, 3)).size} distinct, range {mult.min():.2f}-{mult.max():.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
