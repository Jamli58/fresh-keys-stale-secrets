"""Population counts quoted in the paper that the run summaries do not record: distinct test products, the
D3o perturbation population (every generated product) and the D3 one (products in all evaluation prompts).

    uv run python scripts/dataset_counts.py   # writes results/R047_dataset_counts.json
"""

import json
from pathlib import Path

from numpriv.dataset import EvalConfig, build_eval_set

OUT = Path(__file__).resolve().parents[1] / "results" / "R047_dataset_counts.json"
DEFENSE_SEEDS = 3
D3O_DRAWS = 5


def main() -> None:
    cfg = EvalConfig()
    ev = build_eval_set(cfg)
    test = [p.spec for p in ev.prompts if p.split == "test"]
    test_obs = [(s.firm_id, prod.name) for s in test for prod in s.products]
    all_obs = [(p.spec.firm_id, prod.name) for p in ev.prompts for prod in p.spec.products]
    n_products = sum(len(f.products) for f in ev.firms)
    result = {
        "n_firms": len(ev.firms),
        "test_prompts": len(test),
        "test_product_observations": len(test_obs),
        "test_distinct_products": len(set(test_obs)),
        "test_rows_with_defense_seeds": len(test_obs) * DEFENSE_SEEDS,
        "generated_products_all_firms": n_products,
        "d3o_perturbation_observations": n_products * D3O_DRAWS,
        "prompt_product_observations_all_firms": len(all_obs),
        "d3_perturbation_observations": len(all_obs) * DEFENSE_SEEDS,
    }
    OUT.write_text(json.dumps({"config": {"eval": vars(cfg), "defense_seeds": DEFENSE_SEEDS,
                                          "d3o_draws": D3O_DRAWS}, "result": result}, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
