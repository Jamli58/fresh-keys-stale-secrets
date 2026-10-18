"""Evaluation set: counterfactual firms -> prompts with dev/test split, plus the public prior P_pub."""

from dataclasses import dataclass
from typing import Literal

from numpriv.priors import PriorTable, build_prior_table
from numpriv.records import Firm, generate_firms
from numpriv.templates import TASKS, PromptSpec, build_prompt

Split = Literal["dev", "test"]
PRIOR_DRAW_SIZE = 2000


@dataclass(frozen=True)
class EvalConfig:
    n_firms: int = 300
    n_dev: int = 50
    tasks_per_firm: int = 2
    seed: int = 1001
    prior_seed: int = 2002
    cues: bool = True


@dataclass(frozen=True)
class EvalPrompt:
    split: Split
    spec: PromptSpec


@dataclass(frozen=True)
class EvalSet:
    config: EvalConfig
    firms: tuple[Firm, ...]
    prompts: tuple[EvalPrompt, ...]
    prior: PriorTable
    prior_source_names: frozenset[str]


def _tasks_for(index: int, k: int) -> tuple[str, ...]:
    """Cycle through tasks so every task is used and a firm never repeats one."""
    return tuple(TASKS[(index + j) % len(TASKS)] for j in range(k))


def build_eval_set(config: EvalConfig) -> EvalSet:
    if not 0 < config.n_dev < config.n_firms:
        raise ValueError("need 0 < n_dev < n_firms")
    if not 1 <= config.tasks_per_firm <= len(TASKS):
        raise ValueError(f"tasks_per_firm must be in [1, {len(TASKS)}]")
    firms = generate_firms(config.n_firms, seed=config.seed)
    prior_firms = generate_firms(PRIOR_DRAW_SIZE, seed=config.prior_seed)
    prompts = tuple(
        EvalPrompt("dev" if i < config.n_dev else "test", build_prompt(task, firm, cues=config.cues))
        for i, firm in enumerate(firms)
        for task in _tasks_for(i, config.tasks_per_firm)
    )
    return EvalSet(
        config=config,
        firms=firms,
        prompts=prompts,
        prior=build_prior_table(prior_firms),
        prior_source_names=frozenset(f.name for f in prior_firms),
    )
