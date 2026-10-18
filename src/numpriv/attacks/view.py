"""What an attacker is allowed to see, kept separate from the ground truth used for scoring."""

from dataclasses import dataclass

from numpriv.fields import NumField
from numpriv.templates import PromptSpec, render, render_masked


@dataclass(frozen=True)
class Oracle:
    """Side information given only to oracle attackers (A0 floor, A1+)."""

    sector: str
    product_classes: tuple[str, ...]


@dataclass(frozen=True)
class AttackerView:
    """The provider's view of one prompt: rendered text, shown numbers, and the (public) defense algorithm."""

    task: str
    text: str
    shown: tuple[NumField, ...]
    n_products: int
    defense_name: str
    oracle: Oracle | None
    masked_text: str = ""  # same prose with every number masked (identical under every defense)


@dataclass(frozen=True)
class Estimate:
    product_index: int
    margin: float
    unit_cost: float
    margin_interval: tuple[float, float] | None = None  # 80% interval, LLM attackers only
    cost_interval: tuple[float, float] | None = None
    status: str = "ok"  # ok / llm_failed / invalid_output / llm_failed_fallback


def make_view(
    spec: PromptSpec, shown: tuple[NumField, ...], defense_name: str, with_oracle: bool = False
) -> AttackerView:
    oracle = Oracle(spec.sector, tuple(p.product_class for p in spec.products)) if with_oracle else None
    return AttackerView(spec.task, render(spec, shown), shown, len(spec.products), defense_name, oracle,
                        render_masked(spec))


def truths_for(spec: PromptSpec) -> tuple[Estimate, ...]:
    """Ground truth from the generated records (never from a model)."""
    return tuple(Estimate(i, p.margin, p.unit_cost) for i, p in enumerate(spec.products))
