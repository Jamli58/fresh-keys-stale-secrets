"""Minimal FinQA program DSL: parse, execute (arithmetic ops only), and rewrite literals.

Follows the FinQA reference executor conventions: ``const_X`` constants
(``const_m1`` = -1), ``#k`` step references, ``N%`` literals mean N/100, and
``greater`` yields "yes"/"no". Table ops are detected but not executed.
"""

import math
import re
from typing import Mapping, Union

Answer = Union[float, str]
Step = tuple[str, tuple[str, ...]]

_STEP_RE = re.compile(r"([a-z_]+)\(([^)]*)\)")
_ARITH = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "exp": lambda a, b: a**b,
}


def parse_program(program: str) -> tuple[Step, ...]:
    steps = tuple(
        (op, tuple(arg.strip() for arg in args.split(","))) for op, args in _STEP_RE.findall(program)
    )
    if not steps:
        raise ValueError(f"no steps parsed from program: {program!r}")
    return steps


def is_arithmetic_only(program: str) -> bool:
    return all(op in _ARITH or op == "greater" for op, _ in parse_program(program))


def _is_constant(token: str) -> bool:
    return token.startswith("const_")


def _is_reference(token: str) -> bool:
    return token.startswith("#")


def literal_to_number(token: str) -> float:
    if _is_constant(token):
        raw = token[len("const_"):]
        return -1.0 if raw == "m1" else float(raw)
    if token.endswith("%"):
        return float(token[:-1].replace(",", "")) / 100.0
    return float(token.replace(",", ""))


def _resolve(token: str, results: list[Answer]) -> float:
    if _is_reference(token):
        value = results[int(token[1:])]
        if isinstance(value, str):
            raise ValueError(f"cannot use boolean step result {token} in arithmetic")
        return value
    return literal_to_number(token)


def execute_program(program: str) -> Answer:
    results: list[Answer] = []
    for op, args in parse_program(program):
        if op not in _ARITH and op != "greater":
            raise ValueError(f"unsupported op {op!r} (table ops are not executed)")
        a, b = (_resolve(t, results) for t in args[:2])
        results.append(("yes" if a > b else "no") if op == "greater" else float(_ARITH[op](a, b)))
    return results[-1]


def program_literals(program: str) -> tuple[str, ...]:
    """Document numbers used by the program, in first-seen order (no constants/references)."""
    seen: dict[str, None] = {}
    for _, args in parse_program(program):
        for token in args:
            if not (_is_constant(token) or _is_reference(token) or token == "none"):
                seen.setdefault(token, None)
    return tuple(seen)


def substitute_literals(program: str, mapping: Mapping[str, str]) -> str:
    steps = parse_program(program)
    return ", ".join(f"{op}({', '.join(mapping.get(t, t) for t in args)})" for op, args in steps)


FINQA_DECIMALS = 5  # FinQA stores exe_ans rounded to 5 decimals; its evaluator compares at that precision


def answers_match(predicted: Answer, gold: Answer, rel_tol: float = 1e-4, abs_tol: float = 1e-6) -> bool:
    if isinstance(predicted, str) or isinstance(gold, str):
        return predicted == gold
    if round(predicted, FINQA_DECIMALS) == round(gold, FINQA_DECIMALS):
        return True
    return math.isclose(predicted, gold, rel_tol=rel_tol, abs_tol=abs_tol)
