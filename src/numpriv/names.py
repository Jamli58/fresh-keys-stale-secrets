"""Fictional firm and product names built from random syllables.

The names carry no real-world referent, so an attacker LLM cannot look up a
firm; the prose-only probe (R023) checks this empirically.
"""

import numpy as np

_ONSETS = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "kr", "tr")
_VOWELS = ("a", "e", "i", "o", "u", "ae", "io")
_CODAS = ("", "n", "l", "r", "s", "th", "x")


def pseudo_word(rng: np.random.Generator, syllables: int = 2) -> str:
    parts = [
        _ONSETS[rng.integers(len(_ONSETS))]
        + _VOWELS[rng.integers(len(_VOWELS))]
        + _CODAS[rng.integers(len(_CODAS))]
        for _ in range(syllables)
    ]
    return "".join(parts).capitalize()


def unique_firm_names(rng: np.random.Generator, count: int, suffixes: tuple[str, ...]) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    while len(names) < count:
        candidate = f"{pseudo_word(rng, 2 + int(rng.integers(2)))} {suffixes[rng.integers(len(suffixes))]}"
        if candidate not in seen:
            seen.add(candidate)
            names.append(candidate)
    return tuple(names)
