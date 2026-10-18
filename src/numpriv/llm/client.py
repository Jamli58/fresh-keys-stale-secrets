"""Minimal Ollama /api/chat client: schema-constrained JSON, deterministic decoding, bounded retries.

Model output is external data: every response is parsed and validated against the
schema before use, and failures raise LLMError instead of returning partial results.
"""

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

LOG = logging.getLogger(__name__)
DEFAULT_URL = "http://localhost:11434"
Transport = Callable[[str, dict, float], dict]


class LLMError(RuntimeError):
    """The model could not produce a valid response (transport error or schema violation)."""


def http_post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


_TYPES = {"string": (str,), "number": (int, float), "integer": (int,), "boolean": (bool,), "object": (dict,)}


def validate(obj: Any, schema: dict) -> list[str]:
    """Flat check of an object schema: required keys, primitive types, enums. Returns problems found."""
    if not isinstance(obj, dict):
        return ["response is not a JSON object"]
    problems = [f"missing key: {k}" for k in schema.get("required", []) if k not in obj]
    for key, spec in schema.get("properties", {}).items():
        if key not in obj:
            continue
        value, kind = obj[key], spec.get("type")
        wrong_type = kind in _TYPES and (not isinstance(value, _TYPES[kind]) or (kind != "boolean" and isinstance(value, bool)))
        if wrong_type:
            problems.append(f"{key}: expected {kind}")
        elif "enum" in spec and value not in spec["enum"]:
            problems.append(f"{key}: {value!r} not in {spec['enum']}")
    return problems


class LLMClient(Protocol):
    """What attackers need from a model: schema-valid JSON or an LLMError."""

    model: str

    def generate_json(self, system: str, user: str, schema: dict, max_tokens: int | None = None) -> dict: ...


def generate_validated(model: str, max_retries: int, schema: dict, request: Callable[[], str]) -> dict:
    """Call ``request`` (returns the raw message text) until it parses and validates; else raise LLMError."""
    last_error = "no attempt made"
    for attempt in range(max_retries + 1):
        try:
            obj = json.loads(request())
        except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:  # transport or envelope/JSON failure
            last_error = f"{type(exc).__name__}: {exc}"
            LOG.warning("%s attempt %d failed: %s", model, attempt + 1, last_error)
            continue
        problems = validate(obj, schema)
        if not problems:
            return obj
        last_error = "; ".join(problems)
        LOG.warning("%s attempt %d invalid: %s", model, attempt + 1, last_error)
    raise LLMError(f"{model}: no valid response after {max_retries + 1} attempts ({last_error})")


@dataclass(frozen=True)
class OllamaClient:
    model: str
    seed: int = 0
    temperature: float = 0.0
    base_url: str = DEFAULT_URL
    timeout: float = 300.0
    max_retries: int = 2
    transport: Transport = field(default=http_post, repr=False)

    def generate_json(self, system: str, user: str, schema: dict, max_tokens: int | None = None) -> dict:
        options = {"temperature": self.temperature, "seed": self.seed}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "options": options,
        }
        return generate_validated(self.model, self.max_retries, schema, lambda: self._content(payload))

    def _content(self, payload: dict) -> str:
        return self.transport(f"{self.base_url}/api/chat", payload, self.timeout)["message"]["content"]
