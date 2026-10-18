"""OpenAI-compatible /chat/completions client (vLLM on the cloud GPU; Ollama's /v1 endpoint locally).

The same code path serves local debugging and the paper runs; only ``base_url`` and
``model`` change. Output is constrained with ``response_format: json_schema`` and then
validated again locally (model output is external data).
"""

from dataclasses import dataclass, field

from numpriv.llm.client import Transport, generate_validated, http_post

DEFAULT_BASE_URL = "http://localhost:8000/v1"  # vLLM's default; Ollama is http://localhost:11434/v1
DEFAULT_MAX_TOKENS = 512


@dataclass(frozen=True)
class OpenAICompatClient:
    model: str
    base_url: str = DEFAULT_BASE_URL
    seed: int = 0
    temperature: float = 0.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: float = 300.0
    max_retries: int = 2
    transport: Transport = field(default=http_post, repr=False)

    def generate_json(self, system: str, user: str, schema: dict, max_tokens: int | None = None) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}},
        }
        return generate_validated(self.model, self.max_retries, schema, lambda: self._content(payload))

    def _content(self, payload: dict) -> str:
        reply = self.transport(f"{self.base_url.rstrip('/')}/chat/completions", payload, self.timeout)
        return reply["choices"][0]["message"]["content"]
