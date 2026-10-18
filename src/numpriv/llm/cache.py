"""Append-only JSONL cache of successful LLM responses.

Lets a long cloud run resume after an interruption and lets identical requests
(e.g. A2's masked prose, which is the same under every defense) be paid for once.
Only validated responses are stored; failures are always retried on the next run.
"""

import hashlib
import json
import logging
import threading
from pathlib import Path

from numpriv.llm.client import LLMClient

LOG = logging.getLogger(__name__)


def request_key(client: LLMClient, system: str, user: str, schema: dict, max_tokens: int | None) -> str:
    identity = {
        "model": client.model,
        "seed": getattr(client, "seed", None),
        "temperature": getattr(client, "temperature", None),
        "system": system,
        "user": user,
        "schema": schema,
        "max_tokens": max_tokens,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _load(path: Path) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    if not path.exists():
        return entries
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            record = json.loads(line)
            entries[record["key"]] = record["response"]
        except (ValueError, KeyError, TypeError) as exc:
            LOG.warning("skipping corrupt cache line %d in %s: %s", n, path, exc)
    return entries


class CachedClient:
    """Wraps any LLMClient; thread-safe so the runner can issue requests concurrently."""

    def __init__(self, inner: LLMClient, path: Path):
        self.inner = inner
        self.model = inner.model
        self.path = Path(path)
        self._entries = _load(self.path)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def generate_json(self, system: str, user: str, schema: dict, max_tokens: int | None = None) -> dict:
        key = request_key(self.inner, system, user, schema, max_tokens)
        with self._lock:
            if key in self._entries:
                self._hits += 1
                return dict(self._entries[key])
        response = self.inner.generate_json(system, user, schema, max_tokens=max_tokens)  # LLMError propagates
        with self._lock:
            self._misses += 1
            self._entries[key] = dict(response)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps({"key": key, "model": self.model, "response": response}) + "\n")
        return dict(response)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self._hits, "misses": self._misses}
