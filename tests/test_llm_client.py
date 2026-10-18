import json

import pytest

from numpriv.llm import LLMError, OllamaClient, validate

SCHEMA = {
    "type": "object",
    "properties": {"sector": {"type": "string"}, "price_level": {"type": "number"}},
    "required": ["sector", "price_level"],
}


class FakeTransport:
    """Returns queued Ollama /api/chat responses and records every request payload."""

    def __init__(self, contents):
        self.contents = list(contents)
        self.requests = []

    def __call__(self, url, payload, timeout):
        self.requests.append((url, payload))
        return {"message": {"content": self.contents.pop(0)}, "model": payload["model"]}


def test_request_is_deterministic_and_schema_constrained():
    t = FakeTransport([json.dumps({"sector": "bakery", "price_level": 1.1})])
    out = OllamaClient("llama3.1:8b", seed=7, transport=t).generate_json("sys", "user", SCHEMA)
    assert out == {"sector": "bakery", "price_level": 1.1}
    url, payload = t.requests[0]
    assert url.endswith("/api/chat")
    assert payload["format"] == SCHEMA
    assert payload["options"]["temperature"] == 0 and payload["options"]["seed"] == 7
    assert payload["stream"] is False
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]


def test_retries_on_invalid_json_then_succeeds():
    t = FakeTransport(["not json", json.dumps({"sector": "x", "price_level": 2})])
    out = OllamaClient("m", transport=t, max_retries=2).generate_json("s", "u", SCHEMA)
    assert out["price_level"] == 2 and len(t.requests) == 2


def test_gives_up_after_max_retries_with_clear_error():
    t = FakeTransport(["{}", "{}", "{}"])
    with pytest.raises(LLMError, match="price_level"):
        OllamaClient("m", transport=t, max_retries=2).generate_json("s", "u", SCHEMA)
    assert len(t.requests) == 3


def test_transport_failure_is_reported_not_swallowed():
    def broken(url, payload, timeout):
        raise OSError("connection refused")

    with pytest.raises(LLMError, match="connection refused"):
        OllamaClient("m", transport=broken, max_retries=0).generate_json("s", "u", SCHEMA)


def test_validate_checks_required_keys_and_types():
    assert validate({"sector": "a", "price_level": 1}, SCHEMA) == []
    assert validate({"sector": "a"}, SCHEMA) == ["missing key: price_level"]
    assert validate({"sector": 3, "price_level": 1}, SCHEMA) == ["sector: expected string"]
    assert validate({"sector": "a", "price_level": True}, SCHEMA) == ["price_level: expected number"]
    enum_schema = {"type": "object", "properties": {"p": {"type": "string", "enum": ["a", "b"]}}, "required": ["p"]}
    assert validate({"p": "c"}, enum_schema) == ["p: 'c' not in ['a', 'b']"]
