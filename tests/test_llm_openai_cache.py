import json

import pytest

from numpriv.llm import CachedClient, LLMError, OpenAICompatClient

SCHEMA = {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}


class FakeOpenAI:
    """Returns queued OpenAI-style chat completions and records every request."""

    def __init__(self, contents):
        self.contents = list(contents)
        self.requests = []

    def __call__(self, url, payload, timeout):
        self.requests.append((url, payload))
        return {"choices": [{"message": {"content": self.contents.pop(0)}, "finish_reason": "stop"}]}


def test_openai_request_is_deterministic_and_schema_constrained():
    t = FakeOpenAI([json.dumps({"x": 1.5})])
    client = OpenAICompatClient("m", base_url="http://host:8000/v1", seed=3, max_tokens=64, transport=t)
    assert client.generate_json("sys", "user", SCHEMA) == {"x": 1.5}
    url, payload = t.requests[0]
    assert url == "http://host:8000/v1/chat/completions"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"] == SCHEMA
    assert payload["temperature"] == 0 and payload["seed"] == 3 and payload["max_tokens"] == 64
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]


def test_openai_max_tokens_can_be_overridden_per_call():
    t = FakeOpenAI([json.dumps({"x": 1})])
    OpenAICompatClient("m", transport=t).generate_json("s", "u", SCHEMA, max_tokens=999)
    assert t.requests[0][1]["max_tokens"] == 999


def test_openai_retries_invalid_output_then_raises():
    t = FakeOpenAI(["{}", "oops", "{}"])
    with pytest.raises(LLMError, match="3 attempts"):
        OpenAICompatClient("m", max_retries=2, transport=t).generate_json("s", "u", SCHEMA)
    assert len(t.requests) == 3


def test_openai_malformed_envelope_is_an_llm_error():
    def broken(url, payload, timeout):
        return {"choices": []}

    with pytest.raises(LLMError):
        OpenAICompatClient("m", max_retries=0, transport=broken).generate_json("s", "u", SCHEMA)


class CountingClient:
    model = "m"
    seed = 0
    temperature = 0.0

    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def generate_json(self, system, user, schema, max_tokens=None):
        self.calls += 1
        if self.fail:
            raise LLMError("down")
        return {"x": float(len(user))}


def test_cache_serves_repeats_and_persists_across_instances(tmp_path):
    path = tmp_path / "cache.jsonl"
    inner = CountingClient()
    cached = CachedClient(inner, path)
    first = cached.generate_json("s", "hello", SCHEMA)
    assert cached.generate_json("s", "hello", SCHEMA) == first
    assert inner.calls == 1 and cached.stats() == {"hits": 1, "misses": 1}
    cached.generate_json("s", "other", SCHEMA)
    assert inner.calls == 2

    fresh_inner = CountingClient()
    reloaded = CachedClient(fresh_inner, path)
    assert reloaded.generate_json("s", "hello", SCHEMA) == first
    assert fresh_inner.calls == 0


def test_cache_key_depends_on_max_tokens_and_model(tmp_path):
    inner = CountingClient()
    cached = CachedClient(inner, tmp_path / "c.jsonl")
    cached.generate_json("s", "u", SCHEMA)
    cached.generate_json("s", "u", SCHEMA, max_tokens=10)
    other_model = CountingClient()
    other_model.model = "m2"
    CachedClient(other_model, tmp_path / "c.jsonl").generate_json("s", "u", SCHEMA)
    assert inner.calls == 2 and other_model.calls == 1


def test_cache_does_not_store_failures(tmp_path):
    path = tmp_path / "c.jsonl"
    with pytest.raises(LLMError):
        CachedClient(CountingClient(fail=True), path).generate_json("s", "u", SCHEMA)
    assert not path.exists() or path.read_text() == ""


def test_cache_skips_corrupt_lines_with_a_warning(tmp_path, caplog):
    path = tmp_path / "c.jsonl"
    path.write_text("not json\n")
    cached = CachedClient(CountingClient(), path)
    assert cached.generate_json("s", "u", SCHEMA) == {"x": 1.0}
    assert "corrupt" in caplog.text
