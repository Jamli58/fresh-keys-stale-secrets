"""LLM access for the M2 attackers and the D2 topic shift.

OpenAICompatClient is the main backend (vLLM on the cloud GPU, Ollama's /v1 locally);
OllamaClient uses Ollama's native API (smoke test). CachedClient wraps either.
"""

from numpriv.llm.cache import CachedClient
from numpriv.llm.client import LLMClient, LLMError, OllamaClient, validate
from numpriv.llm.openai_compat import OpenAICompatClient

__all__ = ["CachedClient", "LLMClient", "LLMError", "OllamaClient", "OpenAICompatClient", "validate"]
