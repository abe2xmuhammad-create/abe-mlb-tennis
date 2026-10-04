"""
Provider-agnostic LLM wrapper.
Pick model via LLM_PROVIDER env var (or --llm flag). No code changes needed to switch.

Supported: ollama (local weights, free — also proxies Ollama-cloud models
like deepseek-v4-flash:cloud), gemini, deepseek, anthropic, openai (or any
OpenAI-compatible endpoint: OpenRouter, Groq, vLLM, ...).

Stronger reasoning without a new key: pass an Ollama-cloud model, e.g.
    --llm ollama --llm-model deepseek-v4-flash:cloud

Usage:
    from llm_provider import get_llm
    llm = get_llm()                # reads LLM_PROVIDER from env, default "ollama"
    llm = get_llm("gemini")        # or force one explicitly
    text = llm.generate("Analyze this matchup: ...")

Failures are typed so callers can tell them apart:
    LLMConfigError    bad/missing configuration (no API key) — fail fast, no I/O
    LLMResponseError  provider answered but the answer was unusable (blocked,
                      truncated, empty)
    requests.RequestException  transport failure, after bounded retries

Every provider raises rather than returning an empty string, so "the model said
nothing" can never be mistaken for a successful result.

Endpoints are called through a pooled Session with bounded retry/backoff on
429/5xx (see _session). Timeouts are per-provider and overridable:
    <PROVIDER>_TIMEOUT or LLM_TIMEOUT          (seconds)
    LLM_MAX_RETRIES, LLM_BACKOFF               (retry policy)
"""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Protocol
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class LLMError(RuntimeError):
    """Base class for provider-level failures."""


class LLMConfigError(LLMError):
    """Missing or invalid configuration, e.g. an API key that was never set.

    Subclasses RuntimeError so existing `except RuntimeError` callers still work.
    """


class LLMResponseError(LLMError):
    """The provider answered, but the answer was unusable.

    Covers safety blocks, truncated generations, and empty responses. Kept
    distinct from transport failures: retrying an identical prompt that the
    provider already refused is not useful.
    """


# Transient statuses worth retrying. A generation request has no server-side
# side effect, so repeating one is safe.
RETRY_STATUSES = (429, 500, 502, 503, 504)

DEFAULT_TIMEOUTS = {
    "ollama": 300.0,   # local weights on modest hardware can be slow
    "gemini": 60.0,
    "deepseek": 180.0,  # deepseek-reasoner thinks for a while before answering
    "anthropic": 120.0,
    "openai": 60.0,
}

# Current defaults. gemini-2.0-flash was shut down by Google on 2026-06-01 and
# now 404s; gpt-4o-mini is a legacy model whose family was retired from Azure on
# 2026-03-31. Tests assert these values, so a stale default fails CI instead of
# failing silently at 4am on a live board.
DEFAULT_MODELS = {
    "ollama": "qwen3.5:4b",
    "gemini": "gemini-3.5-flash",
    "deepseek": "deepseek-chat",
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-5-mini",
}


def _model(provider: str, override: str | None) -> str:
    """Explicit override, else <PROVIDER>_MODEL, else the current default.

    `or` rather than getenv's default so an empty env var (a stray "-" line in
    a .env file) falls back instead of sending an empty model name.
    """
    return override or os.getenv(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return max(0, int(value))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return max(0.0, float(value))
    except ValueError:
        return default


def _timeout(provider: str) -> float:
    """Per-provider timeout, overridable by <PROVIDER>_TIMEOUT then LLM_TIMEOUT."""
    for key in (f"{provider.upper()}_TIMEOUT", "LLM_TIMEOUT"):
        value = os.getenv(key)
        if value:
            try:
                return max(1.0, float(value))
            except ValueError:
                break
    return DEFAULT_TIMEOUTS.get(provider, 60.0)


def _new_session() -> requests.Session:
    """A Session with connection pooling and bounded retry/backoff.

    Without this, a single transient 429 or 5xx degraded one game's analysis
    permanently, and every one of a slate's 15+ calls paid for a fresh TLS
    handshake.
    """
    retries = _env_int("LLM_MAX_RETRIES", 3)
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=_env_float("LLM_BACKOFF", 0.5),
        status_forcelist=RETRY_STATUSES,
        allowed_methods=frozenset({"POST"}),
        respect_retry_after_header=True,
        # Return the last response instead of raising MaxRetryError, so the
        # caller's raise_for_status() produces one consistent error type.
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_local = threading.local()


def _session() -> requests.Session:
    """One Session per thread.

    requests.Session is not documented as thread-safe and rag_analyze fans games
    out across a worker pool, so each thread gets its own pooled session.
    """
    session = getattr(_local, "session", None)
    if session is None:
        session = _new_session()
        _local.session = session
    return session


def _json_object(resp: requests.Response, provider: str) -> dict[str, Any]:
    """Decoded JSON body, or a clear error instead of a bare AttributeError."""
    try:
        parsed: Any = resp.json()
    except ValueError as exc:
        raise LLMResponseError(f"{provider} returned a non-JSON response: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMResponseError(f"{provider} returned a non-object JSON response")
    body: dict[str, Any] = parsed
    return body


# Hostnames treated as "local, no API key needed". 0.0.0.0 is included because a
# locally launched server is commonly addressed that way. This is a hostname
# comparison, not a socket bind, so bandit B104 / ruff S104 do not apply.
_LOCAL_HOSTNAMES = ("localhost", "127.0.0.1", "0.0.0.0", "::1")  # nosec B104  # noqa: S104


def _is_local_url(url: str) -> bool:
    """True for loopback endpoints, which need no API key (local vLLM, llama.cpp)."""
    return (urlparse(url).hostname or "") in _LOCAL_HOSTNAMES


class LLMProvider(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        ...


class OllamaProvider(LLMProvider):
    """Local, free. Requires `ollama serve` running."""

    name = "ollama"

    def __init__(self, model: str | None = None):
        self.model = _model("ollama", model)
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.timeout = _timeout(self.name)

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            # Reasoning models (qwen3.5 etc) emit into a separate "thinking"
            # field and leave "response" empty, generating unbounded reasoning
            # until the request times out. Disable it for short factual output.
            "think": os.getenv("OLLAMA_THINK", "").lower() in ("1", "true", "yes"),
        }
        resp = _session().post(
            f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        data = _json_object(resp, "Ollama")
        # Fall back to "thinking" if a model ignores think=False.
        text = data.get("response") or data.get("thinking")
        if not text:
            raise LLMResponseError(
                "Ollama returned an empty response — the model may have spent its "
                "whole budget on reasoning. Set OLLAMA_THINK=1 if that is wanted, "
                "or choose a non-reasoning model."
            )
        return str(text)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str | None = None):
        self.model = _model("gemini", model)
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise LLMConfigError("GEMINI_API_KEY not set")
        self.timeout = _timeout(self.name)

    def generate(self, prompt: str, system: str | None = None) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        # The key travels in a header, never the URL: requests embeds the full
        # URL in every exception message and traceback, which used to leak the
        # credential into logs and into the generated report file.
        headers = {"x-goog-api-key": self.api_key, "content-type": "application/json"}
        payload: dict[str, Any] = {"contents": [{"parts": [{"text": prompt}]}]}
        if system:
            # Dedicated field — stronger instruction adherence than prepending
            # the system text to the user turn.
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        resp = _session().post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return _gemini_text(_json_object(resp, "Gemini"))


def _gemini_text(data: dict[str, Any]) -> str:
    """Pull text out of a Gemini response, or explain why there is none.

    Indexing straight into data["candidates"][0]["content"]["parts"][0]["text"]
    raised a bare KeyError on a safety-blocked or truncated response, which
    reads as an internal bug rather than "the model declined".
    """
    candidates = data.get("candidates") or []
    if not candidates:
        block = (data.get("promptFeedback") or {}).get("blockReason")
        if block:
            raise LLMResponseError(f"Gemini blocked the prompt (blockReason={block})")
        raise LLMResponseError("Gemini returned no candidates")

    candidate = candidates[0] or {}
    finish = candidate.get("finishReason")
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))
    if not text:
        if finish and finish != "STOP":
            raise LLMResponseError(f"Gemini returned no text (finishReason={finish})")
        raise LLMResponseError("Gemini returned an empty response")
    return text


class DeepSeekProvider(LLMProvider):
    name = "deepseek"

    def __init__(self, model: str | None = None):
        self.model = _model("deepseek", model)
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise LLMConfigError("DEEPSEEK_API_KEY not set")
        self.timeout = _timeout(self.name)

    def generate(self, prompt: str, system: str | None = None) -> str:
        url = "https://api.deepseek.com/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        resp = _session().post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = _json_object(resp, "DeepSeek")
        return _chat_message_text(data, "DeepSeek")


def _chat_message_text(data: dict[str, Any], provider: str) -> str:
    """Text from an OpenAI-shaped chat completion, or a clear error.

    Reasoner models put the chain in reasoning_content and may leave content
    empty; prefer the answer, fall back to the reasoning.
    """
    choices = data.get("choices") or []
    if not choices:
        raise LLMResponseError(f"{provider} returned no choices")
    message = (choices[0] or {}).get("message") or {}
    text = message.get("content") or message.get("reasoning_content")
    if not text:
        raise LLMResponseError(f"{provider} returned an empty message")
    return str(text)


class AnthropicProvider(LLMProvider):
    """Claude — strongest reasoning of the wired providers."""

    name = "anthropic"

    def __init__(self, model: str | None = None):
        self.model = _model("anthropic", model)
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise LLMConfigError("ANTHROPIC_API_KEY not set")
        self.timeout = _timeout(self.name)

    def generate(self, prompt: str, system: str | None = None) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": _env_int("ANTHROPIC_MAX_TOKENS", 1024),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            # Top-level system param, not a pseudo-user turn.
            payload["system"] = system
        resp = _session().post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = _json_object(resp, "Anthropic")
        blocks = data.get("content") or []
        text = "".join(
            str(b.get("text", ""))
            for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )
        if not text:
            stop = data.get("stop_reason")
            if stop and stop not in ("end_turn", "stop_sequence"):
                raise LLMResponseError(f"Anthropic returned no text (stop_reason={stop})")
            raise LLMResponseError("Anthropic returned an empty response")
        return text


class OpenAICompatProvider(LLMProvider):
    """Any OpenAI-compatible endpoint (OpenAI itself, OpenRouter, Groq, local vLLM, etc)."""

    name = "openai"

    def __init__(self, model: str | None = None):
        self.model = _model("openai", model)
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not self.api_key and not _is_local_url(self.base_url):
            raise LLMConfigError(
                "OPENAI_API_KEY not set (only a loopback OPENAI_BASE_URL may omit it)"
            )
        self.timeout = _timeout(self.name)

    def generate(self, prompt: str, system: str | None = None) -> str:
        url = f"{self.base_url}/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        resp = _session().post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return _chat_message_text(_json_object(resp, "OpenAI-compatible endpoint"), "OpenAI-compatible endpoint")


class _ProviderFactory(Protocol):
    """Constructor shape every provider shares."""

    def __call__(self, model: str | None = None) -> LLMProvider:
        ...


_PROVIDERS: dict[str, _ProviderFactory] = {
    "ollama": OllamaProvider,
    "gemini": GeminiProvider,
    "deepseek": DeepSeekProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAICompatProvider,
}


def get_llm(provider: str | None = None, model: str | None = None) -> LLMProvider:
    """Return an LLM provider instance. Pick via arg, else LLM_PROVIDER env, else 'ollama'.

    Whitespace is stripped and an empty value falls back to the documented
    default, so a stray `LLM_PROVIDER=` line in a .env file behaves the way the
    contract says instead of failing with "Unknown LLM_PROVIDER ''".
    """
    name = (provider or os.getenv("LLM_PROVIDER") or "ollama").strip().lower()
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Options: {list(_PROVIDERS)}")
    return _PROVIDERS[name](model=model)
