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
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import requests


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        ...


class OllamaProvider(LLMProvider):
    """Local, free. Requires `ollama serve` running."""

    name = "ollama"

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            # Reasoning models (qwen3.5 etc) emit into a separate "thinking"
            # field and leave "response" empty, generating unbounded reasoning
            # until the request times out. Disable it for short factual output.
            "think": os.getenv("OLLAMA_THINK", "").lower() in ("1", "true", "yes"),
        }
        timeout = float(os.getenv("OLLAMA_TIMEOUT", "300"))
        resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # Fall back to "thinking" if a model ignores think=False.
        return data.get("response") or data.get("thinking", "")


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

    def generate(self, prompt: str, system: str | None = None) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload: dict = {"contents": [{"parts": [{"text": prompt}]}]}
        if system:
            # Dedicated field — stronger instruction adherence than prepending
            # the system text to the user turn.
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


class DeepSeekProvider(LLMProvider):
    name = "deepseek"

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")

    def generate(self, prompt: str, system: str | None = None) -> str:
        url = "https://api.deepseek.com/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "messages": messages}
        # deepseek-reasoner thinks for a while before answering.
        resp = requests.post(url, json=payload, headers=headers, timeout=180)
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]
        # Reasoner models put the chain in reasoning_content and may leave
        # content empty; prefer the answer, fall back to the reasoning.
        return message.get("content") or message.get("reasoning_content", "")


class AnthropicProvider(LLMProvider):
    """Claude — strongest reasoning of the wired providers."""

    name = "anthropic"

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

    def generate(self, prompt: str, system: str | None = None) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: dict = {
            "model": self.model,
            "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024")),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            # Top-level system param, not a pseudo-user turn.
            payload["system"] = system
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        blocks = resp.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


class OpenAICompatProvider(LLMProvider):
    """Any OpenAI-compatible endpoint (OpenAI itself, OpenRouter, Groq, local vLLM, etc)."""

    name = "openai"

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

    def generate(self, prompt: str, system: str | None = None) -> str:
        url = f"{self.base_url}/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "messages": messages}
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


_PROVIDERS = {
    "ollama": OllamaProvider,
    "gemini": GeminiProvider,
    "deepseek": DeepSeekProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAICompatProvider,
}


def get_llm(provider: str | None = None, model: str | None = None) -> LLMProvider:
    """Return an LLM provider instance. Pick via arg, else LLM_PROVIDER env, else 'ollama'."""
    name = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Options: {list(_PROVIDERS)}")
    return _PROVIDERS[name](model=model)
