"""Hot-swappable OpenAI-compatible LLM backends.

No provider SDK is imported at module import time.  Configuration is supplied
through environment variables, keeping secrets out of plans, traces and CLI
arguments.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any


class LLMProvider(str, Enum):
    DEEPSEEK = "deepseek"
    MIMO = "mimo"
    VLLM = "vllm"
    OPENAI = "openai"


@dataclass(frozen=True)
class LLMBackendConfig:
    provider: LLMProvider
    model: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.1
    extra_body: dict[str, Any] | None = None
    request_timeout_seconds: float = 45.0
    max_retries: int = 1


class OpenAICompatibleLLM:
    def __init__(self, config: LLMBackendConfig):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The 'openai' package is required for LLM backends") from exc
        kwargs: dict[str, Any] = {
            "api_key": config.api_key,
            "timeout": config.request_timeout_seconds,
            "max_retries": config.max_retries,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self.client = OpenAI(**kwargs)
        self.config = config

    async def __call__(self, prompt: str) -> str:
        def invoke() -> str:
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "stream": False,
            }
            if self.config.extra_body:
                kwargs["extra_body"] = self.config.extra_body
            response = self.client.chat.completions.create(**kwargs)
            return str(response.choices[0].message.content or "")

        return await asyncio.to_thread(invoke)


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def backend_config(provider: str | LLMProvider, model: str | None = None) -> LLMBackendConfig:
    provider = LLMProvider(provider)
    prefix = provider.value.upper()
    provider_defaults: dict[LLMProvider, dict[str, Any]] = {
        LLMProvider.DEEPSEEK: {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
            "extra_body": {"thinking": {"type": "disabled"}},
        },
        LLMProvider.MIMO: {"base_url": None, "model": None, "extra_body": None},
        LLMProvider.VLLM: {
            "base_url": "http://127.0.0.1:8000/v1",
            "model": None,
            "extra_body": None,
        },
        LLMProvider.OPENAI: {"base_url": None, "model": None, "extra_body": None},
    }
    defaults = provider_defaults[provider]

    resolved_model = model or os.getenv(f"{prefix}_MODEL") or defaults["model"]
    if not resolved_model:
        raise ValueError(f"Provide --model or set {prefix}_MODEL")
    base_url = os.getenv(f"{prefix}_BASE_URL") or defaults["base_url"]
    request_timeout_seconds = float(os.getenv(f"{prefix}_TIMEOUT_SECONDS", "45"))
    max_retries = int(os.getenv(f"{prefix}_MAX_RETRIES", "1"))
    if request_timeout_seconds <= 0:
        raise ValueError(f"{prefix}_TIMEOUT_SECONDS must be positive")
    if max_retries < 0:
        raise ValueError(f"{prefix}_MAX_RETRIES must be >= 0")
    if provider is LLMProvider.VLLM:
        api_key = os.getenv("VLLM_API_KEY", "EMPTY")
    else:
        api_key = _required(f"{prefix}_API_KEY")
    return LLMBackendConfig(
        provider=provider,
        model=resolved_model,
        api_key=api_key,
        base_url=base_url,
        extra_body=defaults["extra_body"],
        request_timeout_seconds=request_timeout_seconds,
        max_retries=max_retries,
    )


def build_llm(provider: str | LLMProvider, model: str | None = None) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(backend_config(provider, model=model))


def auto_provider() -> LLMProvider | None:
    for provider in (
        LLMProvider.DEEPSEEK,
        LLMProvider.MIMO,
        LLMProvider.OPENAI,
    ):
        if os.getenv(f"{provider.value.upper()}_API_KEY"):
            return provider
    if os.getenv("VLLM_MODEL"):
        return LLMProvider.VLLM
    return None
