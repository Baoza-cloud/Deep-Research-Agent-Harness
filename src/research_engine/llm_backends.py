"""Hot-swappable OpenAI-compatible LLM backends.

No provider SDK is imported at module import time.  Configuration is supplied
through environment variables, keeping secrets out of plans, traces and CLI
arguments.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
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
    input_cost_per_million_usd: float = 0.0
    output_cost_per_million_usd: float = 0.0


class OpenAICompatibleLLM:
    def __init__(self, config: LLMBackendConfig):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The 'openai' package is required for LLM backends") from exc
        kwargs: dict[str, Any] = {
            "api_key": config.api_key,
            "timeout": config.request_timeout_seconds,
            # Retry here instead of inside the SDK so actual attempts are
            # observable and can be attributed to a Trace span.
            "max_retries": 0,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self.client = OpenAI(**kwargs)
        self.config = config
        self._usage_lock = threading.Lock()
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "retry_count": 0,
            "call_count": 0,
            "duration_ms": 0.0,
            "measurement": "provider",
        }

    async def __call__(self, prompt: str) -> str:
        def invoke() -> Any:
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "stream": False,
            }
            if self.config.extra_body:
                kwargs["extra_body"] = self.config.extra_body
            return self.client.chat.completions.create(**kwargs)

        started = time.monotonic()
        retries = 0
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await asyncio.to_thread(invoke)
                break
            except Exception as exc:
                if attempt >= self.config.max_retries or not self._retryable(exc):
                    raise
                retries += 1
                await asyncio.sleep(min(2.0, 0.25 * (2**attempt)))

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
            or prompt_tokens + completion_tokens
        )
        cost = (
            prompt_tokens * self.config.input_cost_per_million_usd
            + completion_tokens * self.config.output_cost_per_million_usd
        ) / 1_000_000
        elapsed_ms = (time.monotonic() - started) * 1000
        with self._usage_lock:
            self._usage["prompt_tokens"] += prompt_tokens
            self._usage["completion_tokens"] += completion_tokens
            self._usage["total_tokens"] += total_tokens
            self._usage["cost_usd"] += cost
            self._usage["retry_count"] += retries
            self._usage["call_count"] += 1
            self._usage["duration_ms"] += elapsed_ms
        return str(response.choices[0].message.content or "")

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status in {408, 409, 429} or (isinstance(status, int) and status >= 500):
            return True
        name = type(exc).__name__
        return name in {
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "RateLimitError",
        }

    def usage_snapshot(self) -> dict[str, Any]:
        with self._usage_lock:
            snapshot = dict(self._usage)
        snapshot["cost_usd"] = round(float(snapshot["cost_usd"]), 9)
        snapshot["duration_ms"] = round(float(snapshot["duration_ms"]), 3)
        snapshot["pricing"] = {
            "input_per_million_usd": self.config.input_cost_per_million_usd,
            "output_per_million_usd": self.config.output_cost_per_million_usd,
        }
        return snapshot


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
    input_cost = float(os.getenv(f"{prefix}_INPUT_COST_PER_MILLION_USD", "0"))
    output_cost = float(os.getenv(f"{prefix}_OUTPUT_COST_PER_MILLION_USD", "0"))
    if request_timeout_seconds <= 0:
        raise ValueError(f"{prefix}_TIMEOUT_SECONDS must be positive")
    if max_retries < 0:
        raise ValueError(f"{prefix}_MAX_RETRIES must be >= 0")
    if input_cost < 0 or output_cost < 0:
        raise ValueError(f"{prefix} token costs must be >= 0")
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
        input_cost_per_million_usd=input_cost,
        output_cost_per_million_usd=output_cost,
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
