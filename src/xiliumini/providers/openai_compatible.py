from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from xiliumini.config import Settings
from xiliumini.errors import (
    AuthenticationProviderError,
    ProviderError,
    RateLimitProviderError,
    TimeoutProviderError,
)


def create_chat_model(settings: Settings) -> ChatOpenAI:
    """Create a streaming OpenAI-compatible chat model from validated settings."""

    kwargs: dict[str, Any] = {
        "api_key": settings.api_key,
        "model": settings.model,
        "temperature": settings.temperature,
        "timeout": settings.timeout_seconds,
        "streaming": True,
    }
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    return ChatOpenAI(**kwargs)


def _status_code(error: Exception) -> int | None:
    direct = getattr(error, "status_code", None)
    if isinstance(direct, int):
        return direct
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def classify_provider_error(error: Exception) -> ProviderError:
    """Map provider failures to stable, secret-free application errors."""

    status = _status_code(error)
    if status in {401, 403}:
        return AuthenticationProviderError("Provider authentication failed")
    if status == 429:
        return RateLimitProviderError("Provider rate limit exceeded")
    if isinstance(error, (TimeoutError, TimeoutProviderError)):
        return TimeoutProviderError("Provider request timed out")
    return ProviderError("Provider request failed")
