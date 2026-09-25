from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from xiliumini.config import Settings
from xiliumini.errors import (
    AuthenticationProviderError,
    ProviderError,
    RateLimitProviderError,
    TimeoutProviderError,
)
from xiliumini.providers import openai_compatible as provider


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings.from_env(
        {
            "XILIUMINI_API_KEY": "provider-secret",
            "XILIUMINI_MODEL": "deepseek-chat",
            "XILIUMINI_BASE_URL": "https://api.example.test/v1",
            "XILIUMINI_TEMPERATURE": "0.25",
            "XILIUMINI_TIMEOUT_SECONDS": "12",
            "XILIUMINI_DATA_DIR": str(tmp_path),
        }
    )


def test_create_chat_model_maps_validated_settings(monkeypatch, settings: Settings) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(provider, "ChatOpenAI", fake_chat_openai)

    result = provider.create_chat_model(settings)

    assert result is sentinel
    assert captured["model"] == "deepseek-chat"
    assert captured["base_url"] == "https://api.example.test/v1"
    assert captured["temperature"] == 0.25
    assert captured["timeout"] == 12
    assert captured["api_key"] == settings.api_key


@dataclass
class HttpFailure(Exception):
    status_code: int
    message: str

    def __str__(self) -> str:
        return self.message


@pytest.mark.parametrize(
    ("source", "expected_type", "expected_code"),
    [
        (HttpFailure(401, "provider-secret"), AuthenticationProviderError, "authentication"),
        (HttpFailure(429, "provider-secret"), RateLimitProviderError, "rate_limit"),
        (TimeoutError("provider-secret"), TimeoutProviderError, "timeout"),
        (RuntimeError("provider-secret"), ProviderError, "provider_error"),
    ],
)
def test_provider_error_classification_is_typed_and_redacted(
    source: Exception,
    expected_type: type[ProviderError],
    expected_code: str,
) -> None:
    result = provider.classify_provider_error(source)

    assert type(result) is expected_type
    assert result.code == expected_code
    assert "provider-secret" not in str(result)
