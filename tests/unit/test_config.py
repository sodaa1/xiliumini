from __future__ import annotations

from pathlib import Path

import pytest

from xiliumini.config import Settings
from xiliumini.errors import ConfigError


def test_settings_reports_both_missing_required_fields(isolated_cwd: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        Settings.from_env({})

    message = str(caught.value)
    assert "XILIUMINI_API_KEY" in message
    assert "XILIUMINI_MODEL" in message


def test_settings_keeps_secret_out_of_repr(isolated_cwd: Path) -> None:
    settings = Settings.from_env(
        {
            "XILIUMINI_API_KEY": "never-print-me",
            "XILIUMINI_MODEL": "primary",
        }
    )

    assert "never-print-me" not in repr(settings)


@pytest.mark.parametrize("field", ["XILIUMINI_API_KEY", "XILIUMINI_MODEL"])
def test_settings_rejects_whitespace_only_required_values(
    isolated_cwd: Path,
    field: str,
) -> None:
    env = {
        "XILIUMINI_API_KEY": "secret",
        "XILIUMINI_MODEL": "primary",
        field: "   ",
    }

    with pytest.raises(ConfigError, match=field):
        Settings.from_env(env)


def test_settings_builds_an_ordered_unique_model_list(isolated_cwd: Path) -> None:
    settings = Settings.from_env(
        {
            "XILIUMINI_API_KEY": "secret",
            "XILIUMINI_MODEL": "primary",
            "XILIUMINI_MODELS": "primary, backup-a, backup-b, backup-a",
        }
    )

    assert settings.model_names == ("primary", "backup-a", "backup-b")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("XILIUMINI_TIMEOUT_SECONDS", "0"),
        ("XILIUMINI_ANALYSIS_TIMEOUT_SECONDS", "0"),
        ("XILIUMINI_ANALYSIS_MAX_CHARS", "0"),
    ],
)
def test_settings_rejects_invalid_runtime_limits(
    isolated_cwd: Path,
    name: str,
    value: str,
) -> None:
    env = {
        "XILIUMINI_API_KEY": "secret",
        "XILIUMINI_MODEL": "primary",
        name: value,
    }

    with pytest.raises(ConfigError, match=name):
        Settings.from_env(env)


def test_settings_no_longer_exposes_react_max_steps(isolated_cwd: Path) -> None:
    settings = Settings.from_env({"XILIUMINI_API_KEY": "secret", "XILIUMINI_MODEL": "primary"})

    assert not hasattr(settings, "max_steps")


def test_settings_has_memory_limit_defaults_and_overrides(isolated_cwd: Path) -> None:
    defaults = Settings.from_env({"XILIUMINI_API_KEY": "secret", "XILIUMINI_MODEL": "primary"})
    overridden = Settings.from_env(
        {
            "XILIUMINI_API_KEY": "secret",
            "XILIUMINI_MODEL": "primary",
            "XILIUMINI_CONTEXT_WINDOW_TOKENS": "32000",
            "XILIUMINI_COMPRESSION_TRIGGER_RATIO": "0.75",
            "XILIUMINI_COMPRESSION_KEEP_TOKENS": "4000",
        }
    )

    assert (
        defaults.context_window_tokens,
        defaults.compression_trigger_ratio,
        defaults.compression_keep_tokens,
    ) == (64000, 0.8, 8000)
    assert (
        overridden.context_window_tokens,
        overridden.compression_trigger_ratio,
        overridden.compression_keep_tokens,
    ) == (32000, 0.75, 4000)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("XILIUMINI_CONTEXT_WINDOW_TOKENS", "0"),
        ("XILIUMINI_COMPRESSION_TRIGGER_RATIO", "0"),
        ("XILIUMINI_COMPRESSION_TRIGGER_RATIO", "1"),
        ("XILIUMINI_COMPRESSION_KEEP_TOKENS", "0"),
    ],
)
def test_settings_rejects_invalid_memory_limits(isolated_cwd: Path, name: str, value: str) -> None:
    env = {
        "XILIUMINI_API_KEY": "secret",
        "XILIUMINI_MODEL": "primary",
        name: value,
    }

    with pytest.raises(ConfigError, match=name):
        Settings.from_env(env)


def test_settings_rejects_keep_budget_at_trigger_budget(isolated_cwd: Path) -> None:
    with pytest.raises(ConfigError, match="XILIUMINI_COMPRESSION_KEEP_TOKENS"):
        Settings.from_env(
            {
                "XILIUMINI_API_KEY": "secret",
                "XILIUMINI_MODEL": "primary",
                "XILIUMINI_CONTEXT_WINDOW_TOKENS": "100",
                "XILIUMINI_COMPRESSION_TRIGGER_RATIO": "0.5",
                "XILIUMINI_COMPRESSION_KEEP_TOKENS": "50",
            }
        )


def test_tavily_secret_loads_from_dotenv_without_export(monkeypatch, isolated_cwd):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    (isolated_cwd / ".env").write_text(
        "XILIUMINI_API_KEY=fake\nXILIUMINI_MODEL=fake\nTAVILY_API_KEY=private-tavily\n"
    )
    settings = Settings.from_env()
    assert settings.tavily_api_key is not None
    assert settings.tavily_api_key.get_secret_value() == "private-tavily"
    assert "private-tavily" not in repr(settings)
