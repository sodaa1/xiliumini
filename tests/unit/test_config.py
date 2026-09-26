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
