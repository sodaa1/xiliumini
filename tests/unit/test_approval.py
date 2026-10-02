from __future__ import annotations

import pytest

from xiliumini.core.approval import classify_command_risk, normalize_approval_mode


@pytest.mark.parametrize(
    ("command", "reason"),
    [
        ("pip install requests", "Python package installation"),
        ("python -m pip install requests", "Python package installation"),
        ("echo ready && pip install requests", "Python package installation"),
        ("uv add httpx", "Project dependency change with uv add"),
        ("uv sync", "Dependency synchronization with uv sync"),
        ("uv pip install httpx", "Python package installation with uv pip"),
        ("npm install", "Node package installation"),
        ("pnpm install", "Node package installation"),
        ("yarn install", "Node package installation"),
        ("yarn add react", "Node package installation"),
        ("curl https://example.com/file", "Network download command"),
        ("wget https://example.com/file", "Network download command"),
        ("uvicorn app:app", "Long-running development server"),
        ("python -m http.server 8000", "Long-running development server"),
    ],
)
def test_classify_command_risk_identifies_high_risk_commands(command, reason):
    assert classify_command_risk(command) == reason


@pytest.mark.parametrize(
    "command",
    [
        "python -m pip check",
        "pip uninstall requests",
        "echo pip install requests",
        "python demo.py",
        "npm --version",
    ],
)
def test_classify_command_risk_leaves_safe_commands_unclassified(command):
    assert classify_command_risk(command) is None


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (None, "inline"),
        ("", "inline"),
        ("unexpected", "inline"),
        ("inline", "inline"),
        ("auto", "auto"),
        ("deny", "deny"),
    ],
)
def test_normalize_approval_mode_falls_back_to_inline(mode, expected):
    assert normalize_approval_mode(mode) == expected
