from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import xiliumini.cli as cli_module
from xiliumini import __version__
from xiliumini.cli import app
from xiliumini.config import Settings
from xiliumini.errors import ConfigError
from xiliumini.events import (
    ActorEvent,
    ErrorEvent,
    FinalEvent,
    PlannerEvent,
    ProgressEvent,
    VerifierEvent,
)

runner = CliRunner()


def test_cli_reconfigures_legacy_output_for_stage_icons(monkeypatch) -> None:
    class LegacyStream:
        encoding = "gbk"

        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def reconfigure(self, **kwargs) -> None:
            self.calls.append(kwargs)

    stdout = LegacyStream()
    stderr = LegacyStream()
    monkeypatch.setattr(cli_module.sys, "stdout", stdout)
    monkeypatch.setattr(cli_module.sys, "stderr", stderr)

    cli_module._ensure_utf8_output()

    assert stdout.calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert stderr.calls == [{"encoding": "utf-8", "errors": "replace"}]


def configured_settings(tmp_path: Path) -> Settings:
    return Settings.from_env(
        {
            "XILIUMINI_API_KEY": "secret",
            "XILIUMINI_MODEL": "fake-model",
            "XILIUMINI_DATA_DIR": str(tmp_path),
        }
    )


class FakeRuntime:
    def __init__(self, events) -> None:
        self.events = events
        self.calls: list[tuple[str, str, int]] = []

    def stream(self, question: str, session_id: str, max_attempts: int = 3):
        self.calls.append((question, session_id, max_attempts))
        yield from self.events


def install_runtime(monkeypatch, tmp_path: Path, runtime: FakeRuntime) -> None:
    monkeypatch.setattr(cli_module, "load_settings", lambda: configured_settings(tmp_path))
    monkeypatch.setattr(cli_module, "create_runtime", lambda _settings: runtime)


def test_cli_help_and_version() -> None:
    help_result = runner.invoke(app, ["--help"])
    version_result = runner.invoke(app, ["--version"])

    assert help_result.exit_code == 0
    for command in ("doctor", "ask", "chat", "sessions"):
        assert command in help_result.stdout
    assert version_result.exit_code == 0
    assert __version__ in version_result.stdout


def test_doctor_missing_config_exits_two_without_traceback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("XILIUMINI_API_KEY", "XILIUMINI_MODEL", "XILIUMINI_MODELS"):
        monkeypatch.delenv(name, raising=False)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 2
    assert "XILIUMINI_API_KEY" in result.stdout
    assert "Traceback" not in result.stdout


def test_ask_uses_default_and_explicit_max_attempts(monkeypatch, tmp_path: Path) -> None:
    runtime = FakeRuntime([FinalEvent(text="done", session_id="session")])
    install_runtime(monkeypatch, tmp_path, runtime)

    default = runner.invoke(app, ["ask", "first"])
    explicit = runner.invoke(app, ["ask", "second", "--max-attempts", "5"])

    assert default.exit_code == explicit.exit_code == 0
    assert [call[2] for call in runtime.calls] == [3, 5]


def test_ask_rejects_zero_max_attempts() -> None:
    result = runner.invoke(app, ["ask", "question", "--max-attempts", "0"])

    assert result.exit_code == 2
    assert "--max-attempts" in result.stderr


def test_ask_renders_each_graph_stage(monkeypatch, tmp_path: Path) -> None:
    runtime = FakeRuntime(
        [
            PlannerEvent(todo=["test first", "implement"]),
            ProgressEvent(stage="actor", message="Starting: tests"),
            ActorEvent(result="red then green", attempt=1),
            VerifierEvent(passed=True, reason="all evidence present", attempt=1),
            FinalEvent(text="passed", session_id="session"),
        ]
    )
    install_runtime(monkeypatch, tmp_path, runtime)

    result = runner.invoke(app, ["ask", "build"])

    assert result.exit_code == 0
    assert "📋 Planner: test first → implement" in result.stdout
    assert "Starting: tests" in result.stdout
    assert "🔧 Actor (attempt 1/3): red then green" in result.stdout
    assert "✅ Verifier: all evidence present" in result.stdout
    assert "📝 Final: passed" in result.stdout


def test_ask_renders_failed_verifier_icon(monkeypatch, tmp_path: Path) -> None:
    runtime = FakeRuntime(
        [
            VerifierEvent(passed=False, reason="demo missing", attempt=1),
            FinalEvent(text="failed", session_id="session"),
        ]
    )
    install_runtime(monkeypatch, tmp_path, runtime)

    result = runner.invoke(app, ["ask", "build"])

    assert result.exit_code == 0
    assert "❌ Verifier: demo missing" in result.stdout


def test_ask_no_stream_hides_intermediate_events(monkeypatch, tmp_path: Path) -> None:
    runtime = FakeRuntime(
        [
            PlannerEvent(todo=["hidden"]),
            ActorEvent(result="hidden", attempt=1),
            VerifierEvent(passed=True, reason="hidden", attempt=1),
            FinalEvent(text="complete", session_id="session"),
        ]
    )
    install_runtime(monkeypatch, tmp_path, runtime)

    result = runner.invoke(app, ["ask", "answer", "--no-stream"])

    assert result.exit_code == 0
    assert result.stdout == "📝 Final: complete\n"


def test_chat_uses_default_attempts_and_reuses_session(monkeypatch, tmp_path: Path) -> None:
    runtime = FakeRuntime([FinalEvent(text="done", session_id="session")])
    install_runtime(monkeypatch, tmp_path, runtime)

    result = runner.invoke(app, input="hello\nfollow up\n/exit\n")

    assert result.exit_code == 0
    assert [call[0] for call in runtime.calls] == ["hello", "follow up"]
    assert runtime.calls[0][1] == runtime.calls[1][1]
    assert [call[2] for call in runtime.calls] == [3, 3]


def test_interactive_new_changes_session(monkeypatch, tmp_path: Path) -> None:
    runtime = FakeRuntime([FinalEvent(text="done", session_id="session")])
    install_runtime(monkeypatch, tmp_path, runtime)

    result = runner.invoke(app, input="first\n/new\nsecond\n/exit\n")

    assert result.exit_code == 0
    assert runtime.calls[0][1] != runtime.calls[1][1]


def test_ask_runtime_error_exits_one_without_traceback(monkeypatch, tmp_path: Path) -> None:
    runtime = FakeRuntime([ErrorEvent(code="provider_error", message="Provider request failed")])
    install_runtime(monkeypatch, tmp_path, runtime)

    result = runner.invoke(app, ["ask", "answer"])

    assert result.exit_code == 1
    assert "Provider request failed" in result.stdout
    assert "Traceback" not in result.stdout


def test_ask_configuration_error_exits_two(monkeypatch) -> None:
    def fail_settings() -> Settings:
        raise ConfigError("Invalid or missing configuration: XILIUMINI_MODEL")

    monkeypatch.setattr(cli_module, "load_settings", fail_settings)

    result = runner.invoke(app, ["ask", "answer"])

    assert result.exit_code == 2
    assert "XILIUMINI_MODEL" in result.stdout
    assert "Traceback" not in result.stdout
