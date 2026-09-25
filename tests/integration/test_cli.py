from __future__ import annotations

import asyncio
from pathlib import Path

from typer.testing import CliRunner

import xiliumini.cli as cli_module
from xiliumini import __version__
from xiliumini.cli import app
from xiliumini.config import Settings
from xiliumini.errors import ConfigError
from xiliumini.events import ErrorEvent, FinalEvent, TokenEvent

runner = CliRunner()


def test_cli_help_lists_task_zero_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("doctor", "ask", "chat", "sessions"):
        assert command in result.stdout


def test_cli_version_reports_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_doctor_missing_config_exits_two_without_traceback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("XILIUMINI_API_KEY", "XILIUMINI_MODEL", "XILIUMINI_MODELS"):
        monkeypatch.delenv(name, raising=False)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 2
    assert "XILIUMINI_API_KEY" in result.stdout
    assert "Traceback" not in result.stdout


def test_doctor_whitespace_config_exits_two_without_traceback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XILIUMINI_API_KEY", "   ")
    monkeypatch.setenv("XILIUMINI_MODEL", "   ")

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 2
    assert "XILIUMINI_API_KEY" in result.stdout
    assert "XILIUMINI_MODEL" in result.stdout
    assert "Traceback" not in result.stdout


def test_chat_session_option_is_present() -> None:
    result = runner.invoke(app, ["chat", "--help"])

    assert result.exit_code == 0
    assert "--session" in result.stdout


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

    async def astream(self, _question: str, _session_id: str):
        for event in self.events:
            yield event


class ConversationalFakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.event_loops: list[asyncio.AbstractEventLoop] = []

    async def astream(self, question: str, session_id: str):
        self.calls.append((question, session_id))
        self.event_loops.append(asyncio.get_running_loop())
        yield TokenEvent(text=f"reply:{question}")
        yield FinalEvent(text=f"reply:{question}", session_id=session_id)


def test_bare_command_starts_conversation_and_reuses_session(monkeypatch, tmp_path: Path) -> None:
    runtime = ConversationalFakeRuntime()
    monkeypatch.setattr(cli_module, "load_settings", lambda: configured_settings(tmp_path))
    monkeypatch.setattr(cli_module, "create_runtime", lambda _settings: runtime, raising=False)

    result = runner.invoke(app, input="hello\nfollow up\n/exit\n")

    assert result.exit_code == 0
    assert [question for question, _session_id in runtime.calls] == ["hello", "follow up"]
    assert runtime.calls[0][1] == runtime.calls[1][1]
    assert "reply:hello" in result.stdout
    assert "reply:follow up" in result.stdout


def test_conversation_reuses_one_event_loop_across_turns(monkeypatch, tmp_path: Path) -> None:
    runtime = ConversationalFakeRuntime()
    monkeypatch.setattr(cli_module, "load_settings", lambda: configured_settings(tmp_path))
    monkeypatch.setattr(cli_module, "create_runtime", lambda _settings: runtime, raising=False)

    result = runner.invoke(app, input="first\nsecond\n/exit\n")

    assert result.exit_code == 0
    assert len(runtime.event_loops) == 2
    assert runtime.event_loops[0] is runtime.event_loops[1]


def test_chat_command_starts_the_same_conversation_loop(monkeypatch, tmp_path: Path) -> None:
    runtime = ConversationalFakeRuntime()
    monkeypatch.setattr(cli_module, "load_settings", lambda: configured_settings(tmp_path))
    monkeypatch.setattr(cli_module, "create_runtime", lambda _settings: runtime, raising=False)

    result = runner.invoke(app, ["chat"], input="hello\n/exit\n")

    assert result.exit_code == 0
    assert [question for question, _session_id in runtime.calls] == ["hello"]
    assert "reply:hello" in result.stdout


def test_interactive_commands_do_not_reach_provider_and_new_changes_session(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = ConversationalFakeRuntime()
    monkeypatch.setattr(cli_module, "load_settings", lambda: configured_settings(tmp_path))
    monkeypatch.setattr(cli_module, "create_runtime", lambda _settings: runtime, raising=False)

    result = runner.invoke(
        app,
        input="first\n/help\n/status\n/new\nsecond\n/exit\n",
    )

    assert result.exit_code == 0
    assert [question for question, _session_id in runtime.calls] == ["first", "second"]
    assert runtime.calls[0][1] != runtime.calls[1][1]
    assert "/help" in result.stdout
    assert "fake-model" in result.stdout


def test_bare_command_exits_cleanly_on_end_of_input(monkeypatch, tmp_path: Path) -> None:
    runtime = ConversationalFakeRuntime()
    monkeypatch.setattr(cli_module, "load_settings", lambda: configured_settings(tmp_path))
    monkeypatch.setattr(cli_module, "create_runtime", lambda _settings: runtime, raising=False)

    result = runner.invoke(app, input="")

    assert result.exit_code == 0
    assert "Traceback" not in result.stdout


def test_ask_streams_tokens_without_repeating_final_text(monkeypatch, tmp_path: Path) -> None:
    runtime = FakeRuntime(
        [
            TokenEvent(text="hel"),
            TokenEvent(text="lo"),
            FinalEvent(text="hello", session_id="session-1"),
        ]
    )
    monkeypatch.setattr(cli_module, "load_settings", lambda: configured_settings(tmp_path))
    monkeypatch.setattr(cli_module, "create_runtime", lambda _settings: runtime, raising=False)

    result = runner.invoke(app, ["ask", "greet"])

    assert result.exit_code == 0
    assert result.stdout == "hello\n"


def test_ask_no_stream_prints_only_final_text(monkeypatch, tmp_path: Path) -> None:
    runtime = FakeRuntime(
        [TokenEvent(text="ignored"), FinalEvent(text="complete", session_id="session-1")]
    )
    monkeypatch.setattr(cli_module, "load_settings", lambda: configured_settings(tmp_path))
    monkeypatch.setattr(cli_module, "create_runtime", lambda _settings: runtime, raising=False)

    result = runner.invoke(app, ["ask", "answer", "--no-stream"])

    assert result.exit_code == 0
    assert result.stdout == "complete\n"


def test_ask_runtime_error_exits_one_without_traceback(monkeypatch, tmp_path: Path) -> None:
    runtime = FakeRuntime([ErrorEvent(code="provider_error", message="Provider request failed")])
    monkeypatch.setattr(cli_module, "load_settings", lambda: configured_settings(tmp_path))
    monkeypatch.setattr(cli_module, "create_runtime", lambda _settings: runtime, raising=False)

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
