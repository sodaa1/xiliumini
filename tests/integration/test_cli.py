from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import xiliumini.cli as cli_module
from xiliumini import __version__
from xiliumini.cli import app
from xiliumini.config import Settings
from xiliumini.core.approval import ApprovalDecision, ApprovalRequest
from xiliumini.errors import ConfigError
from xiliumini.events import (
    ErrorEvent,
    FinalEvent,
    PlannerEvent,
    ProgressEvent,
    VerifierEvent,
)
from xiliumini.tools.bash_tool import BashTool

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

    def stream_workspace(self, question: str, workspace: Path, max_attempts: int = 3):
        self.calls.append((question, str(workspace), max_attempts))
        yield from self.events


def install_runtime(monkeypatch, tmp_path: Path, runtime) -> None:
    monkeypatch.setattr(cli_module, "load_settings", lambda: configured_settings(tmp_path))
    monkeypatch.setattr(cli_module, "create_runtime", lambda _settings, **_kwargs: runtime)


def test_cli_help_and_version() -> None:
    help_result = runner.invoke(app, ["--help"])
    version_result = runner.invoke(app, ["--version"])

    assert help_result.exit_code == 0
    for command in ("doctor", "ask", "chat", "sessions"):
        assert command in help_result.stdout
    for option in (
        "--workspace",
        "--max-attempts",
        "--approval-mode",
        "--checkpoint-mode",
        "--trace-mode",
        "--resume",
    ):
        assert option in help_result.stdout
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


def test_cli_resume_path_with_spaces(monkeypatch, tmp_path):
    class ResumeRuntime(FakeRuntime):
        def resume(self, workspace, *, max_attempts=3):
            assert workspace == tmp_path / "workspace with spaces"
            assert max_attempts == 3
            yield FinalEvent(text="restored", session_id="saved")

    install_runtime(monkeypatch, tmp_path, ResumeRuntime([]))
    result = runner.invoke(app, ["--resume", str(tmp_path / "workspace with spaces")])
    assert result.exit_code == 0
    assert "Final: restored" in result.stdout


def test_cli_resume_uses_root_attempt_and_harness_options(monkeypatch, tmp_path):
    captured = []

    class ResumeRuntime(FakeRuntime):
        def resume(self, workspace, *, max_attempts=3):
            assert workspace == tmp_path / "workspaces" / "saved"
            assert max_attempts == 6
            yield FinalEvent(text="restored", session_id="saved")

    monkeypatch.setattr(cli_module, "load_settings", lambda: configured_settings(tmp_path))

    def create(_settings, **overrides):
        captured.append(overrides)
        return ResumeRuntime([])

    monkeypatch.setattr(cli_module, "create_runtime", create)
    result = runner.invoke(
        app,
        [
            "--max-attempts",
            "6",
            "--approval-mode",
            "auto",
            "--checkpoint-mode",
            "off",
            "--trace-mode",
            "off",
            "--resume",
            str(tmp_path / "workspaces" / "saved"),
        ],
    )

    assert result.exit_code == 0
    assert captured == [
        {
            "approval_mode": "auto",
            "approval_handler": None,
            "checkpoint_mode": "off",
            "trace_mode": "off",
        }
    ]


def test_cli_resume_conflicts_with_subcommand(monkeypatch):
    def forbidden():
        raise AssertionError("configuration should not load")

    monkeypatch.setattr(cli_module, "load_settings", forbidden)
    result = runner.invoke(app, ["--resume", "private path", "ask", "question"])
    assert result.exit_code == 2
    assert "private path" not in result.output
    assert "Traceback" not in result.output


def test_cli_workspace_conflicts_with_resume_before_loading_config(monkeypatch, tmp_path):
    def forbidden():
        raise AssertionError("configuration should not load")

    monkeypatch.setattr(cli_module, "load_settings", forbidden)
    workspace = tmp_path / "private-one"
    resume = tmp_path / "private-two"

    result = runner.invoke(
        app,
        ["--workspace", str(workspace), "--resume", str(resume)],
    )

    assert result.exit_code == 2
    assert "--workspace cannot be combined with --resume" in result.stdout
    assert str(workspace) not in result.output
    assert str(resume) not in result.output


def test_root_harness_options_apply_to_ask_and_explicit_workspace(monkeypatch, tmp_path):
    runtime = FakeRuntime([FinalEvent(text="done", session_id="session")])
    captured = []
    monkeypatch.setattr(cli_module, "load_settings", lambda: configured_settings(tmp_path))

    def create(_settings, **overrides):
        captured.append(overrides)
        return runtime

    monkeypatch.setattr(cli_module, "create_runtime", create)
    workspace = tmp_path / "workspaces" / "manual"

    result = runner.invoke(
        app,
        [
            "--workspace",
            str(workspace),
            "--max-attempts",
            "5",
            "--approval-mode",
            "deny",
            "--checkpoint-mode",
            "strict",
            "--trace-mode",
            "off",
            "ask",
            "question",
        ],
    )

    assert result.exit_code == 0
    assert runtime.calls == [("question", str(workspace), 5)]
    assert captured == [
        {
            "approval_mode": "deny",
            "approval_handler": None,
            "checkpoint_mode": "strict",
            "trace_mode": "off",
        }
    ]


def test_inline_approval_handler_prompts_once_and_returns_decision(monkeypatch, tmp_path):
    runtime = FakeRuntime([FinalEvent(text="done", session_id="session")])
    captured = []
    monkeypatch.setattr(cli_module, "load_settings", lambda: configured_settings(tmp_path))

    def create(_settings, **overrides):
        captured.append(overrides)
        return runtime

    monkeypatch.setattr(cli_module, "create_runtime", create)
    prompts = []

    def confirm(*args, **kwargs):
        prompts.append((args, kwargs))
        return False

    monkeypatch.setattr(cli_module.typer, "confirm", confirm)

    result = runner.invoke(app, ["ask", "question"])
    handler = captured[0]["approval_handler"]
    decision = handler(
        ApprovalRequest(
            id="approval-12345678",
            command="pip install demo",
            risk_reason="Python package installation",
        )
    )

    assert result.exit_code == 0
    assert captured[0]["approval_mode"] == "inline"
    assert decision == ApprovalDecision(approved=False, reason="Denied by user.")
    assert prompts == [(("Approve this command?",), {"default": False})]
    assert captured[0]["checkpoint_mode"] is None
    assert captured[0]["trace_mode"] is None


def test_inline_approval_abort_interrupts_bash_tool(monkeypatch, tmp_path):
    def abort(*args, **kwargs):
        raise cli_module.typer.Abort()

    monkeypatch.setattr(cli_module.typer, "confirm", abort)
    tool = BashTool(
        workspace=tmp_path,
        approval_mode="inline",
        approval_handler=cli_module._inline_approval,
    )

    with pytest.raises(KeyboardInterrupt):
        tool.execute(["pip", "install", "demo"])


@pytest.mark.parametrize("mode", ["auto", "deny"])
def test_non_inline_approval_modes_never_install_prompt_handler(monkeypatch, tmp_path, mode):
    runtime = FakeRuntime([FinalEvent(text="done", session_id="session")])
    captured = []
    monkeypatch.setattr(cli_module, "load_settings", lambda: configured_settings(tmp_path))
    monkeypatch.setattr(
        cli_module.typer,
        "confirm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not prompt")),
    )

    def create(_settings, **overrides):
        captured.append(overrides)
        return runtime

    monkeypatch.setattr(cli_module, "create_runtime", create)

    result = runner.invoke(app, ["--approval-mode", mode, "ask", "question"])

    assert result.exit_code == 0
    assert captured[0]["approval_mode"] == mode
    assert captured[0]["approval_handler"] is None


def test_cli_foreign_workspace_error_is_redacted(monkeypatch, tmp_path):
    from xiliumini.runtime import Runtime

    install_runtime(monkeypatch, tmp_path, Runtime(object(), data_dir=tmp_path))
    foreign = tmp_path.parent / "private-workspace"

    result = runner.invoke(app, ["--workspace", str(foreign), "ask", "question"])

    assert result.exit_code == 1
    assert "workspace_error" in result.stdout
    assert str(foreign) not in result.output


def test_cli_resume_invalid_paths_are_redacted(monkeypatch, tmp_path):
    from xiliumini.runtime import Runtime

    install_runtime(monkeypatch, tmp_path, Runtime(object(), data_dir=tmp_path))
    for workspace in (tmp_path / "missing", tmp_path):
        result = runner.invoke(app, ["--resume", str(workspace)])
        assert result.exit_code == 1
        assert "checkpoint_error" in result.stdout
        assert str(tmp_path) not in result.output
        assert "Traceback" not in result.output


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
            PlannerEvent(todos=[], summary="test first → implement", attempt=1),
            ProgressEvent(stage="actor", message="Starting: tests"),
            ProgressEvent(stage="code_agent", message="codeAgent: red then green"),
            VerifierEvent(passed=True, reason="all evidence present", attempt=1),
            FinalEvent(text="passed", session_id="session"),
        ]
    )
    install_runtime(monkeypatch, tmp_path, runtime)

    result = runner.invoke(app, ["ask", "build"])

    assert result.exit_code == 0
    assert "Planner (attempt 1/3): test first → implement" in result.stdout
    assert "Starting: tests" in result.stdout
    assert "codeAgent: red then green" in result.stdout
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
            PlannerEvent(todos=[], summary="hidden", attempt=1),
            ProgressEvent(stage="code_agent", message="hidden"),
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


def test_chat_uses_root_max_attempts(monkeypatch, tmp_path: Path) -> None:
    runtime = FakeRuntime([FinalEvent(text="done", session_id="session")])
    install_runtime(monkeypatch, tmp_path, runtime)

    result = runner.invoke(app, ["--max-attempts", "6"], input="hello\n/exit\n")

    assert result.exit_code == 0
    assert [call[2] for call in runtime.calls] == [6]


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
