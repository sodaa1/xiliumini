import json
import re
import subprocess
import sys
from typing import cast

import pytest

from xiliumini.core.approval import ApprovalDecision
from xiliumini.tools import bash_tool as module
from xiliumini.tools import build_tools
from xiliumini.tools.approval_context import ApprovalConfig, approval_config


def test_build_tools_binds_run_local_approval_configuration(tmp_path):
    def handler(request):
        return ApprovalDecision(approved=True, reason=request.id)

    default_bash = cast(
        module.BashTool,
        next(tool for tool in build_tools({"workspace": tmp_path}) if tool.name == "bash"),
    )
    assert default_bash.approval_mode == "inline"
    assert default_bash.approval_handler is None

    token = approval_config.set(ApprovalConfig(mode="deny", handler=handler))
    try:
        configured_bash = cast(
            module.BashTool,
            next(tool for tool in build_tools({"workspace": tmp_path}) if tool.name == "bash"),
        )
    finally:
        approval_config.reset(token)

    assert configured_bash.approval_mode == "deny"
    assert configured_bash.approval_handler is handler
    assert approval_config.get() == ApprovalConfig()


def test_bash_runs_script_with_phase(tmp_path):
    (tmp_path / "demo.py").write_text("print('done')", encoding="utf-8")
    result = json.loads(
        module.BashTool(workspace=tmp_path).invoke({"argv": ["python", "demo.py"], "phase": "demo"})
    )
    assert result["ok"] and result["stdout"] == "done\n"
    assert result["phase"] == "demo"


def test_bash_runs_script_from_workspace_subdirectory(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "demo.py").write_text("print('nested')", encoding="utf-8")

    result = json.loads(
        module.BashTool(workspace=tmp_path).invoke(
            {"argv": ["python", "demo.py"], "cwd": "project", "phase": "demo"}
        )
    )

    assert result["ok"] is True
    assert result["stdout"] == "nested\n"


@pytest.mark.parametrize(
    "argv",
    [
        ["python", "-m", "ruff", "check", "."],
        ["python", "-m", "ruff", "format", "--check", "."],
        ["python", "-m", "pyright", "."],
    ],
)
def test_bash_check_commands_use_fixed_cwd_and_no_shell(monkeypatch, tmp_path, argv):
    observed = []

    def run(command, **kwargs):
        observed.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, b"checked", b"")

    monkeypatch.setattr(subprocess, "run", run)
    result = module.BashTool(workspace=tmp_path).execute(argv)
    assert result.ok
    assert observed[0][0][1:] == argv[1:]
    assert observed[0][1]["cwd"] == tmp_path
    assert observed[0][1]["shell"] is False


@pytest.mark.parametrize(
    "argv",
    [
        ["python", "-m", "compileall", "-q", "."],
        ["python", "-m", "pip", "check"],
    ],
)
def test_bash_runs_allowlisted_diagnostics_in_workspace_subdirectory(monkeypatch, tmp_path, argv):
    project = tmp_path / "project"
    project.mkdir()
    observed = []

    def run(command, **kwargs):
        observed.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, b"checked", b"")

    monkeypatch.setattr(subprocess, "run", run)

    result = module.BashTool(workspace=tmp_path).execute(argv, cwd="project")

    assert result.ok is True
    assert observed[0][0][1:] == argv[1:]
    assert observed[0][1]["cwd"] == project


@pytest.mark.parametrize("cwd", ["../outside", "C:/outside", "missing"])
def test_bash_rejects_invalid_working_directory(tmp_path, cwd):
    result = module.BashTool(workspace=tmp_path).execute(["python", "-m", "pip", "check"], cwd=cwd)

    assert result.ok is False
    assert result.exit_code is None
    assert result.stderr.startswith("command rejected:")


@pytest.mark.parametrize(
    "argv",
    [
        ["cmd", "/c", "dir"],
        ["python", "-m", "ruff", "check", "../outside"],
        ["python", "-m", "ruff", "check", "--fix"],
        ["python", "-m", "ruff", "format", "."],
        ["python", "-m", "pyright", "--pythonpath", "C:/outside"],
        ["python", "-m", "pip", "install", "flask"],
        ["python", "-m", "compileall", "--invalidation-mode", "unchecked-hash", "."],
        ["python", "-m", "pytest", ";", "dir"],
    ],
)
def test_bash_rejects_unsupported_commands(tmp_path, argv):
    result = module.BashTool(workspace=tmp_path).execute(argv)
    assert not result.ok and result.exit_code is None


def test_bash_auto_approves_risky_command_and_marks_result(monkeypatch, tmp_path):
    observed = []

    def run(command, **kwargs):
        observed.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, b"installed", b"")

    monkeypatch.setattr(subprocess, "run", run)

    result = module.BashTool(workspace=tmp_path, approval_mode="auto").execute(
        ["python", "-m", "pip", "install", "flask"]
    )

    assert result.ok is True
    assert result.requires_approval is True
    assert observed[0][0] == [sys.executable, "-m", "pip", "install", "flask"]
    assert observed[0][1]["shell"] is False


def test_bash_deny_mode_rejects_risky_command_without_execution(monkeypatch, tmp_path):
    def fail_if_run(*args, **kwargs):
        pytest.fail("denied command must not start a subprocess")

    monkeypatch.setattr(subprocess, "run", fail_if_run)

    result = module.BashTool(workspace=tmp_path, approval_mode="deny").execute(
        ["curl", "https://example.com/file"]
    )

    assert result.ok is False
    assert result.requires_approval is True
    assert result.exit_code is None
    assert "approval denied" in result.stderr


def test_bash_inline_approval_passes_request_to_handler(monkeypatch, tmp_path):
    requests = []

    def approve(request):
        requests.append(request)
        return ApprovalDecision(approved=True, reason="approved for test")

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, b"downloaded", b"")

    monkeypatch.setattr(subprocess, "run", run)

    result = module.BashTool(
        workspace=tmp_path,
        approval_mode="inline",
        approval_handler=approve,
    ).execute(["curl", "https://example.com/file"])

    assert result.ok is True
    assert result.requires_approval is True
    assert len(requests) == 1
    request = requests[0]
    assert re.fullmatch(r"approval-[0-9a-f]{8}", request.id)
    assert request.command == "curl https://example.com/file"
    assert request.risk_reason == "Network download command"
    assert request.tool_name == "BashTool"


def test_bash_inline_rejection_returns_decision_reason_without_execution(monkeypatch, tmp_path):
    def fail_if_run(*args, **kwargs):
        pytest.fail("rejected command must not start a subprocess")

    monkeypatch.setattr(subprocess, "run", fail_if_run)
    tool = module.BashTool(
        workspace=tmp_path,
        approval_handler=lambda request: ApprovalDecision(False, "not allowed today"),
    )

    result = tool.execute(["uv", "sync"])

    assert result.ok is False
    assert result.requires_approval is True
    assert "not allowed today" in result.stderr


def test_bash_inline_without_handler_rejects_risky_command(monkeypatch, tmp_path):
    def fail_if_run(*args, **kwargs):
        pytest.fail("command awaiting approval must not start a subprocess")

    monkeypatch.setattr(subprocess, "run", fail_if_run)

    result = module.BashTool(workspace=tmp_path).execute(["npm", "install"])

    assert result.ok is False
    assert result.requires_approval is True
    assert "approval handler" in result.stderr


def test_bash_inline_handler_failure_rejects_risky_command(monkeypatch, tmp_path):
    def fail_if_run(*args, **kwargs):
        pytest.fail("command with failed approval must not start a subprocess")

    def fail_approval(request):
        raise RuntimeError("approval UI unavailable")

    monkeypatch.setattr(subprocess, "run", fail_if_run)

    result = module.BashTool(workspace=tmp_path, approval_handler=fail_approval).execute(
        ["pnpm", "install"]
    )

    assert result.ok is False
    assert result.requires_approval is True
    assert "approval handler failed" in result.stderr


@pytest.mark.parametrize(
    "argv",
    [
        ["python", "-c", "print('unsafe');curl"],
        ["cmd", "/c", "echo unsafe&&curl"],
    ],
)
def test_bash_risk_text_in_argument_does_not_bypass_allowlist(monkeypatch, tmp_path, argv):
    def fail_if_run(*args, **kwargs):
        pytest.fail("risk text inside an argument must not bypass the command allowlist")

    monkeypatch.setattr(subprocess, "run", fail_if_run)

    result = module.BashTool(workspace=tmp_path, approval_mode="auto").execute(argv)

    assert result.ok is False
    assert result.exit_code is None


def test_bash_executes_the_argv_snapshot_that_was_approved(monkeypatch, tmp_path):
    argv = ["curl", "https://example.com/file"]
    observed = []

    def approve(request):
        argv[:] = ["python", "-c", "print('changed');curl"]
        return ApprovalDecision(True)

    def run(command, **kwargs):
        observed.append(command)
        return subprocess.CompletedProcess(command, 0, b"downloaded", b"")

    monkeypatch.setattr(subprocess, "run", run)

    result = module.BashTool(workspace=tmp_path, approval_handler=approve).execute(argv)

    assert result.ok is True
    assert result.argv == ["curl", "https://example.com/file"]
    assert observed == [["curl", "https://example.com/file"]]


def test_bash_malformed_approval_decision_fails_closed(monkeypatch, tmp_path):
    def fail_if_run(*args, **kwargs):
        pytest.fail("malformed approval must not start a subprocess")

    def return_malformed_decision(request):
        return cast(ApprovalDecision, None)

    monkeypatch.setattr(subprocess, "run", fail_if_run)

    result = module.BashTool(
        workspace=tmp_path,
        approval_handler=return_malformed_decision,
    ).execute(["uv", "sync"])

    assert result.ok is False
    assert result.requires_approval is True
    assert "approval handler failed" in result.stderr


def test_bash_non_boolean_approval_value_fails_closed(monkeypatch, tmp_path):
    def fail_if_run(*args, **kwargs):
        pytest.fail("only the boolean value True may approve a command")

    def return_non_boolean_decision(request):
        return ApprovalDecision(approved=cast(bool, "false"))

    monkeypatch.setattr(subprocess, "run", fail_if_run)

    result = module.BashTool(
        workspace=tmp_path,
        approval_handler=return_non_boolean_decision,
    ).execute(["uv", "sync"])

    assert result.ok is False
    assert result.requires_approval is True
    assert "approval denied" in result.stderr


def test_bash_safe_command_does_not_require_approval(tmp_path):
    (tmp_path / "demo.py").write_text("print('safe')", encoding="utf-8")

    result = module.BashTool(workspace=tmp_path, approval_mode="deny").execute(
        ["python", "demo.py"]
    )

    assert result.ok is True
    assert result.requires_approval is False


def test_build_tools_binds_files_and_notes_to_workspace(tmp_path):
    from xiliumini.tools import build_tools

    tools = build_tools({"workspace": tmp_path})
    assert {tool.name for tool in tools} == {
        "file_read",
        "file_write",
        "file_edit",
        "grep",
        "bash",
        "notepad_read",
        "notepad_append",
    }
    assert all(vars(tool)["workspace"] == tmp_path for tool in tools)
