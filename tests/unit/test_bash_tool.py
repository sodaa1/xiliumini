import json
import subprocess

import pytest

from xiliumini.tools import bash_tool as module


def test_bash_runs_script_with_phase(tmp_path):
    (tmp_path / "demo.py").write_text("print('done')", encoding="utf-8")
    result = json.loads(
        module.BashTool(workspace=tmp_path).invoke({"argv": ["python", "demo.py"], "phase": "demo"})
    )
    assert result["ok"] and result["stdout"] == "done\n"
    assert result["phase"] == "demo"


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
        ["cmd", "/c", "dir"],
        ["python", "-m", "ruff", "check", "../outside"],
        ["python", "-m", "ruff", "check", "--fix"],
        ["python", "-m", "ruff", "format", "."],
        ["python", "-m", "pyright", "--pythonpath", "C:/outside"],
        ["python", "-m", "pytest", ";", "dir"],
    ],
)
def test_bash_rejects_unsupported_commands(tmp_path, argv):
    result = module.BashTool(workspace=tmp_path).execute(argv)
    assert not result.ok and result.exit_code is None


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
