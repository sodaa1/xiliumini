import json
import subprocess
from pathlib import Path

import pytest

from xiliumini.tools.command import CommandTool


def test_command_runs_workspace_python_script(tmp_path: Path) -> None:
    (tmp_path / "demo.py").write_text("print('life-demo')\n", encoding="utf-8")
    result = CommandTool(workspace=tmp_path).execute(["python", "demo.py"])
    assert result.ok is True
    assert result.exit_code == 0
    assert result.stdout == "life-demo\n"


def test_command_runs_pytest_module(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text(
        "def test_sample():\n    assert 1 + 1 == 2\n", encoding="utf-8"
    )
    result = CommandTool(workspace=tmp_path).execute(["python", "-m", "pytest", "-q"])
    assert result.ok is True
    assert "1 passed" in result.stdout


def test_command_runs_pytest_for_workspace_relative_target(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text(
        "def test_sample():\n    assert True\n", encoding="utf-8"
    )

    result = CommandTool(workspace=tmp_path).execute(
        ["python", "-m", "pytest", "-q", "test_sample.py::test_sample"]
    )

    assert result.ok is True
    assert "1 passed" in result.stdout


@pytest.mark.parametrize(
    "argv",
    [
        ["powershell", "-Command", "Get-ChildItem"],
        ["cmd", "/c", "dir"],
        ["bash", "-c", "ls"],
        ["python", "-c", "print(1)"],
        ["python", "-m", "http.server"],
        ["python", "C:/outside.py"],
        ["python", "../outside.py"],
        ["python", "demo.py", "&&", "cmd"],
        ["python", "-m", "pytest", "../outside.py"],
        ["python", "-m", "pytest", "C:/outside/test_sample.py"],
        ["python", "-m", "pytest", "--basetemp", "../outside"],
        ["python", "-m", "pytest", "--junitxml=../outside.xml"],
        ["python", "-m", "pytest", "-p", "no:cacheprovider"],
    ],
)
def test_command_rejects_non_allowlisted_execution(tmp_path: Path, argv: list[str]) -> None:
    payload = json.loads(CommandTool(workspace=tmp_path).invoke({"argv": argv}))
    assert payload["ok"] is False
    assert payload["exit_code"] is None
    assert "rejected" in payload["stderr"].lower()


def test_command_reports_nonzero_and_truncates_output(tmp_path: Path) -> None:
    (tmp_path / "fail.py").write_text("print('x' * 100)\nraise SystemExit(7)\n", encoding="utf-8")
    result = CommandTool(workspace=tmp_path, max_output_bytes=16).execute(["python", "fail.py"])
    assert result.ok is False
    assert result.exit_code == 7
    assert result.truncated is True
    assert len(result.stdout.encode()) <= 16


def test_command_applies_one_combined_output_budget(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "demo.py").write_text("print('demo')\n", encoding="utf-8")

    def completed(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, stdout=b"a" * 12, stderr=b"b" * 12)

    monkeypatch.setattr(subprocess, "run", completed)

    result = CommandTool(workspace=tmp_path, max_output_bytes=16).execute(["python", "demo.py"])

    assert result.truncated is True
    assert len(result.stdout.encode()) + len(result.stderr.encode()) <= 16


def test_command_converts_timeout_to_evidence(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "slow.py").write_text("print('slow')\n", encoding="utf-8")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    result = CommandTool(workspace=tmp_path, timeout_seconds=0.01).execute(["python", "slow.py"])
    assert result.ok is False
    assert result.timed_out is True
    assert result.exit_code is None
