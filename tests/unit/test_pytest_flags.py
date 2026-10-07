from pathlib import Path

from xiliumini.tools.command import CommandTool


def test_command_allows_pytest_failure_summary_flag(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text(
        "def test_sample():\n    assert True\n", encoding="utf-8"
    )
    result = CommandTool(workspace=tmp_path).execute(
        ["python", "-m", "pytest", "-rf", "test_sample.py"]
    )
    assert result.ok is True
