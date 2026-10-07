from __future__ import annotations

from typer.testing import CliRunner

from xiliumini.cli import app


def test_extension_subcommands_appear_in_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("skills", "mcp", "automation"):
        assert command in result.output


def test_automation_cli_adds_daily_and_date_tasks_without_model_key(tmp_path, monkeypatch):
    monkeypatch.setenv("XILIUMINI_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("XILIUMINI_API_KEY", raising=False)
    runner = CliRunner()
    daily = runner.invoke(app, [
        "automation", "add", "daily-brief", "--name", "Brief", "--prompt", "Read news",
        "--daily", "09:00", "--timezone", "Asia/Shanghai",
    ])
    assert daily.exit_code == 0, daily.output
    once = runner.invoke(app, [
        "automation", "add", "once", "--name", "Once", "--prompt", "Task",
        "--at", "2026-10-08T09:00:00+08:00",
    ])
    assert once.exit_code == 0, once.output
    listing = runner.invoke(app, ["automation", "list"])
    assert "daily-brief" in listing.output
    assert "once" in listing.output
