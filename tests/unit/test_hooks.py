from __future__ import annotations

from types import SimpleNamespace
import json

from xiliumini.execution.gateway import ExecutionGateway, RunContext
from xiliumini.hooks.engine import HookEngine
from xiliumini.policy.engine import PolicyEngine
from xiliumini.tools.file_edit import FileEditTool
from xiliumini.tools.file_write import FileWriteTool


def test_ruff_hook_runs_once_after_successful_python_write(tmp_path):
    calls = []
    events = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    context = RunContext(
        run_id="run", session_id="session", workspace=tmp_path,
        event_sink=events.append, hook_engine=HookEngine(tmp_path / "hooks.json", runner=runner),
        policy_engine=PolicyEngine(tmp_path / "policy.json"),
    )
    gateway = ExecutionGateway()
    wrapped = gateway.wrap_tool(FileWriteTool(workspace=tmp_path), context)
    assert wrapped.invoke({"path": "pkg/demo.py", "content": "x = 1\n"}).startswith("Wrote")
    assert len(calls) == 1
    assert calls[0][0][2:5] == ["ruff", "format", "--check"]
    assert any(event.get("type") == "hook_result" and event["status"] == "passed" for event in events)


def test_ruff_hook_skips_failed_edit_and_markdown(tmp_path):
    calls = []

    def runner(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    context = RunContext(
        run_id="run", session_id="session", workspace=tmp_path,
        hook_engine=HookEngine(tmp_path / "hooks.json", runner=runner),
        policy_engine=PolicyEngine(tmp_path / "policy.json"),
    )
    gateway = ExecutionGateway()
    gateway.wrap_tool(FileWriteTool(workspace=tmp_path), context).invoke(
        {"path": "readme.md", "content": "hello"}
    )
    output = gateway.wrap_tool(FileEditTool(workspace=tmp_path), context).invoke(
        {"path": "missing.py", "old_string": "x", "new_string": "y"}
    )
    assert output.startswith("Error:")
    assert calls == []


def test_hook_process_is_policy_guarded(tmp_path):
    calls = []

    def runner(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    (tmp_path / "policy.json").write_text(
        '{"default":"deny","profiles":{"interactive":{"filesystem.write.workspace":"allow"}}}',
        encoding="utf-8",
    )
    events = []
    context = RunContext(
        run_id="run", session_id="session", workspace=tmp_path,
        event_sink=events.append, hook_engine=HookEngine(tmp_path / "hooks.json", runner=runner),
        policy_engine=PolicyEngine(tmp_path / "policy.json"),
    )
    ExecutionGateway().wrap_tool(FileWriteTool(workspace=tmp_path), context).invoke(
        {"path": "demo.py", "content": "x = 1\n"}
    )
    assert calls == []
    assert any(e.get("type") == "hook_result" and e["status"] == "denied" for e in events)


def test_blocking_before_hook_vetoes_write(tmp_path):
    (tmp_path / "hooks.json").write_text(json.dumps({"hooks": [{
        "id": "before-check", "event": "tool.before",
        "match": {"tools": ["file_write"], "path_glob": "**/*.py"},
        "action": {"type": "builtin.ruff_format_check", "target": "changed_file"},
        "on_failure": "block",
    }]}), encoding="utf-8")
    events = []
    context = RunContext(
        run_id="run", session_id="session", workspace=tmp_path,
        event_sink=events.append, hook_engine=HookEngine(tmp_path / "hooks.json"),
        policy_engine=PolicyEngine(tmp_path / "policy.json"),
    )
    result = ExecutionGateway().wrap_tool(FileWriteTool(workspace=tmp_path), context).invoke(
        {"path": "new.py", "content": "x = 1\n"}
    )
    assert json.loads(result)["error"] == "hook_blocked"
    assert not (tmp_path / "new.py").exists()
