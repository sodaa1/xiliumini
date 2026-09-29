import importlib
import json

from tests.agent_fakes import ScriptedModel, call, state
from xiliumini.tools.todo import TodoStore


def install(monkeypatch, tmp_path, responses):
    module = importlib.import_module("xiliumini.agents.code_agent")
    model = ScriptedModel(responses)
    monkeypatch.setattr(module, "create_agent_model", lambda: model)
    TodoStore(tmp_path).write([{"id": "impl", "content": "implement"}])
    return module, model


def test_code_writes_runs_checks_and_persists_todos(monkeypatch, tmp_path):
    module, model = install(
        monkeypatch,
        tmp_path,
        [
            call("todo_update", {"todo_id": "impl", "status": "in_progress"}),
            call("file_write", {"path": "demo.py", "content": "print('demo')"}),
            call("bash", {"argv": ["python", "demo.py"], "phase": "demo"}),
            call("todo_update", {"todo_id": "impl", "status": "completed"}),
            "created demo.py and ran demo",
        ],
    )
    events = []
    result = module.run_code_agent(state(tmp_path), "implement a demo", writer=events.append)
    assert result["ok"]
    assert (tmp_path / "demo.py").read_text() == "print('demo')"
    assert result["todos"][0]["status"] == "completed"
    assert json.loads(result["tool_events"][2]["output"])["exit_code"] == 0
    assert {"todo_update", "tool_call", "tool_result"} <= {e["type"] for e in events}
    assert "implement a demo" in model.calls[0][1].content


def test_code_loop_limit_keeps_in_progress_todo(monkeypatch, tmp_path):
    module, _ = install(
        monkeypatch, tmp_path, [call("todo_update", {"todo_id": "impl", "status": "in_progress"})]
    )
    result = module.run_code_agent(state(tmp_path), "work", max_loops=1)
    assert not result["ok"]
    assert result["todos"][0]["status"] == "in_progress"


def test_code_blocked_todo_is_not_success(monkeypatch, tmp_path):
    module, _ = install(
        monkeypatch,
        tmp_path,
        [
            call(
                "todo_update",
                {"todo_id": "impl", "status": "blocked", "note": "missing dependency"},
            ),
            "blocked",
        ],
    )
    result = module.run_code_agent(state(tmp_path), "work")
    assert not result["ok"]
    assert result["todos"][0]["note"] == "missing dependency"


def test_expected_red_followed_by_green_is_success(monkeypatch, tmp_path):
    module, _ = install(
        monkeypatch,
        tmp_path,
        [
            call("todo_update", {"todo_id": "impl", "status": "in_progress"}),
            call(
                "file_write",
                {"path": "test_it.py", "content": "def test_it():\n    assert False\n"},
            ),
            call("bash", {"argv": ["python", "-m", "pytest", "-q"], "phase": "test_red"}),
            call("file_read", {"path": "test_it.py"}),
            call("file_edit", {"path": "test_it.py", "old_string": "False", "new_string": "True"}),
            call("bash", {"argv": ["python", "-m", "pytest", "-q"], "phase": "test_green"}),
            call("todo_update", {"todo_id": "impl", "status": "completed"}),
            "done",
        ],
    )
    result = module.run_code_agent(state(tmp_path), "test")
    assert result["ok"]
    assert not result["tool_events"][2]["ok"]
    assert result["tool_events"][5]["ok"]


def test_rejected_exploration_can_be_recovered_with_valid_execution(monkeypatch, tmp_path):
    module, _ = install(
        monkeypatch,
        tmp_path,
        [
            call("todo_update", {"todo_id": "impl", "status": "in_progress"}),
            call("bash", {"argv": ["ls"], "phase": "other"}),
            call("file_write", {"path": "demo.py", "content": "print('ok')"}),
            call("bash", {"argv": ["python", "demo.py"], "phase": "demo"}),
            call("todo_update", {"todo_id": "impl", "status": "completed"}),
            "done",
        ],
    )
    result = module.run_code_agent(state(tmp_path), "implement demo")
    assert result["tool_events"][1]["ok"] is False
    assert result["ok"] is True


def test_executed_failure_not_cleared_by_unrelated_success(monkeypatch, tmp_path):
    module, _ = install(
        monkeypatch,
        tmp_path,
        [
            call("todo_update", {"todo_id": "impl", "status": "in_progress"}),
            call("file_write", {"path": "broken.py", "content": "raise RuntimeError('failed')"}),
            call("bash", {"argv": ["python", "broken.py"], "phase": "check"}),
            call("file_write", {"path": "demo.py", "content": "print('ok')"}),
            call("bash", {"argv": ["python", "demo.py"], "phase": "demo"}),
            call("todo_update", {"todo_id": "impl", "status": "completed"}),
            "done",
        ],
    )
    assert module.run_code_agent(state(tmp_path), "implement demo")["ok"] is False


def test_red_failure_after_final_green_remains_unresolved(monkeypatch, tmp_path):
    module, _ = install(
        monkeypatch,
        tmp_path,
        [
            call("todo_update", {"todo_id": "impl", "status": "in_progress"}),
            call(
                "file_write",
                {"path": "test_it.py", "content": "def test_it():\n    assert True\n"},
            ),
            call("bash", {"argv": ["python", "-m", "pytest", "-q"], "phase": "test_green"}),
            call("file_read", {"path": "test_it.py"}),
            call("file_edit", {"path": "test_it.py", "old_string": "True", "new_string": "False"}),
            call("bash", {"argv": ["python", "-m", "pytest", "-q"], "phase": "test_red"}),
            call("todo_update", {"todo_id": "impl", "status": "completed"}),
            "done",
        ],
    )
    assert module.run_code_agent(state(tmp_path), "implement with tests")["ok"] is False


def test_later_full_pytest_success_repairs_earlier_green_failure(monkeypatch, tmp_path):
    module, _ = install(
        monkeypatch,
        tmp_path,
        [
            call("todo_update", {"todo_id": "impl", "status": "in_progress"}),
            call(
                "file_write",
                {"path": "test_it.py", "content": "def test_it():\n    assert False\n"},
            ),
            call(
                "bash",
                {"argv": ["python", "-m", "pytest", "-x"], "phase": "test_green"},
            ),
            call("file_read", {"path": "test_it.py"}),
            call("file_edit", {"path": "test_it.py", "old_string": "False", "new_string": "True"}),
            call("bash", {"argv": ["python", "-m", "pytest", "-q"], "phase": "test_green"}),
            call("todo_update", {"todo_id": "impl", "status": "completed"}),
            "done",
        ],
    )
    assert module.run_code_agent(state(tmp_path), "implement with tests")["ok"] is True
