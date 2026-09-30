import json
from pathlib import Path

import pytest
import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tests.agent_fakes import ScriptedModel, call
from xiliumini.errors import MemoryBudgetError
from xiliumini.graph.memory import (
    FIXED_RULES,
    LayeredMemory,
    MemoryLimits,
    MemoryManager,
    estimate_message_tokens,
)
from xiliumini.tools.notepad import NotepadAppendTool
from xiliumini.tools.preferences import UserPreferenceStore


def test_layered_memory_contract_is_json_serializable() -> None:
    memory: LayeredMemory = {
        "rules": {
            "fixed_rules": ["workspace only"],
            "user_preferences": [
                {
                    "key": "test_style",
                    "content": "Prefer pytest.",
                    "updated_at": "2026-09-29T00:00:00+00:00",
                }
            ],
        },
        "working": {
            "current_node": "planner",
            "task": "build",
            "session_id": "session",
            "plan_summary": "plan",
            "todos": [],
            "acceptance_criteria": ["tests pass"],
            "research_notes": [],
            "sources": [],
            "agent_handoffs": [],
            "code_agent_summary": "",
            "verifier_summary": "",
            "last_error": "",
            "attempts": {"current": 1, "max": 3},
        },
        "history": {
            "history_summary": "older work",
            "notepad_summary": "decision",
            "context_summary": "compressed",
            "compression_events": [
                {
                    "timestamp": "2026-09-29T00:00:00+00:00",
                    "before_tokens": 100,
                    "after_tokens": 30,
                    "compressed_messages": 4,
                    "attempt": 1,
                }
            ],
        },
    }

    assert json.loads(json.dumps(memory, ensure_ascii=False))["working"]["task"] == "build"


def test_memory_limits_are_immutable() -> None:
    limits = MemoryLimits(64000, 0.8, 8000)

    with pytest.raises((AttributeError, TypeError)):
        limits.keep_tokens = 1  # type: ignore[misc]


def layered_state(workspace: Path) -> dict:
    handoffs = [
        {
            "agent": "code_agent",
            "instruction": f"step {index}",
            "ok": True,
            "summary": f"done {index}",
            "attempt": 1,
        }
        for index in range(7)
    ]
    events = [
        {
            "timestamp": f"2026-09-29T00:00:0{index}+00:00",
            "before_tokens": 100 + index,
            "after_tokens": 20,
            "compressed_messages": 2,
            "attempt": index,
        }
        for index in range(4)
    ]
    return {
        "task": "Use tabs for this task only",
        "session_id": "session",
        "workspace": workspace,
        "plan_summary": "implement memory",
        "todos": [{"id": "one", "content": "work", "status": "pending", "note": ""}],
        "acceptance_criteria": ["tests pass"],
        "research_notes": [
            {
                "summary": "docs",
                "queries": ["docs"],
                "sources": ["https://one.test", "https://two.test"],
                "attempt": 1,
            },
            {
                "summary": "more",
                "queries": ["more"],
                "sources": ["https://one.test", "https://three.test"],
                "attempt": 1,
            },
        ],
        "agent_handoffs": handoffs,
        "code_agent_summary": "code done",
        "verifier_summary": "needs demo",
        "last_error": "missing demo",
        "attempt": 2,
        "max_attempts": 3,
        "context_summary": "prior context",
        "compression_events": events,
        "api_key": "must-not-leak",
        "model": object(),
        "tool_events": [{"args": {"secret": "do-not-copy"}}],
    }


def test_memory_manager_assembles_three_bounded_layers(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    preferences = UserPreferenceStore(tmp_path / "data")
    preferences.upsert("indentation", "Always use spaces.")
    NotepadAppendTool(workspace=workspace).invoke({"entry": "n" * 9000})
    (workspace / "HISTORY_SUMMARY.md").write_text("h" * 9000, encoding="utf-8")
    state = layered_state(workspace)

    memory = MemoryManager(
        workspace,
        preferences,
        MemoryLimits(64000, 0.8, 8000),
        model_name="unknown-model",
    ).assemble(state, current_node="verifier")

    assert memory["rules"]["fixed_rules"] == list(FIXED_RULES)
    assert memory["rules"]["user_preferences"][0]["key"] == "indentation"
    assert memory["working"]["current_node"] == "verifier"
    assert memory["working"]["task"] == state["task"]
    assert memory["working"]["plan_summary"] == "implement memory"
    assert memory["working"]["acceptance_criteria"] == ["tests pass"]
    assert memory["working"]["sources"] == [
        "https://one.test",
        "https://two.test",
        "https://three.test",
    ]
    assert [h["instruction"] for h in memory["working"]["agent_handoffs"]] == [
        f"step {index}" for index in range(1, 7)
    ]
    assert memory["working"]["code_agent_summary"] == "code done"
    assert memory["working"]["verifier_summary"] == "needs demo"
    assert memory["working"]["last_error"] == "missing demo"
    assert memory["working"]["attempts"] == {"current": 2, "max": 3}
    assert memory["history"]["notepad_summary"] == "n" * 7999 + "\n"
    assert memory["history"]["history_summary"] == "h" * 8000
    assert memory["history"]["context_summary"] == "prior context"
    assert [e["attempt"] for e in memory["history"]["compression_events"]] == [1, 2, 3]


def test_memory_snapshot_is_isolated_json_safe_and_obeys_precedence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    preferences = UserPreferenceStore(tmp_path / "data")
    saved = preferences.upsert("indentation", "Always use spaces.")
    state = layered_state(workspace)
    manager = MemoryManager(
        workspace,
        preferences,
        MemoryLimits(64000, 0.8, 8000),
        model_name="unknown-model",
    )

    memory = manager.assemble(state, current_node="planner")
    serialized = json.dumps(memory, ensure_ascii=False)
    memory["working"]["todos"][0]["status"] = "completed"
    memory["working"]["research_notes"].clear()

    assert state["todos"][0]["status"] == "pending"
    assert len(state["research_notes"]) == 2
    assert "must-not-leak" not in serialized
    assert "do-not-copy" not in serialized
    assert str(workspace) not in serialized
    assert "fixed safety rules > current explicit task instructions" in serialized
    assert "Use tabs for this task only" in serialized
    assert "Always use spaces" in serialized
    assert preferences.read() == saved


class PreferenceStub:
    def read(self) -> list:
        return []


def compression_manager(tmp_path: Path, counter, *, limits=None) -> MemoryManager:
    return MemoryManager(
        tmp_path,
        PreferenceStub(),
        limits or MemoryLimits(10, 0.8, 2),
        model_name="test-model",
        token_counter=counter,
    )


def compression_state(tmp_path: Path) -> dict:
    return {
        "task": "keep this complete current task",
        "session_id": "session",
        "workspace": tmp_path,
        "attempt": 2,
        "max_attempts": 3,
        "context_summary": "previous summary",
        "compression_events": [
            {
                "timestamp": f"2026-09-29T00:00:0{i}+00:00",
                "before_tokens": i + 1,
                "after_tokens": 1,
                "compressed_messages": 1,
                "attempt": i,
            }
            for i in range(3)
        ],
    }


def test_token_estimation_is_deterministic_for_english_chinese_and_unknown_model(
    monkeypatch,
) -> None:
    encoding = tiktoken.Encoding(
        "test_cl100k_base",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([value]): value for value in range(256)},
        special_tokens={},
    )

    def encoding_for_model(model_name: str):
        if model_name == "not-a-real-model":
            raise KeyError(model_name)
        return encoding

    monkeypatch.setattr(tiktoken, "encoding_for_model", encoding_for_model)
    monkeypatch.setattr(
        tiktoken,
        "get_encoding",
        lambda name: encoding if name == "cl100k_base" else None,
    )
    messages = [SystemMessage(content="rules"), HumanMessage(content="你好, build it")]

    known = estimate_message_tokens(messages, "gpt-4o-mini")
    unknown = estimate_message_tokens(messages, "not-a-real-model")

    assert known > 0
    assert unknown > 0
    assert known == estimate_message_tokens(messages, "gpt-4o-mini")
    assert unknown == estimate_message_tokens(messages, "not-a-real-model")


@pytest.mark.parametrize("before", [8, 9])
def test_compression_triggers_at_and_above_threshold(tmp_path: Path, before: int) -> None:
    def counter(messages, _model):
        if any("Compressed history summary" in str(message.content) for message in messages):
            return 3
        return {4: before, 2: 3, 1: 1}.get(len(messages), 3)

    manager = compression_manager(tmp_path, counter, limits=MemoryLimits(10, 0.8, 1))
    messages = [
        SystemMessage(content="rules"),
        HumanMessage(content="current task"),
        HumanMessage(content="old"),
        AIMessage(content="recent"),
    ]
    state = compression_state(tmp_path)

    result = manager.prepare_planner_messages(
        messages, state, model=ScriptedModel(["condensed history"])
    )

    assert result != messages
    assert state["context_summary"] == "condensed history"


def test_no_compression_one_token_below_threshold(tmp_path: Path) -> None:
    manager = compression_manager(tmp_path, lambda _messages, _model: 7)
    messages = [SystemMessage(content="rules"), HumanMessage(content="current task")]
    state = compression_state(tmp_path)

    result = manager.prepare_planner_messages(messages, state, model=ScriptedModel([]))

    assert result == messages
    assert state["compression_events"][-1]["attempt"] == 2


def test_compression_retains_anchors_recent_messages_and_persists(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def counter(messages, _model):
        contents = [str(message.content) for message in messages]
        calls.append(contents)
        if len(calls) == 1:
            return 8
        if contents == ["rules", "current task"]:
            return 3
        if len(contents) == 1:
            return 2
        return 4

    manager = compression_manager(tmp_path, counter)
    messages = [
        SystemMessage(content="rules"),
        HumanMessage(content="current task"),
        HumanMessage(content="old one"),
        AIMessage(content="old two"),
        HumanMessage(content="recent"),
    ]
    state = compression_state(tmp_path)

    result = manager.prepare_planner_messages(
        messages, state, model=ScriptedModel(["condensed history"])
    )

    assert result[0] is messages[0]
    assert result[1] is messages[1]
    assert isinstance(result[2], SystemMessage)
    assert "condensed history" in str(result[2].content)
    assert result[-1] is messages[-1]
    assert "condensed history" in (tmp_path / "HISTORY_SUMMARY.md").read_text("utf-8")
    assert state["context_summary"] == "condensed history"
    assert len(state["compression_events"]) == 3
    assert state["compression_events"][-1]["before_tokens"] == 8
    assert state["compression_events"][-1]["after_tokens"] == 4
    assert state["compression_events"][-1]["compressed_messages"] == 2
    assert state["compression_events"][-1]["attempt"] == 2


def test_compression_never_retains_an_orphan_tool_message(tmp_path: Path) -> None:
    def counter(messages, _model):
        return {4: 8, 3: 4, 2: 5, 1: 1}[len(messages)]

    manager = compression_manager(tmp_path, counter)
    tool_call = call("preference_write", {"action": "remove", "key": "style"})
    messages = [
        SystemMessage(content="rules"),
        HumanMessage(content="current task"),
        tool_call,
        ToolMessage(content='{"ok":true}', tool_call_id="call-1"),
    ]
    state = compression_state(tmp_path)

    result = manager.prepare_planner_messages(
        messages, state, model=ScriptedModel(["preference was removed"])
    )

    assert not any(isinstance(message, ToolMessage) for message in result)
    assert state["compression_events"][-1]["compressed_messages"] == 2


def test_compression_reduces_context_when_all_recent_messages_fit_keep_budget(
    tmp_path: Path,
) -> None:
    calls = 0

    def counter(messages, _model):
        nonlocal calls
        calls += 1
        if len(messages) == 4:
            return 8
        if len(messages) == 3:
            return 6
        if len(messages) == 2:
            return 5
        return 1

    manager = compression_manager(tmp_path, counter)
    model = ScriptedModel(["first summary", "complete summary"])
    messages = [
        SystemMessage(content="rules"),
        HumanMessage(content="current task"),
        HumanMessage(content="old"),
        AIMessage(content="recent"),
    ]
    state = compression_state(tmp_path)

    result = manager.prepare_planner_messages(messages, state, model=model)

    assert result != messages
    assert len(model.calls) == 2
    assert state["compression_events"][-1]["after_tokens"] == 6
    assert state["compression_events"][-1]["compressed_messages"] == 2


@pytest.mark.parametrize("response", [RuntimeError("offline"), "   "])
def test_compression_model_failure_is_lossless(tmp_path: Path, response: object) -> None:
    def counter(messages, _model):
        return {3: 8, 2: 3, 1: 3}[len(messages)]

    manager = compression_manager(tmp_path, counter)
    messages = [
        SystemMessage(content="rules"),
        HumanMessage(content="current task"),
        HumanMessage(content="old"),
    ]
    state = compression_state(tmp_path)
    original_events = list(state["compression_events"])

    result = manager.prepare_planner_messages(messages, state, model=ScriptedModel([response]))

    assert result == messages
    assert state["compression_events"] == original_events
    assert state["context_summary"] == "previous summary"


def test_history_write_failure_is_lossless(tmp_path: Path, monkeypatch) -> None:
    from xiliumini.graph import memory as memory_module

    def fail_write(_self, _summary):
        raise OSError("disk full")

    monkeypatch.setattr(memory_module.HistorySummaryStore, "write", fail_write)

    def counter(messages, _model):
        return {3: 8, 2: 3, 1: 3}[len(messages)]

    manager = compression_manager(tmp_path, counter)
    messages = [
        SystemMessage(content="rules"),
        HumanMessage(content="current task"),
        HumanMessage(content="old"),
    ]
    state = compression_state(tmp_path)
    original_events = list(state["compression_events"])

    result = manager.prepare_planner_messages(
        messages, state, model=ScriptedModel(["condensed history"])
    )

    assert result == messages
    assert state["compression_events"] == original_events
    assert state["context_summary"] == "previous summary"


def test_anchor_messages_over_budget_raise_before_summary_model(tmp_path: Path) -> None:
    def counter(messages, _model):
        return 8 if len(messages) >= 2 else 1

    manager = compression_manager(tmp_path, counter)
    model = ScriptedModel(["must not be called"])
    messages = [
        SystemMessage(content="very large fixed rules"),
        HumanMessage(content="very large current task"),
        HumanMessage(content="old"),
    ]

    with pytest.raises(MemoryBudgetError):
        manager.prepare_planner_messages(messages, compression_state(tmp_path), model=model)

    assert model.calls == []
