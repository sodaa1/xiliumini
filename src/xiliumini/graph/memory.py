from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import tiktoken
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from typing_extensions import TypedDict

from xiliumini.agents.react import content_text
from xiliumini.errors import MemoryBudgetError, MemorySystemError, WorkspaceError

HISTORY_SUMMARY_PATH = "HISTORY_SUMMARY.md"
MAX_MEMORY_FILE_BYTES = 1_000_000
SUMMARY_CHAR_LIMIT = 8_000
FIXED_RULES = (
    "Only operate inside the current session workspace.",
    "Use workspace-relative paths for files and commands.",
    "TODO.md is the canonical work plan and status file.",
    "NOTEPAD.md stores durable findings and decisions.",
    "HISTORY_SUMMARY.md stores Runtime-generated compressed history.",
    "Runtime owns Memory assembly; Agents must not write Memory directly.",
    "Rule precedence: fixed safety rules > current explicit task instructions > "
    "saved user preferences.",
)


def encode_markdown_json(title: str, payload: Mapping[str, Any]) -> str:
    return (
        f"# {title}\n\n```json\n"
        + json.dumps(dict(payload), ensure_ascii=False, indent=2)
        + "\n```\n"
    )


def decode_markdown_json(title: str, text: str) -> dict[str, Any]:
    pattern = re.compile(
        rf"\A# {re.escape(title)}\n\n```json\n(?P<payload>.*)\n```\n\Z",
        re.DOTALL,
    )
    match = pattern.fullmatch(text)
    if match is None:
        raise ValueError("invalid Markdown JSON document")
    result = json.loads(match.group("payload"))
    if not isinstance(result, dict):
        raise ValueError("invalid Markdown JSON document")
    return result


class HistorySummaryStore:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    @property
    def path(self) -> Path:
        from xiliumini.tools.workspace import resolve_workspace_path

        return resolve_workspace_path(self.workspace, HISTORY_SUMMARY_PATH)

    def read(self) -> str:
        from xiliumini.tools.workspace import read_utf8_text

        if not self.path.exists():
            return ""
        return read_utf8_text(self.path, MAX_MEMORY_FILE_BYTES)

    def write(self, summary: str) -> str:
        from xiliumini.tools.workspace import atomic_write_utf8, utf8_size

        if utf8_size(summary) > MAX_MEMORY_FILE_BYTES:
            raise ValueError(f"history exceeds the {MAX_MEMORY_FILE_BYTES}-byte limit")
        atomic_write_utf8(self.path, summary)
        return summary


class UserPreference(TypedDict):
    key: str
    content: str
    updated_at: str


class RulesLayer(TypedDict):
    fixed_rules: list[str]
    user_preferences: list[UserPreference]


class AttemptSummary(TypedDict):
    current: int
    max: int


class AgentHandoff(TypedDict):
    agent: Literal["search_agent", "code_agent"]
    instruction: str
    ok: bool
    summary: str
    attempt: int


class WorkingMemory(TypedDict):
    current_node: Literal["planner", "verifier", "final"]
    task: str
    session_id: str
    plan_summary: str
    todos: list[dict[str, Any]]
    acceptance_criteria: list[str]
    research_notes: list[dict[str, Any]]
    sources: list[str]
    agent_handoffs: list[AgentHandoff]
    code_agent_summary: str
    verifier_summary: str
    last_error: str
    attempts: AttemptSummary


class CompressionEvent(TypedDict):
    timestamp: str
    before_tokens: int
    after_tokens: int
    compressed_messages: int
    attempt: int


class HistorySummaryLayer(TypedDict):
    history_summary: str
    notepad_summary: str
    context_summary: str
    compression_events: list[CompressionEvent]


class LayeredMemory(TypedDict):
    rules: RulesLayer
    working: WorkingMemory
    history: HistorySummaryLayer


@dataclass(frozen=True)
class MemoryLimits:
    context_window_tokens: int
    trigger_ratio: float
    keep_tokens: int


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _tail(text: str) -> str:
    return text[-SUMMARY_CHAR_LIMIT:]


class MemoryManager:
    def __init__(
        self,
        workspace: Path,
        preference_store: Any,
        limits: MemoryLimits,
        *,
        model_name: str,
        token_counter: Callable[[Sequence[BaseMessage], str], int] | None = None,
    ) -> None:
        self.workspace = workspace
        self.preference_store = preference_store
        self.limits = limits
        self.model_name = model_name
        self.token_counter = token_counter or estimate_message_tokens

    def assemble(
        self,
        state: Mapping[str, Any],
        *,
        current_node: Literal["planner", "verifier", "final"],
    ) -> LayeredMemory:
        from xiliumini.tools.notepad import read_notepad

        try:
            preferences = self.preference_store.read()
            history_summary = HistorySummaryStore(self.workspace).read()
            notepad_summary = read_notepad(self.workspace)
        except (OSError, ValueError, WorkspaceError):
            raise MemorySystemError("memory data is invalid") from None

        research_notes = _json_copy(state.get("research_notes", []))
        sources: list[str] = []
        for note in research_notes:
            if not isinstance(note, dict):
                continue
            for source in note.get("sources", []):
                if isinstance(source, str) and source not in sources:
                    sources.append(source)

        memory: LayeredMemory = {
            "rules": {
                "fixed_rules": list(FIXED_RULES),
                "user_preferences": _json_copy(preferences),
            },
            "working": {
                "current_node": current_node,
                "task": str(state.get("task", "")),
                "session_id": str(state.get("session_id", "")),
                "plan_summary": str(state.get("plan_summary", "")),
                "todos": _json_copy(state.get("todos", [])),
                "acceptance_criteria": _json_copy(state.get("acceptance_criteria", [])),
                "research_notes": research_notes,
                "sources": sources,
                "agent_handoffs": _json_copy(state.get("agent_handoffs", [])[-6:]),
                "code_agent_summary": str(state.get("code_agent_summary", "")),
                "verifier_summary": str(state.get("verifier_summary", "")),
                "last_error": str(state.get("last_error", "")),
                "attempts": {
                    "current": int(state.get("attempt", 0)),
                    "max": int(state.get("max_attempts", 0)),
                },
            },
            "history": {
                "history_summary": _tail(history_summary),
                "notepad_summary": _tail(notepad_summary),
                "context_summary": str(state.get("context_summary", "")),
                "compression_events": _json_copy(state.get("compression_events", [])[-3:]),
            },
        }
        return memory

    def prepare_planner_messages(
        self,
        messages: list[BaseMessage],
        state: dict[str, Any],
        *,
        model: Any,
    ) -> list[BaseMessage]:
        before_tokens = self.token_counter(messages, self.model_name)
        trigger_tokens = int(self.limits.context_window_tokens * self.limits.trigger_ratio)
        if before_tokens < trigger_tokens:
            return messages
        if len(messages) < 2:
            raise MemoryBudgetError("required planner context exceeds the token budget")

        anchors = messages[:2]
        if self.token_counter(anchors, self.model_name) >= trigger_tokens:
            raise MemoryBudgetError("required planner context exceeds the token budget")

        chunks = _message_chunks(messages[2:])
        recent_chunks: list[list[BaseMessage]] = []
        recent_tokens = 0
        for chunk in reversed(chunks):
            chunk_tokens = self.token_counter(chunk, self.model_name)
            if recent_tokens + chunk_tokens > self.limits.keep_tokens:
                break
            recent_chunks.insert(0, chunk)
            recent_tokens += chunk_tokens
        split = len(chunks) - len(recent_chunks)
        compressed = [message for chunk in chunks[:split] for message in chunk]
        recent = [message for chunk in recent_chunks for message in chunk]
        if not compressed and recent_chunks:
            compressed = list(recent_chunks.pop(0))
            recent = [message for chunk in recent_chunks for message in chunk]

        try:
            summary = _summarize_messages(model, compressed, state)
            if not summary:
                return messages
            replacement = SystemMessage(content=f"Compressed history summary:\n{summary}")
            result = [*anchors, replacement, *recent]
            after_tokens = self.token_counter(result, self.model_name)
            if after_tokens >= trigger_tokens and recent:
                compressed.extend(recent)
                recent = []
                summary = _summarize_messages(model, compressed, state)
                if not summary:
                    return messages
                replacement = SystemMessage(content=f"Compressed history summary:\n{summary}")
                result = [*anchors, replacement]
                after_tokens = self.token_counter(result, self.model_name)
            display_summary = summary
            while after_tokens >= trigger_tokens and display_summary:
                display_summary = display_summary[: len(display_summary) * 3 // 4]
                result = (
                    [
                        *anchors,
                        SystemMessage(content=f"Compressed history summary:\n{display_summary}"),
                    ]
                    if display_summary
                    else list(anchors)
                )
                after_tokens = self.token_counter(result, self.model_name)
            HistorySummaryStore(self.workspace).write(summary)
        except Exception:
            return messages

        event: CompressionEvent = {
            "timestamp": datetime.now(UTC).isoformat(),
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "compressed_messages": len(compressed),
            "attempt": int(state.get("attempt", 0)),
        }
        state["context_summary"] = summary
        state["compression_events"] = [
            *list(state.get("compression_events", [])),
            event,
        ][-3:]
        return result


def _message_chunks(messages: Sequence[BaseMessage]) -> list[list[BaseMessage]]:
    chunks: list[list[BaseMessage]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        chunk = [message]
        index += 1
        if isinstance(message, AIMessage) and message.tool_calls:
            call_ids = {call.get("id") for call in message.tool_calls}
            while index < len(messages):
                candidate = messages[index]
                if not isinstance(candidate, ToolMessage):
                    break
                if candidate.tool_call_id not in call_ids:
                    break
                chunk.append(candidate)
                index += 1
        chunks.append(chunk)
    return chunks


def _summarize_messages(
    model: Any, messages: Sequence[BaseMessage], state: Mapping[str, Any]
) -> str:
    summary_input = {
        "previous_context_summary": str(state.get("context_summary", "")),
        "messages": [_message_payload(message) for message in messages],
    }
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Summarize the supplied older planner conversation faithfully and "
                    "compactly. Preserve decisions, requirements, unresolved errors, "
                    "tool evidence, file names, and next actions. Return summary text only."
                )
            ),
            HumanMessage(
                content=json.dumps(summary_input, ensure_ascii=False, separators=(",", ":"))
            ),
        ]
    )
    return content_text(response.content).strip()


def _message_payload(message: BaseMessage) -> dict[str, Any]:
    return {
        "role": message.type,
        "content": message.content,
        "tool_calls": getattr(message, "tool_calls", None),
        "name": getattr(message, "name", None),
    }


def estimate_message_tokens(messages: Sequence[BaseMessage], model_name: str) -> int:
    try:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoding = None
    count = 2
    for message in messages:
        serialized = json.dumps(
            _message_payload(message), ensure_ascii=False, separators=(",", ":"), default=str
        )
        encoded_count = (
            len(encoding.encode(serialized))
            if encoding is not None
            else max(1, (len(serialized.encode("utf-8")) + 2) // 3)
        )
        count += 4 + encoded_count
    return count
