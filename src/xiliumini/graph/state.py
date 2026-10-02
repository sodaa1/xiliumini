from pathlib import Path
from typing import Any, Literal

from typing_extensions import TypedDict

from xiliumini.graph.memory import AgentHandoff, CompressionEvent, LayeredMemory

GraphStatus = Literal["planning", "verifying", "passed", "failed"]
TodoStatus = Literal["pending", "in_progress", "completed", "blocked"]


class TodoDraft(TypedDict):
    id: str
    content: str


class TodoItem(TodoDraft):
    status: TodoStatus
    note: str


class ResearchNote(TypedDict):
    summary: str
    queries: list[str]
    sources: list[str]
    attempt: int


class ToolEvent(TypedDict):
    agent: Literal["planner", "search_agent", "code_agent"]
    tool: str
    args: dict[str, Any]
    output: str
    ok: bool
    attempt: int
    phase: str | None


class AgentResult(TypedDict):
    agent: Literal["search_agent", "code_agent"]
    instruction: str
    ok: bool
    summary: str
    attempt: int


class GraphState(TypedDict):
    resume_node: Literal["planner", "verifier", "final"]
    supervisor_ok: bool
    task: str
    todos: list[TodoItem]
    research_notes: list[ResearchNote]
    agent_results: list[AgentResult]
    tool_events: list[ToolEvent]
    result: str
    graph_state: GraphStatus
    verification: str
    attempt: int
    max_attempts: int
    final_answer: str
    session_id: str
    workspace: Path
    memory: LayeredMemory
    current_node: Literal["planner", "verifier", "final"]
    plan_summary: str
    acceptance_criteria: list[str]
    agent_handoffs: list[AgentHandoff]
    code_agent_summary: str
    verifier_summary: str
    last_error: str
    context_summary: str
    compression_events: list[CompressionEvent]
