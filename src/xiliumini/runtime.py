from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from xiliumini.agents.analysis import AnalysisAgent
from xiliumini.config import Settings
from xiliumini.core.agent import stream_agent
from xiliumini.errors import WorkspaceError, XiliuminiError
from xiliumini.events import ErrorEvent, RuntimeEvent
from xiliumini.graph.state import GraphState
from xiliumini.providers.openai_compatible import classify_provider_error, create_chat_model
from xiliumini.tools import get_builtin_tools, get_workspace_tools
from xiliumini.tools.delegate_analysis import make_delegate_analysis
from xiliumini.tools.workspace import create_session_workspace

ToolFactory = Callable[[Path], Sequence[BaseTool]]


class Runtime:
    """Own session workspaces and expose synchronous, stable graph events."""

    def __init__(
        self,
        model: Any,
        tools: Sequence[BaseTool] = (),
        checkpointer: Any | None = None,
        data_dir: Path = Path(".xiliumini"),
        workspace_tool_factory: ToolFactory | None = None,
    ) -> None:
        self._model = model
        self._tools = list(tools)
        self._checkpointer = checkpointer or InMemorySaver()
        self._data_dir = data_dir
        self._workspace_tool_factory = workspace_tool_factory or (lambda _workspace: ())

    def stream(
        self,
        task: str,
        session_id: str,
        max_attempts: int = 3,
    ) -> Iterator[RuntimeEvent]:
        try:
            workspace = create_session_workspace(self._data_dir, session_id)
            tools = [*self._tools, *self._workspace_tool_factory(workspace)]
        except WorkspaceError as exc:
            yield ErrorEvent(code=exc.code, message=str(exc))
            return

        inputs: GraphState = {
            "task": task,
            "todo": [],
            "result": "",
            "execution": [],
            "graph_state": "planning",
            "verification": "",
            "attempt": 0,
            "max_attempts": max_attempts,
            "final_answer": "",
            "session_id": session_id,
            "workspace": workspace,
        }
        try:
            yield from stream_agent(
                self._model,
                tools,
                inputs,
                checkpointer=self._checkpointer,
            )
        except XiliuminiError as exc:
            yield ErrorEvent(code=getattr(exc, "code", "runtime_error"), message=str(exc))
        except Exception as exc:
            error = classify_provider_error(exc)
            yield ErrorEvent(code=error.code, message=str(error))


def create_runtime(settings: Settings) -> Runtime:
    """Construct the main graph runtime and isolated analysis Agent."""

    main_model = create_chat_model(settings)
    analysis_agent = AnalysisAgent(create_chat_model(settings))
    delegate = make_delegate_analysis(
        analysis_agent.analyze,
        timeout_seconds=settings.analysis_timeout_seconds,
        max_chars=settings.analysis_max_chars,
    )
    return Runtime(
        main_model,
        tools=[*get_builtin_tools(), delegate],
        data_dir=settings.data_dir,
        workspace_tool_factory=get_workspace_tools,
    )
