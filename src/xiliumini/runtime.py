from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from xiliumini.agents.analysis import AnalysisAgent
from xiliumini.config import Settings
from xiliumini.core.agent import build_actor
from xiliumini.core.state import RuntimeState
from xiliumini.errors import WorkspaceError
from xiliumini.events import (
    ErrorEvent,
    FinalEvent,
    RuntimeEvent,
)
from xiliumini.providers.openai_compatible import classify_provider_error, create_chat_model
from xiliumini.tools import get_builtin_tools, get_workspace_tools
from xiliumini.tools.delegate_analysis import make_delegate_analysis
from xiliumini.tools.workspace import create_session_workspace

ToolFactory = Callable[[Path], Sequence[BaseTool]]


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


class Runtime:
    """Own a checkpointed workflow and expose stable runtime events."""

    def __init__(
        self,
        model: Any,
        tools: Sequence[BaseTool] = (),
        max_steps: int = 8,
        checkpointer: Any | None = None,
        data_dir: Path = Path(".xiliumini"),
        workspace_tool_factory: ToolFactory | None = None,
    ) -> None:
        self._model = model
        self._tools = list(tools)
        self._max_steps = max_steps
        self._checkpointer = checkpointer or InMemorySaver()
        self._data_dir = data_dir
        self._workspace_tool_factory = workspace_tool_factory or (lambda _workspace: ())

    async def astream(self, task: str, session_id: str) -> AsyncIterator[RuntimeEvent]:
        try:
            workspace = create_session_workspace(self._data_dir, session_id)
            tools = [*self._tools, *self._workspace_tool_factory(workspace)]
        except WorkspaceError as exc:
            yield ErrorEvent(code=exc.code, message=str(exc))
            return

        queue: asyncio.Queue[RuntimeEvent | None] = asyncio.Queue()
        result_holder: dict[str, Any] = {}

        async def emit(event: RuntimeEvent) -> None:
            await queue.put(event)

        workflow = build_actor(
            self._model,
            tools,
            self._max_steps,
            checkpointer=self._checkpointer,
            event_sink=emit,
        )
        initial_state: RuntimeState = {
            "messages": [HumanMessage(content=task)],
            "session_id": session_id,
            "workspace": workspace,
            "step_count": 0,
        }

        async def execute() -> None:
            try:
                result_holder["result"] = await workflow.ainvoke(
                    initial_state,
                    config={"configurable": {"thread_id": session_id}},
                )
            except Exception as exc:
                error = classify_provider_error(exc)
                await emit(ErrorEvent(code=error.code, message=str(error)))
            finally:
                await queue.put(None)

        task_handle = asyncio.create_task(execute())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
            await task_handle
        finally:
            if not task_handle.done():
                task_handle.cancel()
                with suppress(asyncio.CancelledError):
                    await task_handle

        result = result_holder.get("result")
        if result is None:
            return

        final = next(
            (
                message
                for message in reversed(result["messages"])
                if isinstance(message, AIMessage) and not message.tool_calls
            ),
            AIMessage(content=""),
        )
        text = _message_text(final.content)
        yield FinalEvent(text=text, session_id=session_id)


def create_runtime(settings: Settings) -> Runtime:
    """Construct the main and isolated analysis Agents from application settings."""

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
        max_steps=settings.max_steps,
        data_dir=settings.data_dir,
        workspace_tool_factory=get_workspace_tools,
    )
