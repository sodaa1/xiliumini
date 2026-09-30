from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, cast

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

from xiliumini.agents.react import model_factory
from xiliumini.config import Settings
from xiliumini.core.agent import stream_agent
from xiliumini.errors import WorkspaceError, XiliuminiError
from xiliumini.events import ErrorEvent, RuntimeEvent
from xiliumini.graph.memory import MemoryLimits, MemoryManager
from xiliumini.graph.state import GraphState
from xiliumini.providers.openai_compatible import classify_provider_error, create_chat_model
from xiliumini.tools.preferences import UserPreferenceStore
from xiliumini.tools.todo import TodoStore
from xiliumini.tools.web_search_tool import search_api_key
from xiliumini.tools.workspace import create_session_workspace


class Runtime:
    """Own session workspaces and expose synchronous, stable graph events."""

    def __init__(
        self,
        model: Any,
        checkpointer: Any | None = None,
        data_dir: Path = Path(".xiliumini"),
        agent_model_factory: Callable[[], Any] | None = None,
        tavily_api_key: SecretStr | None = None,
        memory_limits: MemoryLimits | None = None,
        model_name: str = "unknown-model",
        token_counter: Callable[[Sequence[BaseMessage], str], int] | None = None,
    ) -> None:
        self._model = model
        self._checkpointer = checkpointer or InMemorySaver()
        self._data_dir = data_dir
        self._agent_model_factory = agent_model_factory
        self._tavily_api_key = tavily_api_key
        self._memory_limits = memory_limits or MemoryLimits(64_000, 0.8, 8_000)
        self._model_name = model_name
        self._token_counter = token_counter

    def stream(
        self,
        task: str,
        session_id: str,
        max_attempts: int = 3,
    ) -> Iterator[RuntimeEvent]:
        try:
            workspace = create_session_workspace(self._data_dir, session_id)
            TodoStore(workspace).start_task()
        except WorkspaceError as exc:
            yield ErrorEvent(code=exc.code, message=str(exc))
            return
        except OSError:
            yield ErrorEvent(code="workspace_error", message="could not initialize task progress")
            return

        try:
            memory_manager = MemoryManager(
                workspace,
                UserPreferenceStore(self._data_dir),
                self._memory_limits,
                model_name=self._model_name,
                token_counter=self._token_counter,
            )
            initial: dict[str, Any] = {
                "task": task,
                "todos": [],
                "research_notes": [],
                "agent_results": [],
                "tool_events": [],
                "supervisor_ok": False,
                "result": "",
                "graph_state": "planning",
                "verification": "",
                "attempt": 0,
                "max_attempts": max_attempts,
                "final_answer": "",
                "session_id": session_id,
                "workspace": workspace,
                "current_node": "planner",
                "plan_summary": "",
                "acceptance_criteria": [],
                "agent_handoffs": [],
                "code_agent_summary": "",
                "verifier_summary": "",
                "last_error": "",
                "context_summary": "",
                "compression_events": [],
            }
            inputs = cast(
                GraphState,
                {
                    **initial,
                    "memory": memory_manager.assemble(initial, current_node="planner"),
                },
            )
            token = model_factory.set(self._agent_model_factory)
            search_token = search_api_key.set(self._tavily_api_key)
            try:
                events = iter(
                    stream_agent(
                        self._model,
                        inputs,
                        checkpointer=self._checkpointer,
                        memory_manager=memory_manager,
                    )
                )
            finally:
                model_factory.reset(token)
                search_api_key.reset(search_token)
            while True:
                token = model_factory.set(self._agent_model_factory)
                search_token = search_api_key.set(self._tavily_api_key)
                try:
                    event = next(events)
                except StopIteration:
                    return
                finally:
                    model_factory.reset(token)
                    search_api_key.reset(search_token)
                yield event
        except XiliuminiError as exc:
            yield ErrorEvent(code=getattr(exc, "code", "runtime_error"), message=str(exc))
        except Exception as exc:
            error = classify_provider_error(exc)
            yield ErrorEvent(code=error.code, message=str(error))


def create_runtime(settings: Settings) -> Runtime:
    """Construct the Supervisor runtime with independent specialist models."""

    main_model = create_chat_model(settings)
    return Runtime(
        main_model,
        data_dir=settings.data_dir,
        agent_model_factory=lambda: create_chat_model(settings),
        tavily_api_key=settings.tavily_api_key,
        memory_limits=MemoryLimits(
            settings.context_window_tokens,
            settings.compression_trigger_ratio,
            settings.compression_keep_tokens,
        ),
        model_name=settings.model,
    )
