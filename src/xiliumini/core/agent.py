from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from xiliumini.core.approval import ApprovalDecision, ApprovalRequest
from xiliumini.core.session import (
    MAX_TURN_CONTENT,
    SESSION_ROOT,
    append_assistant_turn,
    append_user_turn,
    build_session_context,
    load_or_create_session,
    save_session,
)
from xiliumini.events import (
    ErrorEvent,
    FinalEvent,
    PlannerEvent,
    ProgressEvent,
    RuntimeEvent,
    VerifierEvent,
)
from xiliumini.execution.gateway import ExecutionGateway, active_run
from xiliumini.graph.state import GraphState
from xiliumini.graph.workflow import build_workflow


def _update_event(node: str, update: Any, session_id: str) -> RuntimeEvent | None:
    if not isinstance(update, dict):
        return None
    if node == "planner":
        todos = update.get("todos")
        result = update.get("result")
        attempt = update.get("attempt")
        if isinstance(todos, list) and isinstance(result, str) and isinstance(attempt, int):
            return PlannerEvent(todos=todos, summary=result, attempt=attempt)
    elif node == "verifier":
        status = update.get("graph_state")
        reason = update.get("verification")
        attempt = update.get("attempt")
        if status in {"passed", "failed"} and isinstance(reason, str) and isinstance(attempt, int):
            return VerifierEvent(passed=status == "passed", reason=reason, attempt=attempt)
    elif node == "final":
        answer = update.get("final_answer")
        if isinstance(answer, str):
            return FinalEvent(text=answer, session_id=session_id)
    return None


def _chunk_events(chunk: Any, session_id: str) -> Iterator[RuntimeEvent]:
    if not isinstance(chunk, tuple) or len(chunk) != 2:
        return
    mode, payload = chunk
    if not isinstance(payload, dict):
        return
    if mode == "custom":
        event_type = payload.get("type")
        stage = payload.get("stage")
        message = payload.get("message")
        if event_type == "handoff" and not isinstance(stage, str):
            stage = "planner"
        if event_type == "handoff" and not isinstance(message, str):
            agent = payload.get("agent")
            summary = payload.get("summary")
            target = agent if isinstance(agent, str) else "unknown"
            outcome = summary if isinstance(summary, str) else "Delegation completed"
            message = f"planner → {target}: {outcome}"
        if isinstance(stage, str) and isinstance(message, str):
            yield ProgressEvent(
                stage=stage,
                message=message,
                event_type=event_type if isinstance(event_type, str) else "progress",
                details=deepcopy(payload),
            )
        return
    if mode != "updates":
        return
    for node, update in payload.items():
        event = _update_event(node, update, session_id)
        if event is not None:
            yield event


def stream_agent(
    model: Any,
    inputs: GraphState,
    *,
    memory_manager: Any,
    checkpointer: Any | None = None,
    checkpoint_manager: Any | None = None,
    trace_recorder: Any | None = None,
    resumed: bool = False,
    resume_event: Any = None,
) -> Generator[RuntimeEvent, None, None]:
    """Run the graph and translate LangGraph chunks into stable runtime events."""

    if inputs["max_attempts"] < 1:
        raise ValueError("max_attempts must be at least 1")
    # Runtime owns live services; persistence observes only independent state data.
    latest_state = deepcopy({key: value for key, value in inputs.items() if key != "runtime"})
    latest_node = None

    def save(status: str, event: Any = None) -> ProgressEvent | None:
        if checkpoint_manager is None or checkpoint_manager.mode == "off":
            return None
        saved = checkpoint_manager.save(
            deepcopy(latest_state), status=status, latest_node=latest_node, event=deepcopy(event)
        )
        if saved is not None and trace_recorder is not None:
            trace_recorder.record_custom_event(deepcopy(saved))
        if not isinstance(saved, dict):
            return None
        saved_type = saved.get("type")
        return ProgressEvent(
            stage="checkpoint",
            message=f"Checkpoint saved: {status}",
            event_type=saved_type if isinstance(saved_type, str) else "checkpoint_saved",
            details=deepcopy(saved),
        )

    status = "completed"
    primary: BaseException | None = None
    chunks: Any = None
    try:
        if trace_recorder is not None:
            trace_recorder.start(
                deepcopy(latest_state), resumed=resumed, resume_event=deepcopy(resume_event)
            )
        run_context = active_run.get()
        if run_context is not None and run_context.hook_engine is not None:
            for hook_event in run_context.hook_engine.lifecycle(
                "run.start", context=run_context, gateway=ExecutionGateway()
            ):
                if run_context.event_sink is not None:
                    run_context.event_sink(hook_event)
        save("started")
        workflow = build_workflow(model, memory_manager, checkpointer=checkpointer)
        chunks = workflow.stream(
            inputs,
            config={
                "configurable": {"thread_id": inputs["session_id"]},
                "recursion_limit": 2 * inputs["max_attempts"] + 5,
            },
            stream_mode=["updates", "custom"],
        )
        for chunk in chunks:
            checkpoint_events: list[ProgressEvent] = []
            if isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[1], dict):
                mode, payload = chunk
                if mode == "custom":
                    if trace_recorder is not None:
                        trace_recorder.record_custom_event(deepcopy(payload))
                    if checkpoint_manager is not None and checkpoint_manager.mode == "strict":
                        saved_event = save("running", payload)
                        if saved_event is not None:
                            checkpoint_events.append(saved_event)
                elif mode == "updates":
                    for node, update in payload.items():
                        if isinstance(update, dict):
                            latest_state.update(deepcopy(update))
                            latest_node = node
                        node_event = {node: update}
                        if trace_recorder is not None:
                            trace_recorder.record_graph_update(deepcopy(node_event))
                        saved_event = save("running", node_event)
                        if saved_event is not None:
                            checkpoint_events.append(saved_event)
            yield from _chunk_events(chunk, inputs["session_id"])
            yield from checkpoint_events
    except BaseException as error:
        primary = error
        status = (
            "interrupted" if isinstance(error, (KeyboardInterrupt, GeneratorExit)) else "failed"
        )
        raise
    finally:
        # Closing LangGraph waits for its background executor. Snapshot only
        # after running nodes have stopped writing the workspace.
        failure: BaseException | None = None
        try:
            close = getattr(chunks, "close", None)
            if callable(close):
                close()
        except BaseException as error:
            failure = error
            if primary is None:
                status = (
                    "interrupted"
                    if isinstance(error, (KeyboardInterrupt, GeneratorExit))
                    else "failed"
                )
        # Attempt each terminal operation once, retaining the original exception.
        try:
            save(status)
        except BaseException as error:
            if failure is None:
                failure = error
            if status == "completed":
                status = "failed"
        if trace_recorder is not None:
            if status == "failed":
                try:
                    trace_recorder.record_custom_event({"type": "error", "message": "Run failed."})
                except BaseException as error:
                    if failure is None:
                        failure = error
            run_context = active_run.get()
            if run_context is not None and run_context.hook_engine is not None:
                terminal = (
                    "run.success"
                    if status == "completed" and latest_state.get("graph_state") == "passed"
                    else "run.failure"
                )
                try:
                    for hook_event in run_context.hook_engine.lifecycle(
                        terminal, context=run_context, gateway=ExecutionGateway()
                    ):
                        if run_context.event_sink is not None:
                            run_context.event_sink(hook_event)
                except BaseException as error:
                    if failure is None:
                        failure = error
            try:
                trace_recorder.end(
                    status=status, latest_node=latest_node, final_state=deepcopy(latest_state)
                )
            except BaseException as error:
                if failure is None:
                    failure = error
        if primary is None and failure is not None:
            raise failure


def stream_agent_events(
    task: str,
    *,
    workspace: Path,
    max_attempts: int = 3,
    approval_mode: str = "inline",
    approval_handler: Callable[[ApprovalRequest], ApprovalDecision] | None = None,
    checkpoint_mode: str = "light",
    resume_workspace: Path | None = None,
    trace_mode: str = "on",
) -> Generator[dict[str, Any], None, None]:
    """Run through Runtime and adapt stable events to legacy event dictionaries."""

    if resume_workspace is not None and workspace.resolve(strict=False) != resume_workspace.resolve(
        strict=False
    ):
        event = ErrorEvent(code="workspace_error", message="workspace conflicts with resume")
        yield {"type": "graph_event", "event": asdict(event)}
        return

    # Local imports preserve the intentionally lazy public core package boundary.
    from xiliumini.config import load_settings
    from xiliumini.runtime import create_runtime

    runtime = create_runtime(
        load_settings(),
        approval_mode=approval_mode,
        approval_handler=approval_handler,
        checkpoint_mode=checkpoint_mode,
        trace_mode="full" if trace_mode == "on" else trace_mode,
    )
    events = (
        runtime.resume(resume_workspace, task=task, max_attempts=max_attempts)
        if resume_workspace is not None
        else runtime.stream_workspace(task, workspace, max_attempts=max_attempts)
    )
    try:
        for event in events:
            yield {
                "type": "custom_event" if isinstance(event, ProgressEvent) else "graph_event",
                "event": asdict(event),
            }
    finally:
        close = getattr(events, "close", None)
        if callable(close):
            close()


def stream_session_events(
    task: str,
    *,
    session_workspace: Path | None = None,
    max_attempts: int = 3,
    approval_mode: str = "inline",
    approval_handler: Callable[[ApprovalRequest], ApprovalDecision] | None = None,
    checkpoint_mode: str = "light",
    trace_mode: str = "on",
) -> Generator[dict[str, Any], None, None]:
    """Persist and route one multi-turn session message."""
    from xiliumini.config import load_settings
    from xiliumini.runtime import create_runtime

    workspace = (session_workspace or Path(SESSION_ROOT).parent).expanduser().resolve()
    session = load_or_create_session(workspace)
    user_turn = append_user_turn(session, task)
    save_session(workspace, session)
    context = build_session_context(workspace, session)
    runtime = create_runtime(
        load_settings(),
        approval_mode=approval_mode,
        approval_handler=approval_handler,
        checkpoint_mode=checkpoint_mode,
        trace_mode="full" if trace_mode == "on" else trace_mode,
        data_dir=workspace,
    )
    events = runtime.stream_session(
        task,
        session["session_id"],
        context,
        max_attempts=max_attempts,
    )
    route = "chat"
    planner_summary = ""
    verifier_summary = ""
    try:
        for event in events:
            if isinstance(event, (ProgressEvent, PlannerEvent, VerifierEvent)):
                route = "workflow"
            if isinstance(event, PlannerEvent) and event.summary:
                planner_summary = event.summary
            elif isinstance(event, VerifierEvent) and event.reason:
                verifier_summary = event.reason
            if isinstance(event, FinalEvent):
                if verifier_summary:
                    available = MAX_TURN_CONTENT - len(verifier_summary) - 2
                    summary = (
                        verifier_summary[:MAX_TURN_CONTENT]
                        if available <= 0 or not planner_summary
                        else f"{planner_summary[:available]}; {verifier_summary}"
                    )
                else:
                    summary = planner_summary[:MAX_TURN_CONTENT]
                append_assistant_turn(
                    session,
                    turn=user_turn + 1,
                    route=route,
                    content=event.text,
                    summary=summary,
                )
                save_session(workspace, session)
            yield {
                "type": "custom_event" if isinstance(event, ProgressEvent) else "graph_event",
                "event": asdict(event),
            }
    finally:
        close = getattr(events, "close", None)
        if callable(close):
            close()
