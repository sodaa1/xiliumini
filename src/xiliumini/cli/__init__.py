from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import typer

from xiliumini import __version__
from xiliumini.config import Settings, load_settings
from xiliumini.core.approval import ApprovalDecision, ApprovalRequest
from xiliumini.errors import ConfigError, XiliuminiError
from xiliumini.events import (
    ErrorEvent,
    FinalEvent,
    PlannerEvent,
    ProgressEvent,
    RuntimeEvent,
    VerifierEvent,
)
from xiliumini.providers.openai_compatible import create_chat_model
from xiliumini.runtime import Runtime, create_runtime

app = typer.Typer(
    no_args_is_help=False,
    invoke_without_command=True,
    pretty_exceptions_show_locals=False,
)


@dataclass(frozen=True, slots=True)
class _CliOptions:
    workspace: Path | None = None
    max_attempts: int = 3
    approval_mode: Literal["inline", "auto", "deny"] = "inline"
    checkpoint_mode: Literal["light", "strict", "off"] | None = None
    trace_mode: Literal["on", "off"] | None = None


def _options(context: typer.Context) -> _CliOptions:
    return context.obj if isinstance(context.obj, _CliOptions) else _CliOptions()


def _option_was_defaulted(context: typer.Context, name: str) -> bool:
    source = context.get_parameter_source(name)
    return source is not None and source.name == "DEFAULT"


def _inline_approval(request: ApprovalRequest) -> ApprovalDecision:
    typer.echo(f"Approval required: {request.risk_reason}")
    typer.echo(f"Command: {request.command}")
    try:
        approved = typer.confirm("Approve this command?", default=False)
    except typer.Abort:
        raise KeyboardInterrupt from None
    return ApprovalDecision(
        approved=approved,
        reason="Approved by user." if approved else "Denied by user.",
    )


def _create_cli_runtime(settings: Settings, options: _CliOptions) -> Runtime:
    return create_runtime(
        settings,
        approval_mode=options.approval_mode,
        approval_handler=_inline_approval if options.approval_mode == "inline" else None,
        checkpoint_mode=options.checkpoint_mode,
        trace_mode=(
            None if options.trace_mode is None else "full" if options.trace_mode == "on" else "off"
        ),
    )


def _ensure_utf8_output() -> None:
    """Allow required stage icons on legacy Windows code pages."""

    for stream in (sys.stdout, sys.stderr):
        encoding = str(getattr(stream, "encoding", "")).lower().replace("_", "-")
        reconfigure = getattr(stream, "reconfigure", None)
        if encoding not in {"utf-8", "utf8", "utf-8-sig"} and callable(reconfigure):
            with suppress(OSError, ValueError):
                reconfigure(encoding="utf-8", errors="replace")


@app.callback()
def main(
    context: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the installed version.", is_eager=True),
    ] = False,
    resume: Annotated[
        Path | None, typer.Option("--resume", help="Resume a checkpoint workspace.")
    ] = None,
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", "-w", help="Use a named workspace under the data directory."),
    ] = None,
    max_attempts: Annotated[
        int,
        typer.Option("--max-attempts", min=1, help="Maximum Supervisor attempts."),
    ] = 3,
    approval_mode: Annotated[
        Literal["inline", "auto", "deny"],
        typer.Option("--approval-mode", help="Risky command approval policy."),
    ] = "inline",
    checkpoint_mode: Annotated[
        Literal["light", "strict", "off"],
        typer.Option("--checkpoint-mode", help="Checkpoint persistence level."),
    ] = "light",
    trace_mode: Annotated[
        Literal["on", "off"],
        typer.Option("--trace-mode", help="Execution trace persistence."),
    ] = "on",
) -> None:
    _ensure_utf8_output()
    options = _CliOptions(
        workspace=workspace,
        max_attempts=max_attempts,
        approval_mode=approval_mode,
        checkpoint_mode=(
            None if _option_was_defaulted(context, "checkpoint_mode") else checkpoint_mode
        ),
        trace_mode=(None if _option_was_defaulted(context, "trace_mode") else trace_mode),
    )
    context.obj = options
    if version:
        typer.echo(f"xiliumini {__version__}")
        raise typer.Exit()
    if workspace is not None and resume is not None:
        typer.echo("--workspace cannot be combined with --resume.")
        raise typer.Exit(code=2)
    if resume is not None:
        if context.invoked_subcommand is not None:
            typer.echo("--resume cannot be combined with a subcommand.")
            raise typer.Exit(code=2)
        try:
            runtime = _create_cli_runtime(load_settings(), options)
            exit_code = _render_events(
                runtime.resume(resume, max_attempts=max_attempts),
                no_stream=False,
                max_attempts=max_attempts,
            )
        except ConfigError as exc:
            typer.echo(f"Configuration error: {exc}")
            raise typer.Exit(code=2) from None
        except XiliuminiError as exc:
            typer.echo(f"Runtime error [{getattr(exc, 'code', 'runtime_error')}]: {exc}")
            raise typer.Exit(code=1) from None
        except KeyboardInterrupt:
            typer.echo("\nInterrupted.")
            raise typer.Exit(code=1) from None
        if exit_code:
            raise typer.Exit(code=exit_code)
        return
    if context.invoked_subcommand is None:
        _start_chat(options)


@app.command()
def doctor() -> None:
    """Check Python, configuration, and the local data directory."""

    try:
        if sys.version_info < (3, 12):  # noqa: UP036 - reports the explicit contract
            raise ConfigError("Python 3.12 or newer is required")
        settings = load_settings()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        create_chat_model(settings)
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}")
        raise typer.Exit(code=2) from None
    except OSError:
        typer.echo("Configuration error: data directory is unavailable")
        raise typer.Exit(code=2) from None

    typer.echo(f"OK: Python {sys.version_info.major}.{sys.version_info.minor}")
    typer.echo(f"OK: {len(settings.model_names)} model(s) configured")
    typer.echo(f"OK: data directory {settings.data_dir}")


@app.command()
def ask(
    context: typer.Context,
    question: Annotated[str, typer.Argument(help="Question for the agent.")],
    no_stream: Annotated[bool, typer.Option("--no-stream")] = False,
    max_attempts: Annotated[
        int | None,
        typer.Option("--max-attempts", min=1, help="Maximum Supervisor attempts."),
    ] = None,
) -> None:
    """Ask one question."""

    try:
        settings = load_settings()
        options = _options(context)
        attempts = options.max_attempts if max_attempts is None else max_attempts
        runtime = _create_cli_runtime(settings, options)
        exit_code = _run_question(
            runtime,
            question,
            str(uuid4()),
            no_stream,
            attempts,
            workspace=options.workspace,
        )
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}")
        raise typer.Exit(code=2) from None
    except XiliuminiError as exc:
        typer.echo(f"Runtime error: {exc}")
        raise typer.Exit(code=1) from None
    except KeyboardInterrupt:
        typer.echo("\nInterrupted.")
        raise typer.Exit(code=1) from None
    if exit_code:
        raise typer.Exit(code=exit_code)


def _run_question(
    runtime: Runtime,
    question: str,
    session_id: str,
    no_stream: bool,
    max_attempts: int = 3,
    *,
    workspace: Path | None = None,
) -> int:
    events = (
        runtime.stream(question, session_id, max_attempts=max_attempts)
        if workspace is None
        else runtime.stream_workspace(question, workspace, max_attempts=max_attempts)
    )
    return _render_events(
        events,
        no_stream=no_stream,
        max_attempts=max_attempts,
    )


def _render_events(
    events: Iterator[RuntimeEvent], *, no_stream: bool, max_attempts: int = 3
) -> int:
    for event in events:
        if isinstance(event, PlannerEvent) and not no_stream:
            typer.echo(f"📋 Planner (attempt {event.attempt}/{max_attempts}): {event.summary}")
        elif isinstance(event, ProgressEvent) and not no_stream:
            typer.echo(f"  ↳ {event.message}")
        elif isinstance(event, VerifierEvent) and not no_stream:
            icon = "✅" if event.passed else "❌"
            typer.echo(f"{icon} Verifier: {event.reason}")
        elif isinstance(event, FinalEvent):
            typer.echo(f"📝 Final: {event.text}")
        elif isinstance(event, ErrorEvent):
            typer.echo(f"Error [{event.code}]: {event.message}")
            return 1
    return 0


@app.command()
def chat(
    context: typer.Context,
    session: Annotated[str | None, typer.Option("--session")] = None,
) -> None:
    """Start an in-memory conversation."""

    if session:
        typer.echo("Session persistence is scheduled for Task 3.")
        raise typer.Exit(code=1)
    _start_chat(_options(context))


def _start_chat(options: _CliOptions | None = None) -> None:
    options = options or _CliOptions()
    try:
        settings = load_settings()
        runtime = _create_cli_runtime(settings, options)
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}")
        raise typer.Exit(code=2) from None
    except XiliuminiError as exc:
        typer.echo(f"Runtime error: {exc}")
        raise typer.Exit(code=1) from None

    try:
        _chat_loop(runtime, settings, options)
    except KeyboardInterrupt:
        typer.echo("\nGoodbye.")


def _chat_loop(runtime: Runtime, settings: Settings, options: _CliOptions | None = None) -> None:
    options = options or _CliOptions()
    session_id = str(uuid4())
    typer.echo(f"xiliumini {__version__} · {settings.model}")
    typer.echo("Type /help for commands. Ctrl+C or Ctrl+D exits.")

    while True:
        try:
            question = typer.prompt("❯", prompt_suffix=" ").strip()
        except (EOFError, KeyboardInterrupt, typer.Abort):
            typer.echo("\nGoodbye.")
            return

        if not question:
            continue
        if question == "/exit":
            typer.echo("Goodbye.")
            return
        if question == "/help":
            typer.echo("/help  /status  /new  /exit")
            continue
        if question == "/status":
            typer.echo(f"Model: {settings.model}\nSession: {session_id}")
            continue
        if question == "/new":
            session_id = str(uuid4())
            typer.echo(f"New session: {session_id}")
            continue
        if question.startswith("/"):
            typer.echo(f"Unknown command: {question}. Type /help.")
            continue

        _run_question(
            runtime,
            question,
            session_id,
            no_stream=False,
            max_attempts=options.max_attempts,
            workspace=options.workspace,
        )


@app.command("sessions")
def list_sessions() -> None:
    """List saved sessions when persistence is available."""

    typer.echo("No sessions yet; persistence is scheduled for Task 3.")
