from __future__ import annotations

import asyncio
import sys
from typing import Annotated
from uuid import uuid4

import typer

from xiliumini import __version__
from xiliumini.config import Settings, load_settings
from xiliumini.errors import ConfigError, XiliuminiError
from xiliumini.events import ErrorEvent, FinalEvent, TokenEvent
from xiliumini.providers.openai_compatible import create_chat_model
from xiliumini.runtime import Runtime, create_runtime

app = typer.Typer(
    no_args_is_help=False,
    invoke_without_command=True,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def main(
    context: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the installed version.", is_eager=True),
    ] = False,
) -> None:
    if version:
        typer.echo(f"xiliumini {__version__}")
        raise typer.Exit()
    if context.invoked_subcommand is None:
        _start_chat()


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
    question: Annotated[str, typer.Argument(help="Question for the agent.")],
    no_stream: Annotated[bool, typer.Option("--no-stream")] = False,
) -> None:
    """Ask one question."""

    try:
        settings = load_settings()
        runtime = create_runtime(settings)
        exit_code = asyncio.run(_run_question(runtime, question, str(uuid4()), no_stream))
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


async def _run_question(runtime, question: str, session_id: str, no_stream: bool) -> int:
    emitted_token = False
    async for event in runtime.astream(question, session_id):
        if isinstance(event, TokenEvent):
            if not no_stream:
                typer.echo(event.text, nl=False)
                emitted_token = True
        elif isinstance(event, FinalEvent):
            if no_stream or not emitted_token:
                typer.echo(event.text)
            else:
                typer.echo()
        elif isinstance(event, ErrorEvent):
            if emitted_token:
                typer.echo()
            typer.echo(f"Error [{event.code}]: {event.message}")
            return 1
    return 0


@app.command()
def chat(
    session: Annotated[str | None, typer.Option("--session")] = None,
) -> None:
    """Start an in-memory conversation."""

    if session:
        typer.echo("Session persistence is scheduled for Task 3.")
        raise typer.Exit(code=1)
    _start_chat()


def _start_chat() -> None:
    try:
        settings = load_settings()
        runtime = create_runtime(settings)
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}")
        raise typer.Exit(code=2) from None
    except XiliuminiError as exc:
        typer.echo(f"Runtime error: {exc}")
        raise typer.Exit(code=1) from None

    try:
        asyncio.run(_chat_loop(runtime, settings))
    except KeyboardInterrupt:
        typer.echo("\nGoodbye.")


async def _chat_loop(runtime: Runtime, settings: Settings) -> None:
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

        await _run_question(runtime, question, session_id, no_stream=False)


@app.command("sessions")
def list_sessions() -> None:
    """List saved sessions when persistence is available."""

    typer.echo("No sessions yet; persistence is scheduled for Task 3.")
