"""Textual interface for persistent xiliumini sessions."""

from __future__ import annotations

from pathlib import Path


def run_tui(
    *,
    session_workspace: Path | None = None,
    max_attempts: int = 3,
    approval_mode: str = "inline",
    checkpoint_mode: str = "light",
    trace_mode: str = "on",
) -> None:
    """Start the Textual UI without importing Textual during normal CLI startup."""
    from xiliumini.cli.tui.app import run_tui as start

    start(
        session_workspace=session_workspace,
        max_attempts=max_attempts,
        approval_mode=approval_mode,
        checkpoint_mode=checkpoint_mode,
        trace_mode=trace_mode,
    )


__all__ = ["run_tui"]
