"""Core primitives shared across the xiliumini runtime."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xiliumini.core.agent import stream_agent, stream_agent_events, stream_session_events


def __getattr__(name: str) -> Any:
    # Tools import core primitives while workflow nodes are still loading.
    if name in {"stream_agent", "stream_agent_events", "stream_session_events"}:
        from xiliumini.core.agent import stream_agent, stream_agent_events, stream_session_events

        return {
            "stream_agent": stream_agent,
            "stream_agent_events": stream_agent_events,
            "stream_session_events": stream_session_events,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["stream_agent", "stream_agent_events", "stream_session_events"]
