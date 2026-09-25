from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool

Clock = Callable[[], datetime]


def current_time_value(timezone: str, clock: Clock | None = None) -> str:
    """Return the current time in an IANA timezone, with an injectable clock."""

    try:
        zone = ZoneInfo(timezone)
    except (ValueError, ZoneInfoNotFoundError):
        return f"Error: invalid timezone '{timezone}'"

    now = (clock or (lambda: datetime.now(UTC)))()
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(zone).strftime("%Y-%m-%d %H:%M:%S %Z%z")


@tool
def current_time(timezone: str) -> str:
    """Return the current date and time for an IANA timezone such as UTC."""

    return current_time_value(timezone)
