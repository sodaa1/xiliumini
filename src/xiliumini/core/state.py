from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class RuntimeState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    session_id: str
    workspace: Path
    step_count: int
