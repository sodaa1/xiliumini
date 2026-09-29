from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xiliumini.tools.notepad import read_notepad


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def build_memory_snapshot(state: Mapping[str, Any]) -> dict[str, object]:
    """Build the current JSON-safe layered-memory boundary."""

    workspace = state.get("workspace")
    notepad = read_notepad(workspace) if isinstance(workspace, Path) else ""
    return {
        "session_id": str(state.get("session_id", "")),
        "todos": _json_copy(state.get("todos", [])),
        "research_notes": _json_copy(state.get("research_notes", [])),
        "notepad": notepad,
    }
