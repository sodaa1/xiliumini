"""Shared, explicit persistence encoding for checkpoint and trace recorders."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from pydantic import SecretStr

from xiliumini.tools.workspace import atomic_write_utf8

_TAG = "__harness_type__"
_SENSITIVE = re.compile(r"token|secret|password|authorization|api_key", re.IGNORECASE)
_MESSAGE_TYPES = {"human", "ai", "system", "tool", "function", "chat"}


def normalize_checkpoint_mode(mode: str | None) -> str:
    return mode if mode in {"light", "strict", "off"} else "light"


def normalize_trace_mode(mode: str | None) -> str:
    return mode if mode in {"full", "summary", "off"} else "full"


def sanitize_for_persistence(value: Any, workspace: Path, *, max_text: int = 20_000) -> Any:
    """Encode known values without executing or printing arbitrary objects.

    ValueError signals unsupported data; recorders translate it to their controlled error.
    Dataclasses restore as field mappings rather than dynamically imported classes.
    """
    if max_text < 1:
        raise ValueError("invalid text limit")
    root = workspace.resolve()
    workspace_pattern = re.compile(
        "[\\\\/]".join(re.escape(part) for part in root.as_posix().split("/")),
        re.IGNORECASE if os.name == "nt" else 0,
    )

    def redact_workspace(text: str) -> str:
        return workspace_pattern.sub("<workspace>", text)

    def encode(item: Any) -> Any:
        if isinstance(item, SecretStr):
            return "[REDACTED]"
        if item is None or isinstance(item, (bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("non-finite number")
            return item
        if isinstance(item, str):
            text = redact_workspace(item)
            return text if len(text) <= max_text else text[:max_text] + "[TRUNCATED]"
        if isinstance(item, Path):
            try:
                relative = item.resolve().relative_to(root).as_posix()
            except ValueError:
                raise ValueError("path outside workspace") from None
            return {_TAG: "path", "value": relative if relative != "." else "<workspace>"}
        if isinstance(item, BaseMessage):
            if item.type not in _MESSAGE_TYPES:
                raise ValueError("unsupported message")
            return {_TAG: "message", "value": encode(message_to_dict(item))}
        if is_dataclass(item) and not isinstance(item, type):
            return {
                _TAG: "dataclass",
                "value": encode({field.name: getattr(item, field.name) for field in fields(item)}),
            }
        if isinstance(item, tuple):
            return {_TAG: "tuple", "value": [encode(part) for part in item]}
        if isinstance(item, list):
            return [encode(part) for part in item]
        if isinstance(item, dict):
            if _TAG in item or any(not isinstance(key, str) for key in item):
                raise ValueError("invalid mapping keys")
            result = {}
            for key, part in item.items():
                safe_key = redact_workspace(key)
                if safe_key in result:
                    raise ValueError("persistence key collision")
                result[safe_key] = "[REDACTED]" if _SENSITIVE.search(key) else encode(part)
            return result
        raise ValueError("unsupported persistence value")

    return encode(value)


def restore_persisted_value(value: Any, workspace: Path) -> Any:
    """Decode only known tags; never dynamically import saved objects."""
    if isinstance(value, list):
        return [restore_persisted_value(part, workspace) for part in value]
    if isinstance(value, dict):
        if _TAG not in value:
            return {key: restore_persisted_value(part, workspace) for key, part in value.items()}
        if set(value) != {_TAG, "value"}:
            raise ValueError("invalid persistence tag")
        kind, payload = value[_TAG], value["value"]
        if kind == "path":
            if not isinstance(payload, str):
                raise ValueError("invalid path")
            root = workspace.resolve()
            if payload == "<workspace>":
                return root
            if (
                Path(payload).is_absolute()
                or PureWindowsPath(payload).drive
                or ".." in Path(payload).parts
            ):
                raise ValueError("invalid path")
            path = (root / payload).resolve()
            if not path.is_relative_to(root):
                raise ValueError("invalid path")
            return path
        if kind == "tuple" and isinstance(payload, list):
            return tuple(restore_persisted_value(part, workspace) for part in payload)
        if kind == "dataclass" and isinstance(payload, dict):
            return restore_persisted_value(payload, workspace)
        if kind == "message" and isinstance(payload, dict):
            decoded = restore_persisted_value(payload, workspace)
            message_type = decoded.get("type")
            if (
                not isinstance(message_type, str)
                or message_type not in _MESSAGE_TYPES
                or not isinstance(decoded.get("data"), dict)
            ):
                raise ValueError("invalid message")
            try:
                return messages_from_dict([decoded])[0]
            except (TypeError, ValueError, KeyError):
                raise ValueError("invalid message") from None
        raise ValueError("unknown persistence tag")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("invalid persistence value")


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace one JSON document atomically; caller owns error classification."""
    atomic_write_utf8(path, json.dumps(dict(payload), ensure_ascii=False, allow_nan=False) + "\n")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one pre-encoded JSON record and sync it before returning."""
    line = json.dumps(dict(payload), ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
