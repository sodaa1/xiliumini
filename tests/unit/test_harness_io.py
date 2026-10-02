import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import SecretStr

from xiliumini.core.harness_io import normalize_checkpoint_mode, normalize_trace_mode
from xiliumini.errors import CheckpointError, TraceError


def test_checkpoint_mode_defaults_and_invalid_values_fall_back_to_light():
    assert [normalize_checkpoint_mode(x) for x in (None, "bad", "light", "strict", "off")] == [
        "light",
        "light",
        "light",
        "strict",
        "off",
    ]


def test_trace_mode_defaults_and_invalid_values_fall_back_to_full():
    assert [normalize_trace_mode(x) for x in (None, "bad", "full", "summary", "off")] == [
        "full",
        "full",
        "full",
        "summary",
        "off",
    ]


def test_checkpoint_and_trace_errors_have_stable_codes():
    assert CheckpointError.code == "checkpoint_error"
    assert TraceError.code == "trace_error"


def test_safe_serialization_roundtrip_and_redaction(tmp_path):
    from xiliumini.core.harness_io import restore_persisted_value, sanitize_for_persistence

    @dataclass
    class Example:
        path: Path

    messages = [
        HumanMessage(content="hello"),
        AIMessage(content="answer"),
        ToolMessage(content="done", tool_call_id="id"),
    ]
    value = {
        "path": tmp_path / "file",
        "root": tmp_path,
        "tuple": (1, "two"),
        "data": Example(tmp_path / "file"),
        "messages": messages,
        "api_key": "hidden",
        "secret": SecretStr("hidden"),
        "output": str(tmp_path) + "a" * 100,
    }
    encoded = sanitize_for_persistence(value, tmp_path, max_text=30)
    assert "hidden" not in str(encoded)
    assert str(tmp_path) not in str(encoded)
    assert "[TRUNCATED]" in encoded["output"]
    restored = restore_persisted_value(encoded, tmp_path)
    assert restored["path"] == tmp_path / "file"
    assert restored["root"] == tmp_path
    assert restored["tuple"] == (1, "two")
    assert restored["data"] == {"path": tmp_path / "file"}
    assert restored["messages"] == messages


def test_unknown_objects_and_type_tags_are_rejected(tmp_path):
    from xiliumini.core.harness_io import restore_persisted_value, sanitize_for_persistence

    with pytest.raises(ValueError):
        sanitize_for_persistence(object(), tmp_path)
    with pytest.raises(ValueError):
        restore_persisted_value({"__harness_type__": "unknown"}, tmp_path)
    with pytest.raises(ValueError, match="unknown persistence tag"):
        restore_persisted_value({"__harness_type__": "unknown", "value": {}}, tmp_path)
    with pytest.raises(ValueError):
        restore_persisted_value({"__harness_type__": "path", "value": "../escape"}, tmp_path)


def test_workspace_text_and_keys_are_redacted_across_windows_spellings(tmp_path):
    from xiliumini.core.harness_io import sanitize_for_persistence

    variants = [str(tmp_path), tmp_path.as_posix()]
    if os.name == "nt":
        variants.extend([str(tmp_path).swapcase(), tmp_path.as_posix().swapcase()])
    for path in variants:
        assert sanitize_for_persistence(path + "/file", tmp_path) == "<workspace>/file"
        assert sanitize_for_persistence({path: "value"}, tmp_path) == {"<workspace>": "value"}
    with pytest.raises(ValueError, match="key collision"):
        sanitize_for_persistence({str(tmp_path): 1, "<workspace>": 2}, tmp_path)


@pytest.mark.parametrize("message_type", [[], {}, None, 3])
def test_invalid_message_type_is_controlled_value_error(tmp_path, message_type):
    from xiliumini.core.harness_io import restore_persisted_value

    with pytest.raises(ValueError, match="invalid message"):
        restore_persisted_value(
            {"__harness_type__": "message", "value": {"type": message_type, "data": {}}},
            tmp_path,
        )


def test_atomic_json_and_complete_jsonl_lines(tmp_path):
    import json

    from xiliumini.core.harness_io import append_jsonl, write_json_atomic

    path = tmp_path / "nested" / "state.json"
    write_json_atomic(path, {"value": "中文"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": "中文"}
    log = path.with_suffix(".jsonl")
    append_jsonl(log, {"line": 1})
    append_jsonl(log, {"line": 2})
    assert [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] == [
        {"line": 1},
        {"line": 2},
    ]
