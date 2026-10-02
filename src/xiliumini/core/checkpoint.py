"""Durable checkpoint metadata and independent workspace Git snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from langchain_core.messages import BaseMessage
from typing_extensions import is_typeddict

from xiliumini.core.harness_io import (
    append_jsonl,
    normalize_checkpoint_mode,
    restore_persisted_value,
    sanitize_for_persistence,
    write_json_atomic,
)
from xiliumini.errors import CheckpointError
from xiliumini.graph.state import GraphState
from xiliumini.tools.workspace import atomic_write_utf8

_ERROR = "Checkpoint persistence failed."
_CONTROL = {".git", ".xiliumini"}


def _safe_directory(path: Path, workspace: Path) -> None:
    if not path.resolve().is_relative_to(workspace.resolve()):
        raise CheckpointError(_ERROR)
    relative = path.relative_to(workspace)
    current = workspace
    for part in relative.parts:
        current = current / part
        if current.is_symlink() or current.is_junction():
            raise CheckpointError(_ERROR)


def workspace_manifest(workspace: Path) -> list[dict[str, Any]]:
    """Hash ordinary files, rejecting links instead of following them."""
    try:
        root = workspace.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("workspace must be directory")
        entries = []

        def scan_error(error: OSError) -> None:
            raise error

        for directory, dirs, files in os.walk(root, followlinks=False, onerror=scan_error):
            parent = Path(directory)
            if parent == root:
                dirs[:] = [name for name in dirs if name.casefold() not in _CONTROL]
                files = [name for name in files if name.casefold() not in _CONTROL]
            for name in [*dirs, *files]:
                path = parent / name
                if (
                    path.is_symlink()
                    or path.is_junction()
                    or not path.resolve().is_relative_to(root)
                ):
                    raise ValueError("invalid workspace entry")
            for name in files:
                path = parent / name
                digest = hashlib.sha256()
                size = 0
                with path.open("rb") as stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
                        size += len(block)
                entries.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "size": size,
                        "sha256": digest.hexdigest(),
                    }
                )
        return sorted(entries, key=lambda entry: entry["path"])
    except (OSError, ValueError):
        raise CheckpointError(_ERROR) from None


def _git(
    workspace: Path, root: Path, *args: str, output: bool = False, absent_ok: bool = False
) -> str:
    repo = root / "repo.git"
    command = [
        "git",
        "--git-dir",
        str(repo),
        "--work-tree",
        str(workspace),
        "-c",
        "user.name=Xiliumini Checkpoint",
        "-c",
        "user.email=checkpoint@xiliumini.local",
        "-c",
        "core.autocrlf=false",
        "-c",
        "core.hooksPath=" + str(repo / "disabled-hooks"),
        *args,
    ]
    if args and args[0] == "init":
        del command[3:5]  # Git rejects a work tree while initializing a bare repository.
    try:
        result = subprocess.run(
            command,
            shell=False,
            timeout=30,
            check=True,
            stdout=subprocess.PIPE if output else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip() if output else ""
    except subprocess.CalledProcessError as error:
        if absent_ok and error.returncode == 1:
            return "<absent>"
        raise CheckpointError(_ERROR) from None
    except (OSError, subprocess.SubprocessError):
        raise CheckpointError(_ERROR) from None


def snapshot_workspace_git(workspace: Path, root: Path, *, message: str) -> str:
    """Commit all ordinary files into a private bare repository."""
    try:
        _safe_directory(root, workspace)
        root.mkdir(parents=True, exist_ok=True)
        _safe_directory(root / "repo.git", workspace)
        if not (root / "repo.git").exists():
            _git(workspace, root, "init", "--bare")
            _git(workspace, root, "symbolic-ref", "HEAD", "refs/heads/checkpoint")
        if _git(workspace, root, "symbolic-ref", "HEAD", output=True) != "refs/heads/checkpoint":
            raise CheckpointError(_ERROR)
        exists = _git(
            workspace,
            root,
            "show-ref",
            "--verify",
            "--quiet",
            "refs/heads/checkpoint",
            absent_ok=True,
        )
        parent = (
            None
            if exists == "<absent>"
            else _git(
                workspace,
                root,
                "rev-parse",
                "--verify",
                "refs/heads/checkpoint^{commit}",
                output=True,
            )
        )
        if not parent and (root / "checkpoint.json").exists():
            raise CheckpointError(_ERROR)
        manifest = workspace_manifest(workspace)
        _git(workspace, root, "read-tree", "--empty")
        for entry in manifest:
            blob = _git(
                workspace,
                root,
                "hash-object",
                "-w",
                "--no-filters",
                "--",
                str(workspace / entry["path"]),
                output=True,
            )
            _git(
                workspace,
                root,
                "update-index",
                "--add",
                "--cacheinfo",
                "100644",
                blob,
                entry["path"],
            )
        tree = _git(workspace, root, "write-tree", output=True)
        parent_args = ["-p", parent] if parent else []
        commit = _git(
            workspace, root, "commit-tree", tree, *parent_args, "-m", message, output=True
        )
        _git(workspace, root, "update-ref", "HEAD", commit)
        return commit
    except (OSError, ValueError):
        raise CheckpointError(_ERROR) from None


@contextmanager
def _staging_directory(root: Path, workspace: Path) -> Iterator[tuple[Path, dict[str, bool]]]:
    recovery = root / "recovery"
    _safe_directory(recovery, workspace)
    recovery.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="save-", dir=recovery))
    policy = {"preserve": False}
    try:
        yield stage, policy
    finally:
        if not policy["preserve"]:
            shutil.rmtree(stage)


def resume_command(workspace: Path) -> str:
    path = str(workspace.resolve())
    quoted = "'" + path.replace("'", "''") + "'" if os.name == "nt" else shlex.quote(path)
    return "xiliumini --resume " + quoted


def build_recovery_markdown(payload: Mapping[str, Any]) -> str:
    files = "\n".join("- " + entry["path"] for entry in payload["workspace_manifest"])
    return (
        "# Checkpoint recovery\n\n"
        f"Task: {payload['task']}\n\nStatus: {payload['status']}\n\n"
        f"Git commit: `{payload['git_commit']}`\n\nFiles:\n{files or '(empty)'}\n\n"
        f"Resume command:\n\n```powershell\n{payload['resume_command']}\n```\n"
    )


def _next_node(state: Mapping[str, Any], latest_node: str | None) -> str:
    if latest_node == "planner":
        return "verifier"
    if latest_node == "verifier":
        return (
            "planner"
            if state.get("graph_state") == "failed"
            and state.get("attempt", 0) < state.get("max_attempts", 3)
            else "final"
        )
    if latest_node == "final":
        return "final"
    return state.get("resume_node", "planner")


def _relative_file(name: str) -> Path:
    path = Path(name)
    if (
        not name
        or path.is_absolute()
        or PureWindowsPath(name).drive
        or "\\" in name
        or any(part in {"", ".", ".."} for part in name.split("/"))
        or name.split("/")[0].casefold() in _CONTROL
        or any(
            ":" in part or part.endswith((".", " ")) or PureWindowsPath(part).is_reserved()
            for part in name.split("/")
        )
    ):
        raise ValueError("invalid file path")
    return path


def _read_tree(workspace: Path, root: Path, commit: str) -> dict[str, bytes]:
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("invalid commit")
    _safe_directory(root / "repo.git", workspace)
    if _git(workspace, root, "symbolic-ref", "HEAD", output=True) != "refs/heads/checkpoint":
        raise ValueError("foreign repository")
    _git(workspace, root, "merge-base", "--is-ancestor", commit, "refs/heads/checkpoint")
    command = ["git", "--git-dir", str(root / "repo.git"), "--work-tree", str(workspace)]

    def read(*args: str) -> bytes:
        try:
            return subprocess.run(
                command + list(args),
                shell=False,
                timeout=30,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            raise CheckpointError(_ERROR) from None

    result = {}
    for line in read("ls-tree", "-r", "-z", commit).split(b"\0"):
        if not line:
            continue
        info, filename = line.split(b"\t", 1)
        mode, kind, blob = info.split(b" ")
        name = filename.decode("utf-8")
        _relative_file(name)
        if mode != b"100644" or kind != b"blob":
            raise ValueError("unsupported tree entry")
        if name in result:
            raise ValueError("duplicate tree entry")
        result[name] = read("cat-file", "blob", blob.decode("ascii"))
    _validate_target_tree(result)
    return result


def _validate_target_tree(tree: Mapping[str, bytes]) -> None:
    """Reject collisions across all Windows-normalized path segments."""
    seen: dict[str, tuple[str, str]] = {}
    for name in tree:
        _relative_file(name)
        parts = name.split("/")
        for index in range(1, len(parts) + 1):
            spelling = "/".join(parts[:index])
            normalized = spelling.casefold()
            kind = "file" if index == len(parts) else "directory"
            previous = seen.get(normalized)
            if previous is not None and (previous != (spelling, kind) or kind == "file"):
                raise ValueError("ambiguous Windows tree path")
            seen[normalized] = (spelling, kind)


def _workspace_directories(workspace: Path) -> set[str]:
    """Retain ordinary directory topology; Git only records file blobs."""
    directories: set[str] = set()

    def scan_error(error: OSError) -> None:
        raise error

    for directory, dirs, _ in os.walk(workspace, followlinks=False, onerror=scan_error):
        parent = Path(directory)
        if parent == workspace:
            dirs[:] = [name for name in dirs if name.casefold() not in _CONTROL]
        for name in dirs:
            path = parent / name
            _safe_directory(path, workspace)
            directories.add(path.relative_to(workspace).as_posix())
    return directories


def _restore_directory_topology(workspace: Path, original: set[str]) -> None:
    for name in sorted(original, key=lambda value: (value.count("/"), value)):
        path = workspace / name
        _safe_directory(path, workspace)
        path.mkdir(parents=True, exist_ok=True)
    added = _workspace_directories(workspace) - original
    for name in sorted(added, key=lambda value: (value.count("/"), value), reverse=True):
        path = workspace / name
        _safe_directory(path, workspace)
        path.rmdir()


def _apply_tree(workspace: Path, tree: Mapping[str, bytes]) -> None:
    # No checkout command: Git attributes and smudge filters never execute.
    for entry in workspace_manifest(workspace):
        if entry["path"] in tree:
            continue
        path = workspace / entry["path"]
        _safe_directory(path, workspace)
        path.unlink()
    for name, content in tree.items():
        target = workspace / _relative_file(name)
        _safe_directory(target, workspace)
        if target.is_dir():
            # Only a directory occupying a target file path needs removal.
            # All ordinary files within it were absent from the validated target.
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        _safe_directory(target, workspace)
        with target.open("wb") as stream:
            stream.write(content)


def restore_workspace_git(workspace: Path, root: Path, commit: str) -> None:
    """Restore raw snapshot files exactly, retaining a rollback commit."""
    try:
        if root != workspace / ".xiliumini" / "checkpoints":
            raise ValueError("invalid checkpoint root")
        target = _read_tree(workspace, root, commit)
        _validate_target_tree(target)
        workspace_manifest(workspace)
        original_directories = _workspace_directories(workspace)
        previous = snapshot_workspace_git(workspace, root, message="pre-restore")
        rollback = _read_tree(workspace, root, previous)
        try:
            _apply_tree(workspace, target)
        except (OSError, ValueError, CheckpointError):
            with suppress(OSError, ValueError, CheckpointError):
                _apply_tree(workspace, rollback)
                _restore_directory_topology(workspace, original_directories)
            raise
    except (OSError, ValueError, TypeError, subprocess.SubprocessError, CheckpointError):
        raise CheckpointError(_ERROR) from None


def _validate_state_type(value: Any, annotation: Any) -> bool:
    """Check trusted static GraphState types, without loading saved classes."""
    if annotation is Any:
        return True
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Literal:
        return any(type(value) is type(option) and value == option for option in args)
    if origin in (Union, UnionType):
        return any(_validate_state_type(value, option) for option in args)
    if origin is list:
        return isinstance(value, list) and all(
            _validate_state_type(item, args[0]) for item in value
        )
    if origin is dict:
        return isinstance(value, dict) and all(
            _validate_state_type(key, args[0]) and _validate_state_type(item, args[1])
            for key, item in value.items()
        )
    if is_typeddict(annotation):
        fields = get_type_hints(annotation)
        return (
            isinstance(value, dict)
            and not (set(value) - set(fields))
            and not (set(annotation.__required_keys__) - set(value))
            and all(
                (key in {"before_tokens", "after_tokens"} and item == "[REDACTED]")
                or _validate_state_type(item, fields[key])
                for key, item in value.items()
            )
        )
    if annotation in (bool, int, str, type(None)):
        return type(value) is annotation
    return isinstance(value, annotation)


def _validate_payload(payload: Any, workspace: Path, root: Path) -> dict[str, Any]:
    fields = {
        "schema_version",
        "mode",
        "task",
        "status",
        "latest_node",
        "next_node",
        "attempt",
        "saved_at",
        "workspace_manifest",
        "git_commit",
        "state_summary",
        "trace_id",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("invalid checkpoint schema")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported checkpoint schema")
    if payload["mode"] not in ("light", "strict") or payload["status"] not in (
        "running",
        "completed",
        "failed",
        "interrupted",
    ):
        raise ValueError("invalid checkpoint mode/status")
    if (
        not isinstance(payload["task"], str)
        or not isinstance(payload["saved_at"], str)
        or type(payload["attempt"]) is not int
        or payload["attempt"] < 0
    ):
        raise ValueError("invalid checkpoint fields")
    datetime.fromisoformat(payload["saved_at"])
    if payload["latest_node"] not in (None, "planner", "verifier", "final"):
        raise ValueError("invalid checkpoint node")
    if payload["trace_id"] is not None and (
        not isinstance(payload["trace_id"], str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", payload["trace_id"])
    ):
        raise ValueError("invalid trace identifier")
    encoded = payload["state_summary"]
    if payload["mode"] == "strict":
        _safe_directory(root / "state.json", workspace)
        strict = json.loads((root / "state.json").read_text(encoding="utf-8"))
        if strict != encoded:
            raise ValueError("inconsistent strict state")
    state = restore_persisted_value(encoded, workspace)
    if not isinstance(state, dict) or set(state) - (
        set(GraphState.__annotations__) | {"messages", "resume_node"}
    ):
        raise ValueError("unknown state fields")
    if set(GraphState.__required_keys__) - set(state):
        raise ValueError("missing required graph state")
    lists = {
        "todos",
        "research_notes",
        "agent_results",
        "tool_events",
        "acceptance_criteria",
        "agent_handoffs",
        "compression_events",
        "messages",
    }
    texts = {
        "task",
        "result",
        "verification",
        "final_answer",
        "session_id",
        "plan_summary",
        "code_agent_summary",
        "verifier_summary",
        "last_error",
        "context_summary",
    }
    for key, value in state.items():
        if (
            (key in lists and not isinstance(value, list))
            or (key in texts and not isinstance(value, str))
            or (key == "memory" and not isinstance(value, dict))
            or (key == "supervisor_ok" and type(value) is not bool)
        ):
            raise ValueError("invalid state field type")
    annotations = get_type_hints(GraphState)
    for key, value in state.items():
        if key in annotations and not _validate_state_type(value, annotations[key]):
            raise ValueError("invalid nested state type")
    if "messages" in state and any(
        not isinstance(message, BaseMessage) for message in state["messages"]
    ):
        raise ValueError("invalid state messages")
    for key in ("attempt", "max_attempts"):
        if key in state and (
            type(state[key]) is not int or state[key] < (1 if key == "max_attempts" else 0)
        ):
            raise ValueError("invalid attempt")
    if (
        state.get("attempt", 0) != payload["attempt"]
        or state.get("task", payload["task"]) != payload["task"]
    ):
        raise ValueError("inconsistent state metadata")
    if "workspace" in state and state["workspace"] != workspace:
        raise ValueError("foreign workspace")
    if "graph_state" in state and state["graph_state"] not in (
        "planning",
        "verifying",
        "passed",
        "failed",
    ):
        raise ValueError("invalid graph status")
    for key in ("resume_node", "current_node"):
        if key in state and state[key] not in ("planner", "verifier", "final"):
            raise ValueError("invalid state node")
    if payload["next_node"] not in ("planner", "verifier", "final") or payload[
        "next_node"
    ] != _next_node(state, payload["latest_node"]):
        raise ValueError("inconsistent checkpoint route")
    tree = _read_tree(workspace, root, payload["git_commit"])
    manifest = payload["workspace_manifest"]
    if not isinstance(manifest, list):
        raise ValueError("invalid manifest")
    names = []
    for entry in manifest:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            raise ValueError("invalid manifest entry")
        name = entry["path"]
        if not isinstance(name, str):
            raise ValueError("invalid manifest path")
        _relative_file(name)
        if (
            name not in tree
            or type(entry["size"]) is not int
            or entry["size"] != len(tree[name])
            or entry["sha256"] != hashlib.sha256(tree[name]).hexdigest()
        ):
            raise ValueError("inconsistent manifest")
        names.append(name)
    if names != sorted(tree) or len({name.casefold() for name in names}) != len(names):
        raise ValueError("inconsistent manifest tree")
    return state


class CheckpointManager:
    def __init__(self, runtime: Any, task: str = ""):
        self.workspace = Path(runtime.workspace).absolute()
        self.mode = normalize_checkpoint_mode(runtime.checkpoint_mode)
        self.task = task
        self.trace_id = getattr(runtime, "trace_id", None)
        self.root = self.workspace / ".xiliumini" / "checkpoints"

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @classmethod
    def load_resume_inputs(
        cls, runtime: Any, task: str | None = None, max_attempts: int = 3
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            manager = cls(runtime)
            if (
                not manager.enabled
                or type(max_attempts) is not int
                or max_attempts < 1
                or (task is not None and not isinstance(task, str))
            ):
                raise ValueError("invalid resume configuration")
            workspace = manager.workspace.resolve(strict=True)
            lexical_allowed = Path(runtime.data_dir).absolute() / "workspaces"
            _safe_directory(lexical_allowed, Path(runtime.data_dir).absolute())
            allowed = lexical_allowed.resolve(strict=True)
            if (
                not workspace.is_dir()
                or workspace == allowed
                or not workspace.is_relative_to(allowed)
            ):
                raise ValueError("invalid workspace")
            _safe_directory(manager.workspace, allowed)
            _safe_directory(manager.root / "checkpoint.json", workspace)
            payload = json.loads((manager.root / "checkpoint.json").read_text(encoding="utf-8"))
            inputs = _validate_payload(payload, workspace, manager.root)
            inputs.update(
                runtime=getattr(runtime, "owner", runtime),
                workspace=workspace,
                max_attempts=max_attempts,
                resume_node=payload["next_node"],
            )
            inputs["task"] = task if task is not None else payload["task"]
            restore_workspace_git(workspace, manager.root, payload["git_commit"])
            return inputs, {
                "type": "resume",
                "git_commit": payload["git_commit"],
                "resume_node": payload["next_node"],
                "trace_id": payload["trace_id"],
            }
        except (OSError, ValueError, TypeError, KeyError, AttributeError, CheckpointError):
            raise CheckpointError(_ERROR) from None

    def save(
        self,
        state: Mapping[str, Any],
        *,
        status: str = "running",
        latest_node: str | None = None,
        event: Any = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        try:
            _safe_directory(self.root, self.workspace)
            self.root.mkdir(parents=True, exist_ok=True)
            for filename in ("state.json", "events.jsonl", "checkpoint.json", "RECOVERY.md"):
                _safe_directory(self.root / filename, self.workspace)
            safe_state = sanitize_for_persistence(
                {key: value for key, value in state.items() if key != "runtime"}, self.workspace
            )
            commit = snapshot_workspace_git(
                self.workspace, self.root, message="checkpoint " + status
            )
            # The working tree can change while graph custom events are consumed.
            # Publish hashes of immutable committed blobs, never a separate scan.
            tree = _read_tree(self.workspace, self.root, commit)
            manifest = [
                {"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                for name, content in sorted(tree.items())
            ]
            payload = {
                "schema_version": 1,
                "mode": self.mode,
                "task": sanitize_for_persistence(state.get("task", self.task), self.workspace),
                "status": status,
                "latest_node": latest_node,
                "next_node": _next_node(state, latest_node),
                "attempt": state.get("attempt", 0),
                "saved_at": datetime.now(UTC).isoformat(),
                "workspace_manifest": manifest,
                "git_commit": commit,
                "state_summary": safe_state,
                "trace_id": sanitize_for_persistence(self.trace_id, self.workspace),
            }
            with _staging_directory(self.root, self.workspace) as (stage, policy):
                names = ["RECOVERY.md", "checkpoint.json"]
                if self.mode == "strict":
                    names = ["state.json", "events.jsonl", *names]
                    write_json_atomic(stage / "state.json", safe_state)
                    if (self.root / "events.jsonl").exists():
                        shutil.copyfile(self.root / "events.jsonl", stage / "events.jsonl")
                    else:
                        atomic_write_utf8(stage / "events.jsonl", "")
                    if event is not None:
                        append_jsonl(
                            stage / "events.jsonl", sanitize_for_persistence(event, self.workspace)
                        )
                atomic_write_utf8(
                    stage / "RECOVERY.md",
                    build_recovery_markdown(
                        {**payload, "resume_command": resume_command(self.workspace)}
                    ),
                )
                write_json_atomic(stage / "checkpoint.json", payload)
                for name in names:
                    if (self.root / name).exists():
                        shutil.copyfile(self.root / name, stage / (name + ".backup"))
                published = []
                try:
                    policy["preserve"] = True
                    for name in names:
                        os.replace(stage / name, self.root / name)
                        published.append(name)
                except OSError:
                    rollback_complete = True
                    for name in reversed(published):
                        backup = stage / (name + ".backup")
                        try:
                            if backup.exists():
                                os.replace(backup, self.root / name)
                            else:
                                (self.root / name).unlink()
                        except OSError:
                            rollback_complete = False
                    policy["preserve"] = not rollback_complete
                    raise
                policy["preserve"] = False
            return {
                "type": "checkpoint_saved",
                "status": status,
                "latest_node": latest_node,
                "git_commit": commit,
            }
        except CheckpointError:
            raise
        except (OSError, ValueError, TypeError):
            raise CheckpointError(_ERROR) from None
