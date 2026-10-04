from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
from tempfile import mkstemp
from uuid import UUID

from xiliumini.errors import WorkspaceError

MAX_FILE_BYTES = 1_000_000
RUNTIME_MANAGED_WORKSPACE_FILES = frozenset({"todo.md", "notepad.md", "history_summary.md"})


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def create_explicit_workspace(data_dir: Path, workspace: Path) -> Path:
    """Create/reuse a named workspace contained by the configured data directory."""

    try:
        if ".." in workspace.parts:
            raise WorkspaceError("workspace must stay inside the data directory")
        data_root = data_dir.expanduser().resolve()
        workspace_root = data_root / "workspaces"
        if workspace_root.exists() and _is_link(workspace_root):
            raise WorkspaceError("workspace must stay inside the data directory")
        workspace_root.mkdir(parents=True, exist_ok=True)
        workspace_root = workspace_root.resolve(strict=True)
        if not workspace_root.is_relative_to(data_root):
            raise WorkspaceError("workspace must stay inside the data directory")

        requested = workspace.expanduser()
        lexical_candidate = requested.absolute()
        try:
            lexical_relative = lexical_candidate.relative_to(workspace_root)
        except ValueError:
            raise WorkspaceError("workspace must stay inside the data directory") from None
        current = workspace_root
        for part in lexical_relative.parts:
            current /= part
            if _is_link(current):
                raise WorkspaceError("workspace must stay inside the data directory")

        candidate = lexical_candidate.resolve(strict=False)
        if candidate == workspace_root or not candidate.is_relative_to(workspace_root):
            raise WorkspaceError("workspace must stay inside the data directory")

        candidate.mkdir(parents=True, exist_ok=True)
        resolved = candidate.resolve(strict=True)
        if resolved == workspace_root or not resolved.is_relative_to(workspace_root):
            raise WorkspaceError("workspace must stay inside the data directory")
        return resolved
    except WorkspaceError:
        raise
    except OSError:
        raise WorkspaceError("could not create explicit workspace") from None


def create_session_workspace(data_dir: Path, session_id: str) -> Path:
    """Create and return the persistent workspace for one UUID session."""

    try:
        canonical_id = str(UUID(session_id))
    except (ValueError, AttributeError, TypeError):
        raise WorkspaceError("session_id must be a valid UUID") from None
    if canonical_id != session_id.lower():
        raise WorkspaceError("session_id must be a canonical UUID")

    try:
        data_root = data_dir.expanduser().resolve()
        workspace_root = data_root / "workspaces"
        workspace_root.mkdir(parents=True, exist_ok=True)
        workspace_root = workspace_root.resolve(strict=True)
        try:
            workspace_root.relative_to(data_root)
        except ValueError:
            raise WorkspaceError("workspace must stay inside the data directory") from None

        session_root = workspace_root / canonical_id
        session_root.mkdir(exist_ok=True)
        session_root = session_root.resolve(strict=True)
        try:
            session_root.relative_to(workspace_root)
        except ValueError:
            raise WorkspaceError("workspace must stay inside the data directory") from None
        return session_root
    except OSError:
        raise WorkspaceError("could not create session workspace") from None


def resolve_workspace_path(
    workspace: Path,
    relative_path: str,
    *,
    allow_root: bool = False,
) -> Path:
    """Resolve a relative path and reject every route outside ``workspace``."""

    if not isinstance(relative_path, str) or not relative_path.strip():
        raise WorkspaceError("path must be a non-empty workspace-relative path")

    windows_path = PureWindowsPath(relative_path)
    is_unc = relative_path.startswith(("//", "\\\\"))
    if Path(relative_path).is_absolute() or windows_path.drive or is_unc:
        raise WorkspaceError("path must stay inside the workspace")

    root = workspace.resolve(strict=True)
    candidate = (root / relative_path).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError:
        raise WorkspaceError("path must stay inside the workspace") from None

    if candidate == root and not allow_root:
        raise WorkspaceError("path must name an item inside the workspace")
    return candidate


def ensure_workspace_file_mutable(workspace: Path, target: Path) -> None:
    """Reject generic writes to top-level files owned by Runtime stores."""

    root = workspace.resolve(strict=True)
    if target.parent == root and target.name.casefold() in RUNTIME_MANAGED_WORKSPACE_FILES:
        raise WorkspaceError("path is managed by Runtime; use its dedicated tool")


def read_utf8_text(path: Path, max_bytes: int) -> str:
    """Read a bounded UTF-8 text file."""

    if not path.is_file():
        raise WorkspaceError("path is not a readable file")
    if path.stat().st_size > max_bytes:
        raise WorkspaceError(f"file exceeds the {max_bytes}-byte limit")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise WorkspaceError("file is not valid UTF-8 text") from None


def utf8_size(content: str) -> int:
    """Return the encoded size used by workspace text limits."""

    return len(content.encode("utf-8"))


def atomic_write_utf8(path: Path, content: str) -> None:
    """Atomically replace a UTF-8 text file in its destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = mkstemp(prefix=".xiliumini-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
