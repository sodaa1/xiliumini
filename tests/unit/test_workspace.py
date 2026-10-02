import os
import subprocess
from pathlib import Path
from uuid import UUID

import pytest

from xiliumini.errors import WorkspaceError
from xiliumini.tools.workspace import (
    create_explicit_workspace,
    create_session_workspace,
    resolve_workspace_path,
)

SESSION_ID = "11111111-1111-4111-8111-111111111111"


def test_create_session_workspace_uses_uuid_directory(tmp_path: Path) -> None:
    first = create_session_workspace(tmp_path, SESSION_ID)
    second = create_session_workspace(tmp_path, str(UUID(SESSION_ID)))

    assert first == second == (tmp_path / "workspaces" / SESSION_ID).resolve()
    assert first.is_dir()


@pytest.mark.parametrize("session_id", ["", "session-1", "../escape", "not-a-uuid"])
def test_create_session_workspace_rejects_invalid_session_id(
    tmp_path: Path, session_id: str
) -> None:
    with pytest.raises(WorkspaceError, match="valid UUID"):
        create_session_workspace(tmp_path, session_id)


@pytest.mark.parametrize(
    "path",
    ["../outside.txt", "/etc/passwd", r"C:\outside.txt", r"\\server\share\x"],
)
def test_resolve_workspace_path_rejects_escape_forms(tmp_path: Path, path: str) -> None:
    workspace = create_session_workspace(tmp_path, SESSION_ID)

    with pytest.raises(WorkspaceError, match="workspace"):
        resolve_workspace_path(workspace, path)


def test_resolve_workspace_path_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace = create_session_workspace(tmp_path, SESSION_ID)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = workspace / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit symlink creation")

    with pytest.raises(WorkspaceError, match="workspace"):
        resolve_workspace_path(workspace, "link/secret.txt")


def test_create_session_workspace_rejects_symlinked_workspace_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside-workspaces"
    outside.mkdir()
    workspace_link = tmp_path / "workspaces"
    try:
        workspace_link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit symlink creation")

    with pytest.raises(WorkspaceError, match="workspace"):
        create_session_workspace(tmp_path, SESSION_ID)

    assert not (outside / SESSION_ID).exists()


def test_create_explicit_workspace_creates_and_reuses_contained_directory(tmp_path: Path) -> None:
    requested = tmp_path / "workspaces" / "named-task"

    first = create_explicit_workspace(tmp_path, requested)
    (first / "keep.txt").write_text("keep", encoding="utf-8")
    second = create_explicit_workspace(tmp_path, requested)

    assert first == second == requested.resolve()
    assert (second / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_create_explicit_workspace_resolves_relative_path_from_current_directory(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = Path(".xiliumini")
    requested = Path(".xiliumini/workspaces/named-task")

    workspace = create_explicit_workspace(data_dir, requested)

    assert workspace == (tmp_path / requested).resolve()


@pytest.mark.parametrize("relative", [".", "../outside", "workspaces/../outside"])
def test_create_explicit_workspace_rejects_root_and_escape(tmp_path: Path, relative: str) -> None:
    requested = tmp_path / "workspaces" / relative

    with pytest.raises(WorkspaceError, match="workspace") as captured:
        create_explicit_workspace(tmp_path, requested)

    assert str(requested) not in str(captured.value)


def test_create_explicit_workspace_rejects_link_component(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces"
    workspace_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = workspace_root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit symlink creation")

    with pytest.raises(WorkspaceError, match="workspace"):
        create_explicit_workspace(tmp_path, link / "child")

    assert not (outside / "child").exists()


def test_create_explicit_workspace_rejects_link_that_resolves_inside_root(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces"
    target = workspace_root / "target"
    target.mkdir(parents=True)
    link = workspace_root / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            pytest.skip("host does not permit symlink creation")
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=False,
        )
        if created.returncode != 0:
            pytest.skip("host does not permit junction creation")

    with pytest.raises(WorkspaceError, match="workspace"):
        create_explicit_workspace(tmp_path, link / "child")

    assert not (target / "child").exists()
