import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from xiliumini.core.checkpoint import CheckpointManager, workspace_manifest
from xiliumini.errors import CheckpointError


def resume_fixture(tmp_path, mode="light"):
    from tests.agent_fakes import state as complete_state

    class CompleteStateCheckpoint(CheckpointManager):
        def save(self, state, **kwargs):
            return super().save({**complete_state(self.workspace), **state}, **kwargs)

    workspace = tmp_path / "workspaces" / "session"
    workspace.mkdir(parents=True)
    runtime = context(workspace, mode)
    runtime.data_dir = tmp_path
    return runtime, CompleteStateCheckpoint(runtime)


@pytest.mark.parametrize("mode", ["light", "strict"])
def test_restore_exact_files_and_rebuild_inputs(tmp_path, mode):
    from langchain_core.messages import HumanMessage

    runtime, manager = resume_fixture(tmp_path, mode)
    workspace = runtime.workspace
    (workspace / "nested").mkdir()
    original = b"first\r\nsecond\r\n\x00\xff"
    (workspace / "nested/a.bin").write_bytes(original)
    (workspace / ".git").mkdir()
    (workspace / ".git/keep").write_text("user git")
    state = {
        "task": "saved",
        "workspace": workspace,
        "attempt": 1,
        "max_attempts": 3,
        "messages": [HumanMessage(content="hello")],
        "graph_state": "verifying",
    }
    event = manager.save(state, latest_node="planner")
    assert event is not None
    (workspace / "nested/a.bin").unlink()
    (workspace / "nested").rmdir()
    (workspace / "nested").write_text("file replaces directory")
    (workspace / "new.txt").write_text("new")
    inputs, resumed = manager.load_resume_inputs(runtime, task="override", max_attempts=5)
    assert (workspace / "nested/a.bin").read_bytes() == original
    assert not (workspace / "new.txt").exists()
    assert (workspace / ".git/keep").read_text() == "user git"
    assert (manager.root / "checkpoint.json").exists()
    assert inputs["runtime"] is runtime
    assert inputs["workspace"] == workspace
    assert inputs["task"] == "override"
    assert inputs["max_attempts"] == 5
    assert inputs["attempt"] == 1
    assert inputs["resume_node"] == "verifier"
    assert inputs["messages"] == [HumanMessage(content="hello")]
    assert resumed["type"] == "resume"
    assert resumed["git_commit"] == event["git_commit"]


def test_restore_saved_task_and_final_route(tmp_path):
    runtime, manager = resume_fixture(tmp_path)
    manager.save(
        {"task": "saved", "attempt": 2, "max_attempts": 2, "graph_state": "failed"},
        latest_node="verifier",
    )
    inputs, _ = manager.load_resume_inputs(runtime)
    assert inputs["task"] == "saved"
    assert inputs["resume_node"] == "final"
    assert inputs["max_attempts"] == 3


def test_restore_preserves_empty_directories_created_before_and_after_checkpoint(tmp_path):
    runtime, manager = resume_fixture(tmp_path)
    (runtime.workspace / "before/nested").mkdir(parents=True)
    (runtime.workspace / "a.txt").write_text("saved")
    manager.save({"task": "saved"})
    (runtime.workspace / "after/nested").mkdir(parents=True)
    (runtime.workspace / "unrelated/leaf").mkdir(parents=True)
    (runtime.workspace / "unrelated/new.txt").write_text("remove this file")
    manager.load_resume_inputs(runtime)
    assert (runtime.workspace / "before/nested").is_dir()
    assert (runtime.workspace / "after/nested").is_dir()
    assert (runtime.workspace / "unrelated/leaf").is_dir()
    assert not (runtime.workspace / "unrelated/new.txt").exists()


def test_restore_rollback_preserves_empty_directory_topology_after_conflict(tmp_path, monkeypatch):
    runtime, manager = resume_fixture(tmp_path)
    (runtime.workspace / "target").write_text("saved file")
    manager.save({"task": "saved"})
    (runtime.workspace / "target").unlink()
    (runtime.workspace / "target/empty/deep").mkdir(parents=True)
    (runtime.workspace / "unrelated/empty").mkdir(parents=True)
    (runtime.workspace / "current.txt").write_text("current")
    real_open = Path.open
    failed = False

    def fail_once(path, mode="r", *args, **kwargs):
        nonlocal failed
        if path == runtime.workspace / "target" and mode == "wb" and not failed:
            failed = True
            raise OSError("failed writing target")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_once)
    with pytest.raises(CheckpointError):
        manager.load_resume_inputs(runtime)
    assert (runtime.workspace / "target/empty/deep").is_dir()
    assert (runtime.workspace / "unrelated/empty").is_dir()
    assert (runtime.workspace / "current.txt").read_text() == "current"


@pytest.mark.parametrize(
    "names",
    [
        ["A.txt", "a.txt"],
        ["Foo/a.txt", "foo/b.txt"],
        ["A", "a/b"],
        ["a", "A/b"],
        ["a/b", "A"],
        ["foo/b.txt", "Foo/a.txt"],
    ],
)
def test_restore_rejects_windows_path_tree_collisions_before_snapshot(tmp_path, monkeypatch, names):
    import xiliumini.core.checkpoint as module

    runtime, manager = resume_fixture(tmp_path)
    (runtime.workspace / "keep.txt").write_text("current")
    (runtime.workspace / "empty").mkdir()
    event = manager.save({"task": "saved"})
    assert event is not None
    head = manager.root / "repo.git/refs/heads/checkpoint"
    before = head.read_bytes()
    real_read = module._read_tree

    def malicious_tree(workspace, root, commit):
        if commit == event["git_commit"]:
            return {name: b"evil" for name in names}
        return real_read(workspace, root, commit)

    monkeypatch.setattr(module, "_read_tree", malicious_tree)
    with pytest.raises(CheckpointError):
        module.restore_workspace_git(runtime.workspace, manager.root, event["git_commit"])
    assert (runtime.workspace / "keep.txt").read_text() == "current"
    assert (runtime.workspace / "empty").is_dir()
    assert head.read_bytes() == before


def test_restore_directory_to_file_conflict_only_removes_target_directory(tmp_path):
    runtime, manager = resume_fixture(tmp_path)
    (runtime.workspace / "target").write_text("saved")
    manager.save({"task": "saved"})
    (runtime.workspace / "target").unlink()
    (runtime.workspace / "target/empty").mkdir(parents=True)
    (runtime.workspace / "other/empty").mkdir(parents=True)
    manager.load_resume_inputs(runtime)
    assert (runtime.workspace / "target").read_text() == "saved"
    assert (runtime.workspace / "other/empty").is_dir()


def test_restore_rollback_removes_new_directories_but_retains_original_empty_ones(
    tmp_path, monkeypatch
):
    runtime, manager = resume_fixture(tmp_path)
    (runtime.workspace / "new/deep").mkdir(parents=True)
    (runtime.workspace / "new/deep/saved.txt").write_text("saved")
    manager.save({"task": "saved"})
    (runtime.workspace / "new/deep/saved.txt").unlink()
    (runtime.workspace / "new/deep").rmdir()
    (runtime.workspace / "new").rmdir()
    (runtime.workspace / "original/empty").mkdir(parents=True)
    (runtime.workspace / "current.txt").write_text("current")
    real_open = Path.open
    failed = False

    def fail_once(path, mode="r", *args, **kwargs):
        nonlocal failed
        if path == runtime.workspace / "new/deep/saved.txt" and mode == "wb" and not failed:
            failed = True
            raise OSError("failed writing new nested file")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_once)
    with pytest.raises(CheckpointError):
        manager.load_resume_inputs(runtime)
    assert not (runtime.workspace / "new").exists()
    assert (runtime.workspace / "original/empty").is_dir()
    assert (runtime.workspace / "current.txt").read_text() == "current"


@pytest.mark.parametrize(
    "fault",
    [
        "schema",
        "extra",
        "mode",
        "status",
        "attempt",
        "node",
        "route",
        "task",
        "state",
        "tag",
        "relative",
        "absolute",
        "hash",
        "size",
        "missing_commit",
        "foreign_commit",
        "strict_mismatch",
        "corrupt_json",
        "runtime_off",
        "outside",
        "override",
    ],
)
def test_restore_rejects_invalid_checkpoint_before_mutation(tmp_path, fault):
    runtime, manager = resume_fixture(tmp_path, "strict")
    (runtime.workspace / "keep.txt").write_bytes(b"saved")
    manager.save(
        {"task": "saved", "attempt": 1, "max_attempts": 3, "graph_state": "verifying"},
        latest_node="planner",
    )
    path = manager.root / "checkpoint.json"
    payload = json.loads(path.read_text())
    mutations = {
        "schema": ("schema_version", 2),
        "extra": ("unknown", True),
        "mode": ("mode", "evil"),
        "status": ("status", "evil"),
        "attempt": ("attempt", True),
        "node": ("latest_node", "actor"),
        "route": ("next_node", "planner"),
        "task": ("task", []),
        "state": ("state_summary", {"unknown": "value"}),
        "tag": ("state_summary", {"task": {"__xiliumini_type__": "evil", "value": []}}),
        "missing_commit": ("git_commit", "0" * 40),
    }
    if fault in mutations:
        key, value = mutations[fault]
        payload[key] = value
    if fault in {"relative", "absolute", "hash", "size"}:
        key, value = {
            "relative": ("path", "../escape"),
            "absolute": ("path", str(tmp_path / "escape")),
            "hash": ("sha256", "0" * 64),
            "size": ("size", False),
        }[fault]
        payload["workspace_manifest"][0][key] = value
    if fault == "strict_mismatch":
        (manager.root / "state.json").write_text(json.dumps({"task": "other"}))
    if fault == "foreign_commit":
        command = ["git", "--git-dir", str(manager.root / "repo.git")]
        tree = subprocess.run(
            command + ["rev-parse", payload["git_commit"] + "^{tree}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        payload["git_commit"] = subprocess.run(
            command
            + [
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@test",
                "commit-tree",
                tree,
                "-m",
                "foreign orphan",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    path.write_text("{" if fault == "corrupt_json" else json.dumps(payload))
    if fault == "runtime_off":
        runtime.checkpoint_mode = "off"
    if fault == "outside":
        runtime.data_dir = tmp_path / "other"
        (runtime.data_dir / "workspaces").mkdir(parents=True)
    (runtime.workspace / "keep.txt").write_bytes(b"current")
    head_path = manager.root / "repo.git/refs/heads/checkpoint"
    before = head_path.read_bytes()
    with pytest.raises(CheckpointError) as caught:
        manager.load_resume_inputs(runtime, max_attempts=0 if fault == "override" else 3)
    assert "private" not in str(caught.value)
    assert str(tmp_path) not in str(caught.value)
    assert (runtime.workspace / "keep.txt").read_bytes() == b"current"
    assert head_path.read_bytes() == before


def test_restore_rollback_preserves_original_files_after_write_failure(tmp_path, monkeypatch):
    runtime, manager = resume_fixture(tmp_path)
    (runtime.workspace / "saved.txt").write_bytes(b"saved")
    manager.save({"task": "saved"})
    (runtime.workspace / "saved.txt").unlink()
    (runtime.workspace / "current.txt").write_bytes(b"current\x00\xff")
    real_open = Path.open
    failed = False

    def fail_once(path, mode="r", *args, **kwargs):
        nonlocal failed
        if path == runtime.workspace / "saved.txt" and mode == "wb" and not failed:
            failed = True
            raise OSError("private path")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_once)
    with pytest.raises(CheckpointError):
        manager.load_resume_inputs(runtime)
    assert (runtime.workspace / "current.txt").read_bytes() == b"current\x00\xff"
    assert not (runtime.workspace / "saved.txt").exists()


@pytest.mark.parametrize(
    "state",
    [
        {"messages": [{"__harness_type__": "evil", "value": []}]},
        {"workspace": {"__harness_type__": "path", "value": "../escape"}},
        {"tool_events": "bad"},
        {"supervisor_ok": 1},
        {"result": []},
        {"tool_events": [{"ok": 1}]},
        {"todos": [{"status": "evil"}]},
        {"todos": [{"status": "pending"}]},
        {"memory": {"working": {"attempts": {"current": "bad"}}}},
    ],
)
def test_restore_rejects_bad_state_values_before_snapshot(tmp_path, state):
    runtime, manager = resume_fixture(tmp_path)
    manager.save({"task": "saved"})
    path = manager.root / "checkpoint.json"
    payload = json.loads(path.read_text())
    payload["state_summary"].update(state)
    path.write_text(json.dumps(payload))
    head = manager.root / "repo.git/refs/heads/checkpoint"
    previous = head.read_bytes()
    with pytest.raises(CheckpointError):
        manager.load_resume_inputs(runtime)
    assert head.read_bytes() == previous


def test_restore_raw_bytes_never_executes_smudge_filter(tmp_path):
    runtime, manager = resume_fixture(tmp_path)
    original = b"a\r\nb\r\n\xff"
    (runtime.workspace / "a.txt").write_bytes(original)
    (runtime.workspace / ".gitattributes").write_text("*.txt text filter=evil\n")
    manager.save({"task": "saved"})
    marker = runtime.workspace / "FILTER_RAN"
    subprocess.run(
        [
            "git",
            "--git-dir",
            str(manager.root / "repo.git"),
            "config",
            "filter.evil.smudge",
            f'echo ran > "{marker}"',
        ],
        check=True,
    )
    (runtime.workspace / "a.txt").write_bytes(b"changed")
    manager.load_resume_inputs(runtime)
    assert (runtime.workspace / "a.txt").read_bytes() == original
    assert not marker.exists()


def test_restore_rejects_current_symlink_before_snapshot(tmp_path):
    runtime, manager = resume_fixture(tmp_path)
    manager.save({"task": "saved"})
    outside = tmp_path / "outside"
    outside.write_text("outside")
    try:
        (runtime.workspace / "link").symlink_to(outside)
    except OSError:
        pytest.skip("platform cannot create symlinks")
    head = manager.root / "repo.git/refs/heads/checkpoint"
    before = head.read_bytes()
    with pytest.raises(CheckpointError):
        manager.load_resume_inputs(runtime)
    assert outside.read_text() == "outside"
    assert head.read_bytes() == before


def test_restore_rejects_blob_read_timeout_without_leaking_or_mutating(tmp_path, monkeypatch):
    runtime, manager = resume_fixture(tmp_path)
    (runtime.workspace / "a.txt").write_text("saved")
    manager.save({"task": "saved"})
    (runtime.workspace / "a.txt").write_text("current")
    real_run = subprocess.run

    def timeout(command, **kwargs):
        if "cat-file" in command:
            raise subprocess.TimeoutExpired("private command", 30)
        return real_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(CheckpointError) as caught:
        manager.load_resume_inputs(runtime)
    assert "private" not in str(caught.value)
    assert (runtime.workspace / "a.txt").read_text() == "current"


def test_restore_accepts_sanitized_compression_event_counts(tmp_path):
    runtime, manager = resume_fixture(tmp_path)
    manager.save(
        {
            "task": "saved",
            "compression_events": [
                {
                    "timestamp": "now",
                    "before_tokens": 100,
                    "after_tokens": 50,
                    "compressed_messages": 2,
                    "attempt": 1,
                }
            ],
        }
    )
    inputs, _ = manager.load_resume_inputs(runtime)
    assert inputs["compression_events"][0]["before_tokens"] == "[REDACTED]"
    assert inputs["compression_events"][0]["compressed_messages"] == 2


def context(workspace, mode="light"):
    return SimpleNamespace(workspace=workspace, checkpoint_mode=mode, trace_id=None)


def test_checkpoint_off_creates_no_files_and_returns_none(tmp_path):
    manager = CheckpointManager(context(tmp_path, "off"))
    assert not manager.enabled
    assert manager.save({"runtime": object()}) is None
    assert list(tmp_path.iterdir()) == []


def test_manifest_is_sorted_hashed_and_excludes_control_directories(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "z.txt").write_bytes(b"hello")
    (tmp_path / "a.txt").write_bytes(b"a")
    for name in (".git", ".xiliumini"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "hidden").write_text("secret")
    assert workspace_manifest(tmp_path) == [
        {"path": "a.txt", "size": 1, "sha256": hashlib.sha256(b"a").hexdigest()},
        {"path": "nested/z.txt", "size": 5, "sha256": hashlib.sha256(b"hello").hexdigest()},
    ]


def test_light_save_writes_metadata_recovery_and_detached_git_snapshot(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    manager = CheckpointManager(context(tmp_path), "build a tool")
    event = manager.save({"task": "build a tool", "attempt": 1}, latest_node="planner")
    assert event is not None
    payload = json.loads((manager.root / "checkpoint.json").read_text())
    assert event["type"] == "checkpoint_saved"
    assert payload["schema_version"] == 1
    assert payload["latest_node"] == "planner"
    assert payload["next_node"] == "verifier"
    assert payload["workspace_manifest"] == workspace_manifest(tmp_path)
    assert not (tmp_path / ".git").exists()
    assert not (manager.root / "state.json").exists()
    assert not (manager.root / "events.jsonl").exists()
    result = subprocess.run(
        [
            "git",
            "--git-dir",
            str(manager.root / "repo.git"),
            "ls-tree",
            "-r",
            "--name-only",
            payload["git_commit"],
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == ["a.txt"]
    recovery = (manager.root / "RECOVERY.md").read_text()
    assert all(
        text in recovery
        for text in (
            "build a tool",
            "running",
            "a.txt",
            payload["git_commit"],
            "xiliumini --resume",
        )
    )


def test_strict_save_preserves_full_state_and_appends_events(tmp_path):
    manager = CheckpointManager(context(tmp_path, "strict"))
    state = {"task": "hello", "workspace": tmp_path, "attempt": 2, "api_key": "hidden"}
    manager.save(state, event={"type": "tool_call", "password": "hidden"})
    manager.save(state, event={"type": "tool_result", "ok": True})
    saved = json.loads((manager.root / "state.json").read_text())
    assert saved["attempt"] == 2
    assert saved["api_key"] == "[REDACTED]"
    lines = (manager.root / "events.jsonl").read_text().splitlines()
    assert [json.loads(line)["type"] for line in lines] == ["tool_call", "tool_result"]
    assert "hidden" not in "".join(lines)


def test_snapshot_tracks_deleted_and_ignored_files(tmp_path):
    manager = CheckpointManager(context(tmp_path))
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    (tmp_path / "ignored.txt").write_text("must snapshot")
    manager.save({})
    (tmp_path / "ignored.txt").unlink()
    event = manager.save({})
    assert event is not None
    result = subprocess.run(
        [
            "git",
            "--git-dir",
            str(manager.root / "repo.git"),
            "ls-tree",
            "-r",
            "--name-only",
            event["git_commit"],
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [".gitignore"]


def test_failed_artifact_write_does_not_publish_new_metadata(tmp_path, monkeypatch):
    import xiliumini.core.checkpoint as module

    manager = CheckpointManager(context(tmp_path))
    manager.save({"task": "first"})
    previous = (manager.root / "checkpoint.json").read_bytes()

    def fail(*args):
        raise OSError(str(tmp_path) + " private")

    monkeypatch.setattr(module, "atomic_write_utf8", fail)
    with pytest.raises(CheckpointError) as caught:
        manager.save({"task": "second"})
    assert str(tmp_path) not in str(caught.value)
    assert (manager.root / "checkpoint.json").read_bytes() == previous


def test_git_failure_is_controlled_and_does_not_publish_metadata(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired("secret command", 30)

    monkeypatch.setattr(subprocess, "run", fail)
    manager = CheckpointManager(context(tmp_path))
    with pytest.raises(CheckpointError, match="Checkpoint persistence failed"):
        manager.save({})
    assert not (manager.root / "checkpoint.json").exists()


def test_snapshot_preserves_raw_crlf_and_never_runs_clean_filter(tmp_path):
    manager = CheckpointManager(context(tmp_path))
    manager.save({})
    marker = tmp_path / "FILTER_RAN"
    subprocess.run(
        [
            "git",
            "--git-dir",
            str(manager.root / "repo.git"),
            "config",
            "filter.evil.clean",
            f'echo ran > "{marker}"',
        ],
        check=True,
    )
    (tmp_path / ".gitattributes").write_text("*.txt text filter=evil\n")
    original = b"first\r\nsecond\r\n"
    (tmp_path / "a.txt").write_bytes(original)
    event = manager.save({})
    assert event is not None
    (tmp_path / "a.txt").write_bytes(b"changed")
    (tmp_path / "a.txt").unlink()
    blob = subprocess.run(
        [
            "git",
            "--git-dir",
            str(manager.root / "repo.git"),
            "show",
            event["git_commit"] + ":a.txt",
        ],
        capture_output=True,
        check=True,
    ).stdout
    assert blob == original
    assert not marker.exists()


def test_scan_error_keeps_previous_checkpoint(tmp_path, monkeypatch):
    import xiliumini.core.checkpoint as module

    manager = CheckpointManager(context(tmp_path))
    manager.save({"task": "first"})
    previous = (manager.root / "checkpoint.json").read_bytes()
    real_walk = module.os.walk

    def broken_walk(*args, **kwargs):
        if "onerror" in kwargs:
            kwargs["onerror"](PermissionError("hidden path"))
        yield from real_walk(*args, **kwargs)

    monkeypatch.setattr(module.os, "walk", broken_walk)
    with pytest.raises(CheckpointError):
        manager.save({"task": "second"})
    assert (manager.root / "checkpoint.json").read_bytes() == previous


@pytest.mark.parametrize("failure", ["git", "publish"])
def test_failed_strict_save_preserves_all_previous_artifacts(tmp_path, monkeypatch, failure):
    import xiliumini.core.checkpoint as module

    manager = CheckpointManager(context(tmp_path, "strict"))
    manager.save({"task": "first"}, event={"type": "first"})
    names = ("checkpoint.json", "state.json", "events.jsonl", "RECOVERY.md")
    previous = {name: (manager.root / name).read_bytes() for name in names}
    if failure == "git":

        def fail_git(*args, **kwargs):
            raise CheckpointError("failed")

        monkeypatch.setattr(module, "snapshot_workspace_git", fail_git)
    else:
        real_replace = module.os.replace
        failed = False

        def fail_publish(source, target):
            nonlocal failed
            if str(target).endswith("checkpoint.json") and not failed:
                failed = True
                raise OSError("failed")
            return real_replace(source, target)

        monkeypatch.setattr(module.os, "replace", fail_publish)
    with pytest.raises(CheckpointError):
        manager.save({"task": "second"}, event={"type": "second"})
    assert {name: (manager.root / name).read_bytes() for name in names} == previous


def test_json_excludes_absolute_resume_path_but_guide_quotes_it(tmp_path):
    workspace = tmp_path / "space and 'quote"
    workspace.mkdir()
    manager = CheckpointManager(context(workspace))
    manager.save({})
    text = (manager.root / "checkpoint.json").read_text()
    assert "resume_command" not in json.loads(text)
    assert str(workspace) not in text
    from xiliumini.core.checkpoint import resume_command

    assert resume_command(workspace) in (manager.root / "RECOVERY.md").read_text()


def test_off_never_calls_persistence(tmp_path, monkeypatch):
    import xiliumini.core.checkpoint as module

    def forbidden(*args, **kwargs):
        pytest.fail("off performed persistence")

    for name in (
        "workspace_manifest",
        "snapshot_workspace_git",
        "write_json_atomic",
        "append_jsonl",
    ):
        monkeypatch.setattr(module, name, forbidden)
    assert CheckpointManager(context(tmp_path, "off")).save({}) is None


def test_manifest_rejects_links_when_supported(tmp_path):
    target = tmp_path / "target"
    target.write_text("data")
    try:
        (tmp_path / "link").symlink_to(target)
    except OSError:
        pytest.skip("platform cannot create symlinks")
    with pytest.raises(CheckpointError):
        workspace_manifest(tmp_path)


@pytest.mark.parametrize("fault", ["timeout", "failure", "corrupt_head", "corrupt_ref"])
def test_parent_lookup_fault_never_advances_checkpoint(tmp_path, monkeypatch, fault):
    manager = CheckpointManager(context(tmp_path))
    first = manager.save({"task": "first"})
    assert first is not None
    before = (manager.root / "checkpoint.json").read_bytes()
    repo = manager.root / "repo.git"
    if fault == "corrupt_head":
        (repo / "HEAD").write_text("corrupt\n")
    elif fault == "corrupt_ref":
        ref = subprocess.run(
            ["git", "--git-dir", str(repo), "symbolic-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        (repo / ref).write_text("corrupt\n")
    else:
        real_run = subprocess.run

        def broken_run(command, **kwargs):
            if "rev-parse" in command or "show-ref" in command:
                if fault == "timeout":
                    raise subprocess.TimeoutExpired("private", 30)
                raise subprocess.CalledProcessError(128, "private")
            if "commit-tree" in command or "update-ref" in command:
                pytest.fail("parent failure created or advanced commit")
            return real_run(command, **kwargs)

        monkeypatch.setattr(subprocess, "run", broken_run)
    with pytest.raises(CheckpointError):
        manager.save({"task": "second"})
    assert (manager.root / "checkpoint.json").read_bytes() == before
    assert first["git_commit"]


def test_partial_rollback_continues_and_retains_failed_file_backup(tmp_path, monkeypatch):
    import xiliumini.core.checkpoint as module

    manager = CheckpointManager(context(tmp_path, "strict"))
    manager.save({"task": "first"}, event={"type": "first"})
    names = ("checkpoint.json", "state.json", "events.jsonl", "RECOVERY.md")
    previous = {name: (manager.root / name).read_bytes() for name in names}
    real_replace = module.os.replace

    def broken_replace(source, target):
        source = str(source)
        if (str(target) == str(manager.root / "checkpoint.json")) or source.endswith(
            "RECOVERY.md.backup"
        ):
            raise OSError("private")
        return real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", broken_replace)
    with pytest.raises(CheckpointError):
        manager.save({"task": "second"}, event={"type": "second"})
    for name in ("checkpoint.json", "state.json", "events.jsonl"):
        assert (manager.root / name).read_bytes() == previous[name]
    backups = list(manager.root.glob("recovery/save-*/RECOVERY.md.backup"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == previous["RECOVERY.md"]
