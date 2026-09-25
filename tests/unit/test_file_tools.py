from pathlib import Path

import pytest

from xiliumini.tools.file_edit import FileEditTool
from xiliumini.tools.file_read import FileReadTool
from xiliumini.tools.file_write import FileWriteTool
from xiliumini.tools.grep import GrepTool


def test_file_write_then_read_round_trip(tmp_path: Path) -> None:
    writer = FileWriteTool(workspace=tmp_path)
    reader = FileReadTool(workspace=tmp_path)

    result = writer.invoke({"path": "notes/item.txt", "content": "alpha\nbeta\n"})
    output = reader.invoke({"path": "notes/item.txt", "offset": 2, "limit": 1})

    assert "notes/item.txt" in result
    assert "2: beta" in output
    assert "1: alpha" not in output


@pytest.mark.parametrize(
    "path",
    ["../escape.txt", r"C:\escape.txt", r"\\server\share\x"],
)
def test_file_tools_reject_paths_outside_workspace(tmp_path: Path, path: str) -> None:
    write_result = FileWriteTool(workspace=tmp_path).invoke({"path": path, "content": "x"})
    read_result = FileReadTool(workspace=tmp_path).invoke({"path": path})

    assert write_result.startswith("Error:")
    assert read_result.startswith("Error:")


def test_file_read_rejects_oversized_and_invalid_utf8_files(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_bytes(b"x" * 1_000_001)
    (tmp_path / "binary.txt").write_bytes(b"\xff\xfe")
    reader = FileReadTool(workspace=tmp_path)

    assert "limit" in reader.invoke({"path": "large.txt"})
    assert "UTF-8" in reader.invoke({"path": "binary.txt"})


def test_file_write_revalidates_parent_after_creation(tmp_path: Path) -> None:
    writer = FileWriteTool(workspace=tmp_path)

    result = writer.invoke({"path": "new/child.txt", "content": "safe"})

    assert not result.startswith("Error:")
    assert (tmp_path / "new" / "child.txt").read_text(encoding="utf-8") == "safe"


def test_file_write_limits_encoded_utf8_bytes(tmp_path: Path) -> None:
    target = tmp_path / "large.txt"

    result = FileWriteTool(workspace=tmp_path).invoke(
        {"path": "large.txt", "content": "😀" * 250_001}
    )

    assert result.startswith("Error:")
    assert "byte limit" in result
    assert not target.exists()


def test_read_and_write_reject_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-file.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "linked.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("host does not permit symlink creation")

    read_result = FileReadTool(workspace=tmp_path).invoke({"path": "linked.txt"})
    write_result = FileWriteTool(workspace=tmp_path).invoke(
        {"path": "linked.txt", "content": "changed"}
    )

    assert read_result.startswith("Error:")
    assert write_result.startswith("Error:")
    assert outside.read_text(encoding="utf-8") == "secret"


def test_file_edit_requires_unique_match_by_default(tmp_path: Path) -> None:
    target = tmp_path / "item.txt"
    target.write_text("old old", encoding="utf-8")
    editor = FileEditTool(workspace=tmp_path)

    ambiguous = editor.invoke({"path": "item.txt", "old_string": "old", "new_string": "new"})

    assert "2 matches" in ambiguous
    assert target.read_text(encoding="utf-8") == "old old"

    replaced = editor.invoke(
        {
            "path": "item.txt",
            "old_string": "old",
            "new_string": "new",
            "replace_all": True,
        }
    )

    assert "2 replacements" in replaced
    assert target.read_text(encoding="utf-8") == "new new"


def test_file_edit_reports_missing_text(tmp_path: Path) -> None:
    (tmp_path / "item.txt").write_text("alpha", encoding="utf-8")

    result = FileEditTool(workspace=tmp_path).invoke(
        {"path": "item.txt", "old_string": "beta", "new_string": "gamma"}
    )

    assert result.startswith("Error:")
    assert "not found" in result


def test_file_edit_rejects_result_larger_than_byte_limit(tmp_path: Path) -> None:
    target = tmp_path / "item.txt"
    target.write_text("x" * 1_000, encoding="utf-8")

    result = FileEditTool(workspace=tmp_path).invoke(
        {
            "path": "item.txt",
            "old_string": "x",
            "new_string": "y" * 2_000,
            "replace_all": True,
        }
    )

    assert result.startswith("Error:")
    assert "byte limit" in result
    assert target.read_text(encoding="utf-8") == "x" * 1_000


def test_grep_filters_by_glob_and_limits_results(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("needle\nneedle\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("needle\n", encoding="utf-8")

    result = GrepTool(workspace=tmp_path).invoke(
        {"pattern": "needle", "glob": "*.py", "max_results": 1}
    )

    assert "a.py:1:needle" in result
    assert "b.txt" not in result
    assert "truncated" in result


def test_grep_returns_readable_invalid_regex_error(tmp_path: Path) -> None:
    result = GrepTool(workspace=tmp_path).invoke({"pattern": "["})

    assert result.startswith("Error: invalid regular expression")


def test_grep_skips_oversized_and_invalid_utf8_files(tmp_path: Path) -> None:
    (tmp_path / "valid.txt").write_text("needle", encoding="utf-8")
    (tmp_path / "large.txt").write_bytes(b"needle" + b"x" * 1_000_001)
    (tmp_path / "binary.txt").write_bytes(b"needle\xff")

    result = GrepTool(workspace=tmp_path).invoke({"pattern": "needle"})

    assert "valid.txt:1:needle" in result
    assert "large.txt" not in result
    assert "binary.txt" not in result


def test_edit_and_grep_reject_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside-dir"
    outside.mkdir()
    (outside / "secret.txt").write_text("needle", encoding="utf-8")
    link = tmp_path / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit symlink creation")

    edit_result = FileEditTool(workspace=tmp_path).invoke(
        {"path": "linked/secret.txt", "old_string": "needle", "new_string": "changed"}
    )
    grep_result = GrepTool(workspace=tmp_path).invoke({"pattern": "needle", "path": "linked"})

    assert edit_result.startswith("Error:")
    assert grep_result.startswith("Error:")


@pytest.mark.parametrize(
    "glob",
    ["../*.py", r"C:\*.py", r"\\server\share\*"],
)
def test_grep_rejects_glob_escape_forms(tmp_path: Path, glob: str) -> None:
    result = GrepTool(workspace=tmp_path).invoke({"pattern": "x", "glob": glob})

    assert result.startswith("Error:")
