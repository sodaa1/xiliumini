from __future__ import annotations

from pathlib import Path

import pytest

from xiliumini.capabilities.skills import SkillRegistry


def make_skill(root: Path, name: str, body: str = "Use the workflow.") -> Path:
    skill = root / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use when debugging Python.\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return skill


def test_skill_scan_reads_metadata_only(tmp_path):
    root = tmp_path / "skills"
    make_skill(root, "debugging", body="Secret body loaded on demand.")
    registry = SkillRegistry(root)
    specs = registry.scan()
    assert [(item.id, item.name) for item in specs] == [("skill:debugging", "debugging")]
    assert "Secret body" not in repr(specs)
    assert "Secret body" in registry.load("debugging")


def test_skill_resource_rejects_escape_and_symlink(tmp_path):
    root = tmp_path / "skills"
    skill = make_skill(root, "debugging")
    (skill / "references").mkdir()
    (skill / "references" / "guide.md").write_text("Guide", encoding="utf-8")
    registry = SkillRegistry(root)
    assert registry.read_resource("debugging", "references/guide.md") == "Guide"
    with pytest.raises(ValueError):
        registry.read_resource("debugging", "../outside.txt")
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    link = skill / "references" / "link.md"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError):
        registry.read_resource("debugging", "references/link.md")


def test_install_from_workspace_is_atomic_and_rejects_name_collision(tmp_path):
    workspace = tmp_path / "workspace"
    draft = make_skill(workspace / "draft-skills", "debugging")
    registry = SkillRegistry(tmp_path / "data" / "skills")
    result = registry.install(draft, workspace=workspace)
    assert result.name == "debugging"
    assert (registry.root / "debugging" / "SKILL.md").exists()
    with pytest.raises(ValueError):
        registry.install(draft, workspace=workspace)


def test_skill_validation_rejects_missing_frontmatter(tmp_path):
    root = tmp_path / "skills"
    skill = root / "bad"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# No metadata", encoding="utf-8")
    with pytest.raises(ValueError):
        SkillRegistry(root).validate(skill)
