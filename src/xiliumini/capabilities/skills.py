from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path

import yaml

from xiliumini.capabilities.models import CapabilitySpec

MAX_SKILL_BYTES = 256_000
MAX_RESOURCE_BYTES = 1_000_000
_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


class SkillRegistry:
    """Index installed metadata and load skill contents only when requested."""

    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def _contained(base: Path, candidate: Path) -> Path:
        base = base.resolve(strict=False)
        if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
            raise ValueError("invalid skill path")
        current = base
        for part in candidate.parts:
            current = current / part
            if current.is_symlink() or current.is_junction():
                raise ValueError("skill path contains a link")
        resolved = current.resolve(strict=False)
        if not resolved.is_relative_to(base):
            raise ValueError("skill path escapes its root")
        return resolved

    @staticmethod
    def _read(path: Path, max_bytes: int) -> str:
        if path.stat().st_size > max_bytes:
            raise ValueError("skill file exceeds size limit")
        return path.read_text(encoding="utf-8")

    def validate(self, directory: Path) -> CapabilitySpec:
        if not directory.is_dir() or directory.is_symlink() or directory.is_junction():
            raise ValueError("skill directory is invalid")
        name = directory.name
        if not _NAME.fullmatch(name):
            raise ValueError("invalid skill name")
        skill_file = self._contained(directory, Path("SKILL.md"))
        text = self._read(skill_file, MAX_SKILL_BYTES)
        frontmatter = re.match(r"\A---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
        if frontmatter is None:
            raise ValueError("SKILL.md frontmatter is missing")
        metadata = yaml.safe_load(frontmatter.group(1))
        if not isinstance(metadata, dict):
            raise ValueError("invalid skill metadata")
        if metadata.get("name") != name:
            raise ValueError("skill name must match directory")
        description = metadata.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("skill description is required")
        return CapabilitySpec(
            id=f"skill:{name}",
            kind="skill",
            name=name,
            description=description.strip(),
            source_id="local",
        )

    def scan(self) -> list[CapabilitySpec]:
        if not self.root.exists():
            return []
        if self.root.is_symlink() or self.root.is_junction():
            raise ValueError("skill registry root is a link")
        return [self.validate(path) for path in sorted(self.root.iterdir()) if path.is_dir()]

    def load(self, name: str) -> str:
        directory = self._contained(self.root, Path(name))
        self.validate(directory)
        return self._read(self._contained(directory, Path("SKILL.md")), MAX_SKILL_BYTES)

    def read_resource(self, name: str, relative_path: str) -> str:
        directory = self._contained(self.root, Path(name))
        self.validate(directory)
        candidate = self._contained(directory, Path(relative_path))
        if candidate.name == "SKILL.md" or not candidate.is_file():
            raise ValueError("resource is not an explicit skill file")
        return self._read(candidate, MAX_RESOURCE_BYTES)

    def install(
        self, source: Path, *, workspace: Path | None = None, trusted: bool = False
    ) -> CapabilitySpec:
        if workspace is None and not trusted:
            raise ValueError("installation source must be trusted or inside a workspace")
        if workspace is not None:
            resolved = source.resolve(strict=False)
            if not resolved.is_relative_to(workspace.resolve()):
                raise ValueError("installation source escapes workspace")
            relative = resolved.relative_to(workspace.resolve())
            self._contained(workspace, relative)
        spec = self.validate(source)
        for entry in source.rglob("*"):
            if entry.is_symlink() or entry.is_junction():
                raise ValueError("skill installation cannot contain links")
            if entry.is_file() and entry.stat().st_size > MAX_RESOURCE_BYTES:
                raise ValueError("skill installation contains an oversized file")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or self.root.is_junction():
            raise ValueError("skill registry root is a link")
        destination = self._contained(self.root, Path(spec.name))
        if destination.exists():
            raise ValueError("skill already installed")
        temporary = Path(tempfile.mkdtemp(prefix=".skill-", dir=self.root))
        try:
            staged = temporary / spec.name
            shutil.copytree(source, staged)
            self.validate(staged)
            os.replace(staged, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return spec

    def remove(self, name: str) -> None:
        destination = self._contained(self.root, Path(name))
        self.validate(destination)
        shutil.rmtree(destination)
