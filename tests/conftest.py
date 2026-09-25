from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def isolated_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Run configuration tests away from the developer's real ``.env`` file."""

    monkeypatch.chdir(tmp_path)
    yield tmp_path
