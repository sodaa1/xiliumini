from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CapabilityKind = Literal["builtin", "skill", "mcp"]


@dataclass(frozen=True)
class CapabilityRef:
    id: str
    kind: CapabilityKind
    source_id: str


@dataclass(frozen=True)
class CapabilitySpec(CapabilityRef):
    name: str
    description: str
    enabled: bool = True
    schema_digest: str = ""
