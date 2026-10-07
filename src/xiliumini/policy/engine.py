from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Effect = Literal["allow", "ask", "deny"]

DEFAULT_PROFILES: dict[str, dict[str, Effect]] = {
    "interactive": {
        "compute.safe": "allow",
        "time.read": "allow",
        "filesystem.read.workspace": "allow",
        "filesystem.write.workspace": "allow",
        "network.read": "allow",
        "process.execute.safe": "allow",
        "process.execute": "ask",
        "agent.delegate": "allow",
        "package.install": "ask",
        "remote.write": "ask",
        "automation.manage": "allow",
    },
    "background": {
        "compute.safe": "allow",
        "time.read": "allow",
        "filesystem.read.workspace": "allow",
        "filesystem.write.workspace": "allow",
        "network.read": "allow",
        "process.execute.safe": "allow",
        "process.execute": "deny",
        "agent.delegate": "allow",
        "package.install": "deny",
        "remote.write": "deny",
        "automation.manage": "deny",
    },
}


@dataclass(frozen=True)
class Decision:
    effect: Effect
    reason: str


class PolicyEngine:
    def __init__(self, config_path: Path, *, trusted_mcp_servers: set[str] | None = None):
        self.config_path = config_path
        self.trusted_mcp_servers = trusted_mcp_servers or set()
        if config_path.exists():
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not isinstance(raw.get("profiles"), dict):
                raise ValueError("invalid policy configuration")
            if raw.get("default", "deny") != "deny":
                raise ValueError("policy default must be deny")
            self.profiles = raw["profiles"]
        else:
            self.profiles = DEFAULT_PROFILES

    def evaluate(
        self,
        *,
        actor: str,
        source: str,
        capability_id: str,
        permission: str,
        arguments: dict[str, Any],
        profile: str,
    ) -> Decision:
        del actor, arguments
        if not capability_id or source not in {"builtin", "mcp", "hook", "automation"}:
            return Decision("deny", "unknown capability source")
        if source == "mcp":
            parts = capability_id.split(":", 2)
            if len(parts) != 3 or parts[0] != "mcp" or parts[1] not in self.trusted_mcp_servers:
                return Decision("deny", "MCP server is not trusted")
        permissions = self.profiles.get(profile, {})
        effect = permissions.get(permission, "deny")
        if effect not in {"allow", "ask", "deny"}:
            return Decision("deny", "invalid policy effect")
        if profile == "background" and effect == "ask":
            return Decision("deny", "background tasks cannot wait for approval")
        reason = f"{profile}: {permission} {effect}"
        return Decision(effect, reason)
