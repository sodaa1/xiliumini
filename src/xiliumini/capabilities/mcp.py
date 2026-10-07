from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import Callable
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from jsonschema import ValidationError, validate

from xiliumini.capabilities.models import CapabilitySpec


@dataclass(frozen=True)
class ServerConfig:
    id: str
    transport: str
    target: str
    args: tuple[str, ...]
    permissions: tuple[str, ...]
    allowlist: frozenset[str]


class MCPManager:
    """Lazy MCP discovery and calls through one dedicated event-loop thread."""

    def __init__(
        self,
        mcp_root: Path,
        *,
        client_factory: Callable[..., Any] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.mcp_root = mcp_root
        self.timeout_seconds = timeout_seconds
        self._client_factory = client_factory
        self._servers = self._load_config()
        self._tools: dict[str, tuple[CapabilitySpec, dict[str, Any]]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _load_config(self) -> dict[str, ServerConfig]:
        if not self.mcp_root.exists():
            return {}
        if not self.mcp_root.is_dir():
            raise ValueError("invalid MCP configuration")
        servers: dict[str, ServerConfig] = {}
        for provider_dir in sorted(self.mcp_root.iterdir()):
            if not provider_dir.is_dir():
                continue
            config_path = provider_dir / "mcp.json"
            if not config_path.is_file():
                continue
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            server_id = provider_dir.name
            if not isinstance(raw, dict) or not isinstance(raw.get("servers"), dict):
                raise ValueError("invalid MCP configuration")
            configured = raw["servers"]
            if set(configured) != {server_id}:
                raise ValueError("invalid MCP configuration")
            item = configured[server_id]
            if not isinstance(item, dict):
                raise ValueError("invalid MCP server")
            if not item.get("enabled", False):
                continue
            if item.get("trust") != "explicit":
                raise ValueError("MCP server requires explicit trust")
            transport = item.get("transport")
            if transport == "http":
                target = item.get("url")
                parsed = urlparse(target) if isinstance(target, str) else None
                if parsed is None or parsed.scheme != "https" or not parsed.netloc:
                    raise ValueError("MCP HTTP URL must use HTTPS")
            elif transport == "stdio":
                target = item.get("command")
                if not isinstance(target, str) or not Path(target).is_absolute():
                    raise ValueError("MCP stdio command must be an absolute path")
            else:
                raise ValueError("unsupported MCP transport")
            assert isinstance(target, str)
            permissions = item.get("permissions", [])
            allowlist = item.get("tool_allowlist", [])
            args = item.get("args", [])
            if not all(
                isinstance(values, list) and all(isinstance(value, str) for value in values)
                for values in (permissions, allowlist, args)
            ):
                raise ValueError("invalid MCP server list")
            servers[server_id] = ServerConfig(
                server_id,
                transport,
                target,
                tuple(args),
                tuple(permissions),
                frozenset(allowlist),
            )
        return servers

    def servers(self) -> list[str]:
        return sorted(self._servers)

    def transport(self, server_id: str) -> str:
        return self._servers[server_id].transport

    def server_permissions(self, server_id: str) -> tuple[str, ...]:
        return self._servers[server_id].permissions

    def permission_for_tool(self, capability_id: str) -> str:
        pieces = capability_id.split(":", 2)
        if len(pieces) != 3 or pieces[0] != "mcp":
            raise ValueError("invalid MCP capability ID")
        server = self._servers.get(pieces[1])
        if server is None or pieces[2] not in server.allowlist:
            raise ValueError("MCP tool is not allowlisted")
        if len(server.permissions) != 1:
            raise ValueError("MCP tool requires exactly one local permission")
        return server.permissions[0]

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=loop.run_forever, name="xiliumini-mcp", daemon=True
                )
                thread.start()
                self._loop, self._thread = loop, thread
            return self._loop

    def _run(self, coroutine: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(coroutine, self._ensure_loop())
        try:
            return future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError:
            future.cancel()
            raise TimeoutError(
                "MCP request timed out; remote side effects may have occurred"
            ) from None
        except Exception as exc:
            raise RuntimeError(f"MCP request failed ({type(exc).__name__})") from None

    def _target(self, server: ServerConfig) -> Any:
        if server.transport == "http":
            return server.target
        from mcp import StdioServerParameters

        return StdioServerParameters(command=server.target, args=list(server.args))

    def _client(self, server: ServerConfig) -> Any:
        if self._client_factory is None:
            from mcp import Client

            return Client(self._target(server))
        return self._client_factory(self._target(server))

    async def _list(self, server: ServerConfig) -> Any:
        async with self._client(server) as client:
            return await client.list_tools()

    def tools(self, server_id: str) -> list[CapabilitySpec]:
        server = self._servers.get(server_id)
        if server is None:
            raise ValueError("MCP server is not configured")
        listing = self._run(self._list(server))
        specs: list[CapabilitySpec] = []
        for tool in listing.tools:
            if tool.name not in server.allowlist:
                continue
            schema = tool.input_schema
            digest = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
            spec = CapabilitySpec(
                id=f"mcp:{server_id}:{tool.name}",
                kind="mcp",
                name=tool.name,
                description=tool.description or tool.name,
                source_id=server_id,
                schema_digest=digest,
            )
            self._tools[spec.id] = (spec, schema)
            specs.append(spec)
        return specs

    def schema(self, capability_id: str) -> dict[str, Any]:
        if capability_id not in self._tools:
            self._discover_id(capability_id)
        return self._tools[capability_id][1]

    def _discover_id(self, capability_id: str) -> ServerConfig:
        pieces = capability_id.split(":", 2)
        if len(pieces) != 3 or pieces[0] != "mcp":
            raise ValueError("invalid MCP capability ID")
        server = self._servers.get(pieces[1])
        if server is None or pieces[2] not in server.allowlist:
            raise ValueError("MCP tool is not allowlisted")
        if capability_id not in self._tools:
            self.tools(server.id)
        if capability_id not in self._tools:
            raise ValueError("MCP tool was not discovered")
        return server

    async def _call(self, server: ServerConfig, name: str, arguments: dict[str, Any]) -> Any:
        async with self._client(server) as client:
            return await client.call_tool(name, arguments)

    def call(self, capability_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        server = self._discover_id(capability_id)
        schema = self._tools[capability_id][1]
        try:
            validate(arguments, schema)
        except ValidationError:
            raise ValueError("MCP arguments do not match tool schema") from None
        name = capability_id.split(":", 2)[2]
        result = self._run(self._call(server, name, arguments))
        text = "\n".join(
            block.text for block in result.content if getattr(block, "type", None) == "text"
        )[:16_000]
        return {
            "ok": not result.is_error,
            "text": text,
            "structured_content": result.structured_content,
            "is_error": result.is_error,
        }

    def close(self) -> None:
        with self._lock:
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
                if self._thread is not None:
                    self._thread.join(timeout=2)
                self._loop.close()
                self._loop = None
                self._thread = None
