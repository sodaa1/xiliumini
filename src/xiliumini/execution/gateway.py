from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field

from xiliumini.capabilities.manager import BUILTIN_PERMISSIONS, CapabilityManager
from xiliumini.capabilities.models import CapabilitySpec
from xiliumini.core.approval import ApprovalDecision, ApprovalRequest
from xiliumini.hooks.engine import HookEngine
from xiliumini.policy.engine import PolicyEngine


@dataclass(frozen=True)
class RunContext:
    run_id: str
    session_id: str
    workspace: Path
    data_dir: Path | None = None
    actor_type: str = "interactive"
    policy_profile: str = "interactive"
    event_sink: Callable[[dict[str, Any]], None] | None = None
    capability_manager: CapabilityManager | None = None
    policy_engine: PolicyEngine | None = None
    approval_handler: Callable[[ApprovalRequest], ApprovalDecision] | None = None
    approval_mode: str = "inline"
    hook_engine: HookEngine | None = None
    hook_failures: list[str] = field(default_factory=list)


active_run: ContextVar[RunContext | None] = ContextVar("xiliumini_capability_run", default=None)


class _GatewayTool(BaseTool):
    base_tool: BaseTool = Field(exclude=True)
    run_context: RunContext = Field(exclude=True)
    gateway: ExecutionGateway = Field(exclude=True)

    def _run(self, **kwargs: Any) -> Any:
        return self.gateway.invoke(self.base_tool, kwargs, self.run_context)


class ExecutionGateway:
    def discover_mcp(self, server_id: str, context: RunContext) -> dict[str, Any]:
        manager = context.capability_manager
        if manager is None or manager.mcp_manager is None or context.policy_engine is None:
            return {"ok": False, "error": "policy_denied", "reason": "MCP is unavailable"}
        try:
            transport = manager.mcp_manager.transport(server_id)
        except KeyError:
            return {"ok": False, "error": "policy_denied", "reason": "MCP server is unknown"}
        permission = "process.execute" if transport == "stdio" else "network.read"
        judged = context.policy_engine.evaluate(
            actor=context.actor_type,
            source="mcp",
            capability_id=f"mcp:{server_id}:discovery",
            permission=permission,
            arguments={},
            profile=context.policy_profile,
        )
        if judged.effect != "allow":
            if context.event_sink is not None:
                context.event_sink(
                    {
                        "type": "mcp_discovery",
                        "run_id": context.run_id,
                        "server_id": server_id,
                        "decision": judged.effect,
                        "success": False,
                        "reason": judged.reason,
                    }
                )
            return {"ok": False, "error": "policy_denied", "reason": judged.reason}
        started = time.monotonic()
        success = False
        try:
            specs = manager.mcp_tools(server_id)
            success = True
            return {"ok": True, "tools": specs}
        except Exception as exc:
            return {
                "ok": False,
                "error": "mcp_discovery_failed",
                "reason": type(exc).__name__,
            }
        finally:
            if context.event_sink is not None:
                context.event_sink(
                    {
                        "type": "mcp_discovery",
                        "run_id": context.run_id,
                        "server_id": server_id,
                        "decision": "allow",
                        "success": success,
                        "duration_ms": int((time.monotonic() - started) * 1000),
                    }
                )

    @staticmethod
    def authorize_hook(context: RunContext, hook_id: str) -> bool:
        if context.policy_engine is None:
            return False
        decision = context.policy_engine.evaluate(
            actor="hook",
            source="hook",
            capability_id=f"hook:{hook_id}",
            permission="process.execute.safe",
            arguments={},
            profile=context.policy_profile,
        )
        return decision.effect == "allow"

    @staticmethod
    def _policy_target(
        tool: BaseTool, arguments: dict[str, Any], context: RunContext
    ) -> tuple[str, str, str]:
        if tool.name == "automation_manage":
            return "automation", "automation:manage", "automation.manage"
        if tool.name == "mcp_call":
            capability_id = arguments.get("capability_id")
            if not isinstance(capability_id, str):
                return "mcp", "mcp:invalid", "unknown"
            try:
                manager = context.capability_manager
                if manager is None or manager.mcp_manager is None:
                    raise ValueError("MCP is unavailable")
                permission = manager.mcp_manager.permission_for_tool(capability_id)
            except ValueError:
                permission = "unknown"
            return "mcp", capability_id, permission
        if tool.name == "bash":
            argv = arguments.get("argv", [])
            if isinstance(argv, list) and all(isinstance(part, str) for part in argv):
                from xiliumini.tools.bash_tool import RISKY_ARGV_PREFIXES

                for prefix in RISKY_ARGV_PREFIXES:
                    if tuple(argv[: len(prefix)]) == prefix:
                        permission = (
                            "package.install"
                            if "install" in prefix or prefix == ("uv", "add")
                            else "process.execute"
                        )
                        return "builtin", "builtin:bash", permission
        permission = BUILTIN_PERMISSIONS.get(tool.name, "unknown")
        return "builtin", f"builtin:{tool.name}", permission

    def invoke(self, tool: BaseTool, arguments: dict[str, Any], context: RunContext) -> Any:
        started = time.monotonic()
        success = False
        decision = "allow"
        hook_results: list[dict[str, Any]] = []
        source, capability_id, permission = self._policy_target(tool, arguments, context)
        try:
            if tool.name == "mcp_call" and context.policy_engine is not None:
                manager = context.capability_manager
                pieces = capability_id.split(":", 2)
                if manager is not None and manager.mcp_manager is not None and len(pieces) == 3:
                    try:
                        transport = manager.mcp_manager.transport(pieces[1])
                    except KeyError:
                        transport = "unknown"
                    if transport == "stdio":
                        process_decision = context.policy_engine.evaluate(
                            actor=context.actor_type,
                            source="mcp",
                            capability_id=capability_id,
                            permission="process.execute",
                            arguments=arguments,
                            profile=context.policy_profile,
                        )
                        if process_decision.effect != "allow":
                            decision = "deny"
                            return json.dumps(
                                {
                                    "ok": False,
                                    "error": "policy_denied",
                                    "retryable": False,
                                    "reason": process_decision.reason,
                                }
                            )
            if context.policy_engine is not None:
                judged = context.policy_engine.evaluate(
                    actor=context.actor_type,
                    source=source,
                    capability_id=capability_id,
                    permission=permission,
                    arguments=arguments,
                    profile=context.policy_profile,
                )
                decision = judged.effect
                if decision == "ask":
                    handler = context.approval_handler
                    if (
                        source == "builtin"
                        and tool.name == "bash"
                        and context.approval_mode == "auto"
                    ):
                        decision = "allow"
                    elif handler is None:
                        decision = "deny"
                    else:
                        request = ApprovalRequest(
                            id=f"policy-{context.run_id}",
                            command=capability_id,
                            risk_reason=judged.reason,
                        )
                        try:
                            approved = handler(request)
                            decision = "allow" if approved.approved else "deny"
                        except Exception:
                            decision = "deny"
                if decision == "deny":
                    return json.dumps(
                        {
                            "ok": False,
                            "error": "policy_denied",
                            "retryable": False,
                            "reason": judged.reason,
                        }
                    )
            if context.hook_engine is not None:
                before_events, blocked = context.hook_engine.before(
                    tool_name=tool.name, arguments=arguments, context=context, gateway=self
                )
                hook_results.extend(before_events)
                if context.event_sink is not None:
                    for event in before_events:
                        context.event_sink(event)
                if blocked:
                    decision = "deny"
                    return json.dumps(
                        {
                            "ok": False,
                            "error": "hook_blocked",
                            "retryable": False,
                            "reason": "blocking before hook failed",
                        }
                    )
            result = tool.invoke(arguments)
            success = not (isinstance(result, str) and result.startswith("Error:"))
            if isinstance(result, str) and result.startswith("{"):
                try:
                    data = json.loads(result)
                except ValueError:
                    pass
                else:
                    success = success and not (isinstance(data, dict) and data.get("ok") is False)
            if success and context.hook_engine is not None:
                after_events = context.hook_engine.after(
                    tool_name=tool.name,
                    arguments=arguments,
                    output=result,
                    context=context,
                    gateway=self,
                )
                hook_results.extend(after_events)
                if context.event_sink is not None:
                    for event in after_events:
                        context.event_sink(event)
            return result
        finally:
            if context.event_sink is not None:
                context.event_sink(
                    {
                        "type": "capability_execution",
                        "capability_id": capability_id,
                        "run_id": context.run_id,
                        "tool_name": tool.name,
                        "duration_ms": int((time.monotonic() - started) * 1000),
                        "decision": decision,
                        "success": success,
                        "hook_results": [item["status"] for item in hook_results],
                    }
                )

    def wrap_tool(self, tool: BaseTool, context: RunContext) -> BaseTool:
        manager = context.capability_manager
        if manager is not None and tool.name in BUILTIN_PERMISSIONS:
            raw_schema = tool.args_schema
            if isinstance(raw_schema, dict):
                schema = raw_schema
            elif raw_schema is None:
                schema = {}
            else:
                factory = getattr(raw_schema, "model_json_schema", None)
                schema = factory() if callable(factory) else raw_schema.schema()
            digest = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
            identifier = f"builtin:{tool.name}"
            if manager.get(identifier) is None:
                manager.register(
                    CapabilitySpec(
                        id=identifier,
                        kind="builtin",
                        name=tool.name,
                        description=tool.description,
                        source_id="builtin",
                        schema_digest=digest,
                    )
                )
        return _GatewayTool(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            base_tool=tool,
            run_context=context,
            gateway=self,
        )


def wrap_tools(tools: list[BaseTool]) -> list[BaseTool]:
    context = active_run.get()
    if context is None:
        return tools
    gateway = ExecutionGateway()
    return [gateway.wrap_tool(tool, context) for tool in tools]
