from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from xiliumini.execution.gateway import ExecutionGateway, active_run


def build_meta_tools(*, role: str) -> list[BaseTool]:
    """Expose a small fixed tool surface when a workflow run is active."""
    context = active_run.get()
    if context is None or context.capability_manager is None:
        return []
    manager = context.capability_manager

    def capability_search(query: str, kind: str = "", limit: int = 5) -> str:
        """Find relevant built-in, Skill or MCP capabilities by a short query."""
        if kind not in {"", "builtin", "skill", "mcp"}:
            return json.dumps({"ok": False, "error": "invalid_capability_kind"})
        if manager.mcp_manager is not None and kind in {"", "mcp"}:
            for server_id in manager.mcp_manager.servers():
                ExecutionGateway().discover_mcp(server_id, context)
        selected = manager.search(query, kind=kind or None, limit=limit)  # type: ignore[arg-type]
        results: list[dict[str, Any]] = []
        for spec in selected:
            entry: dict[str, Any] = {
                "id": spec.id,
                "name": spec.name,
                "description": spec.description[:300],
            }
            if spec.kind == "mcp" and manager.mcp_manager is not None:
                schema = manager.mcp_manager.schema(spec.id)
                entry["schema_preview"] = {
                    "properties": list(schema.get("properties", {}))[:20],
                    "required": schema.get("required", [])[:20],
                }
            results.append(entry)
        return json.dumps({"ok": True, "results": results}, ensure_ascii=False)

    def skill_load(skill_id: str) -> str:
        """Load the complete SKILL.md instructions for one selected Skill."""
        try:
            content = manager.load_skill(skill_id)
            if context.event_sink is not None:
                context.event_sink(
                    {"type": "skill_activation", "run_id": context.run_id, "skill_id": skill_id}
                )
            return json.dumps(
                {"ok": True, "skill_id": skill_id, "content": content}, ensure_ascii=False
            )
        except ValueError as exc:
            return json.dumps({"ok": False, "error": "skill_load_failed", "reason": str(exc)})

    def skill_resource_read(skill_id: str, relative_path: str) -> str:
        """Read one explicitly named reference file inside a loaded Skill."""
        try:
            content = manager.read_skill_resource(skill_id, relative_path)
            return json.dumps({"ok": True, "content": content}, ensure_ascii=False)
        except ValueError as exc:
            return json.dumps({"ok": False, "error": "skill_resource_failed", "reason": str(exc)})

    def mcp_call(capability_id: str, arguments: dict[str, Any]) -> str:
        """Invoke one discovered and locally allowlisted MCP tool."""
        if manager.mcp_manager is None:
            return json.dumps({"ok": False, "error": "mcp_unavailable"})
        try:
            result = manager.mcp_manager.call(capability_id, arguments)
            return json.dumps(result, ensure_ascii=False)
        except (ValueError, TimeoutError, RuntimeError) as exc:
            return json.dumps({"ok": False, "error": "mcp_call_failed", "reason": str(exc)})

    def automation_manage(
        action: str, task_id: str = "", definition: dict[str, Any] | None = None
    ) -> str:
        """Create, list, pause, resume or remove a persisted scheduled task."""
        if context.data_dir is None:
            return json.dumps({"ok": False, "error": "automation_unavailable"})
        from xiliumini.automation.models import AutomationTask
        from xiliumini.automation.store import AutomationStore

        store = AutomationStore(context.data_dir)
        try:
            if action == "add" and definition is not None:
                task = AutomationTask.model_validate(definition)
                if task_id and task.id != task_id:
                    raise ValueError("task ID conflicts with definition")
                store.add(task)
                return json.dumps({"ok": True, "id": task.id, "persisted": True})
            if action == "list":
                tasks = [task.model_dump(mode="json") for task in store.list()]
                return json.dumps({"ok": True, "tasks": tasks}, ensure_ascii=False)
            if action in {"pause", "resume", "remove"} and task_id:
                changed = (
                    store.remove(task_id)
                    if action == "remove"
                    else store.set_enabled(task_id, action == "resume")
                )
                return json.dumps({"ok": changed, "id": task_id, "persisted": changed})
            raise ValueError("invalid automation action")
        except (ValueError, OSError) as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": "automation_manage_failed",
                    "reason": str(exc),
                }
            )

    functions = [capability_search, skill_load, skill_resource_read, mcp_call]
    if role == "planner":
        functions.append(automation_manage)
    return [StructuredTool.from_function(function) for function in functions]
