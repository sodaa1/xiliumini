from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from xiliumini.automation.models import AutomationTask
from xiliumini.automation.runner import AutomationRunner
from xiliumini.automation.scheduler import AutomationScheduler
from xiliumini.automation.store import AutomationStore
from xiliumini.capabilities.langchain_tools import build_meta_tools
from xiliumini.capabilities.manager import CapabilityManager
from xiliumini.capabilities.mcp import MCPManager
from xiliumini.capabilities.skills import SkillRegistry
from xiliumini.config import load_settings
from xiliumini.execution.gateway import ExecutionGateway, RunContext, active_run
from xiliumini.policy.engine import PolicyEngine

skills_app = typer.Typer(help="Manage local Agent Skills.")
mcp_app = typer.Typer(help="Inspect configured MCP servers.")
automation_app = typer.Typer(help="Manage persistent scheduled tasks.")


def _data_dir() -> Path:
    return Path(os.environ.get("XILIUMINI_DATA_DIR", ".xiliumini")).expanduser()


def _skills() -> SkillRegistry:
    return SkillRegistry(_data_dir() / "skills")


@skills_app.command("list")
def skills_list() -> None:
    for spec in _skills().scan():
        typer.echo(f"{spec.name}\t{spec.description}")


@skills_app.command("validate")
def skills_validate(path: Path) -> None:
    try:
        spec = _skills().validate(path)
    except (ValueError, OSError) as exc:
        typer.echo(f"Invalid Skill: {exc}")
        raise typer.Exit(code=2) from None
    typer.echo(f"Valid Skill: {spec.name}")


@skills_app.command("install")
def skills_install(
    source: Path,
    from_workspace: Annotated[Path | None, typer.Option("--from-workspace")] = None,
    trusted_source: Annotated[bool, typer.Option("--trusted-source")] = False,
) -> None:
    try:
        spec = _skills().install(source, workspace=from_workspace, trusted=trusted_source)
    except (ValueError, OSError) as exc:
        typer.echo(f"Skill installation failed: {exc}")
        raise typer.Exit(code=2) from None
    typer.echo(f"Installed Skill: {spec.name}")


@skills_app.command("remove")
def skills_remove(name: str) -> None:
    try:
        _skills().remove(name)
    except (ValueError, OSError) as exc:
        typer.echo(f"Skill removal failed: {exc}")
        raise typer.Exit(code=2) from None
    typer.echo(f"Removed Skill: {name}")


@mcp_app.command("list")
def mcp_list() -> None:
    manager = MCPManager(_data_dir() / "mcp")
    for server_id in manager.servers():
        typer.echo(server_id)


def _mcp_context(manager: MCPManager) -> RunContext:
    data_dir = _data_dir()
    return RunContext(
        run_id="cli-mcp",
        session_id="cli-mcp",
        workspace=data_dir,
        data_dir=data_dir,
        capability_manager=CapabilityManager(mcp_manager=manager),
        policy_engine=PolicyEngine(
            data_dir / "policy.json", trusted_mcp_servers=set(manager.servers())
        ),
    )


@mcp_app.command("tools")
def mcp_tools(server_id: str) -> None:
    manager = MCPManager(_data_dir() / "mcp")
    try:
        result = ExecutionGateway().discover_mcp(server_id, _mcp_context(manager))
        if not result["ok"]:
            raise RuntimeError(result["reason"])
        for spec in result["tools"]:
            schema = manager.schema(spec.id)
            typer.echo(
                json.dumps(
                    {
                        "id": spec.id,
                        "description": spec.description,
                        "schema": schema,
                    }
                )
            )
    except (ValueError, TimeoutError, RuntimeError, OSError) as exc:
        typer.echo(f"MCP discovery failed: {exc}")
        raise typer.Exit(code=1) from None
    finally:
        manager.close()


@mcp_app.command("test")
def mcp_test(
    server_id: str,
    call: Annotated[str | None, typer.Option("--call")] = None,
    arguments: Annotated[str, typer.Option("--arguments")] = "{}",
) -> None:
    manager = MCPManager(_data_dir() / "mcp")
    try:
        context = _mcp_context(manager)
        gateway = ExecutionGateway()
        discovery = gateway.discover_mcp(server_id, context)
        if not discovery["ok"]:
            raise RuntimeError(discovery["reason"])
        specs = discovery["tools"]
        typer.echo(f"Discovered {len(specs)} allowlisted tool(s)")
        if call is not None:
            token = active_run.set(context)
            try:
                tool = next(
                    tool
                    for tool in build_meta_tools(role="search_agent")
                    if tool.name == "mcp_call"
                )
            finally:
                active_run.reset(token)
            output = gateway.wrap_tool(tool, context).invoke(
                {
                    "capability_id": f"mcp:{server_id}:{call}",
                    "arguments": json.loads(arguments),
                }
            )
            result = json.loads(output)
            typer.echo(json.dumps(result, ensure_ascii=False))
            if not result["ok"]:
                raise typer.Exit(code=1)
    except (ValueError, TimeoutError, RuntimeError, OSError) as exc:
        typer.echo(f"MCP test failed: {exc}")
        raise typer.Exit(code=1) from None
    finally:
        manager.close()


@automation_app.command("add")
def automation_add(
    task_id: str,
    name: Annotated[str, typer.Option("--name")],
    prompt: Annotated[str, typer.Option("--prompt")],
    daily: Annotated[str | None, typer.Option("--daily")] = None,
    at: Annotated[str | None, typer.Option("--at")] = None,
    every_seconds: Annotated[int | None, typer.Option("--every-seconds")] = None,
    timezone: Annotated[str, typer.Option("--timezone")] = "Asia/Shanghai",
) -> None:
    if sum(value is not None for value in (daily, at, every_seconds)) != 1:
        raise typer.BadParameter("choose exactly one of --daily, --at, or --every-seconds")
    if daily is not None:
        try:
            hour, minute = (int(part) for part in daily.split(":"))
        except ValueError:
            raise typer.BadParameter("--daily must be HH:MM") from None
        trigger_type, schedule = "cron", {"hour": hour, "minute": minute}
    elif at is not None:
        trigger_type, schedule = "date", {"at": at}
    else:
        trigger_type, schedule = "interval", {"seconds": every_seconds}
    try:
        task = AutomationTask(
            id=task_id,
            name=name,
            prompt=prompt,
            trigger_type=trigger_type,
            schedule=schedule,
            timezone=timezone,
        )
        AutomationStore(_data_dir()).add(task)
    except (ValueError, OSError) as exc:
        typer.echo(f"Automation add failed: {exc}")
        raise typer.Exit(code=2) from None
    typer.echo(f"Added automation: {task_id}")


@automation_app.command("list")
def automation_list() -> None:
    for task in AutomationStore(_data_dir()).list():
        enabled = "on" if task.enabled else "off"
        typer.echo(f"{task.id}\t{task.trigger_type}\t{task.schedule}\t{enabled}")


@automation_app.command("pause")
def automation_pause(task_id: str) -> None:
    if not AutomationStore(_data_dir()).set_enabled(task_id, False):
        raise typer.Exit(code=2)
    typer.echo(f"Paused automation: {task_id}")


@automation_app.command("resume")
def automation_resume(task_id: str) -> None:
    if not AutomationStore(_data_dir()).set_enabled(task_id, True):
        raise typer.Exit(code=2)
    typer.echo(f"Resumed automation: {task_id}")


@automation_app.command("remove")
def automation_remove(task_id: str) -> None:
    if not AutomationStore(_data_dir()).remove(task_id):
        raise typer.Exit(code=2)
    typer.echo(f"Removed automation: {task_id}")


@automation_app.command("run")
def automation_run(task_id: str, once: Annotated[bool, typer.Option("--once")] = False) -> None:
    if not once:
        raise typer.BadParameter("use --once for an immediate run")
    try:
        result = AutomationRunner(AutomationStore(_data_dir()), load_settings()).run_once(task_id)
    except (ValueError, OSError) as exc:
        typer.echo(f"Automation run failed: {exc}")
        raise typer.Exit(code=1) from None
    typer.echo(
        f"{result.status}: {result.run_id} trace={result.trace_id} report={result.report_path}"
    )
    if result.status != "success":
        raise typer.Exit(code=1)


@automation_app.command("start")
def automation_start() -> None:
    try:
        store = AutomationStore(_data_dir())
        service = AutomationScheduler(store, AutomationRunner(store, load_settings()))
        typer.echo("Automation scheduler started")
        service.serve_forever()
    except KeyboardInterrupt:
        typer.echo("Automation scheduler stopped")
    except (RuntimeError, OSError) as exc:
        typer.echo(f"Automation scheduler failed: {exc}")
        raise typer.Exit(code=1) from None
