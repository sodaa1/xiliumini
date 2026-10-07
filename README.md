[简体中文](README.zh-CN.md)

# xiliumini

> A local-first, recoverable, and auditable Python Agent CLI.

`xiliumini` uses OpenAI-compatible models as its reasoning core. LangGraph coordinates
intent routing, planning, specialist Agent execution, and result verification, while a
Textual full-screen interface supports continuous conversations, research, file changes,
and controlled command execution inside local workspaces.

## About

`xiliumini` is a local-first project for learning and practicing production-oriented Agent
engineering.

Starting from a basic **ReAct + Tool Calling** loop, the project progressively adds:

- LangGraph workflow orchestration
- A Plan-Act-Verify execution loop
- A Supervisor and specialist Agents
- Layered Memory and context compression
- Human-in-the-loop Approval
- Checkpoint / Resume
- Trace observability
- Persistent Sessions
- A Textual TUI
- Capability Harness with Skill, MCP, policy/hook, and automation interfaces

Rather than wrapping a single model call, `xiliumini` develops an Agent from simply being
able to call tools into a complete system capable of **planning, execution, verification,
memory, recovery, and continuous interaction**.

> [!NOTE]
> During its design and implementation, this project referenced the Agent evolution path of
> [Wood-Q/MokioAgent](https://github.com/Wood-Q/MokioAgent), while independently implementing
> workspace isolation, Memory, Sessions, Harness engineering, and the TUI.

The current version is `0.1.0`. It requires Python `>=3.12` and uses `uv` for dependency
management.

### Engineering Capabilities

- **Local workspace isolation**: every Session uses a dedicated directory; file tools reject absolute paths, parent-directory traversal, and symlink escapes.
- **Persistent multi-turn interaction**: recent Sessions are saved so later requests can reuse the Workspace, Todo list, Memory, and execution records.
- **Layered memory**: manages Rules, Working Memory, History Summary, Notepad, and long-term preferences.
- **Safe execution**: BashTool accepts only allowlisted commands, and risky operations can require human approval.
- **Recoverable runs**: Checkpoints preserve state and file snapshots, allowing execution to resume from safe graph nodes.
- **Structured tracing**: Traces record nodes, tools, approvals, handoffs, duration, and failures while redacting sensitive fields.
- **Web research**: uses a Tavily API Key when available and can try keyless mode when authentication fails or no key is configured.
- **Extensible capabilities**: discovers Skills and MCP providers on demand, routes execution through policy and hooks, and exposes automation interfaces without loading every capability into model context.

## Features

### Multi-Agent Architecture

The system combines intent routing with collaboration between a Supervisor and specialist
Agents.

| Agent / Node | Responsibility |
| --- | --- |
| `IntentRouter` | Decides whether input should use ordinary chat or the full workflow |
| `ChatResponder` | Handles ordinary conversation and simple questions |
| `Supervisor` | Plans tasks, maintains the Todo list, and dispatches specialist Agents |
| `searchAgent` | Searches online sources and organizes findings and conclusions |
| `codeAgent` | Works with workspace files, runs restricted commands, and completes coding tasks |
| `Verifier` | Uses tests, static checks, and tool evidence to decide whether a task is complete |
| `Final` | Summarizes the final state and produces the response |

Complex tasks are not delegated to a single model call. The Supervisor invokes specialist
Agents according to task state, collects structured evidence, and performs bounded retries
when verification does not pass.

## Workflow

```text
                         ┌──────────────┐
                         │  User Input  │
                         └──────┬───────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ Intent Router │
                        └──────┬────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌────────────────┐          ┌────────────────┐
        │ ChatResponder  │          │   Supervisor   │◄──────────┐
        └───────┬────────┘          └───────┬────────┘           │
                │                 ┌─────────┼─────────┐          │
                │                 │         │         │          │
                │                 ▼         ▼         ▼          │
                │          ┌───────────┐ ┌─────────┐ ┌──────┐   │
                │          │searchAgent│ │codeAgent│ │ Todo │   │
                │          └─────┬─────┘ └────┬────┘ └──────┘   │
                │                └──────┬──────┘                 │
                │                       │ results                │
                │                       ▼                        │
                │               ┌──────────────┐                 │
                │               │  Supervisor  │                 │
                │               └──────┬───────┘                 │
                │                      ▼                         │
                │               ┌──────────────┐                 │
                │               │   Verifier   │── retry ────────┘
                │               └──────┬───────┘
                │                      │ pass
                └──────────┬───────────┘
                           ▼
                    ┌────────────┐
                    │   Final    │
                    └────────────┘

        Automation ────────▶ Runtime ────────▶ Intent Router

        Supervisor / specialist Agent
                    │
                    ▼
            Capability Manager
                    │
                    ▼
          Skill / MCP / Builtin
                    │
                    ▼
            Execution Gateway
                    │
                    ├──▶ Tool Policy
                    ├──▶ Before Hook
                    ├──▶ Tool execution
                    └──▶ After Hook / audit
```

Ordinary conversations receive a fast response through `ChatResponder`. Requests involving
file creation or modification, command execution, web research, coding, tests, error fixes,
or continued workspace tasks enter the full Agent workflow.
Workflow Agents discover Skills and MCP providers when needed. Automation is another trigger
for the same Runtime and Agent workflow.

## Project Structure

```text
xiliumini/
├── src/xiliumini/
│   ├── agents/       # Specialist Agents and the shared ReAct loop
│   ├── automation/   # Scheduled tasks, persistence, and Runtime runner
│   ├── capabilities/ # Capability Manager, Skills, MCP, and meta tools
│   ├── cli/          # Typer CLI and Textual TUI
│   ├── core/         # Approval, Checkpoint, Session, and Trace
│   ├── execution/    # Execution Gateway
│   ├── graph/        # LangGraph state, nodes, routing, and verification
│   ├── hooks/        # Post-execution hooks
│   ├── policy/       # Tool policy decisions
│   ├── providers/    # OpenAI-compatible model adapters
│   ├── storage/      # Session and Trace storage interfaces
│   └── tools/        # File, search, command, Todo, Notepad, and other tools
├── .xiliumini/
│   ├── skills/        # Versioned Agent Skills
│   ├── mcp/<provider>/mcp.json  # Versioned MCP provider configurations
│   └── workspaces/    # Local Runtime workspaces
├── docs/             # Product, specification, evolution, design, and implementation docs
├── .env.example      # Configuration template
├── pyproject.toml    # Project and dependency configuration
└── uv.lock           # Reproducible dependency lockfile
```

For the detailed architecture, see [Project Evolution](docs/项目进程.md).

## Quick Start

### Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)
- An available OpenAI-compatible model, API key, and service endpoint
- Git (required by the default Checkpoint mode; optional when Checkpoints are disabled)

### Installation

```powershell
git clone https://github.com/sodaa1/xiliumini.git
cd xiliumini
uv sync --no-dev
Copy-Item .env.example .env
```

Edit the local `.env` file and provide at least the model configuration:

```dotenv
XILIUMINI_API_KEY=your-provider-key
XILIUMINI_MODEL=your-model-name
XILIUMINI_BASE_URL=https://your-provider.example/v1
```

`XILIUMINI_BASE_URL` can be omitted when using the default OpenAI service. A Tavily key is
optional:

```dotenv
TAVILY_API_KEY=your-tavily-key
```

The `.env` file is ignored by Git. Never commit real credentials.

Check the environment and start the TUI:

```powershell
uv run xiliumini doctor
uv run xiliumini
```

You can also install it as a standalone command:

```powershell
uv tool install --editable .
xiliumini doctor
xiliumini
```

## Usage

### Start the TUI

```powershell
uv run xiliumini
uv run xiliumini chat
```

A Session continuously reuses its conversation, Workspace, Todo list, and Memory. Each run
records the corresponding Checkpoint and Trace in that workspace, making it suitable for
multi-step work:

```text
Create a Todo API
Add SQLite support
Run the tests
Fix the previous error
```

### One-shot Task

```powershell
uv run xiliumini ask "Analyze the current project structure and suggest improvements"
uv run xiliumini ask "Calculate (17 + 5) * 3" --no-stream
```

### Automation

```powershell
uv run xiliumini automation add daily-brief --name "Daily brief" --prompt "Summarize AI news" --daily 09:00
uv run xiliumini automation list
uv run xiliumini automation start
```

### Workspace and Resume

```powershell
uv run xiliumini --workspace ".xiliumini/workspaces/demo" `
  --checkpoint-mode strict --trace-mode on ask "Complete this task"

uv run xiliumini --resume ".xiliumini/workspaces/<session-id>"
```

Root options must appear before `ask` or `chat`. Common options include:

| Option | Values | Description |
| --- | --- | --- |
| `--max-attempts` | Positive integer | Maximum Supervisor attempts; defaults to `3` |
| `--approval-mode` | `inline` / `auto` / `deny` | Approval policy for risky commands |
| `--checkpoint-mode` | `light` / `strict` / `off` | Checkpoint persistence level |
| `--trace-mode` | `on` / `off` | Whether to save execution Traces |
| `--workspace` | Path under the data directory | Use the specified workspace |
| `--resume` | Checkpoint workspace path | Resume a saved run |

Run `uv run xiliumini --help`, `uv run xiliumini chat --help`, or
`uv run xiliumini ask --help` for the complete option reference.

## Data and Security Boundaries

Runtime data is written to the following location by default:

```text
.xiliumini/
├── session/                         # Multi-turn Sessions
├── USER_PREFERENCES.md              # Project-level long-term preferences
└── workspaces/<session-id>/
    ├── TODO.md
    ├── NOTEPAD.md
    ├── HISTORY_SUMMARY.md
    └── .xiliumini/
        ├── checkpoints/             # State, recovery instructions, and file snapshots
        └── traces/<trace-id>/        # Events, statistics, and timeline
```

- File operations can only occur inside the current Workspace.
- Risky Bash commands can require human approval before execution.
- Runtime control files such as `TODO.md`, `NOTEPAD.md`, and `HISTORY_SUMMARY.md` are protected.
- API keys are not written to Agent State, and sensitive fields are redacted from events, errors, and Traces.
- BashTool uses argument arrays, an allowlist, time limits, and output limits, but it is not an operating-system sandbox.
- Checkpoint Git snapshots preserve original workspace bytes and may contain sensitive data; treat runtime directories as private.
- Restoring a Checkpoint restores ordinary files and removes ordinary files added after the snapshot; back up important content first.

## Design Principles

- **Explicit over Magic**: planning, tool calls, handoffs, verification, Checkpoints, and Traces are represented by explicit events.
- **Evidence-based Verification**: the Verifier evaluates file state, tests, static checks, command output, and tool evidence.
- **Workspace First**: every file or command operation begins from an explicit Workspace boundary.
- **Recoverable Execution**: long-running work should be saved and resumed instead of restarted after every interruption.
- **Memory != Full History**: retain rules, the active task, working state, long-term notes, and compressed history instead of accumulating messages indefinitely.

## Documentation

- [Product Requirements (PRD)](docs/PRD.md)
- [Technical Specification (SPEC)](docs/SPEC.md)
- [Project Evolution](docs/项目进程.md)

## Current Limitations

- Only Python 3.12 and later are supported.
- Command execution and generated code use the current user's permissions; no container or operating-system isolation is provided.
- Checkpoint recovery occurs at graph-node boundaries and cannot resume in the middle of a tool call.
- `chat --session` does not yet support selecting a Session by name; conversation data can be found under `.xiliumini`.
