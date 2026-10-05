# Session Routing and Multi-Turn Events Design

## Goal

Add durable multi-turn conversation state in `.xiliumini/session` and expose a
`stream_session_events` API that routes each new user message through the lightweight chat
graph or the existing full workflow. The router and chat responder receive bounded context
about the isolated session workspace and recent conversation without gaining tool access.

## Directory Model

`session_workspace` is the xiliumini data root and defaults to `.xiliumini`.

```text
.xiliumini/
├── session/
│   ├── session.json
│   └── SESSION_SUMMARY.md
└── workspaces/
    └── <session_id>/
        ├── TODO.md
        ├── NOTEPAD.md
        ├── HISTORY_SUMMARY.md
        └── <user workspace files>
```

The full workflow and its tools remain confined to the existing per-session workspace. The
router's file inventory also comes from that isolated directory. Session, checkpoint, trace,
and other conversations' metadata are not exposed as workspace files.

## Session Persistence

`src/xiliumini/core/session.py` owns persistence and context assembly. It defines:

- `SESSION_ROOT = ".xiliumini/session"`
- `SESSION_FILE = "session.json"`
- `SESSION_SUMMARY_FILE = "SESSION_SUMMARY.md"`
- `MAX_SESSION_CONTEXT = 7000`
- `MAX_TURN_CONTENT = 4000`

The public helpers accept the xiliumini data workspace (default `.xiliumini`) and use its
`session` child. `SESSION_ROOT` names the complete default relative path; its parent is the
default workspace and its final component is the per-workspace session directory. Therefore
the default final path is exactly `.xiliumini/session`, never
`.xiliumini/.xiliumini/session`.

`load_or_create_session` creates a UUID and UTC ISO timestamps when no file exists. Existing
JSON must contain a valid session ID, non-negative integer turn index, timestamps, and a list
of valid turns. Invalid UTF-8, malformed JSON, or invalid structure raises a stable session
error and does not overwrite the source.

Every message receives a globally increasing sequence number. `append_user_turn` allocates
and returns the next number. The caller passes the following number to
`append_assistant_turn`, which rejects duplicates and gaps. Turn content is limited to 4000
characters. The persisted `recent_turns` list retains the most recent ten messages while
`turn_index` remains monotonic.

`save_session` validates the complete object, updates `updated_at`, then atomically writes
both JSON and a human-readable Markdown snapshot. JSON is the source of truth; the Markdown
file is never parsed during loading.

## Session Context

`build_session_context` returns a string no longer than 7000 characters containing:

1. session ID and current turn index;
2. up to 30 files from the isolated execution workspace, newest modification time first;
3. up to ten recent messages in chronological order.

Only relative file paths are included; file contents are not read. Assistant messages prefer
their non-empty summary and otherwise use content. Budget allocation always preserves the
session header and newest conversation first, then older messages and as much of the file
inventory as fits.

## Runtime and Event Flow

The Runtime remains the owner of the model, memory manager, approval context, checkpoint,
and trace lifecycle. A session-aware Runtime method prepares a fresh `GraphState`, sets
`context_summary`, and invokes the entry graph with the main model.

- For `chat`, the entry graph supplies `final_answer`; Runtime emits one `FinalEvent` and does
  not start the full workflow or tool lifecycle.
- For `workflow`, the routed state is passed to the existing full workflow through the normal
  Runtime harness path. `build_complex_workflow` is a compatibility name for the existing
  `build_workflow` implementation, not a second graph.

`stream_session_events` in `core/agent.py` performs the outer transaction:

1. resolve the default or explicit session data root;
2. load/create the session and append/save the user message;
3. build bounded context from the isolated workspace;
4. consume the Runtime's routed event stream;
5. forward events in the existing `graph_event` / `custom_event` dictionary format;
6. after a real final answer, append and save an assistant message with its route.

An error event, provider exception, or consumer closing the generator before a final answer
does not create a fabricated assistant turn. Nested generators are always closed.

The API mirrors the existing maximum-attempt, approval, checkpoint, and trace options. It is
lazy-exported from `xiliumini.core`. Existing `stream_agent_events`, CLI behavior, and resume
behavior remain unchanged.

## Error Handling

Session validation and persistence failures surface as stable session errors without leaking
paths or raw corrupt content. Atomic replacement protects the last valid file. Workspace and
provider failures continue through the existing redacted `ErrorEvent` mapping. The user turn
is saved before model execution so a failed request is still represented accurately.

## Verification

Unit coverage will verify creation/reload, UUID and timestamp validation, monotonic turns,
content truncation, recent-turn retention, atomic JSON/Markdown output, corrupt-file handling,
file ordering, and the 7000-character context limit.

Runtime and agent coverage will verify chat short-circuiting, workflow handoff with session
context, prior-turn visibility, final-response persistence, error behavior, generator close
behavior, stable event dictionaries, and unchanged legacy entry points. Completion requires
the focused tests, full pytest suite, Ruff lint and formatting, Pyright, and `git diff --check`.
