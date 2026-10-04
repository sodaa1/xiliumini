class XiliuminiError(Exception):
    """Base class for safe, user-facing application errors."""


class CheckpointError(XiliuminiError):
    """Raised when checkpoint persistence or recovery fails."""

    code = "checkpoint_error"


class TraceError(XiliuminiError):
    """Raised when execution tracing fails."""

    code = "trace_error"


class SessionError(XiliuminiError):
    """Raised when durable conversation state is invalid or cannot be saved."""

    code = "session_error"


class ConfigError(XiliuminiError):
    """Raised when runtime configuration is invalid."""


class ProviderError(XiliuminiError):
    """Raised when an OpenAI-compatible provider request fails."""

    code = "provider_error"


class AuthenticationProviderError(ProviderError):
    code = "authentication"


class RateLimitProviderError(ProviderError):
    code = "rate_limit"


class TimeoutProviderError(ProviderError):
    code = "timeout"


class ToolExecutionError(XiliuminiError):
    """Raised for a controlled tool failure."""


class WorkspaceError(ToolExecutionError):
    """Raised when a workspace or workspace-relative path is invalid."""

    code = "workspace_error"


class CommandExecutionError(ToolExecutionError):
    """Raised when a requested command violates the bounded execution contract."""

    code = "command_error"


class NodeOutputError(XiliuminiError):
    """Raised when a model node cannot produce its required structured output."""

    code = "node_output_error"


class MemorySystemError(XiliuminiError):
    """Raised when managed memory cannot be safely read or assembled."""

    code = "memory_error"


class MemoryBudgetError(MemorySystemError):
    """Raised when non-compressible planner context exceeds its token budget."""
