class XiliuminiError(Exception):
    """Base class for safe, user-facing application errors."""


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
