from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass

from xiliumini.core.approval import ApprovalDecision, ApprovalRequest


@dataclass(frozen=True, slots=True)
class ApprovalConfig:
    """Approval policy active for one Runtime generator step."""

    mode: str = "inline"
    handler: Callable[[ApprovalRequest], ApprovalDecision] | None = None


_DEFAULT_APPROVAL_CONFIG = ApprovalConfig()

approval_config: ContextVar[ApprovalConfig] = ContextVar(
    "approval_config",
    default=_DEFAULT_APPROVAL_CONFIG,
)


__all__ = ["ApprovalConfig", "approval_config"]
