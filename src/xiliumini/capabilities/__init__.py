"""Capability discovery without eagerly loading external content."""

from xiliumini.capabilities.manager import CapabilityManager
from xiliumini.capabilities.models import CapabilityRef, CapabilitySpec

__all__ = ["CapabilityManager", "CapabilityRef", "CapabilitySpec"]
