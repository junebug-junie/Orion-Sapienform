"""
Abstract base class for all Orion signal adapters.
"""
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Optional

from orion.signals.models import OrionSignalV1, OrionOrganRegistryEntry
from orion.signals.normalization import NormalizationContext


class OrionSignalAdapter(ABC):
    """Base class for all organ adapters."""

    organ_id: ClassVar[str]

    @abstractmethod
    def can_handle(self, channel: str, payload: dict) -> bool:
        """Return True if this adapter should process this bus event."""

    @abstractmethod
    def adapt(
        self,
        channel: str,
        payload: dict,
        registry: Dict[str, OrionOrganRegistryEntry],
        prior_signals: Dict[str, OrionSignalV1],
        norm_ctx: NormalizationContext,
    ) -> Optional[OrionSignalV1]:
        """Transform a raw bus event into a hardened signal. Return None to drop."""
