"""Per-organ NormalizationContext registry, owned by the gateway."""
from typing import Dict

from orion.signals.normalization import NormalizationContext


class NormalizationStateRegistry:
    """Lazy registry of NormalizationContext per organ_id."""

    def __init__(self):
        self._contexts: Dict[str, NormalizationContext] = {}

    def get(self, organ_id: str) -> NormalizationContext:
        if organ_id not in self._contexts:
            self._contexts[organ_id] = NormalizationContext()
        return self._contexts[organ_id]
