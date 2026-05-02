"""In-memory recent signals by organ_id with TTL eviction."""
from datetime import datetime, timezone
from typing import Dict, Optional

from orion.signals.models import OrionSignalV1


class SignalWindow:
    """Keeps the most recent OrionSignalV1 per organ_id, evicting entries older than window_sec."""

    def __init__(self, window_sec: float = 30.0):
        self._window_sec = window_sec
        self._signals: Dict[str, OrionSignalV1] = {}

    def put(self, signal: OrionSignalV1) -> None:
        """Store the most recent signal for an organ. Evict stale entries."""
        self._evict()
        self._signals[signal.organ_id] = signal

    def get_all(self) -> Dict[str, OrionSignalV1]:
        """Return all non-stale signals keyed by organ_id."""
        self._evict()
        return dict(self._signals)

    def get(self, organ_id: str) -> Optional[OrionSignalV1]:
        """Return the most recent non-stale signal for organ_id, or None."""
        self._evict()
        return self._signals.get(organ_id)

    def _evict(self) -> None:
        now = datetime.now(timezone.utc)
        stale = [
            k for k, v in self._signals.items()
            if (now - v.emitted_at.replace(tzinfo=timezone.utc) if v.emitted_at.tzinfo is None
                else now - v.emitted_at).total_seconds() > self._window_sec
        ]
        for k in stale:
            del self._signals[k]
