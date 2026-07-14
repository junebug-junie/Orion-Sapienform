"""Bound the tension stream: per-source cap, dedup, storm safety (spec §5).

Even with deviation gating a misbehaving organ could burst. ``TensionRateLimiter``
keeps a sliding window of recent emits per ``(source_kind, drive-signature)`` and
drops anything over the cap. State is bounded: at most ``max_keys`` windows, and
each window holds at most ``cap`` timestamps. Pure/synchronous; ``now`` is passed
in (monotonic seconds) so it is deterministic and testable.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

from orion.core.schemas.drives import TensionEventV1


def _signature(t: TensionEventV1) -> Tuple[str, Tuple[str, ...]]:
    """Source identity: kind + the set of drives it pushes. Two tensions with the
    same kind and drive-set are the same recurring pressure.

    Includes relief (negative-weight) drives, not just growth (positive)
    ones -- every producer before the P3 satisfaction tension only ever
    emitted non-negative weights, so this is a no-op for them (w != 0.0 vs.
    w > 0.0 select the same drives when no weight is ever negative).
    Without this, an all-negative-weight tension collides into an empty-
    tuple signature shared with any other zero-positive-weight tension of
    the same kind, rate-limiting them together instead of independently.
    """
    drives = tuple(sorted(d for d, w in (t.drive_impacts or {}).items() if w != 0.0))
    return (t.kind, drives)


@dataclass
class TensionRateLimiter:
    cap: int = 3
    window_sec: float = 60.0
    max_keys: int = 512
    _seen: "OrderedDict[Tuple[str, Tuple[str, ...]], Deque[float]]" = field(
        default_factory=OrderedDict
    )

    def _evict_if_needed(self) -> None:
        while len(self._seen) > self.max_keys:
            self._seen.popitem(last=False)  # drop oldest key

    def bounded(self, candidates: List[TensionEventV1], now: float) -> List[TensionEventV1]:
        """Return the subset of ``candidates`` allowed through, updating windows.

        Within one call, identical-signature candidates also count against the
        cap (a 100-event storm in one tick keeps at most ``cap``).
        """
        kept: List[TensionEventV1] = []
        for t in candidates:
            key = _signature(t)
            window = self._seen.get(key)
            if window is None:
                window = deque()
                self._seen[key] = window
            # Expire old timestamps.
            cutoff = now - self.window_sec
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) >= self.cap:
                continue  # over cap -> drop
            window.append(now)
            self._seen.move_to_end(key)  # LRU freshness
            kept.append(t)
        self._evict_if_needed()
        return kept

    def key_count(self) -> int:
        return len(self._seen)
