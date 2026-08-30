"""Durable arm/clear state for the vision liveness watcher.

Why this exists. `VisionLivenessWatcher` held `_alerting`, `_failing_since` and
`_last_alert_at` in memory only, so any restart reset it to "not alerting".
Recovery is emitted exclusively on the `alerting -> clear` transition, so a
restart between the alert and the recovery loses the recovery permanently.

That is not theoretical. Live on 2026-08-29: eight `vision_blind` attention
records since 2026-08-21 and **zero `vision_recovered` records, ever**. The
signature is in the alert bodies -- three alerts that day (20:25, 21:00, 22:13)
each reported failing "for 3m" with ~88 samples. 3m is exactly `sustain_sec`,
so the sustain clock was fresh every time. A live process structurally cannot
re-alert (`if self._alerting: ... return` blocks it) and a genuine clear would
have left a recovery record. A restarted one re-arms from scratch, which is
what those three identical bodies are.

Consequences of the missing recovery, all confirmed live:
  - the attention store accumulates alerts that never close;
  - `orion-actions`' capability-gap journal seed (PR #1965) saw nine
    permanently-open vision episodes going back to 08-21;
  - nothing downstream can tell "still blind" from "blind, then fine, then
    blind again".

Design notes:

*Wall clock on disk, monotonic in memory.* The watcher uses `time.monotonic()`,
which is process-relative and meaningless across a restart. Persisting it
directly would restore a cooldown deadline from another process's epoch, so
timestamps are converted to wall clock on save and back on load.

*Samples are deliberately not persisted.* The deque is a rolling
`window_sec` view; after a restart it is stale by definition and the watcher
should rebuild it from live traffic.

*Restoring `alerting=True` is the point, not a risk.* If the service comes back
and traffic is healthy, the first records clear the state and emit exactly the
recovery that was being lost.

*Nothing here may stop the service from seeing.* Every operation swallows its
own errors -- a missing, unreadable, corrupt, or unwritable state file degrades
to clean in-memory behaviour with a log line, matching the policy already
stated in `liveness.build_watcher_or_default`.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("orion-vision-host.liveness_state")

STATE_VERSION = 1

# A restored alert older than this is dropped rather than trusted. Guards the
# case where the service was down for days: re-arming from week-old state would
# emit a recovery for an incident nobody remembers, and the sustain clock would
# be meaningless. One day is far longer than any real outage here and far
# shorter than "stale".
MAX_STATE_AGE_SEC = 86_400.0


@dataclass(frozen=True)
class PersistedLivenessState:
    """Watcher state in WALL-CLOCK seconds (`time.time()`), not monotonic."""

    alerting: bool = False
    failing_since_wall: Optional[float] = None
    last_alert_at_wall: Optional[float] = None

    def to_json(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "alerting": bool(self.alerting),
            "failing_since_wall": self.failing_since_wall,
            "last_alert_at_wall": self.last_alert_at_wall,
            "saved_at_wall": time.time(),
        }


def _coerce_ts(raw: Any) -> Optional[float]:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
        # Reject nonsense rather than propagate it into clock arithmetic.
        return value if value > 0 else None
    return None


class LivenessStateStore:
    """Atomic, best-effort JSON state file. Never raises."""

    def __init__(self, path: str) -> None:
        self._path = str(path)

    @property
    def path(self) -> str:
        return self._path

    def load(self, *, now_wall: Optional[float] = None) -> PersistedLivenessState:
        now = float(now_wall if now_wall is not None else time.time())
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            return PersistedLivenessState()
        except Exception as exc:
            logger.warning("liveness state unreadable (%s): %s -- starting clean", self._path, exc)
            return PersistedLivenessState()

        try:
            version_ok = isinstance(raw, dict) and int(raw.get("version") or 0) == STATE_VERSION
        except (TypeError, ValueError):
            # A non-numeric "version" (e.g. a list) used to raise straight out of
            # load(), contradicting this method's "Never raises" contract.
            version_ok = False
        if not version_ok:
            logger.warning("liveness state version mismatch in %s -- starting clean", self._path)
            return PersistedLivenessState()

        saved_at = _coerce_ts(raw.get("saved_at_wall"))
        if saved_at is None or (now - saved_at) > MAX_STATE_AGE_SEC or saved_at > now + 60.0:
            # Also rejects a file from the future, which a clock step can produce
            # and which would otherwise park a cooldown deadline permanently ahead.
            logger.warning("liveness state stale or future-dated in %s -- starting clean", self._path)
            return PersistedLivenessState()

        return PersistedLivenessState(
            alerting=bool(raw.get("alerting")),
            failing_since_wall=_coerce_ts(raw.get("failing_since_wall")),
            last_alert_at_wall=_coerce_ts(raw.get("last_alert_at_wall")),
        )

    def save(self, state: PersistedLivenessState) -> bool:
        """Atomic write via tmp+rename. Returns False on failure, never raises."""
        try:
            directory = os.path.dirname(self._path) or "."
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=directory, prefix=".liveness_state.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(state.to_json(), fh)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, self._path)
            except Exception:
                # Never leave a partial temp file behind on a full/read-only volume.
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            return True
        except Exception as exc:
            logger.warning("liveness state not saved (%s): %s", self._path, exc)
            return False
