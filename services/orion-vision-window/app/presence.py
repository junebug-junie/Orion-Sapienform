"""Embodied presence — "is someone at the camera, and for how long" as a
self-state observable.

Mirrors `services/orion-hub/scripts/hub_presence.py`'s shape exactly: a
per-key state machine kept in process, a small snapshot mirrored to a
single-row Postgres upsert (`substrate_embodied_presence`, keyed by
`presence_id = stream_id` here instead of the fixed `'hub'`), best-effort and
rate-limited so a write can never cost a window flush.

**Reuses SceneBeliefTracker's output, does not re-smooth.** The input to this
module is `believed_labels` -- the vote-gated, flicker-resistant set
`scene_belief.py` already computes (`WINDOW_BELIEF_VOTE_N=3`,
`ENTER_VOTES=3`, `EXIT_VOTES=0` live). Building a second smoothing layer on
top of raw per-window detections would duplicate hysteresis this service
already pays for and already tuned.

**`subject` narrows from `"unknown"` only on a fresh, hedged identity hint.**
2026-08-26: `identity_face` shipped (PR #1886/#1890) but is dispatched
separately by `orion-vision-frame-router` and reaches this service over its
own dedicated channel (`orion-vision-host`'s `CHANNEL_VISIONHOST_IDENTITY_PUB`
-- see `main.py`'s `_consume_identity`), not through the `believed_labels`
this tracker already reads. `person` alone still only proves *a* person, not
*Juniper*; `observe()`'s `identity_hint` param is how a real identity
hypothesis narrows that, and only when it is itself hedged as `"probable"`
or `"possible"` (never `"unsure"`, never absent) and the caller has already
checked it is fresh enough to speak to *now*. No hint, or a stale/unsure one,
and this stays exactly the `"unknown"` the design doc's own honesty
discipline requires.

`state` mirrors `hub_presence`'s `active | idle | dormant` with camera-shaped
names: `present` (seen in the most recent window), `recent` (not seen just
now, but within `grace_sec` -- covers a bathroom break without flapping to
absent), `absent` (nothing for longer than that). `since_sec` is the number
nothing in this pipeline computed before this module: how long the CURRENT
state has held, which is the substrate for "you've been at that desk five
hours and it's 2am."
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger("orion-vision-window.presence")

# Default subject label. GroundingDINO's default_prompts (config/
# vision_profiles.yaml) always includes "person", so this does not need to be
# per-stream configurable yet -- it becomes a real question once a stream
# tracks something other than a person (an object-permanence target, say),
# not before.
_DEFAULT_SUBJECT_LABEL = "person"


class PresenceTracker:
    """One stream's presence state machine. Pure logic, no I/O."""

    def __init__(self, *, grace_sec: float, subject_label: str = _DEFAULT_SUBJECT_LABEL) -> None:
        self._grace_sec = float(grace_sec)
        self._subject_label = subject_label
        self._state = "absent"
        self._state_since: Optional[float] = None
        self._last_present_ts: Optional[float] = None
        self._last_snapshot: Optional[dict[str, Any]] = None

    def last_snapshot(self) -> Optional[dict[str, Any]]:
        """Read-only: the most recent observe() result, or None before the
        first call. Never triggers a new observation."""
        return self._last_snapshot

    def observe(
        self,
        believed_labels: frozenset[str],
        *,
        now: float,
        identity_hint: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """``identity_hint``, when given, is ``{"subject": ..., "state": ...}``
        from a FRESH identity_face hypothesis -- staleness is the caller's
        job (this method has no clock of its own to judge freshness by; see
        WindowService's own age gate before it calls this). Only ever
        narrows `"unknown"` to a real subject, and only when the hint's own
        state is `"probable"` or `"possible"` -- an `"unsure"` hypothesis is
        the same honesty-preserving no-op as no hint at all. Never overrides
        `"none"` (nobody believed present): an identity hint that arrived a
        beat late for a person who already left is not evidence anyone is
        here now.
        """
        present_now = self._subject_label in believed_labels
        if present_now:
            self._last_present_ts = now

        last_seen_sec = (
            round(now - self._last_present_ts, 1) if self._last_present_ts is not None else None
        )

        if present_now:
            new_state = "present"
        elif last_seen_sec is not None and last_seen_sec <= self._grace_sec:
            new_state = "recent"
        else:
            new_state = "absent"

        if new_state != self._state or self._state_since is None:
            self._state = new_state
            self._state_since = now

        subject = "unknown" if present_now or last_seen_sec is not None else "none"
        if (
            subject == "unknown"
            and identity_hint
            and identity_hint.get("state") in ("probable", "possible")
            and identity_hint.get("subject")
            and identity_hint["subject"] != "unknown"
        ):
            subject = str(identity_hint["subject"])

        snapshot = {
            "state": self._state,
            "since_sec": round(now - self._state_since, 1),
            "last_seen_sec": last_seen_sec,
            "subject": subject,
        }
        self._last_snapshot = snapshot
        return snapshot


class PresenceRegistry:
    """Per-stream trackers plus the rate-limited Postgres mirror.

    `record()` is called from the window service's main loop right after the
    scene-belief transition it depends on; the Postgres write happens off that
    path in a background thread so a slow or unreachable database can never
    delay a window flush.
    """

    def __init__(self, *, grace_sec: float, write_min_interval_sec: float = 5.0) -> None:
        self._grace_sec = grace_sec
        self._write_min_interval_sec = write_min_interval_sec
        self._trackers: dict[str, PresenceTracker] = {}
        self._last_write_at: dict[str, float] = {}

    def _tracker(self, stream_id: str) -> PresenceTracker:
        if stream_id not in self._trackers:
            self._trackers[stream_id] = PresenceTracker(grace_sec=self._grace_sec)
        return self._trackers[stream_id]

    def record(
        self,
        stream_id: str,
        believed_labels: frozenset[str],
        *,
        now: Optional[float] = None,
        identity_hint: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Update the tracker and return the snapshot ONLY when a Postgres
        write is due (rate-limited) -- unchanged contract, never raises.

        For the ALWAYS-fresh snapshot (presence's own current state on
        every flush, not just the write cadence) call current_snapshot()
        right after this; record() already updated the tracker for this
        call regardless of what it returns. Note: council's window
        evidence (summary.evidence.identity_hypothesis) does NOT go
        through this registry at all -- it reads WindowService's own
        _identity_by_stream directly (main.py's _get_fresh_identity_hint).
        This registry is presence-only; current_snapshot() exists for any
        other same-flush consumer of presence's live state (a debug
        surface, say), not for that specific field.
        """
        try:
            ts = float(now if now is not None else time.time())
            snapshot = self._tracker(stream_id).observe(believed_labels, now=ts, identity_hint=identity_hint)
            last_write = self._last_write_at.get(stream_id, 0.0)
            due = (ts - last_write) >= self._write_min_interval_sec
            if due:
                self._last_write_at[stream_id] = ts
            return snapshot if due else None
        except Exception as exc:
            logger.warning("presence_record_failed stream=%s error=%s", stream_id, exc)
            return None

    def current_snapshot(self, stream_id: str) -> Optional[dict[str, Any]]:
        """The tracker's most recent observe() result for this stream,
        regardless of record()'s write-rate-limit gate. None if record()
        has never been called for this stream_id yet. Never triggers a new
        observation -- call record() first for this flush."""
        tracker = self._trackers.get(stream_id)
        return tracker.last_snapshot() if tracker is not None else None


def write_snapshot_to_postgres(stream_id: str, snapshot: dict[str, Any], *, postgres_uri: str) -> None:
    """Blocking upsert. Callers MUST run this off the event loop.

    Same shape as `hub_presence._write_snapshot_to_postgres`: `presence_json`
    is a JSONB blob rather than typed columns, matching the adopted precedent
    for this exact "self-state observable, single-row upsert" pattern rather
    than inventing a second one.
    """
    if not postgres_uri:
        return
    try:
        import json

        from sqlalchemy import create_engine, text

        engine = create_engine(postgres_uri, pool_pre_ping=True)
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_embodied_presence
                            (presence_id, generated_at, presence_json, updated_at)
                        VALUES (:presence_id, now(), CAST(:presence_json AS jsonb), now())
                        ON CONFLICT (presence_id) DO UPDATE SET
                            generated_at = EXCLUDED.generated_at,
                            presence_json = EXCLUDED.presence_json,
                            updated_at = EXCLUDED.updated_at
                        """
                    ),
                    {"presence_id": stream_id, "presence_json": json.dumps(snapshot)},
                )
        finally:
            engine.dispose()
    except Exception as exc:
        logger.warning("embodied_presence_write_failed stream=%s error=%s", stream_id, exc)
