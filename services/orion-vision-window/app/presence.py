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

**`identity_uncertain` is the mirror-image signal, added the same day the
unified turn started reading `subject`** (Juniper, 2026-08-26: "have Orion
say something ... this is not Juniper with confidence" -- a friendly
clarifying question, never asked when it IS Juniper, never asked when the
subsystem isn't running). `identity_confidence` (from `projection.
identity_confidence_from_artifact`, `"confirmed"` | `"uncertain"` | `None`)
carries the one distinction `identity_hint` above deliberately throws away:
a face that was DETECTED and did NOT match, vs. no signal at all. `observe()`
only ever sets `identity_uncertain=True` when a person is believed present
RIGHT NOW (`present_now`, not merely `"recent"`) -- asking about someone who
already stepped out of frame is exactly the awkwardness this exists to
avoid. The caller (WindowService) also owns a cross-process cooldown so this
question gets asked at most once per sit-down, not every turn -- see
`orion.situational.identity_ask_cooldown`.

**`identity_confirmed` is the POSITIVE fact, added 2026-08-29** (Juniper:
"orion never bites when they can't recognize me (eg I close the camera
lid). they are supposed to ask who dis"). `identity_uncertain` above can
only ever be True when a face was DETECTED and did not match -- closing a
laptop lid produces no frames, so no face, so `identity_confidence` is
`None` rather than `"uncertain"` and the whole ask was structurally
unreachable for exactly the case that prompted it. Asking "do I have a
fresh confirmed read of Juniper RIGHT NOW" is the question that actually
covers lid-closed, camera-off, person-absent AND unmatched-face, and it
needs a field that says "yes I do" rather than one that says "no I
explicitly don't" -- absence of `identity_uncertain` conflates "matched"
with "never looked". Consumers decide staleness themselves from the row's
own `updated_at` (see `orion.situational.perception_reader.fetch_presence`):
a camera that goes dark stops UPDATING this row rather than writing
`identity_confirmed=False` into it, so the row content alone cannot
distinguish "confirmed a moment ago" from "confirmed yesterday".

`state` mirrors `hub_presence`'s `active | idle | dormant` with camera-shaped
names: `present` (seen in the most recent window), `recent` (not seen just
now, but within `grace_sec` -- covers a bathroom break without flapping to
absent), `absent` (nothing for longer than that). `since_sec` is the number
nothing in this pipeline computed before this module: how long the CURRENT
state has held, which is the substrate for "you've been at that desk five
hours and it's 2am."

**Transitions are the replayable history (Temporal Self seam S1, 2026-10-14).**
The upsert above keeps only the latest state, so "someone was with me from
9:02 to 11:15" could not be replayed. `PresenceRegistry` now also yields one
append-only `vision_presence_transition` row per state change (`from_state`,
`to_state`, `occurred_at` = the tracker's own `state_since`), consumed by the
Temporal Self chronicle as `company` arcs. Two rows that are not plain state
changes, both stated so a reader is never surprised:

* `to_state='unknown'` when no window has been observed for a stream for longer
  than `unknown_after_sec` (the service's own `STALE_AFTER_MS`, 120 s; live
  2026-10-14 an empty room still flushes every 5-10 s, so 120 s of silence is a
  dark camera or a stopped pipeline, not an empty room). `occurred_at` is the
  LAST observation, the last moment anything was actually known. Also written
  for every stream on a graceful shutdown.
* `present -> present` once per company stretch, when identity becomes
  confirmed after the stretch opened (a face match usually lands a few seconds
  after the person is first seen). `occurred_at` is the confirming observation.

`from_state` is NULL on the first row after a process start: the producer
cannot know what happened while it was down, and the reducer reads that NULL as
"close whatever was open, unobserved". `label` is the enrolled identity label
and is stored ONLY when `identity_confirmed` is true, which for this history
means a fresh `probable` face match (a `possible` match is not a name) -- never a guess.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from datetime import datetime, timezone
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

    @property
    def state(self) -> str:
        return self._state

    @property
    def state_since(self) -> Optional[float]:
        return self._state_since

    def mark_unknown(self, at: float) -> None:
        """The stream went dark: nothing is known after ``at`` (the last
        observation). The next observe() then sees a real state change and
        stamps a fresh ``state_since``, so a resumed camera opens a new
        stretch instead of silently continuing the one before the gap."""
        self._state = "unknown"
        self._state_since = at

    def observe(
        self,
        believed_labels: frozenset[str],
        *,
        now: float,
        identity_hint: Optional[dict[str, Any]] = None,
        identity_confidence: Optional[str] = None,
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

        ``identity_confidence`` (``"confirmed"`` | `"uncertain"`` | `None``,
        same freshness contract as ``identity_hint`` -- caller's job) drives
        the separate ``identity_uncertain`` output field. Only ever True when
        `present_now` -- see the module docstring for why `"recent"` doesn't
        qualify.
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

        identity_uncertain = bool(present_now and identity_confidence == "uncertain")
        identity_confirmed = bool(present_now and identity_confidence == "confirmed")

        snapshot = {
            "state": self._state,
            "since_sec": round(now - self._state_since, 1),
            "last_seen_sec": last_seen_sec,
            "subject": subject,
            "identity_uncertain": identity_uncertain,
            "identity_confirmed": identity_confirmed,
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

    def __init__(
        self,
        *,
        grace_sec: float,
        write_min_interval_sec: float = 5.0,
        transitions_enabled: bool = False,
        unknown_after_sec: float = 120.0,
    ) -> None:
        self._grace_sec = grace_sec
        self._write_min_interval_sec = write_min_interval_sec
        self._trackers: dict[str, PresenceTracker] = {}
        self._last_write_at: dict[str, float] = {}
        # Transition history (seam S1). All per stream, all in process.
        self._transitions_enabled = bool(transitions_enabled)
        self._unknown_after_sec = float(unknown_after_sec)
        self._written_state: dict[str, str] = {}  # last to_state yielded
        self._last_transition_at: dict[str, float] = {}
        self._last_observed_at: dict[str, float] = {}
        # True once a confirmed row exists for the current company stretch
        # (present .. absent/unknown); reset when the stretch ends.
        self._stretch_confirmed: dict[str, bool] = {}
        self._pending: list[dict[str, Any]] = []
        self._closed = False  # set by close(): no transition after the shutdown unknown rows

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
        identity_confidence: Optional[str] = None,
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
            snapshot = self._tracker(stream_id).observe(
                believed_labels, now=ts, identity_hint=identity_hint, identity_confidence=identity_confidence
            )
            if self._transitions_enabled and not self._closed:
                self._note_transition(stream_id, snapshot, ts, identity_hint=identity_hint)
            last_write = self._last_write_at.get(stream_id, 0.0)
            due = (ts - last_write) >= self._write_min_interval_sec
            if due:
                self._last_write_at[stream_id] = ts
            return snapshot if due else None
        except Exception as exc:
            logger.warning("presence_record_failed stream=%s error=%s", stream_id, exc)
            return None

    # ------------------------------------------------------------------ transitions

    def _emit(
        self,
        stream_id: str,
        to_state: str,
        occurred_at: float,
        *,
        identity_confirmed: bool,
        label: Optional[str],
    ) -> None:
        # Two rows of one stream never share an instant: the reducer orders by
        # occurred_at, and a tie would be broken by the hashed id, not by order.
        prev_at = self._last_transition_at.get(stream_id)
        if prev_at is not None and occurred_at <= prev_at:
            occurred_at = prev_at + 0.001
        row = transition_row(
            stream_id,
            from_state=self._written_state.get(stream_id),
            to_state=to_state,
            occurred_at=occurred_at,
            identity_confirmed=identity_confirmed,
            label=label,
        )
        self._written_state[stream_id] = to_state
        self._last_transition_at[stream_id] = occurred_at
        self._pending.append(row)

    def _note_transition(
        self, stream_id: str, snapshot: dict[str, Any], ts: float, *, identity_hint: Optional[dict[str, Any]] = None
    ) -> None:
        tracker = self._trackers[stream_id]
        self._last_observed_at[stream_id] = ts
        state = snapshot["state"]
        label = _confirmed_label(snapshot, identity_hint)
        confirmed = label is not None
        written = self._written_state.get(stream_id)
        if state != written:
            since = tracker.state_since if tracker.state_since is not None else ts
            if state in ("absent", "unknown") or written in (None, "absent", "unknown"):
                # A stretch ended, or a new one may be starting: identity starts over.
                self._stretch_confirmed[stream_id] = False
            self._emit(stream_id, state, since, identity_confirmed=confirmed, label=label)
            if confirmed:
                self._stretch_confirmed[stream_id] = True
        elif state == "present" and confirmed and not self._stretch_confirmed.get(stream_id):
            self._emit(stream_id, "present", ts, identity_confirmed=True, label=label)
            self._stretch_confirmed[stream_id] = True

    def sweep_stale(self, now: float, *, force: bool = False) -> int:
        """Yield an ``unknown`` row for every stream silent longer than
        ``unknown_after_sec`` (every stream when ``force``, for shutdown).
        Returns how many were queued. Never raises."""
        if not self._transitions_enabled:
            return 0
        n = 0
        try:
            for stream_id, last in list(self._last_observed_at.items()):
                if self._written_state.get(stream_id) in (None, "unknown"):
                    continue
                if not force and (now - last) <= self._unknown_after_sec:
                    continue
                self._trackers[stream_id].mark_unknown(last)
                self._stretch_confirmed[stream_id] = False
                self._emit(stream_id, "unknown", last, identity_confirmed=False, label=None)
                n += 1
        except Exception as exc:
            logger.warning("presence_sweep_failed error=%s", exc)
        return n

    def close(self, now: float) -> int:
        """Shutdown: queue ``unknown`` for every stream, then refuse further transitions so a
        late flush cannot write a row after the stream was declared dark."""
        n = self.sweep_stale(now, force=True)
        self._closed = True
        return n

    def requeue(self, rows: list[dict[str, Any]]) -> None:
        """Put rows whose write failed back at the front of the queue (bounded; ids are
        deterministic and the insert is ON CONFLICT DO NOTHING, so a retry is safe)."""
        merged = list(rows) + self._pending
        dropped = max(0, len(merged) - MAX_PENDING_TRANSITIONS)
        if dropped:
            logger.warning("presence_transition_queue_full dropped=%d", dropped)
        self._pending = merged[-MAX_PENDING_TRANSITIONS:] if dropped else merged

    def has_pending(self) -> bool:
        return bool(self._pending)

    def drain_transitions(self) -> list[dict[str, Any]]:
        """Hand out (and forget) the transition rows queued since the last drain."""
        out, self._pending = self._pending, []
        return out

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

        from sqlalchemy import text

        engine = _engine_for(postgres_uri)
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
    except Exception as exc:
        logger.warning("embodied_presence_write_failed stream=%s error=%s", stream_id, exc)


# ---------------------------------------------------------------------- shared engine

_ENGINES: dict[str, Any] = {}
_ENGINES_LOCK = threading.Lock()


def _engine_for(postgres_uri: str) -> Any:
    """One pooled engine per URI for the process, shared by the snapshot
    upsert and the transition insert (both run in worker threads; a
    SQLAlchemy engine is thread-safe). Small pool: at most a couple of
    writes are ever in flight."""
    with _ENGINES_LOCK:
        engine = _ENGINES.get(postgres_uri)
        if engine is None:
            from sqlalchemy import create_engine

            engine = create_engine(postgres_uri, pool_pre_ping=True, pool_size=2, max_overflow=2)
            _ENGINES[postgres_uri] = engine
        return engine


# ---------------------------------------------------------------------- transitions

MAX_PENDING_TRANSITIONS = 256


def _confirmed_label(snapshot: dict[str, Any], identity_hint: Optional[dict[str, Any]] = None) -> Optional[str]:
    """The enrolled label, only when identity is confirmed RIGHT NOW by a
    ``probable`` match. ``identity_confirmed`` upstream also counts a
    ``possible`` match (the weaker similarity band, orion-vision-host
    identity_gallery), which is not strong enough to put a name on a stretch
    of company, so the chronology treats it as unconfirmed."""
    if not snapshot.get("identity_confirmed"):
        return None
    if not identity_hint or identity_hint.get("state") != "probable":
        return None
    subject = str(snapshot.get("subject") or "").strip()
    return subject if subject and subject not in ("unknown", "none") else None


def transition_id_for(presence_id: str, occurred_at: datetime, to_state: str) -> str:
    """Deterministic on (presence_id, occurred_at, to_state): a retried or
    duplicated write is the same row (ON CONFLICT DO NOTHING)."""
    raw = f"{presence_id}|{occurred_at.astimezone(timezone.utc).isoformat()}|{to_state}".encode()
    return hashlib.sha256(raw).hexdigest()[:32]


def transition_row(
    presence_id: str,
    *,
    from_state: Optional[str],
    to_state: str,
    occurred_at: float,
    identity_confirmed: bool,
    label: Optional[str],
) -> dict[str, Any]:
    at = datetime.fromtimestamp(round(occurred_at, 6), tz=timezone.utc)
    confirmed = bool(identity_confirmed and label)
    return {
        "transition_id": transition_id_for(presence_id, at, to_state),
        "presence_id": presence_id,
        "from_state": from_state,
        "to_state": to_state,
        "occurred_at": at,
        "identity_confirmed": confirmed,
        "label": label if confirmed else None,
    }


INSERT_TRANSITION_SQL = """
INSERT INTO vision_presence_transition
    (transition_id, presence_id, from_state, to_state, occurred_at, identity_confirmed, label)
VALUES (:transition_id, :presence_id, :from_state, :to_state, :occurred_at, :identity_confirmed, :label)
ON CONFLICT (transition_id) DO NOTHING
"""


def write_transitions_to_postgres(rows: list[dict[str, Any]], *, postgres_uri: str) -> int:
    """Blocking append. Callers MUST run this off the event loop. Never raises:
    a failed write is logged and the rows are lost (the state machine and the
    snapshot upsert are unaffected). Returns rows attempted on success, 0 on
    failure."""
    if not postgres_uri or not rows:
        return 0
    try:
        from sqlalchemy import text

        with _engine_for(postgres_uri).begin() as conn:
            conn.execute(text(INSERT_TRANSITION_SQL), rows)
        return len(rows)
    except Exception as exc:
        logger.warning(
            "presence_transition_write_failed rows=%d streams=%s error=%s",
            len(rows), ",".join(sorted({r["presence_id"] for r in rows})), exc,
        )
        return 0
