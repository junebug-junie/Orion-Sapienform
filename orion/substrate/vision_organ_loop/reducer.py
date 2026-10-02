"""Fold the eye's own window reports into one organ reading.

Two readings reach the field (node:substrate.vision_organ -> capability:vision):

``vision_frame_staleness`` -- can Orion see at all. The staleness of the
FRESHEST stream in the window (``min`` over streams), via the same
``vision_channel_staleness_pressure`` ramp the old artifact tick used (0.0 inside
15 s, 1.0 at 60 s). Rest point exactly 0.0 while any camera delivers. 1.0 when no
stream delivers, when the router lists no stream at all, and -- through
``vision_organ_silence_receipt`` -- when the router stops reporting.
Per-stream staleness (e.g. the carbon webcam going dark while cam0 runs) is on
the projection and the receipt, not in this number: a laptop webcam is off
whenever the laptop is, and nothing models that day-shape yet, so a ``max``
over streams would pin capability:vision at alarm every evening.

``vision_processing_failure_pressure`` -- of the frames handed to the vision host,
how many came back without a usable answer (timeout, invalid reply, ok=false).
Rolling over the last 600 s of windows, ``failures / max(attempts, 10)``, 0.0
until 2 failures -- the RPC delivery bridge's ``hop_pressure`` rule. The worse
of the pooled share and the worst single stream's, so cam0's volume (~120 tasks
per 10 min) cannot dilute a stream whose every task fails. None (no hint, held,
then expired by the digester) when nothing was dispatched in the span.

Record-only (never pressure): detection and caption yield. A sustained zero
object count is equally consistent with a blinded lens and an empty dark room
(prediction_error.py::perceptual_blindness_pressure).
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.reduction_receipt import ProjectionUpdateV1, ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1
from orion.schemas.vision_organ_projection import (
    ORGAN_REPORTING,
    ORGAN_SILENT,
    VisionOrganProjectionV1,
    VisionOrganStreamStateV1,
    VisionOrganWindowCountV1,
)
from orion.substrate.ids import stable_delta_id, stable_receipt_id
from orion.substrate.rpc_delivery import RpcDeliveryConfig, hop_pressure

from .constants import (
    VISION_ORGAN_NODE_ID,
    VISION_ORGAN_PROJECTION_ID,
    VISION_ORGAN_REDUCER_ID,
    VISION_ORGAN_SOURCE_SERVICE,
    VISION_ORGAN_TARGET_KIND,
)
from .extract import extract_vision_organ_window, parse_vision_organ_trace_id

_RPC = RpcDeliveryConfig()
FAILURE_WINDOW_SEC: float = _RPC.window_s
FAILURE_MIN_DENOMINATOR: int = _RPC.min_denominator
FAILURE_MIN_COUNT: int = _RPC.min_timeouts
_MAX_RECENT_WINDOWS = 400
POOLED_SCOPE = "organ"


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _utc_now(now: datetime | None) -> datetime:
    return _aware(now) if now is not None else datetime.now(timezone.utc)


def _noop(events: list[GrammarEventV1], clock: datetime, warnings: list[str] | None = None) -> ReductionReceiptV1:
    noop_ids = [e.event_id for e in events]
    return ReductionReceiptV1(
        receipt_id=stable_receipt_id(
            reducer_id=VISION_ORGAN_REDUCER_ID,
            accepted_event_ids=[],
            rejected_event_ids=[],
            merged_event_ids=[],
            noop_event_ids=noop_ids,
        ),
        noop_event_ids=noop_ids,
        warnings=list(warnings or []),
        created_at=clock,
    )


def fold_counts(
    history: list[VisionOrganWindowCountV1],
    new: list[VisionOrganWindowCountV1],
    *,
    window_sec: float = FAILURE_WINDOW_SEC,
) -> list[VisionOrganWindowCountV1]:
    """Add this window's per-stream counts (a replayed copy of the same
    window/stream replaces, never double counts) and drop anything older than
    ``window_sec`` before the newest. Oldest first."""
    keys = {(c.window_id, c.stream_id) for c in new}
    kept = [c for c in history if (c.window_id, c.stream_id) not in keys] + list(new)
    if not kept:
        return []
    kept.sort(key=lambda c: _aware(c.window_end))
    cutoff = _aware(kept[-1].window_end) - timedelta(seconds=window_sec)
    kept = [c for c in kept if _aware(c.window_end) > cutoff]
    return kept[-_MAX_RECENT_WINDOWS:]


def failure_reading(history: list[VisionOrganWindowCountV1]) -> dict[str, Any]:
    ok = sum(c.replies_ok for c in history)
    failed = sum(c.failed for c in history)
    reading: dict[str, Any] = {
        "pressure": None,
        "scope": None,
        "attempted": ok + failed,
        "failed": failed,
        "window_sec": FAILURE_WINDOW_SEC,
        "min_denominator": FAILURE_MIN_DENOMINATOR,
        "min_failures": FAILURE_MIN_COUNT,
    }
    if ok + failed <= 0:
        return reading
    best = hop_pressure(failed, ok, FAILURE_MIN_DENOMINATOR, FAILURE_MIN_COUNT)
    scope = POOLED_SCOPE
    per_stream: dict[str, tuple[int, int]] = {}
    for c in history:
        s_ok, s_failed = per_stream.get(c.stream_id, (0, 0))
        per_stream[c.stream_id] = (s_ok + c.replies_ok, s_failed + c.failed)
    for stream in sorted(per_stream):
        s_ok, s_failed = per_stream[stream]
        value = hop_pressure(s_failed, s_ok, FAILURE_MIN_DENOMINATOR, FAILURE_MIN_COUNT)
        if value > best:
            best, scope = value, stream
    reading["pressure"] = min(1.0, float(best))
    reading["scope"] = scope
    return reading


def _stream_brief(state: VisionOrganStreamStateV1) -> dict[str, Any]:
    return {
        "status": state.status,
        "configured": state.configured,
        "frame_staleness": state.frame_staleness,
        "last_frame_age_sec": state.last_frame_age_sec,
        "frames": state.frames,
        "dispatched": state.dispatched,
        "replies_ok": state.replies_ok,
        "failed": state.failed,
        "failure_classes": dict(state.failure_classes),
        "detect_replies": state.detect_replies,
        "objects": state.objects,
        "caption_requested": state.caption_requested,
        "captions": state.captions,
    }


def _organ_delta(
    *,
    before: VisionOrganProjectionV1 | None,
    after: VisionOrganProjectionV1,
    caused_by: list[str],
    failure: dict[str, Any] | None,
) -> StateDeltaV1:
    hints: dict[str, float] = {}
    if after.vision_frame_staleness is not None:
        hints["vision_frame_staleness"] = float(after.vision_frame_staleness)
    if after.vision_processing_failure_pressure is not None:
        hints["vision_processing_failure_pressure"] = float(after.vision_processing_failure_pressure)
    operation = "create" if before is None or before.last_window_id is None else "update"
    payload = {
        "node_id": VISION_ORGAN_NODE_ID,
        "status": after.status,
        "router": after.router,
        "window_id": after.last_window_id,
        "vision_frame_staleness": after.vision_frame_staleness,
        "vision_processing_failure_pressure": after.vision_processing_failure_pressure,
        "streams": {sid: _stream_brief(s) for sid, s in sorted(after.streams.items())},
        "failure_window": failure,
        "pressure_hints": hints,
    }
    return StateDeltaV1(
        delta_id=stable_delta_id(
            reducer_id=VISION_ORGAN_REDUCER_ID,
            target_projection=VISION_ORGAN_PROJECTION_ID,
            target_kind=VISION_ORGAN_TARGET_KIND,
            target_id=VISION_ORGAN_NODE_ID,
            operation=operation,
            caused_by_event_ids=caused_by,
        ),
        target_projection=VISION_ORGAN_PROJECTION_ID,
        target_kind=VISION_ORGAN_TARGET_KIND,
        target_id=VISION_ORGAN_NODE_ID,
        operation=operation,
        before=None
        if before is None
        else {
            "status": before.status,
            "vision_frame_staleness": before.vision_frame_staleness,
            "vision_processing_failure_pressure": before.vision_processing_failure_pressure,
        },
        after=payload,
        caused_by_event_ids=caused_by,
        reducer_id=VISION_ORGAN_REDUCER_ID,
    )


def reduce_vision_organ_trace_events(
    *,
    events: list[GrammarEventV1],
    projection: VisionOrganProjectionV1,
    now: datetime | None = None,
) -> tuple[VisionOrganProjectionV1, ReductionReceiptV1]:
    """One router window trace (or the part of it in this batch) -> one receipt.

    Stream atoms update the projection's per-stream states. The organ reading and
    its delta are produced only when the window's closing atom is in hand, over the
    streams reported in that window -- a batch boundary can split a window but can
    never produce a reading over half its streams."""
    clock = _utc_now(now)
    if not events:
        return projection, _noop([], clock)
    trace_id = events[0].trace_id or ""
    if not parse_vision_organ_trace_id(trace_id):
        return projection, _noop(events, clock)
    if any(e.provenance.source_service != VISION_ORGAN_SOURCE_SERVICE for e in events):
        return projection, _noop(events, clock)
    try:
        window = extract_vision_organ_window(events, now=clock)
    except ValueError as exc:
        return projection, _noop(events, clock, warnings=[str(exc)])

    before = deepcopy(projection)
    updated = deepcopy(projection)
    updated.projection_id = VISION_ORGAN_PROJECTION_ID
    updated.generated_at = clock
    updated.router = window.router
    updated.streams.update(window.streams)
    updated.recent_windows = fold_counts(updated.recent_windows, list(window.counts.values()))

    accepted = list(window.stream_event_ids)
    deltas: list[StateDeltaV1] = []
    updates: list[ProjectionUpdateV1] = []
    if window.completed:
        accepted.append(window.closing_event_id or "")
        this_window = {sid: s for sid, s in updated.streams.items() if s.window_id == window.window_id}
        # Streams no longer reported (dropped from config and not seen) leave the projection.
        updated.streams = this_window
        updated.status = ORGAN_REPORTING
        updated.last_window_id = window.window_id
        updated.last_window_end = window.window_end
        updated.vision_frame_staleness = (
            min(s.frame_staleness for s in this_window.values()) if this_window else 1.0
        )
        failure = failure_reading(updated.recent_windows)
        updated.vision_processing_failure_pressure = failure["pressure"]
        caused_by = [e.event_id for e in events]
        deltas.append(_organ_delta(before=before, after=updated, caused_by=caused_by, failure=failure))
        updates.append(
            ProjectionUpdateV1(
                projection_kind=VISION_ORGAN_TARGET_KIND,
                projection_id=VISION_ORGAN_PROJECTION_ID,
                node_id=VISION_ORGAN_NODE_ID,
                operation=deltas[0].operation,
            )
        )

    receipt = ReductionReceiptV1(
        receipt_id=stable_receipt_id(
            reducer_id=VISION_ORGAN_REDUCER_ID,
            accepted_event_ids=[a for a in accepted if a],
            rejected_event_ids=[],
            merged_event_ids=[],
            noop_event_ids=[],
        ),
        accepted_event_ids=[a for a in accepted if a],
        state_deltas=deltas,
        projection_updates=updates,
        created_at=clock,
    )
    return updated, receipt


def silence_age_seconds(
    projection: VisionOrganProjectionV1 | None, *, now: datetime, process_started_at: datetime
) -> float:
    """Seconds since the router's last completed window; with none on record, since
    this process started (a host reboot takes the router and the substrate down
    together, and an unbounded "never heard" must not read as calm)."""
    clock = _aware(now)
    last = projection.last_window_end if projection is not None else None
    anchor = _aware(last) if last is not None else _aware(process_started_at)
    return max(0.0, (clock - anchor).total_seconds())


def vision_organ_silence_receipt(
    projection: VisionOrganProjectionV1,
    *,
    now: datetime,
    silent_for_sec: float,
) -> tuple[VisionOrganProjectionV1, ReductionReceiptV1]:
    """The router stopped reporting: write staleness 1.0 on a clock.

    The failure reading is withdrawn (no hint): with no reports there is nothing to
    count, and the digester expires the channel rather than holding a stale share.
    Each write carries a time-stamped synthetic cause so its delta id is new -- the
    digester applies a delta id once, ever."""
    clock = _aware(now)
    before = deepcopy(projection)
    updated = deepcopy(projection)
    updated.projection_id = VISION_ORGAN_PROJECTION_ID
    updated.generated_at = clock
    updated.status = ORGAN_SILENT
    updated.vision_frame_staleness = 1.0
    updated.vision_processing_failure_pressure = None
    stamp = f"vision_organ_silence:{clock.strftime('%Y%m%dT%H%M%S')}"
    delta = _organ_delta(
        before=before,
        after=updated,
        caused_by=[stamp],
        failure={"silent_for_sec": round(silent_for_sec, 1)},
    )
    receipt = ReductionReceiptV1(
        receipt_id=stable_receipt_id(
            reducer_id=VISION_ORGAN_REDUCER_ID,
            accepted_event_ids=[],
            rejected_event_ids=[],
            merged_event_ids=[],
            noop_event_ids=[],
            emission_id=stamp,
        ),
        state_deltas=[delta],
        projection_updates=[
            ProjectionUpdateV1(
                projection_kind=VISION_ORGAN_TARGET_KIND,
                projection_id=VISION_ORGAN_PROJECTION_ID,
                node_id=VISION_ORGAN_NODE_ID,
                operation=delta.operation,
            )
        ],
        warnings=[f"vision organ silent for {silent_for_sec:.0f}s"],
        created_at=clock,
    )
    return updated, receipt
