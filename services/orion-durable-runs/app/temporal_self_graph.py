"""temporal_self.update: Orion's regulation as a LangGraph thread (Temporal Self rev 4, PR #2369,
order 3 "R2/R3 core").

One invocation per (coalesced) event on the day's thread ``temporal_self:orion:<local date>``:

    ingest -> regulate -> chronicle -> done

* regulate: reads E1/S1/S2 (``deps.read_inputs``), folds them through the pure, I/O-free
  ``orion.regulation.arousal.classify_arousal`` with the previous reading carried by the
  checkpoint (hysteresis lives there), embeds the latest drive readings verbatim, projects the
  ``RegulationStateV1`` to Redis, and records one ``arousal_transition`` event on a level change.

* chronicle (patch 3): runs the pure chronology reducer (``orion.temporal_self``) live through
  ``deps.chronicle`` (``app.temporal_self_chronicle.Chronicler``): reads every bound source up to
  ``now - TEMPORAL_SELF_READ_LAG_SEC``, folds, and commits arcs / closed days / frame / cursors /
  reducer state in one transaction per window. The reducer's state lives in its own table, never
  in this checkpoint (it reaches ~1.2 MB); the checkpoint keeps a one-line summary. A chronicle
  failure is a warning on this step and never touches the regulation reading.

No LLM, no GPU lease. Nothing reads arousal or the chronology yet (spec order 6 and patch 4).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from orion.regulation.arousal import classify_arousal
from orion.schemas.drive_reading import DriveReadingV1
from orion.schemas.regulation import (
    TEMPORAL_SELF_WORKFLOW,
    ArousalInputsV1,
    ArousalReadingV1,
    RegulationStateV1,
)

logger = logging.getLogger("orion-durable-runs.temporal_self_graph")

NODES = ("ingest", "regulate", "chronicle", "done")


class TemporalSelfGraphState(TypedDict, total=False):
    workflow: str                # always TEMPORAL_SELF_WORKFLOW; the runner's resume sweep skips it
    thread_id: str
    day_id: str
    event: dict                  # {event_id, kind, correlation_id, at, juniper}
    regulation: dict             # RegulationStateV1 json, carried across events by the checkpoint
    last_inputs: dict            # ArousalInputsV1 json: seeds the cabinet reflex's hysteresis
    last_juniper_turn_at: Optional[str]   # newest Juniper turn seen on the bus (ISO)
    transition: Optional[dict]   # {"from", "to"} when this step changed the level (transient)
    warnings: list
    chronicle: Optional[dict]    # Chronicler.step() summary: watermark, lag, windows, error


@dataclass
class TemporalSelfDeps:
    read_inputs: Callable[[datetime, Optional[ArousalInputsV1], Optional[datetime]], Awaitable[ArousalInputsV1]]
    read_drives: Callable[[datetime], Awaitable[tuple[list[DriveReadingV1], list[str]]]]
    project: Callable[[RegulationStateV1], Awaitable[None]]
    record_transition: Callable[[Optional[ArousalReadingV1], ArousalReadingV1, str], Awaitable[bool]]
    now: Callable[[], datetime]
    arousal_enabled: bool = True
    engaged_minutes: float = 45.0
    gpu_queue_floor: int = 2
    gpu_sustain_sec: float = 300.0
    clear_sec: float = 600.0
    max_prev_gap_sec: float = 360.0
    # Patch 3: one chronology step (``Chronicler.step``); None = TEMPORAL_SELF_CHRONICLE_ENABLED off.
    chronicle: Optional[Callable[[], Awaitable[dict]]] = None
    # Hard ceiling on one chronicle call. The chronicler itself stops starting windows after 30 s
    # and every statement has a 30 s timeout; this catches anything else (review finding: a hung
    # read would otherwise hold every queued tick and chat turn, and the regulate checkpoint).
    chronicle_timeout_sec: float = 120.0


def _dt(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _prev_reading(regulation: Optional[dict]) -> Optional[ArousalReadingV1]:
    """The checkpointed reading, or None if the stored shape no longer validates (a stale
    checkpoint must not crash-loop the writer; it costs one step of hysteresis memory)."""
    if not regulation:
        return None
    try:
        return ArousalReadingV1.model_validate(regulation.get("arousal") or {})
    except Exception:  # noqa: BLE001
        logger.warning("regulation_prev_reading_invalid", exc_info=True)
        return None


def _prev_inputs(raw: Optional[dict]) -> Optional[ArousalInputsV1]:
    if not raw:
        return None
    try:
        return ArousalInputsV1.model_validate(raw)
    except Exception:  # noqa: BLE001
        return None


def build_temporal_self_graph(deps: TemporalSelfDeps, checkpointer: Any):
    async def ingest(state: TemporalSelfGraphState) -> dict:
        ev = dict(state.get("event") or {})
        last_turn = state.get("last_juniper_turn_at")
        if ev.get("kind") == "chat_turn" and ev.get("juniper") and ev.get("at"):
            at = _dt(ev["at"])
            if at is not None and (last_turn is None or at > (_dt(last_turn) or at)):
                last_turn = at.isoformat()
        return {"workflow": TEMPORAL_SELF_WORKFLOW, "transition": None, "warnings": [],
                "last_juniper_turn_at": last_turn}

    async def regulate(state: TemporalSelfGraphState) -> dict:
        now = deps.now()
        prev = _prev_reading(state.get("regulation"))
        # The cabinet reflex's hysteresis seed obeys the same gap rule as the previous reading:
        # after an outage, an old "hot" must not hold the reflex hot (review finding).
        prev_inputs = _prev_inputs(state.get("last_inputs"))
        if prev_inputs is not None:
            gap = (now - prev_inputs.observed_at).total_seconds()
            if gap < 0 or gap > deps.max_prev_gap_sec:
                prev_inputs = None
        inputs = await deps.read_inputs(now, prev_inputs, _dt(state.get("last_juniper_turn_at")))
        reading = classify_arousal(
            prev, inputs, enabled=deps.arousal_enabled, engaged_minutes=deps.engaged_minutes,
            gpu_queue_floor=deps.gpu_queue_floor, gpu_sustain_sec=deps.gpu_sustain_sec,
            clear_sec=deps.clear_sec, max_prev_gap_sec=deps.max_prev_gap_sec)
        drives, warnings = await deps.read_drives(now)
        warnings = list(warnings)
        transition = None
        # A new level episode began: the level changed, or the previous reading was too old to
        # carry (``since`` restarted after an outage). Either way ``since`` moved.
        if prev is None or prev.since != reading.since:
            transition = {"from": prev.arousal_level if prev else None, "to": reading.arousal_level}
            if not await deps.record_transition(prev, reading, state.get("day_id") or ""):
                warnings.append("transition_write_failed")
        model = RegulationStateV1(generated_at=now, arousal=reading, drives=drives, warnings=warnings)
        try:
            await deps.project(model)
        except Exception:  # noqa: BLE001 - Redis down: readers see missing -> unknown, by design
            logger.warning("regulation_project_failed", exc_info=True)
            model = model.model_copy(update={"warnings": warnings + ["project_failed"]})
        return {"regulation": model.model_dump(mode="json"), "last_inputs": inputs.model_dump(mode="json"),
                "transition": transition, "warnings": model.warnings}

    async def chronicle(state: TemporalSelfGraphState) -> dict:
        if deps.chronicle is None:
            return {"chronicle": None}
        try:
            summary = await asyncio.wait_for(deps.chronicle(), timeout=deps.chronicle_timeout_sec)
        except asyncio.TimeoutError:
            logger.warning("temporal_self_chronicle_node_timeout sec=%s", deps.chronicle_timeout_sec)
            summary = {"error": f"timeout after {deps.chronicle_timeout_sec:.0f} s"}
        except Exception as exc:  # noqa: BLE001 - the chronology must never break regulation
            logger.warning("temporal_self_chronicle_node_failed", exc_info=True)
            summary = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        warnings = list(state.get("warnings") or [])
        if summary.get("error"):
            warnings.append("chronicle_failed")
        return {"chronicle": summary, "warnings": warnings}

    async def done(state: TemporalSelfGraphState) -> dict:
        return {}

    g = StateGraph(TemporalSelfGraphState)
    for name, fn in (("ingest", ingest), ("regulate", regulate), ("chronicle", chronicle), ("done", done)):
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "regulate")
    g.add_edge("regulate", "chronicle")
    g.add_edge("chronicle", "done")
    g.add_edge("done", END)
    return g.compile(checkpointer=checkpointer)
