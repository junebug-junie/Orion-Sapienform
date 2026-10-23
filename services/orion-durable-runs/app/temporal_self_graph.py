"""temporal_self.update: Orion's regulation as a LangGraph thread (Temporal Self rev 4, PR #2369,
order 3 "R2/R3 core").

One invocation per (coalesced) event on the day's thread ``temporal_self:orion:<local date>``:

    ingest -> regulate -> immune -> chronicle -> done

* regulate: reads E1/S1/S2 (``deps.read_inputs``), folds them through the pure, I/O-free
  ``orion.regulation.arousal.classify_arousal`` with the previous reading carried by the
  checkpoint (hysteresis lives there), embeds the latest drive readings verbatim, projects the
  ``RegulationStateV1`` to Redis, and records one ``arousal_transition`` event on a level change.

* immune (R4): while arousal is ``idle`` and at most once per ``immune_interval_sec`` (6 h), reads
  24 h of the glossary's semantic field channels (``deps.read_immune_series``), judges them with the
  pure ``orion.regulation.immune.sweep`` (constant / floor-locked / replayed / decayed-to-zero),
  records each finding as one ``immune_finding`` row (``deps.record_immune``) and sets
  ``RegulationStateV1.immune_findings_open``. It never fixes anything. Last run time and count
  live in this checkpoint (``immune``) and carry over midnight. Own timeout; a failure is a
  warning and never touches the regulation reading.

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

from orion.regulation.arousal import classify_arousal, strain_causes
from orion.regulation.immune import ImmuneConfig, sweep as immune_sweep
from orion.schemas.drive_reading import DriveReadingV1
from orion.schemas.regulation import (
    TEMPORAL_SELF_WORKFLOW,
    ArousalInputsV1,
    ArousalReadingV1,
    RegulationStateV1,
)

logger = logging.getLogger("orion-durable-runs.temporal_self_graph")

NODES = ("ingest", "regulate", "immune", "chronicle", "done")
IMMUNE_TRACE_CAP = 20   # finding ids kept in the checkpoint


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
    immune: dict                 # {last_run_at, open, findings[<=20], coverage, error, skipped}: carried across days
    # Strain causes already recorded for the open strain episode (None: none recorded yet, or not
    # strained). Carried across days so a long episode is not re-seeded at every midnight.
    strain_causes_seen: Optional[list]
    strain_row_at: Optional[str]     # when the last strain row (level, cause, seed or keepalive) was written
    strain_causes_now: Optional[list]  # the strain causes present at that row


@dataclass
class TemporalSelfDeps:
    read_inputs: Callable[[datetime, Optional[ArousalInputsV1], Optional[datetime]], Awaitable[ArousalInputsV1]]
    read_drives: Callable[[datetime], Awaitable[tuple[list[DriveReadingV1], list[str]]]]
    project: Callable[[RegulationStateV1], Awaitable[None]]
    record_transition: Callable[[Optional[ArousalReadingV1], ArousalReadingV1, str], Awaitable[bool]]
    now: Callable[[], datetime]
    # Strain loop part 1: write a row when a strain cause joins an episode, or to seed an episode
    # the chronicle has no onset row for. None (TEMPORAL_SELF_STRAIN_ARCS_ENABLED=false): no rows.
    record_cause: Optional[Callable[[ArousalReadingV1, str, list[str], bool], Awaitable[bool]]] = None
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
    # R4 immune sweep. read: now -> {(node, channel): samples} and {(node, channel): semantics};
    # record: (findings, day_id, now) -> written ok. None = TEMPORAL_SELF_IMMUNE_ENABLED off.
    read_immune_series: Optional[Callable[[datetime], Awaitable[tuple[dict, dict]]]] = None
    record_immune: Optional[Callable[[list, str, datetime], Awaitable[bool]]] = None
    immune_interval_sec: float = 21600.0
    # The read is ~3 s live; the store's statement timeout is 45 s. This is the hard ceiling.
    immune_timeout_sec: float = 90.0
    immune_config: ImmuneConfig = ImmuneConfig()


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


# Strain loop part 1: while strained, a row at least this often (a keepalive, ``joined == []``), so
# the chronology can tell "still strained" from "the closing row was lost". The reducer closes a
# strain arc with no row for 2.5x this (``ReducerConfig.strain_stale_sec``).
STRAIN_KEEPALIVE_SEC = 1800.0


# A cause that LEFT (or came back) is recorded no sooner than this after the last strain row, so a
# cause flapping around its threshold costs at most one row per 10 min; a cause that JOINS the
# episode for the first time is recorded at once.
STRAIN_CAUSE_CHANGE_MIN_SEC = 600.0


async def _strain_causes_step(
    deps: TemporalSelfDeps, reading: ArousalReadingV1, state: TemporalSelfGraphState, level_row_written: bool,
    warnings: list,
) -> tuple[Optional[list], Optional[str], Optional[list]]:
    """Strain loop part 1. A level row is written only when the level changes, so a cause that joins
    an episode already strained (the cabinet going critical mid-episode) was never recorded. While
    strained, write a row when a cause joins, when the causes present now differ from the last row's
    (a cause left, e.g. the cabinet cooled; at most every STRAIN_CAUSE_CHANGE_MIN_SEC), a seed row
    when nothing was recorded for the episode (the first step after deploy, a lost onset row), and a
    keepalive every STRAIN_KEEPALIVE_SEC. Every row carries the causes present now.
    Returns the new ``(strain_causes_seen, strain_row_at, strain_causes_now)``; a failed write keeps
    the old values, so the next step retries it."""
    if deps.record_cause is None or reading.arousal_level != "strained":
        return None, None, None
    now = reading.observed_at
    causes = strain_causes(reading.reasons)
    if level_row_written:
        return causes, now.isoformat(), causes  # the onset row carries this episode's causes
    seen = state.get("strain_causes_seen")
    row_at = _dt(state.get("strain_row_at"))
    recorded_now = state.get("strain_causes_now")
    joined = [c for c in causes if c not in (seen or [])]
    age = None if row_at is None else (now - row_at).total_seconds()
    due = age is None or age >= STRAIN_KEEPALIVE_SEC
    # None: no row since this key existed (the first steps after deploy) -- what the last row
    # carried is unknown here, so write one once the rate limit allows rather than wait for the
    # keepalive.
    changed = ((recorded_now is None or set(causes) != set(recorded_now))
               and age is not None and age >= STRAIN_CAUSE_CHANGE_MIN_SEC)
    if seen is not None and not joined and not due and not changed:
        return seen, state.get("strain_row_at"), recorded_now
    if not await deps.record_cause(reading, state.get("day_id") or "", joined, seen is None):
        warnings.append("strain_cause_write_failed")
        return seen, state.get("strain_row_at"), recorded_now
    return list(seen or []) + joined, now.isoformat(), causes


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
        recorded = False
        # A new level episode began: the level changed, or the previous reading was too old to
        # carry (``since`` restarted after an outage). Either way ``since`` moved.
        if prev is None or prev.since != reading.since:
            transition = {"from": prev.arousal_level if prev else None, "to": reading.arousal_level}
            recorded = await deps.record_transition(prev, reading, state.get("day_id") or "")
            if not recorded:
                warnings.append("transition_write_failed")
        seen, row_at, causes_now = await _strain_causes_step(deps, reading, state, recorded, warnings)
        open_findings = int((state.get("immune") or {}).get("open") or 0)
        model = RegulationStateV1(generated_at=now, arousal=reading, drives=drives, warnings=warnings,
                                  immune_findings_open=max(0, open_findings))
        try:
            await deps.project(model)
        except Exception:  # noqa: BLE001 - Redis down: readers see missing -> unknown, by design
            logger.warning("regulation_project_failed", exc_info=True)
            model = model.model_copy(update={"warnings": warnings + ["project_failed"]})
        return {"regulation": model.model_dump(mode="json"), "last_inputs": inputs.model_dump(mode="json"),
                "transition": transition, "warnings": model.warnings, "strain_causes_seen": seen, "strain_row_at": row_at,
                "strain_causes_now": causes_now}

    async def immune(state: TemporalSelfGraphState) -> dict:
        if deps.read_immune_series is None:
            return {}
        regulation = dict(state.get("regulation") or {})
        prev = dict(state.get("immune") or {})
        level = (regulation.get("arousal") or {}).get("arousal_level")
        if level != "idle":
            # Visible in /health: with arousal disabled or unknown the sweep never runs.
            skipped = f"arousal {level}"
            return {} if prev.get("skipped") == skipped else {"immune": {**prev, "skipped": skipped}}
        now = deps.now()
        last = _dt(prev.get("last_run_at"))
        if last is not None and 0.0 <= (now - last).total_seconds() < deps.immune_interval_sec:
            return {} if not prev.get("skipped") else {"immune": {**prev, "skipped": None}}

        async def run() -> tuple[list, dict]:
            series, semantics = await deps.read_immune_series(now)
            result = immune_sweep(series, semantics, config=deps.immune_config)
            answered = [v for v in result.verdicts if any(c in ("flagged", "clear") or str(c).startswith("clear:")
                                                          for c in v.checks.values())]
            coverage = {"judged": len(answered),
                        "rest_unjudged": sum(1 for v in answered if "cannot_judge:no rest_range" in v.checks.values()),
                        "skipped": sum(1 for v in result.verdicts if v.skipped is not None)}
            return result.findings, coverage

        new_warnings: list[str] = []
        # A failed or timed-out sweep still stamps last_run_at: a broken read must not re-run a
        # 24 h scan every 120 s step. The open count keeps its last known value.
        summary: dict = {"last_run_at": now.isoformat(), "open": int(prev.get("open") or 0),
                         "findings": list(prev.get("findings") or []), "coverage": prev.get("coverage"),
                         "error": None, "skipped": None}
        try:
            findings, coverage = await asyncio.wait_for(run(), timeout=deps.immune_timeout_sec)
        except asyncio.TimeoutError:
            logger.warning("temporal_self_immune_timeout sec=%s", deps.immune_timeout_sec)
            summary["error"] = f"timeout after {deps.immune_timeout_sec:.0f} s"
            findings = None
        except Exception as exc:  # noqa: BLE001 - the sweep must never break regulation
            logger.warning("temporal_self_immune_failed", exc_info=True)
            summary["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            findings = None
        if findings is None:
            new_warnings.append("immune_failed")
        else:
            summary.update(open=len(findings), coverage=coverage,
                           findings=[f"{f.metric_id}:{f.kind}" for f in findings][:IMMUNE_TRACE_CAP])
            logger.info("temporal_self_immune_sweep coverage=%s open=%s findings=%s", coverage, len(findings),
                        ",".join(summary["findings"]))
            if findings and deps.record_immune is not None:
                try:
                    ok = await deps.record_immune(findings, state.get("day_id") or "", now)
                except Exception:  # noqa: BLE001
                    logger.warning("temporal_self_immune_record_failed", exc_info=True)
                    ok = False
                if not ok:
                    # The count still reflects what the sweep saw; the day block shows only rows.
                    new_warnings.append("immune_write_failed")
        out: dict = {"immune": summary, "warnings": list(state.get("warnings") or []) + new_warnings}
        if not regulation:
            return out
        # Re-project so /regulation/state carries the count and any immune warning this step.
        try:
            model = RegulationStateV1.model_validate(regulation)
            model = model.model_copy(update={"immune_findings_open": summary["open"],
                                             "warnings": list(model.warnings) + new_warnings})
        except Exception:  # noqa: BLE001
            logger.warning("temporal_self_immune_state_invalid", exc_info=True)
            return out
        try:
            await deps.project(model)
        except Exception:  # noqa: BLE001 - Redis down: the checkpoint still carries the count
            logger.warning("regulation_project_failed after immune sweep", exc_info=True)
        out["regulation"] = model.model_dump(mode="json")
        return out

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
    for name, fn in (("ingest", ingest), ("regulate", regulate), ("immune", immune), ("chronicle", chronicle),
                     ("done", done)):
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "regulate")
    g.add_edge("regulate", "immune")
    g.add_edge("immune", "chronicle")
    g.add_edge("chronicle", "done")
    g.add_edge("done", END)
    return g.compile(checkpointer=checkpointer)
