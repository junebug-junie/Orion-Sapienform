"""One pure adapter per source kind: a row dict in, a ``TemporalSelfEventV1`` (or None) out.

The patch-3 driver reads rows; this module only shapes them. Each adapter states its
occurrence-time column and its cast (see ``day.py``), its subject ref, and why a row is
dropped. Nothing here copies model prose: labels are the row's own short label or "".

Source drift found live on 2026-10-10 (recorded in the PR report):

* ``reverie_visual_attempt.outcome`` adds ``abandoned`` (31 rows in 30 days, the only
  deferral word in the last 7) to the spec's ``deferred_thermal`` / ``deferred_resource`` /
  ``failed``. Any non-``produced`` outcome is a constraint, kept in the source's own word.
* Reverie thoughts link to chains by ``thought_json.chain_id``, not ``correlation_id``.
  Reverie attention rows carry the *thought's* ``correlation_id``, so the chain adapter
  carries those ids in ``related_refs`` for rule 8.
* Curiosity attention rows carry a uuid5 ``correlation_id`` that matches no ``run_id``
  (0 of 41 over 3 days), so they bind to curiosity arcs by time, labelled co-occurrence.
* ``vision_events.entities`` is non-empty on 74 of 1,806 room rows in 7 days; rows with no
  entities carry no percept and are not emitted.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, Iterable, Mapping

from orion.schemas.temporal_self import LABEL_MAX, TemporalSelfEventV1
from orion.temporal_self.day import DEFAULT_TZ, as_utc, day_id_for

Row = Mapping[str, Any]

# Same predicate as services/orion-cortex-exec/app/admission_cue.py:85-97 (WAIT_THRESHOLD_MS).
GPU_WAIT_THRESHOLD_MS = 500.0
METACOG_SEVERITIES = frozenset({"degraded", "critical"})
ROOM_STREAMS = frozenset({"cam0"})


def clip(text: Any, limit: int = LABEL_MAX) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _json(value: Any) -> Any:
    if isinstance(value, (str, bytes)):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def _event(
    *,
    kind: str,
    table: str,
    ref: Any,
    at: datetime | str | None,
    tz_name: str,
    ended: datetime | str | None = None,
    **fields: Any,
) -> TemporalSelfEventV1 | None:
    occurred = as_utc(at)
    if occurred is None or ref in (None, ""):
        return None
    end = as_utc(ended)
    if end is not None and end < occurred:
        end = occurred
    return TemporalSelfEventV1(
        event_id=f"{kind}:{ref}",
        day_id=day_id_for(occurred, tz_name),
        occurred_at=occurred,
        ended_at=end,
        source_kind=kind,  # type: ignore[arg-type]
        source_table=table,
        source_ref=str(ref),
        **fields,
    )


def _speaker(row: Row) -> str:
    """Who wrote the row, from flags only (the driver selects booleans, never the text).

    Juniper's turn has a non-empty prompt (the same rule as
    scripts/analysis/measure_arousal_replay.py). Orion's outreach has no prompt and
    ``client_meta.unsolicited`` true. Live 10-09: all four rows on the session that looked
    like a conversation were unsolicited outreach with no reply.
    """
    has_prompt = row.get("has_prompt")
    if has_prompt is None:
        has_prompt = bool(str(row.get("prompt") or "").strip())
    meta = _json(row.get("client_meta")) or {}
    unsolicited = row.get("unsolicited")
    if unsolicited is None:
        unsolicited = bool(meta.get("unsolicited")) if isinstance(meta, dict) else False
    if has_prompt:
        return "juniper"
    return "orion_outreach" if unsolicited else "orion"


def chat_turn(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """chat_history_log. Time: created_at (naive UTC). No text copied.

    Subject: session_id, for Juniper's turns only. Orion's own rows (outreach, journal
    replies with no prompt) carry the session in ``related_refs`` and no subject, so they
    bind by time as context and never open or extend a conversation (danger mode 9: Orion's
    outreach must not read as Juniper speaking)."""
    session = row.get("session_id") or None
    speaker = _speaker(row)
    return _event(
        kind="chat_turn", table="chat_history_log", ref=row.get("id"), at=row.get("created_at"),
        tz_name=tz_name, correlation_id=row.get("correlation_id") or None,
        subject_ref=session if speaker == "juniper" else None,
        related_refs=[str(session)] if session else [],
        privacy_class="juniper_chat", payload={"source": row.get("source"), "speaker": speaker},
    )


def curiosity_run(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """curiosity_offer_decisions joined to curiosity_run_outcomes on run_id.

    Emitted only once ``completed_at`` exists, so the arc is born with its interval.
    Time: turn_started_at, else decided_at. Subject: run_id. related_refs: offered prior ids.
    """
    if row.get("completed_at") is None:
        return None
    offered = _json(row.get("offered")) or []
    priors = sorted({str(o.get("prior_id")) for o in offered if isinstance(o, dict) and o.get("prior_id")})
    return _event(
        kind="curiosity_run", table="curiosity_run_outcomes", ref=row.get("run_id"),
        at=row.get("turn_started_at") or row.get("decided_at"), ended=row.get("completed_at"),
        tz_name=tz_name, subject_ref=str(row.get("run_id")), related_refs=priors,
        payload={k: row.get(k) for k in ("turn_ok", "n_tested", "n_moved", "n_formed", "arm")},
    )


def reverie_chain(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """substrate_reverie_chain plus its thoughts (``row['thoughts']``: thought_id, created_at,
    correlation_id; linked by thought_json.chain_id). A chain with no thoughts has no content
    and is not an arc. Time: first thought .. max(last thought, chain created_at)."""
    thoughts = sorted(
        (t for t in (row.get("thoughts") or []) if t.get("thought_id") and t.get("created_at")),
        key=lambda t: (as_utc(t["created_at"]), str(t["thought_id"])),
    )
    if not thoughts:
        return None
    first = as_utc(thoughts[0]["created_at"])
    last = max(as_utc(thoughts[-1]["created_at"]), as_utc(row.get("created_at")) or first)
    return _event(
        kind="reverie_chain", table="substrate_reverie_chain", ref=row.get("chain_id"), at=first,
        ended=last, tz_name=tz_name, subject_ref=str(row.get("chain_id")),
        related_refs=sorted({str(t["correlation_id"]) for t in thoughts if t.get("correlation_id")}),
        label=clip(row.get("theme_key")),
        payload={"terminal_reason": row.get("terminal_reason"),
                 "thought_ids": [str(t["thought_id"]) for t in thoughts]},
    )


def visual_run(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """reverie_visual_chain (created_at is stamped at the END of the run) joined to the
    attempt whose result_json.chain_id matches (``row['attempt_started_at']``,
    ``row['thermal_state']``). Time: attempt start, else a point at created_at."""
    end = row.get("created_at")
    return _event(
        kind="visual_run", table="reverie_visual_chain", ref=row.get("chain_id"),
        at=row.get("attempt_started_at") or end, ended=end, tz_name=tz_name,
        subject_ref=str(row.get("chain_id")), label=clip(row.get("theme_key")),
        payload={"terminal_reason": row.get("terminal_reason"),
                 "thermal_state": row.get("thermal_state")},
    )


def visual_deferral(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """reverie_visual_attempt with any outcome other than ``produced``. Time: started_at."""
    outcome = row.get("outcome")
    if not outcome or outcome == "produced":
        return None
    result = _json(row.get("result_json")) or {}
    detail = result.get("detail") if isinstance(result.get("detail"), dict) else {}
    return _event(
        kind="visual_deferral", table="reverie_visual_attempt", ref=row.get("attempt_id"),
        at=row.get("started_at"), tz_name=tz_name, verdict=str(outcome),
        payload={"reason": result.get("reason"), "thermal_state": detail.get("state"),
                 "refused": result.get("refused")},
    )


def gpu_wait(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """gpu_pool_events: Orion's own background lease that waited >= 500 ms or was never
    served -- the admission cue's deferral predicate, exactly. Time: generated_at."""
    if str(row.get("holder") or "").startswith("http:") or row.get("priority") != "background":
        return None
    event = row.get("event")
    waited = row.get("waited_ms")
    if not (event == "unavailable" or (event == "granted" and waited is not None and waited >= GPU_WAIT_THRESHOLD_MS)):
        return None
    return _event(
        kind="gpu_wait", table="gpu_pool_events", ref=row.get("event_id"), at=row.get("generated_at"),
        tz_name=tz_name, verdict=str(event), correlation_id=row.get("turn_correlation_id") or None,
        payload={"waited_ms": waited, "holder": row.get("holder"), "work_class": row.get("work_class")},
    )


def dream_cycle(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """dream_cycle, completed only. Time: started_at .. ended_at. Subject: cycle_id."""
    if row.get("status") != "completed" or row.get("ended_at") is None:
        return None
    pressure = (_json(row.get("cycle_json")) or {}).get("pressure") or {}
    return _event(
        kind="dream_cycle", table="dream_cycle", ref=row.get("cycle_id"), at=row.get("started_at"),
        ended=row.get("ended_at"), tz_name=tz_name, subject_ref=str(row.get("cycle_id")),
        label=clip(row.get("trigger")),
        payload={"pressure": row.get("pressure"), "replay_count": row.get("replay_count"),
                 "hypothesis_count": row.get("hypothesis_count"),
                 "covered_since": pressure.get("since") if isinstance(pressure, dict) else None},
    )


def dream_hypothesis(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """dream_hypothesis. Time: created_at. Attaches to its sleep arc by cycle_id, never by
    time. ``claim``/``why`` are model prose and are not copied."""
    return _event(
        kind="dream_hypothesis", table="dream_hypothesis", ref=row.get("hypothesis_id"),
        at=row.get("created_at"), tz_name=tz_name, related_refs=[str(row.get("cycle_id"))],
        payload={"arm": row.get("arm"),
                 "expires_at": _iso(row.get("expires_at")), "offered_at": _iso(row.get("offered_at")),
                 "offered_run_id": row.get("offered_run_id")},
    )


def expectation_verdict(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """substrate_reverie_thought once scored. Time: expectation_scored_at. Attaches to its
    reverie arc by chain id (``row['chain_id']`` = thought_json.chain_id)."""
    if row.get("expectation_scored_at") is None or not row.get("expectation_verdict"):
        return None
    chain = row.get("chain_id")
    return _event(
        kind="expectation_verdict", table="substrate_reverie_thought", ref=row.get("thought_id"),
        at=row.get("expectation_scored_at"), tz_name=tz_name, verdict=str(row["expectation_verdict"]),
        related_refs=[str(chain)] if chain else [],
        correlation_id=row.get("correlation_id") or None,
        payload={"committed_at": _iso(row.get("created_at"))},
    )


def action_outcome(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """substrate_action_outcomes. Time: observed_at, else created_at. Verdict: claim_upheld."""
    upheld = row.get("claim_upheld")
    word = "null" if upheld is None else ("true" if upheld else "false")
    return _event(
        kind="action_outcome", table="substrate_action_outcomes", ref=row.get("id"),
        at=row.get("observed_at") or row.get("created_at"), tz_name=tz_name,
        verdict=f"claim_upheld={word}",
        payload={"dispatch_kind": row.get("dispatch_kind"), "target_id": row.get("target_id"),
                 "prediction_error": row.get("prediction_error")},
    )


def metacog_observation(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """orion_metacog (degraded / critical only) left-joined to metacog_trigger on
    correlation_id. Time: the trigger's naive-UTC ``trigger_timestamp`` when joined, else the
    TEXT ``timestamp`` of the metacog row. ``summary``/``mantra`` are LLM prose: not copied."""
    if row.get("severity") not in METACOG_SEVERITIES:
        return None
    return _event(
        kind="metacog_observation", table="orion_metacog", ref=row.get("id"),
        at=row.get("trigger_timestamp") or row.get("timestamp"), tz_name=tz_name,
        verdict=str(row["severity"]), correlation_id=row.get("correlation_id") or None,
        payload={"trigger_kind": row.get("trigger_kind")},
    )


def consolidation_window_close(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """memory_consolidation_windows once closed. Time: closed_at. Binds to the conversation
    arc whose turns it holds, by correlation id. Context only, never closes the arc: windows
    close when the *next* turn arrives, so treating them as authority would split a live
    conversation."""
    if row.get("closed_at") is None:
        return None
    turns = _json(row.get("turn_correlation_ids")) or []
    return _event(
        kind="consolidation_window_close", table="memory_consolidation_windows",
        ref=row.get("memory_window_id"), at=row.get("closed_at"), tz_name=tz_name,
        related_refs=sorted({str(t) for t in turns if t}),
        payload={"close_reason": row.get("close_reason"), "source_platform": row.get("source_platform")},
    )


def attention_row(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """substrate_attention_schema. Time: generated_at. Never a subject (attended_id is a text
    hash); folds into per-arc lane counts and reason words. Labels are not copied."""
    return _event(
        kind="attention_row", table="substrate_attention_schema", ref=row.get("entry_id"),
        at=row.get("generated_at"), tz_name=tz_name, correlation_id=row.get("correlation_id") or None,
        payload={"process": row.get("process"), "reason": row.get("attention_reason")},
    )


def attention_loop_raised(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """attention_salience_trace, scope='chat', and only when its correlation_id resolves to a
    real chat turn (``row['chat_turn']``: the driver's EXISTS against chat_history_log).

    Live 10-09: one chat-scope loop got 381 traces, each with a distinct correlation_id,
    across all 24 hours, and only 6 resolved to a chat turn. The rest are the scorer
    re-emitting on its own cadence; counting them would make concern dwell measure the
    scorer, not Juniper raising the loop. Subject: loop_id. The label is cut from
    Juniper's turn text, so it is marked juniper_chat (outward boundary only)."""
    if row.get("scope") != "chat" or not row.get("loop_id") or not row.get("chat_turn"):
        return None
    return _event(
        kind="attention_loop_raised", table="attention_salience_trace", ref=row.get("trace_id"),
        at=row.get("created_at"), tz_name=tz_name, subject_ref=str(row["loop_id"]),
        correlation_id=row.get("correlation_id") or None, label=clip(row.get("description")),
        privacy_class="juniper_chat",
    )


def attention_loop_verdict(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """attention_loop_outcome. Time: created_at (for decayed_unattended, when the digest ran,
    not when silence began). Subject: loop_id. Verdict: the row's own word."""
    if not row.get("loop_id"):
        return None
    return _event(
        kind="attention_loop_verdict", table="attention_loop_outcome", ref=row.get("outcome_id"),
        at=row.get("created_at"), tz_name=tz_name, subject_ref=str(row["loop_id"]),
        verdict=str(row.get("verdict") or ""), payload={"actor": row.get("actor")},
    )


def field_dominance_run(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """field_dominance_run (seam S2, live since 10-09). Time: started_at .. ended_at.
    Subject: target_id."""
    return _event(
        kind="field_dominance_run", table="field_dominance_run", ref=row.get("run_id"),
        at=row.get("started_at"), ended=row.get("ended_at"), tz_name=tz_name,
        subject_ref=str(row.get("target_id")), label=clip(row.get("target_id")),
        payload={"tick_count": row.get("tick_count"), "min_streak_at_run": row.get("min_streak_at_run"),
                 "target_kind": row.get("target_kind"), "left_censored": bool(row.get("left_censored"))},
    )


def vision_percept(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """vision_events from a room stream (or NULL stream_id), with entities. Time: created_at
    (write time; no observation time exists). event_type and entities only, never narrative."""
    stream = row.get("stream_id")
    if stream is not None and stream not in ROOM_STREAMS:
        return None
    entities = sorted({clip(e, 64) for e in (_json(row.get("entities")) or []) if e})
    if not entities:
        return None
    return _event(
        kind="vision_percept", table="vision_events", ref=row.get("event_id"), at=row.get("created_at"),
        tz_name=tz_name, payload={"event_type": row.get("event_type"), "entities": entities},
    )


def memory_episode(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """episode_memory. Time: occurred_at. episode_id and purpose only; statement is prose."""
    return _event(
        kind="memory_episode", table="episode_memory", ref=row.get("memory_id"),
        at=row.get("occurred_at"), tz_name=tz_name, related_refs=[str(row.get("episode_id"))],
        payload={"purpose": row.get("purpose")},
    )


def _iso(value: Any) -> str | None:
    dt = as_utc(value)
    return dt.isoformat() if dt else None


ADAPTERS: dict[str, Callable[[Row, str], TemporalSelfEventV1 | None]] = {
    "chat_turn": chat_turn,
    "curiosity_run": curiosity_run,
    "reverie_chain": reverie_chain,
    "visual_run": visual_run,
    "visual_deferral": visual_deferral,
    "gpu_wait": gpu_wait,
    "dream_cycle": dream_cycle,
    "dream_hypothesis": dream_hypothesis,
    "expectation_verdict": expectation_verdict,
    "action_outcome": action_outcome,
    "metacog_observation": metacog_observation,
    "consolidation_window_close": consolidation_window_close,
    "attention_row": attention_row,
    "attention_loop_raised": attention_loop_raised,
    "attention_loop_verdict": attention_loop_verdict,
    "field_dominance_run": field_dominance_run,
    "vision_percept": vision_percept,
    "memory_episode": memory_episode,
}


def adapt(kind: str, rows: Iterable[Row], tz_name: str = DEFAULT_TZ) -> list[TemporalSelfEventV1]:
    out = [ADAPTERS[kind](r, tz_name) for r in rows]
    return [e for e in out if e is not None]
