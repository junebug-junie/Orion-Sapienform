"""``build_self_readings`` / ``render_self_readings``: body signals Orion can understand. Pure.

Spec: docs/superpowers/specs/2026-10-07-orion-self-calibration-design.md section C ("Self-readings
Orion can understand"); recommended next patch 4 (stance_react and reverie).

What a reading is, and where each part comes from (no second copy of any rule):

- **Which signals:** every ``node:substrate.*`` prediction_error entry in the semantic layer
  (``config/field/field_channel_glossary.v1.yaml``) whose ``prompt_sites`` names
  ``SELF_READINGS_PROMPT_SITE``, and that the attention seam calls internal (the body). The
  glossary is the allow-list: a signal cannot reach Orion's prompt here without its
  value_kind/rest/sparsity being on record, which the prompt-semantics gate enforces
  (``scripts/check_metric_lineage.py --prompt-semantics``). Placeholder and bucket kinds are never
  a measurement and are skipped even if listed.
- **Calibration:** ``orion.attention.pe_history_cache.NodePeHistoryCache`` (the same 7-day window
  and memoized ``PredictionErrorMagnitudeV1`` the field contest uses) and
  ``orion.attention.world_first.judge_candidate`` (band in the signal's bad direction, freshness,
  event-written orienting window). A reading is ``unusual`` exactly when the attention seam would
  let that body signal compete: high or unusual for itself, fresh, still news.
- **Trigger / binary kinds** (cabinet warming): the value is "it fired", so the reading is how
  many times it fired in the last hour against the same trailing count sampled every minute over
  7 days (``world_first.chat_rate_reading``, the chat-rate math), never a magnitude.
- **score_0_10** is the bad-direction percentile x 10, rounded (Juniper: "what is 2 vs 10000").
  Ranking uses the mid-rank percentile, prompts show the 0-10 score.

Rendering (``render_self_readings``): only unusual readings get a line, at most
``MAX_SHOWN`` and ``SELF_READINGS_MAX_CHARS``. A calm body renders nothing ("creatures notice
chaos"); no reading, stale, or too-little-history signals never render as calm -- when a block is
shown they are named as unreadable. Every line carries ``[pe:<name>@<observed_at>]`` naming the
persisted row (``substrate_node_prediction_error_history`` primary key) its numbers came from.

What it tends to lead to (spec section E): when the caller passes the signal's renderable
``self_signal_outcome_association`` row (``orion.attention.self_signal_outcomes.index_by_signal``:
min n met, 95% interval excludes 1), the line gets "; when this is high, my next hour had ... Nx as
often (...) [assoc:<association_id>]". The clause is dropped before a whole reading would be.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping

from orion.attention.pe_history_cache import NodePeHistoryCache
from orion.attention.world_first import (
    SUBSTRATE_NODE_PREFIX,
    chat_rate_reading,
    judge_candidate,
    node_candidate,
    node_prediction_error_continuous,
    node_prediction_error_semantics,
    source_kind_for_node,
)
from orion.schemas.attention_candidate import AttentionCandidateV1
from orion.schemas.self_reading import (
    SELF_READINGS_MAX_CHARS,
    SelfReadingsBlockV1,
    SelfReadingV1,
)

logger = logging.getLogger(__name__)

SELF_READINGS_PROMPT_SITE = "orion.attention.self_readings:render_self_readings"
HISTORY_TABLE = "substrate_node_prediction_error_history"
MAX_SHOWN = 3
TRIGGER_RATE_WINDOW = timedelta(hours=1)
# Fewer real firings than this in 7 days: "more than usual" has nothing to stand on.
TRIGGER_MIN_FIRES_7D = 5

_PERCENTILE_KINDS = frozenset({"level", "count"})
_RATE_KINDS = frozenset({"trigger", "binary"})

# Plain words for the signals that can reach a prompt today. Cosmetic only: which signals render
# is decided by the glossary's prompt_sites, and an unlisted node falls back to its short name.
# Units go on the reading (schema), not in the line: the line's scale is the 0-10 score.
_WORDS: dict[str, tuple[str, str]] = {
    "node:substrate.biometrics": ("how much my machines' load readings are moving", "share of a 0.30 swing"),
    "node:substrate.bus_synaptic": ("how many of my message-bus links look abnormal", "fraction of links"),
    "node:substrate.execution": ("surprise in how my task execution is going", "clipped z / 3"),
    "node:substrate.chat": ("surprise in how conversation load and repair are moving", "clipped z / 3"),
    "node:substrate.route": ("how much my routing decisions are changing", "share of 4 decisions changed"),
    "node:substrate.codebase": ("how unusual the latest change to my own code was", "clipped z / 3"),
    "node:substrate.cabinet": ("my cabinet's warming alarm", "firings in the last hour"),
}

_STATE_WORDS = {
    "no_reading": "no reading",
    "stale": "stale",
    "insufficient_history": "too little history",
}

_NODES_CACHE: list[str] | None = None


def _event_written(node_id: str) -> bool:
    from orion.attention.world_first import node_prediction_error_event_written

    return node_prediction_error_event_written(node_id)


def short_name(node_id: str) -> str:
    return node_id[len(SUBSTRATE_NODE_PREFIX):] if node_id.startswith(SUBSTRATE_NODE_PREFIX) else node_id


def prompt_reaching_nodes() -> list[str]:
    """Body nodes whose glossary prediction_error entry declares this module's prompt site.

    An unreadable glossary yields [] (render nothing) and is not cached."""
    global _NODES_CACHE
    if _NODES_CACHE is not None:
        return list(_NODES_CACHE)
    try:
        from orion.field.channel_glossary import load_glossary

        entries = load_glossary()["entries"]
    except Exception as exc:  # noqa: BLE001 -- a missing glossary renders nothing, never raises
        logger.warning("self_readings_glossary_unreadable err=%s", exc)
        return []
    out: list[str] = []
    for entry in entries:
        if entry.channel != "prediction_error" or not entry.node:
            continue
        sem = dict(entry.semantics)
        sites = sem.get("prompt_sites") or ()
        if SELF_READINGS_PROMPT_SITE not in sites:
            continue
        if source_kind_for_node(entry.node) != "internal":
            continue
        if sem.get("value_kind") not in (_PERCENTILE_KINDS | _RATE_KINDS):
            continue
        out.append(entry.node)
    _NODES_CACHE = sorted(set(out))
    return list(_NODES_CACHE)


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def _z(ts: datetime) -> str:
    return _aware(ts).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def score_0_10(percentile: float | None) -> int | None:
    if percentile is None:
        return None
    return int(min(10, max(0, math.floor(float(percentile) * 10.0 + 0.5))))


def _state(candidate: AttentionCandidateV1, band: str, reason: str) -> str:
    if candidate.absent:
        return "no_reading"
    if band == "insufficient_history" or candidate.unusualness.percentile_now is None:
        return "insufficient_history"
    if reason.startswith("stale"):
        return "stale"
    return "live"


def trigger_onsets(history: list[tuple[datetime, float]]) -> list[datetime]:
    """Each time the trigger turned ON (a >0 row after a 0 row, or the first row). The cabinet
    writes a row every 30 s while warming, so counting >0 rows would count checks, not firings."""
    out: list[datetime] = []
    prev_on = False
    for ts, v in history:
        on = math.isfinite(v) and v > 0.0
        if on and not prev_on:
            out.append(ts)
        prev_on = on
    return out


def _last_on_row(history: list[tuple[datetime, float]]) -> datetime | None:
    for ts, v in reversed(history):
        if math.isfinite(v) and v > 0.0:
            return ts
    return None


def _trigger_candidate(
    node_id: str,
    label: str,
    history: list[tuple[datetime, float]],
    latest: tuple[float, datetime] | None,
    now: datetime,
    value_kind: str,
) -> tuple[AttentionCandidateV1, float]:
    fires = trigger_onsets(history)
    mag, mid = chat_rate_reading(
        fires, now=now, window=TRIGGER_RATE_WINDOW, min_turns=TRIGGER_MIN_FIRES_7D
    )
    absent_reason = None
    if latest is None:
        absent_reason = "no stored reading in 7 days"
    else:
        # Freshness is the node's last WRITE (it writes 0 when not firing), not its last fire.
        age = max(0.0, (now - _aware(latest[1])).total_seconds())
        mag = mag.model_copy(update={"age_sec": round(age, 3)})
    cand = AttentionCandidateV1(
        candidate_id=f"self-reading:{node_id}",
        source_id=node_id,
        source_kind="internal",
        label=label,
        unusualness=mag,
        rank_percentile=mid,
        observed_at=latest[1] if latest else None,
        absent=absent_reason is not None,
        absent_reason=absent_reason,
        value_kind=value_kind,
        polarity=None,
        evidence_refs=[node_id],
    )
    return cand, float(mag.value)


def build_self_readings(
    cache: NodePeHistoryCache,
    *,
    now: datetime,
    nodes: Iterable[str] | None = None,
    associations: Mapping[str, object] | None = None,
    sustain: bool = True,
) -> list[SelfReadingV1]:
    """One reading per prompt-reaching body node. Never raises for a single bad node.

    ``associations`` (signal_id -> ``SelfSignalOutcomeAssociationV1``, from
    ``orion.attention.self_signal_outcomes.index_by_signal``): only renderable rows (a clause, so
    min n met and the interval excludes 1). None = the readings are exactly as before.

    ``sustain`` (``ATTENTION_SUSTAINED_BODY_ENABLED``): a continuous body source (biometrics,
    bus_synaptic) is unusual only when the attention contest would let it compete -- hot for
    more than 300 s of the last 600 s over the same readings (``orion.attention.sustained``)."""
    now = _aware(now)
    node_ids = list(nodes) if nodes is not None else prompt_reaching_nodes()
    mags = cache.magnitudes(now=now)
    current = cache.current()
    readings: list[SelfReadingV1] = []
    for node_id in node_ids:
        value_kind, _ = node_prediction_error_semantics(node_id)
        if value_kind not in (_PERCENTILE_KINDS | _RATE_KINDS):
            # placeholder / bucket (or unknown): never rendered as a measurement.
            continue
        try:
            reading = _reading(cache, node_id, mags, current, now, sustain=sustain)
            assoc = (associations or {}).get(node_id)
            clause = str(getattr(assoc, "clause", "") or "")
            if clause:
                reading = reading.model_copy(update={
                    "tends_to": clause,
                    "outcome_association_id": str(getattr(assoc, "association_id", "") or ""),
                })
                reading = reading.model_copy(update={"plain": plain_line(reading)})
            readings.append(reading)
        except Exception as exc:  # noqa: BLE001 -- one broken node must not hide the others
            logger.warning("self_reading_failed node_id=%s err=%s", node_id, exc)
    return readings


def _reading(cache, node_id, mags, current, now, *, sustain: bool = True) -> SelfReadingV1:
    label, units = _WORDS.get(node_id, (f"my {short_name(node_id)} signal", ""))
    value_kind, _polarity = node_prediction_error_semantics(node_id)
    is_rate = value_kind in _RATE_KINDS
    latest = current.get(node_id)
    ref_at = None
    if is_rate:
        history = cache.history(node_id)
        candidate, value = _trigger_candidate(node_id, label, history, latest, now, str(value_kind))
        observed_at = latest[1] if latest else None
        # The shown ref names the newest ON row (what the count is about), not the last 0 write.
        ref_at = _last_on_row(history)
    else:
        entry = mags.get(node_id)
        magnitude, observed_at, mid = (entry if entry is not None else (None, None, None))
        if magnitude is not None and not math.isfinite(float(magnitude.value)):
            # A non-finite stored value is not a measurement: no reading, never "now inf".
            magnitude, mid = None, None
        candidate = node_candidate(
            node_id=node_id, label=label, magnitude=magnitude, observed_at=observed_at,
            now=now, rank_percentile=mid,
            pe_history=(
                cache.history(node_id) if sustain and node_prediction_error_continuous(node_id) else None
            ),
            sustain=sustain,
        )
        value = float(magnitude.value) if magnitude is not None else None
    verdict = judge_candidate(candidate)
    state = _state(candidate, verdict.band, verdict.reason)
    pct = verdict.score if state == "live" else None
    rank = verdict.rank_score if verdict.rank_score is not None else candidate.rank_percentile
    name = short_name(node_id)
    ref_ts = ref_at if is_rate else observed_at
    ref = f"pe:{name}@{_z(ref_ts)}" if ref_ts is not None else ""
    if observed_at is None:
        source_ref = ""
    elif is_rate:
        start = now - TRIGGER_RATE_WINDOW
        source_ref = (f"{HISTORY_TABLE}:{node_id} off->on transitions with observed_at in "
                      f"({start.isoformat()}, {now.isoformat()}]; last write {_aware(observed_at).isoformat()}")
    else:
        source_ref = f"{HISTORY_TABLE}:{node_id}@{_aware(observed_at).isoformat()}"
    reading = SelfReadingV1(
        source_id=node_id,
        label=label,
        value_kind=str(value_kind or ""),
        value=None if value is None or state == "no_reading" else round(value, 6),
        units=units,
        percentile_now=None if pct is None else round(float(pct), 6),
        rank_percentile=None if rank is None or state != "live" else round(float(rank), 6),
        band=verdict.band if state == "live" else state,
        score_0_10=score_0_10(pct),
        trend=candidate.unusualness.trend,
        state=state,  # type: ignore[arg-type]
        unusual=bool(verdict.eligible) and state == "live",
        reason=verdict.reason,
        observed_at=_aware(observed_at) if observed_at is not None else None,
        age_sec=None if candidate.absent else candidate.unusualness.age_sec,
        n_readings_7d=candidate.unusualness.n_readings_7d,
        rate_window_sec=TRIGGER_RATE_WINDOW.total_seconds() if is_rate else None,
        source_ref=source_ref,
        ref=ref,
    )
    return reading.model_copy(update={"plain": plain_line(reading)})


def _base_line(r: SelfReadingV1) -> str:
    if r.state != "live" or r.percentile_now is None or r.value is None:
        return ""
    pct = int(math.floor(r.percentile_now * 100.0))
    trend = f", {r.trend}" if r.trend in ("rising", "settling") else ""
    if r.rate_window_sec is not None:
        n = int(r.value)
        return (f"{r.label}: {r.score_0_10}/10 for me (went off {n} time{'s' if n != 1 else ''} in the "
                f"last hour; more than in {pct}% of my hours this week{trend}) [{r.ref}]")
    return (f"{r.label}: {r.score_0_10}/10 for me (now {r.value:.2f}; higher than "
            f"{pct}% of my last 7 days{trend}) [{r.ref}]")


def plain_line(r: SelfReadingV1) -> str:
    """One deterministic sentence for a reading. "" for anything not live. A reading with a
    ``tends_to`` clause gets it appended after the reading's own ref."""
    base = _base_line(r)
    if base and r.tends_to:
        return f"{base}; {r.tends_to}"
    return base


def render_self_readings(
    readings: list[SelfReadingV1], *, now: datetime,
    max_shown: int = MAX_SHOWN, max_chars: int = SELF_READINGS_MAX_CHARS,
) -> SelfReadingsBlockV1:
    """The block stance_react / reverie see. ``text`` is "" unless a reading is unusual."""
    now = _aware(now)
    if not readings:
        return SelfReadingsBlockV1(rendered_at=now, skip_reason="no_history")
    unusual = sorted(
        (r for r in readings if r.unusual and r.plain),
        key=lambda r: (-(r.rank_percentile if r.rank_percentile is not None else r.percentile_now or 0.0),
                       r.source_id),
    )
    if not unusual:
        live = any(r.state == "live" for r in readings)
        # "all_calm" only when every signal was actually read (an event signal with no recent
        # event counts as read): the skip log must not call a mostly-unread body calm.
        unread = [r for r in readings if r.state != "live"
                  and not (r.state == "stale" and _event_written(r.source_id))]
        reason = "no_history" if not live else ("calm_partial" if unread else "all_calm")
        return SelfReadingsBlockV1(rendered_at=now, readings=readings, skip_reason=reason)
    # An event-written signal past the staleness horizon had no event to measure (chat with no
    # turns): say so, not "can't read". A per-tick signal gone stale really is unread.
    quiet_events = [r for r in readings if r.state == "stale" and _event_written(r.source_id)]
    # A trigger with too little history has simply not gone off often enough to judge: left out
    # of both lists (never called calm, not "unreadable" either).
    unreadable = [r for r in readings if r.state != "live" and r not in quiet_events
                  and not (r.rate_window_sec is not None and r.state == "insufficient_history")]
    tail = "Nothing else I can read is unusual right now."
    if quiet_events:
        tail += f" No recent events: {', '.join(short_name(r.source_id) for r in quiet_events)}."
    if unreadable:
        names = ", ".join(f"{short_name(r.source_id)} ({_STATE_WORDS[r.state]})" for r in unreadable)
        tail += f" Can't read right now: {names}."
    lines: list[str] = []
    shown: list[SelfReadingV1] = []
    for r in unusual[: max(0, max_shown)]:
        # Selection uses the reading alone: a "tends to" clause never costs Orion a reading.
        candidate = "\n".join([*lines, f"- {_base_line(r)}", tail])
        if len(candidate) > max_chars:
            break
        lines.append(f"- {_base_line(r)}")
        shown.append(r)
    dropped = len(unusual) - len(shown)
    if not lines:
        # Even one line does not fit: say nothing rather than a cut sentence.
        return SelfReadingsBlockV1(rendered_at=now, readings=readings, lines_dropped=dropped,
                                   skip_reason="render_failed")
    if dropped:
        tail = f"{dropped} more unusual signal{'s' if dropped != 1 else ''} not shown. " + tail
    # Then each shown reading's clause, in rank order, while the whole block still fits.
    for i, r in enumerate(shown):
        if r.tends_to and r.plain != _base_line(r):
            trial = [*lines[:i], f"- {r.plain}", *lines[i + 1:]]
            if len("\n".join([*trial, tail])) <= max_chars:
                lines = trial
    text = "\n".join([*lines, tail])
    if len(text) > max_chars:
        text = "\n".join(lines)
    return SelfReadingsBlockV1(
        rendered_at=now, text=text, readings=readings, shown=[r.source_id for r in shown],
        lines_dropped=dropped,
    )


def project_self_readings(
    cache: NodePeHistoryCache, *, now: datetime, associations: Mapping[str, object] | None = None,
    sustain: bool = True,
) -> SelfReadingsBlockV1:
    """build + render. Never raises."""
    now = _aware(now)
    try:
        return render_self_readings(
            build_self_readings(cache, now=now, associations=associations, sustain=sustain), now=now
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("self_readings_render_failed err=%s: %s", type(exc).__name__, exc)
        return SelfReadingsBlockV1(rendered_at=now, skip_reason="render_failed")
