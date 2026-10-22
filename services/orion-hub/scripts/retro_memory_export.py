"""Retro memory export: one deterministic summary per finished Retro session (Orion plays Zelda, F).

Spec: docs/superpowers/specs/2026-10-22-orion-plays-zelda-design.md section F. Called once per
finished session:

- leisure: scripts/retro_leisure_listener.py, with Lane B's ``LeisureOutcome.ended``
  (``export_leisure_outcome``);
- co-play: the co-play session close (Lane B, not wired in this patch) with its journal and atlas
  delta (``export_session(pool=, bus=, session_id=, journal=[...], atlas_delta={...}, kind="co_play",
  game_id=, lineage_id=)``).

What it writes, in order:

1. one ``retro_play_sessions`` row (the claim; the Temporal Self ``retro_play`` source reads it);
2. one ``memory_cards`` row through ``orion.core.storage.memory_cards.insert_card``, provenance and
   trust_source ``retro_play``, linked to session_id and lineage_id (cross-day recall);
3. the card id back onto the session row (or ``card_error`` when the card failed);
4. for a leisure session the motor allocator started (``dispatch_id``): the ``play`` act's close,
   1 when the session had at least one real planner decision, else 0.

The summary is facts and counts only (``build_session_summary``). No model writes it, and it never
says "fun" or "learned". A session where a human drove more than half of the frames is "played
together", never Orion's achievement (#2751 section 7.4). Not exported: frames, RAM, ROM, raw journals.
Nothing here feeds phi, arousal, reward or training. Kill switch: HUB_RETRO_MEMORY_EXPORT_ENABLED
(shipped on); off -> nothing is written, and a leisure act still closes (0, ``export_disabled``).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Literal, Mapping
from uuid import UUID
from zoneinfo import ZoneInfo

from orion.core.contracts.memory_cards import EvidenceItemV1, MemoryCardCreateV1, TimeHorizonV1
from orion.schemas.retro_play import (
    SUMMARY_MAX,
    RetroAtlasDeltaV1,
    RetroJournalEntryV1,
    RetroPlaySessionSummaryV1,
    RetroPortControlV1,
)
from orion.temporal_self.stance_cue import span_words

logger = logging.getLogger("orion-hub.retro_memory_export")

TABLE = "retro_play_sessions"
CARD_SOURCE = "retro_play"
CARD_TZ = "America/Denver"  # Juniper's local day, the same zone the Temporal Self day uses.

# Display names. A game id not listed renders as itself, never as a guessed title.
GAME_TITLES = {"zelda": "The Legend of Zelda"}
# (item, value) -> words. Zelda's $657 sword byte: 1 wooden, 2 white, 3 magical (disassembly
# src/Variables.inc). Anything else renders as its own name with underscores as spaces.
ITEM_WORDS = {("sword", 1): "wooden sword", ("sword", 2): "white sword", ("sword", 3): "magical sword"}


def game_title(game_id: str) -> str:
    """By id segment ("zelda-prg0", "the-legend-of-zelda" -> The Legend of Zelda); else the id."""
    segments = set(game_id.lower().split("-"))
    for key, title in GAME_TITLES.items():
        if key in segments:
            return title
    return game_id


def item_words(item: str, value: int | None) -> str:
    return ITEM_WORDS.get((item, value)) or item.replace("_", " ")


def _times(n: int) -> str:
    return {1: "once", 2: "twice"}.get(n, f"{n} times")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _clip(text: str, limit: int = SUMMARY_MAX) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_session_summary(
    session_id: str,
    journal: Iterable[Mapping[str, Any] | RetroJournalEntryV1],
    atlas_delta: Mapping[str, Any] | RetroAtlasDeltaV1 | None,
    *,
    kind: Literal["co_play", "leisure"],
    game_id: str,
    lineage_id: str | None = None,
    dispatch_id: str | None = None,
    verdict_counts: Mapping[str, int] | None = None,
    decisions: int | None = None,
) -> RetroPlaySessionSummaryV1:
    """Pure and deterministic: the same inputs give the same summary.

    Decisions and verdicts come from ``goal_finished`` journal lines (distinct ``decision_id``), or,
    when the journal has none, from ``verdict_counts`` ({met, refuted, unknown}) and ``decisions``
    (goals the planner produced and Retro accepted; a goal still running at the end has no verdict
    and counts as unknown). Raises ValueError when the journal has no ``session_started`` or
    ``session_ended`` time: a session with no observed bounds is not summarized (no empty-shell
    memory)."""
    entries = [e if isinstance(e, RetroJournalEntryV1) else RetroJournalEntryV1.model_validate(e) for e in journal]
    delta = (atlas_delta if isinstance(atlas_delta, RetroAtlasDeltaV1)
             else RetroAtlasDeltaV1.model_validate(atlas_delta or {}))

    starts = [e.at for e in entries if e.kind == "session_started" and e.at is not None]
    ends = [e for e in entries if e.kind == "session_ended" and e.at is not None]
    if not starts or not ends:
        raise ValueError("journal_incomplete: needs session_started and session_ended times")
    started_at = min(starts)
    end = max(ends, key=lambda e: e.at)
    ended_at = max(end.at, started_at)

    ports: dict[int, list[int]] = {}
    for e in entries:
        if e.kind == "control" and e.port is not None and e.controller is not None and e.frames:
            pair = ports.setdefault(e.port, [0, 0])
            pair[0 if e.controller == "human" else 1] += e.frames
    control = [RetroPortControlV1(port=p, human_frames=h, orion_frames=o) for p, (h, o) in sorted(ports.items())]
    human = sum(c.human_frames for c in control)
    orion = sum(c.orion_frames for c in control)
    total = human + orion
    if total:
        basis, share = "frames", human / total
    elif kind == "leisure":
        # Any human acquire preempts the leisure grant and ends the session: every frame was Orion's.
        basis, share = "leisure_sponsor", 0.0
    else:
        basis, share = "unknown", None
    together = share is not None and share > 0.5

    goals: dict[str, RetroJournalEntryV1] = {}
    for i, e in enumerate(entries):
        if e.kind == "goal_finished":
            goals.setdefault(e.decision_id or f"line:{i}", e)
    if goals:
        verdicts = [g.verdict or "unknown" for g in goals.values()]
        n_decisions, met, refuted = len(goals), verdicts.count("met"), verdicts.count("refuted")
        interventions = sum(1 for g in goals.values() if g.human_intervention)
    else:
        counts = {k: max(0, int(v or 0)) for k, v in dict(verdict_counts or {}).items()}
        met, refuted = counts.get("met", 0), counts.get("refuted", 0)
        n_decisions = max(int(decisions or 0), met + refuted + counts.get("unknown", 0))
        interventions = 0

    items: list[str] = []
    for it in delta.items_gained:
        words = item_words(it.item, it.value)
        if words not in items:
            items.append(words)
    duration = (ended_at - started_at).total_seconds()

    title = game_title(game_id)
    if together:
        lead = f"Played {title} together for {span_words(duration)} (a human drove {round(100 * share)}% of frames)"
    elif basis == "frames" and kind == "co_play":
        lead = f"Played {title} for {span_words(duration)} (Orion drove {round(100 * (1 - share))}% of frames)"
    elif kind == "leisure":
        lead = f"Played {title} solo for {span_words(duration)}"
    else:
        lead = f"Played {title} with a human for {span_words(duration)} (control share not recorded)"
    parts = [lead]
    if delta.rooms_new is not None:
        parts.append(_plural(delta.rooms_new, "new room") + (f" ({_plural(delta.caves_new, 'cave')})"
                                                             if delta.caves_new else ""))
    if items:
        parts.append("got the " + ", the ".join(items))
    if delta.deaths:
        parts.append(f"died {_times(delta.deaths)}")
    parts.append(f"{met} of {n_decisions} predictions met" if n_decisions else "no planner decisions")
    summary = _clip(", ".join(parts) + ".")

    return RetroPlaySessionSummaryV1(
        session_id=session_id, lineage_id=lineage_id, game_id=game_id, game_title=title, kind=kind,
        dispatch_id=dispatch_id, started_at=started_at, ended_at=ended_at, duration_sec=duration,
        end_reason=(end.reason[:200] if end.reason else None), control=control, control_basis=basis,
        human_frames=human, orion_frames=orion, human_share=share, played_together=together,
        rooms_new=delta.rooms_new, caves_new=delta.caves_new, items_gained=items, deaths=delta.deaths,
        death_rooms=sorted(set(delta.death_rooms)), decisions=n_decisions, predictions_met=met,
        predictions_refuted=refuted, predictions_unknown=n_decisions - met - refuted,
        human_interventions=interventions, summary=summary,
    )


def _epoch(value: Any) -> str | None:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    return str(value) if value else None


def leisure_inputs(ended: Any) -> dict[str, Any]:
    """Lane B's ``scripts.retro_session_end.RetroSessionEnded`` (or its ``as_dict()``) -> the
    builder's keyword arguments. Reads: session_id, lineage_id, game_id, started_at/ended_at (epoch
    seconds), end_reason, act_id (the dispatch id), decisions_planned, verdicts, and the optional
    ``atlas_session`` (Lane A's ``zelda.atlas.session`` record: rooms_new, deaths)."""
    d = ended.as_dict() if hasattr(ended, "as_dict") else dict(ended)
    journal = [{"kind": "session_started", "at": _epoch(d.get("started_at"))},
               {"kind": "session_ended", "at": _epoch(d.get("ended_at")), "reason": d.get("end_reason")}]
    atlas = d.get("atlas_session") if isinstance(d.get("atlas_session"), dict) else {}
    return dict(session_id=str(d.get("session_id") or ""), journal=[j for j in journal if j["at"]],
                atlas_delta={k: atlas.get(k) for k in ("rooms_new", "deaths") if atlas.get(k) is not None},
                kind="leisure", game_id=str(d.get("game_id") or ""), lineage_id=d.get("lineage_id"),
                dispatch_id=d.get("act_id"), verdict_counts=d.get("verdicts") or {},
                decisions=d.get("decisions_planned"))


def memory_card_for(s: RetroPlaySessionSummaryV1) -> MemoryCardCreateV1:
    day = s.ended_at.astimezone(ZoneInfo(CARD_TZ)).date().isoformat()
    tags = ["retro_play", "game", "played", s.game_id]
    if s.played_together:
        tags.append("played_together")
    return MemoryCardCreateV1(
        # Unique per session (memory_cards.slug is UNIQUE): the session id is the identity.
        slug=("retro-play-" + (re.sub(r"[^a-z0-9]+", "-", s.session_id.lower()).strip("-") or "session"))[:120],
        types=["event"],
        status="active",
        # Every number is a count of journal or atlas records; nothing is inferred.
        confidence="certain",
        sensitivity="private",
        priority="episodic_detail",
        time_horizon=TimeHorizonV1(kind="timeless", as_of=s.ended_at.isoformat()),
        provenance="retro_play",
        trust_source=CARD_SOURCE,
        project="retro",
        title=f"Played {s.game_title} on {day}",
        summary=s.summary,
        anchors=[f"retro_session:{s.session_id}"] + ([f"retro_lineage:{s.lineage_id}"] if s.lineage_id else []),
        tags=tags,
        evidence=[EvidenceItemV1(source=f"{TABLE}:{s.session_id}", excerpt=s.summary, ts=s.ended_at.isoformat())],
        subschema={"retro_play": {
            "session_id": s.session_id, "lineage_id": s.lineage_id, "game_id": s.game_id, "kind": s.kind,
            "dispatch_id": s.dispatch_id, "human_share": s.human_share, "played_together": s.played_together,
            "control": [c.model_dump() for c in s.control], "control_basis": s.control_basis,
            "decisions": s.decisions,
            "predictions_met": s.predictions_met}},
    )


DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    session_id text PRIMARY KEY, lineage_id text, game_id text NOT NULL, game_title text NOT NULL,
    kind text NOT NULL, dispatch_id text, started_at timestamptz NOT NULL, ended_at timestamptz NOT NULL,
    duration_sec double precision NOT NULL, end_reason text, control jsonb NOT NULL DEFAULT '[]'::jsonb,
    control_basis text NOT NULL, human_frames integer NOT NULL, orion_frames integer NOT NULL,
    human_share double precision, played_together boolean NOT NULL, rooms_new integer, caves_new integer,
    items_gained jsonb NOT NULL DEFAULT '[]'::jsonb, deaths integer,
    death_rooms jsonb NOT NULL DEFAULT '[]'::jsonb, decisions integer NOT NULL,
    predictions_met integer NOT NULL, predictions_refuted integer NOT NULL,
    predictions_unknown integer NOT NULL, human_interventions integer NOT NULL, summary text NOT NULL,
    memory_card_id uuid, card_error text, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS {TABLE}_ended_idx ON {TABLE} (ended_at DESC)
"""

_COLUMNS = ("session_id", "lineage_id", "game_id", "game_title", "kind", "dispatch_id", "started_at", "ended_at",
            "duration_sec", "end_reason", "control", "control_basis", "human_frames", "orion_frames", "human_share",
            "played_together", "rooms_new", "caves_new", "items_gained", "deaths", "death_rooms", "decisions",
            "predictions_met", "predictions_refuted", "predictions_unknown", "human_interventions", "summary")
_JSON_COLUMNS = {"control", "items_gained", "death_rooms"}
INSERT_SQL = (
    f"INSERT INTO {TABLE} ({', '.join(_COLUMNS)}) VALUES ("
    + ", ".join(f"${i + 1}::jsonb" if c in _JSON_COLUMNS else f"${i + 1}" for i, c in enumerate(_COLUMNS))
    + ") ON CONFLICT (session_id) DO NOTHING RETURNING session_id"
)


def _row_values(s: RetroPlaySessionSummaryV1) -> list[Any]:
    data = s.model_dump(mode="python")
    data["control"] = [c.model_dump() for c in s.control]
    return [json.dumps(data[c]) if c in _JSON_COLUMNS else data[c] for c in _COLUMNS]


@dataclass(frozen=True)
class ExportResult:
    state: Literal["exported", "duplicate", "disabled", "skipped", "failed"]
    session_id: str
    card_id: str | None = None
    reason: str | None = None
    summary: RetroPlaySessionSummaryV1 | None = None


def _enabled() -> bool:
    try:
        from scripts.settings import settings

        return bool(getattr(settings, "HUB_RETRO_MEMORY_EXPORT_ENABLED", True))
    except Exception:  # noqa: BLE001
        return True


async def _close_play(bus: Any, dispatch_id: str | None, session_id: str, *, produced: bool, note: str) -> None:
    if not dispatch_id:
        return
    from scripts.act_emit import publish_play_closed

    await publish_play_closed(bus, dispatch_id=dispatch_id, session_id=session_id, produced=produced, note=note)


_table_ready = False


async def _ensure_table(conn: Any) -> None:
    global _table_ready
    if _table_ready:
        return
    async with conn.transaction():
        await conn.execute("SET LOCAL lock_timeout = '3s'")
        await conn.execute(DDL)
    _table_ready = True


async def export_session(
    *,
    pool: Any,
    bus: Any,
    session_id: str,
    journal: Iterable[Mapping[str, Any] | RetroJournalEntryV1],
    atlas_delta: Mapping[str, Any] | RetroAtlasDeltaV1 | None,
    kind: Literal["co_play", "leisure"],
    game_id: str,
    lineage_id: str | None = None,
    dispatch_id: str | None = None,
    verdict_counts: Mapping[str, int] | None = None,
    decisions: int | None = None,
    enabled: bool | None = None,
) -> ExportResult:
    """Build, store and (for a leisure dispatch) close the act. Never raises: a failed export is
    returned and logged, and the session itself is already over."""
    on = _enabled() if enabled is None else enabled
    if not on:
        await _close_play(bus, dispatch_id, session_id, produced=False, note="export_disabled")
        logger.info("retro_memory_export_disabled session_id=%s", session_id)
        return ExportResult("disabled", session_id, reason="HUB_RETRO_MEMORY_EXPORT_ENABLED=false")
    try:
        if not session_id:
            raise ValueError("no_session_id: the session never started in Retro")
        s = build_session_summary(session_id, journal, atlas_delta, kind=kind, game_id=game_id,
                                  lineage_id=lineage_id, dispatch_id=dispatch_id,
                                  verdict_counts=verdict_counts, decisions=decisions)
    except ValueError as exc:
        await _close_play(bus, dispatch_id, session_id, produced=False, note=str(exc)[:200])
        logger.warning("retro_memory_export_skipped session_id=%s reason=%s", session_id, exc)
        return ExportResult("skipped", session_id, reason=str(exc)[:200])
    if pool is None:
        await _close_play(bus, dispatch_id, session_id, produced=False, note="no_memory_pool")
        logger.warning("retro_memory_export_failed session_id=%s reason=no_memory_pool", session_id)
        return ExportResult("failed", session_id, reason="no_memory_pool", summary=s)

    try:
        async with pool.acquire() as conn:
            await _ensure_table(conn)
            claimed = await conn.fetchval(INSERT_SQL, *_row_values(s))
    except Exception as exc:  # noqa: BLE001
        await _close_play(bus, dispatch_id, session_id, produced=False, note="session_row_failed")
        logger.warning("retro_memory_export_failed session_id=%s reason=session_row error=%s", session_id, exc)
        return ExportResult("failed", session_id, reason=f"session_row:{type(exc).__name__}", summary=s)
    if claimed is None:
        # Already exported (a replayed session end): one card per session, never a second.
        logger.info("retro_memory_export_duplicate session_id=%s", session_id)
        return ExportResult("duplicate", session_id, summary=s)

    from orion.core.storage import memory_cards as mc_dal

    card_id: UUID | None = None
    card_error: str | None = None
    try:
        card_id = await mc_dal.insert_card(pool, memory_card_for(s), actor="retro_memory_export", op="create")
    except Exception as exc:  # noqa: BLE001
        card_error = f"{type(exc).__name__}: {exc}"[:300]
        logger.warning("retro_memory_export_card_failed session_id=%s error=%s", session_id, card_error)
    try:
        async with pool.acquire() as conn:
            await conn.execute(f"UPDATE {TABLE} SET memory_card_id = $2, card_error = $3 WHERE session_id = $1",
                               session_id, card_id, card_error)
    except Exception:  # noqa: BLE001
        logger.warning("retro_memory_export_link_failed session_id=%s card_id=%s", session_id, card_id, exc_info=True)

    produced = s.decisions >= 1
    await _close_play(bus, dispatch_id, session_id, produced=produced,
                      note=(f"decisions={s.decisions} met={s.predictions_met}" if produced else "no_planner_decisions"))
    logger.info("retro_memory_export session_id=%s card_id=%s kind=%s human_share=%s together=%s summary=%r",
                session_id, card_id, s.kind, s.human_share, s.played_together, s.summary)
    return ExportResult("exported", session_id, card_id=str(card_id) if card_id else None,
                        reason=card_error, summary=s)


async def export_leisure_outcome(*, pool: Any, bus: Any, ended: Any, enabled: bool | None = None) -> ExportResult:
    """The leisure path: Lane B's ``RetroSessionEnded`` (``LeisureOutcome.ended``) -> export_session."""
    return await export_session(pool=pool, bus=bus, enabled=enabled, **leisure_inputs(ended))


__all__ = ["CARD_SOURCE", "export_leisure_outcome", "leisure_inputs", "DDL", "ExportResult", "INSERT_SQL", "TABLE", "build_session_summary", "export_session",
           "game_title", "item_words", "memory_card_for"]
