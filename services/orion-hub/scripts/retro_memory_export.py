"""Retro memory export: one deterministic summary per finished Retro session (Orion plays Zelda, F).

Spec: docs/superpowers/specs/2026-10-22-orion-plays-zelda-design.md section F. Called once per
finished session:

- leisure: scripts/retro_leisure_listener.py, with Lane B's ``LeisureOutcome.ended``
  (``export_leisure_outcome``);
- co-play: ``app.state.retro_session_end_sink`` (``make_session_end_sink``), which Lane B's
  scripts/retro_routes.py calls with a ``RetroSessionEnded`` at every co-play handoff end.

Both pass Lane B's record through ``session_end_inputs``. The per-port ``control`` column stays
``[]`` on these paths: Retro reports frames per controller, not per port.

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

Learned lines (spec 2026-10-23 embodied experience, "Speaking"): at most 3 plain fact lines read back
from orion-prediction's action bets (``fetch_learned_lines`` -> ``learned_lines_from``), carried on
``RetroSessionEnded.learned_lines`` and appended to the card only (the session row's summary is
unchanged). No calibration line under 10 settled bets. Service unreachable -> no lines, a log line,
and the card is still written.
"""

from __future__ import annotations

import hashlib
import json
import math
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
    human_may_be_in_control: bool = False,
    controller_frames: Mapping[str, int] | None = None,
    retro_session_id: str | None = None,
) -> RetroPlaySessionSummaryV1:
    """Pure and deterministic: the same inputs give the same summary.

    Decisions and verdicts come from ``goal_finished`` journal lines (distinct ``decision_id``), or,
    when the journal has none, from ``verdict_counts`` ({met, refuted, unknown}) and ``decisions``
    (goals the planner produced and Retro accepted; a goal still running at the end has no verdict
    and counts as unknown). None when neither is recorded: no decision count is claimed.
    ``human_may_be_in_control`` (Lane B: a human acted after the start, or presence was unreadable)
    drops a leisure session's sponsor rule: with no counts the session is "played together", never
    Orion's alone. ``controller_frames`` is Retro's ``frames_by_controller`` ({human, script, leisure,
    none}): share = human / (human + script + leisure); ``none`` frames drove nobody. Raises ValueError when the journal has no ``session_started`` or
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
    if not (human + orion) and isinstance(controller_frames, Mapping):
        cf = {str(k): max(0, int(v or 0)) for k, v in controller_frames.items()}
        human, orion = cf.get("human", 0), cf.get("script", 0) + cf.get("leisure", 0)
    total = human + orion
    if total:
        basis, share = "frames", human / total
    elif kind == "leisure" and not human_may_be_in_control:
        # Any human acquire preempts the leisure grant and ends the session: every frame was Orion's.
        basis, share = "leisure_sponsor", 0.0
    else:
        basis, share = "unknown", None
    # More than half of the frames by a human, or a human may have been in control and nothing
    # counted the frames: never recorded as Orion's achievement (#2751 section 7.4).
    together = (share is not None and share > 0.5) or (share is None and human_may_be_in_control)

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
        graded = met + refuted + counts.get("unknown", 0)
        if decisions is None and not verdict_counts:
            n_decisions = None
        else:
            n_decisions = max(int(decisions or 0), graded)
        interventions = 0

    items: list[str] = []
    for it in delta.items_gained:
        words = item_words(it.item, it.value)
        if words not in items:
            items.append(words)
    duration = (ended_at - started_at).total_seconds()

    title = game_title(game_id)
    if together and share is None:
        lead = f"Played {title} together for {span_words(duration)} (control share not recorded)"
    elif together:
        lead = f"Played {title} together for {span_words(duration)} (a human drove {round(100 * share)}% of frames)"
    elif kind == "leisure" and not human_may_be_in_control and (
            basis == "leisure_sponsor" or (basis == "frames" and human == 0)):
        lead = f"Played {title} solo for {span_words(duration)}"
    elif basis == "frames":
        # Floored, so a human's few frames never read as "Orion drove 100%".
        lead = f"Played {title} for {span_words(duration)} (Orion drove {math.floor(100 * (1 - share))}% of frames)"
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
    if n_decisions:
        unknown = n_decisions - met - refuted
        parts.append(f"{met} of {n_decisions} predictions met" + (f", {unknown} unknown" if unknown else ""))
    elif n_decisions == 0:
        parts.append("no planner decisions")
    summary = _clip(", ".join(parts) + ".")

    return RetroPlaySessionSummaryV1(
        session_id=session_id, retro_session_id=retro_session_id, lineage_id=lineage_id, game_id=game_id,
        game_title=title, kind=kind,
        dispatch_id=dispatch_id, started_at=started_at, ended_at=ended_at, duration_sec=duration,
        end_reason=(end.reason[:200] if end.reason else None), control=control, control_basis=basis,
        human_frames=human, orion_frames=orion, human_share=share, played_together=together,
        human_may_be_in_control=human_may_be_in_control,
        rooms_new=delta.rooms_new, caves_new=delta.caves_new, items_gained=items, deaths=delta.deaths,
        death_rooms=sorted(set(delta.death_rooms)), decisions=n_decisions, predictions_met=met,
        predictions_refuted=refuted,
        predictions_unknown=(n_decisions - met - refuted) if n_decisions is not None else None,
        human_interventions=interventions, summary=summary,
    )


def _epoch(value: Any) -> str | None:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    return str(value) if value else None


def session_end_inputs(ended: Any) -> dict[str, Any]:
    """Lane B's ``scripts.retro_session_end.RetroSessionEnded`` (or its ``as_dict()``) -> the builder's
    keyword arguments. Both call sites hand the same record: leisure (``controller='orion_leisure'``,
    from the listener) and co-play (``controller='co_play'``, ``leisure_id=None``, from
    ``app.state.retro_session_end_sink``).

    Reads: session_id, controller, lineage_id, game_id, started_at/ended_at (epoch seconds),
    end_reason, act_id, decisions_planned, verdicts, human_may_be_in_control, rooms_new and deaths
    (None when Retro did not report them: left out, never 0), controller_frames.

    Key: a leisure session is one Retro session. A co-play record is one cognition handoff, and one
    Retro session can hold several, so its row key is ``<retro session id>:<handoff start epoch>``."""
    d = ended.as_dict() if hasattr(ended, "as_dict") else dict(ended)
    co_play = d.get("controller") == "co_play"
    sid = str(d.get("session_id") or "")
    key = f"{sid}:{float(d.get('started_at') or 0):.3f}" if co_play and sid else sid
    journal = [{"kind": "session_started", "at": _epoch(d.get("started_at"))},
               {"kind": "session_ended", "at": _epoch(d.get("ended_at")), "reason": d.get("end_reason")}]
    return dict(session_id=key, retro_session_id=sid or None, journal=[j for j in journal if j["at"]],
                atlas_delta={k: d.get(k) for k in ("rooms_new", "deaths") if d.get(k) is not None},
                kind="co_play" if co_play else "leisure", game_id=str(d.get("game_id") or ""),
                lineage_id=d.get("lineage_id"), dispatch_id=None if co_play else d.get("act_id"),
                verdict_counts=d.get("verdicts") or {}, decisions=d.get("decisions_planned"),
                human_may_be_in_control=bool(d.get("human_may_be_in_control")),
                controller_frames=d.get("controller_frames") if isinstance(d.get("controller_frames"), dict) else None)


LEARNED_MAX = 3
MIN_TRIES = 3            # a place or situation line needs at least this many settled tries
MIN_CALIBRATION = 10     # spec: no calibration line under 10 settled bets
_DIR_WORDS = {"N": "north", "S": "south", "E": "east", "W": "west"}
_SIDE_WORDS = {"open_unvisited": "Moving toward rooms I had not visited",
               "open_visited": "Moving back toward rooms I had visited",
               "closed": "Walking toward a side with no known way out",
               "unknown": "Moving toward a side nobody had checked"}
_SKILL_WORDS = {"traverse": "Moving to another room", "engage": "Fighting", "withdraw": "Retreating",
                "acquire": "Picking things up", "enter": "Entering caves", "hold": "Waiting"}


def _place_words(place: str) -> str | None:
    parts = place.split(":")
    if len(parts) != 4 or parts[0] != "zelda" or parts[3] not in _DIR_WORDS:
        return None
    where = f"Room {parts[2]}" if parts[1] == "0" else f"Level {parts[1]} room {parts[2]}"
    return f"{where}'s {_DIR_WORDS[parts[3]]} side"


def learned_lines_from(session_bets: Any, live: Any) -> list[str]:
    """Pure. ``session_bets``: GET /action-bets?session_id= (``{"bets": [...]}`` or a list);
    ``live``: GET /action-bets/live?domain=retro:zelda. Fields are read defensively: anything
    missing just drops its line. Facts and counts only."""
    bets = session_bets.get("bets") if isinstance(session_bets, Mapping) else session_bets
    bets = [b for b in bets or [] if isinstance(b, Mapping) and b.get("result") in ("success", "failure")]
    live = live if isinstance(live, Mapping) else {}
    lines: list[str] = []
    # 1. a place this session tried again and again with one answer every time
    tries: dict[str, list[int]] = {}
    for b in bets:
        if b.get("skill") == "traverse" and isinstance(b.get("place"), str):
            t = tries.setdefault(b["place"], [0, 0])
            t[0 if b["result"] == "success" else 1] += 1
    clear = sorted(((s + f, place, s, f) for place, (s, f) in tries.items()
                    if s + f >= MIN_TRIES and (s == 0 or f == 0) and _place_words(place)), reverse=True)
    if clear:
        n, place, s, _ = clear[0]
        verb = "led to another room" if s else "has not led anywhere"
        lines.append(f"{_place_words(place)} {verb} ({s} of {n} tries).")
    # 2. how moving went this session by side state, else the busiest skill. Raw per-bet counts from
    # this session's own bets: the service's situation counts weigh each place as one unit, so
    # they count places, not tries.
    sides: dict[str, list[int]] = {}
    by_skill: dict[str, list[int]] = {}
    for b in bets:
        i = 0 if b["result"] == "success" else 1
        if b.get("skill") in _SKILL_WORDS:
            by_skill.setdefault(b["skill"], [0, 0])[i] += 1
        side = (b.get("context") or {}).get("side") if isinstance(b.get("context"), Mapping) else None
        if b.get("skill") == "traverse" and side in _SIDE_WORDS:
            sides.setdefault(side, [0, 0])[i] += 1
    best = max(((s + f, side, s) for side, (s, f) in sides.items() if s + f >= MIN_TRIES), default=None)
    if best:
        n, side, s = best
        lines.append(f"{_SIDE_WORDS[side]} worked {s} of {n} times.")
    else:
        top = max(((s + f, sk, s) for sk, (s, f) in by_skill.items() if s + f >= MIN_TRIES), default=None)
        if top:
            lines.append(f"{_SKILL_WORDS[top[1]]} worked {top[2]} of {top[0]} times.")
    # 3. calibration, only with enough settled bets
    grading = live.get("grading") if isinstance(live.get("grading"), Mapping) else {}
    served = grading.get("served") if isinstance(grading.get("served"), Mapping) else {}
    n = served.get("n")
    # "Always guessing my average" is grade()'s own base-rate Brier, not the skill-rate baseline model.
    if (isinstance(n, int) and n >= MIN_CALIBRATION and isinstance(served.get("brier"), (int, float))
            and isinstance(served.get("brier_base_rate"), (int, float))):
        lines.append(f"My guesses about my own moves scored {served['brier']:.2f}. Always guessing my "
                     f"average would score {served['brier_base_rate']:.2f} (lower is better).")
    return lines[:LEARNED_MAX]


def bets_since(bets: Any, since: float | None) -> Any:
    """Only the bets placed at or after ``since`` (epoch seconds). One Retro session holds many
    co-play handoffs; each handoff's card describes its own moves (live 2026-10-23: four cards in a
    row repeated the whole session's "11 of 15")."""
    if since is None:
        return bets
    rows = bets.get("bets") if isinstance(bets, Mapping) else bets
    from datetime import datetime
    def placed(b: Any) -> float | None:
        try:
            return datetime.fromisoformat(str(b.get("placed_at")).replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError, AttributeError):
            return None
    return [b for b in rows or [] if isinstance(b, Mapping) and (placed(b) or 0) >= since]


async def fetch_learned_lines(retro_session_id: str | None, *, get: Any, since: float | None = None) -> list[str]:
    """``get``: async (path, params) -> (status, json), e.g. ZeldaExperience.get. ``since``: only
    bets placed from then on (the handoff or leisure start). Never raises: an unreachable service
    gives no lines and a log line."""
    if not retro_session_id:
        return []
    try:
        status, bets = await get("/action-bets", {"session_id": retro_session_id})
        if status != 200:
            raise ValueError(f"bets_status_{status}")
        status, live = await get("/action-bets/live", {"domain": "retro:zelda"})
        if status != 200:
            raise ValueError(f"live_status_{status}")
        return learned_lines_from(bets_since(bets, since), live)
    except Exception as exc:  # noqa: BLE001
        logger.warning("retro_memory_export_learned_unavailable session_id=%s reason=%s", retro_session_id,
                       f"{type(exc).__name__}:{exc}"[:200])
        return []


def memory_card_for(s: RetroPlaySessionSummaryV1, learned: Iterable[str] = ()) -> MemoryCardCreateV1:
    learned = [str(x) for x in learned or () if str(x).strip()][:LEARNED_MAX]
    day = s.ended_at.astimezone(ZoneInfo(CARD_TZ)).date().isoformat()
    tags = ["retro_play", "game", "played", s.game_id]
    if s.played_together:
        tags.append("played_together")
    return MemoryCardCreateV1(
        # Unique per session (memory_cards.slug is UNIQUE): the session id is the identity.
        slug=("retro-play-" + (re.sub(r"[^a-z0-9]+", "-", s.session_id.lower()).strip("-") or "session"))[:100]
             + "-" + hashlib.sha256(s.session_id.encode()).hexdigest()[:8],
        types=["event"],
        status="active",
        # Every number is a count of journal or atlas records. A share inferred from the leisure
        # sponsor rule (not counted frames) is "likely", not "certain".
        confidence="certain" if s.control_basis != "leisure_sponsor" else "likely",
        sensitivity="private",
        priority="episodic_detail",
        # A finished session stays true: it happened, with these counts, on that day.
        time_horizon=TimeHorizonV1(kind="timeless", as_of=s.ended_at.isoformat()),
        provenance="retro_play",
        trust_source=CARD_SOURCE,
        project="retro",
        title=f"Played {s.game_title} on {day}",
        summary=s.summary + ((" From my own moves: " + " ".join(learned)) if learned else ""),
        anchors=[f"retro_session:{s.retro_session_id or s.session_id}"] + ([f"retro_lineage:{s.lineage_id}"] if s.lineage_id else []),
        tags=tags,
        evidence=[EvidenceItemV1(source=f"{TABLE}:{s.session_id}", excerpt=s.summary, ts=s.ended_at.isoformat())],
        subschema={"retro_play": {
            "session_id": s.session_id, "lineage_id": s.lineage_id, "game_id": s.game_id, "kind": s.kind,
            "dispatch_id": s.dispatch_id, "human_share": s.human_share, "played_together": s.played_together,
            "control": [c.model_dump() for c in s.control], "control_basis": s.control_basis,
            "decisions": s.decisions,
            "predictions_met": s.predictions_met, "learned_lines": learned}},
    )


DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    session_id text PRIMARY KEY, retro_session_id text, lineage_id text, game_id text NOT NULL, game_title text NOT NULL,
    kind text NOT NULL, dispatch_id text, started_at timestamptz NOT NULL, ended_at timestamptz NOT NULL,
    duration_sec double precision NOT NULL, end_reason text, control jsonb NOT NULL DEFAULT '[]'::jsonb,
    control_basis text NOT NULL, human_may_be_in_control boolean NOT NULL DEFAULT false,
    human_frames integer NOT NULL, orion_frames integer NOT NULL,
    human_share double precision, played_together boolean NOT NULL, rooms_new integer, caves_new integer,
    items_gained jsonb NOT NULL DEFAULT '[]'::jsonb, deaths integer,
    death_rooms jsonb NOT NULL DEFAULT '[]'::jsonb, decisions integer,
    predictions_met integer NOT NULL, predictions_refuted integer NOT NULL,
    predictions_unknown integer, human_interventions integer NOT NULL, summary text NOT NULL,
    memory_card_id uuid, card_error text, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS {TABLE}_ended_idx ON {TABLE} (ended_at DESC)
"""

_COLUMNS = ("session_id", "retro_session_id", "lineage_id", "game_id", "game_title", "kind", "dispatch_id", "started_at", "ended_at",
            "duration_sec", "end_reason", "control", "control_basis", "human_may_be_in_control", "human_frames", "orion_frames", "human_share",
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


def _close_note(s: RetroPlaySessionSummaryV1) -> str:
    if s.decisions:
        return f"decisions={s.decisions} met={s.predictions_met}"
    return "no_planner_decisions" if s.decisions == 0 else "decisions_not_recorded"


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
    human_may_be_in_control: bool = False,
    controller_frames: Mapping[str, int] | None = None,
    retro_session_id: str | None = None,
    enabled: bool | None = None,
    learned: Iterable[str] = (),
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
                                  verdict_counts=verdict_counts, decisions=decisions,
                                  human_may_be_in_control=human_may_be_in_control,
                                  controller_frames=controller_frames, retro_session_id=retro_session_id)
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
        # Already exported (a replayed session end): one card per session, never a second. The act
        # close is repeated (same stable act id): the first one may not have been delivered.
        await _close_play(bus, dispatch_id, session_id, produced=bool(s.decisions),
                          note=_close_note(s))
        logger.info("retro_memory_export_duplicate session_id=%s", session_id)
        return ExportResult("duplicate", session_id, summary=s)

    from orion.core.storage import memory_cards as mc_dal

    card_id: UUID | None = None
    card_error: str | None = None
    try:
        card_id = await mc_dal.insert_card(pool, memory_card_for(s, learned), actor="retro_memory_export", op="create")
    except Exception as exc:  # noqa: BLE001
        card_error = f"{type(exc).__name__}: {exc}"[:300]
        logger.warning("retro_memory_export_card_failed session_id=%s error=%s", session_id, card_error)
    try:
        async with pool.acquire() as conn:
            await conn.execute(f"UPDATE {TABLE} SET memory_card_id = $2, card_error = $3 WHERE session_id = $1",
                               session_id, card_id, card_error)
    except Exception:  # noqa: BLE001
        logger.warning("retro_memory_export_link_failed session_id=%s card_id=%s", session_id, card_id, exc_info=True)

    await _close_play(bus, dispatch_id, session_id, produced=bool(s.decisions), note=_close_note(s))
    logger.info("retro_memory_export session_id=%s card_id=%s kind=%s human_share=%s together=%s summary=%r",
                session_id, card_id, s.kind, s.human_share, s.played_together, s.summary)
    return ExportResult("exported", session_id, card_id=str(card_id) if card_id else None,
                        reason=card_error, summary=s)


async def export_session_end(*, pool: Any, bus: Any, ended: Any, dispatch_id: str | None = None,
                             enabled: bool | None = None) -> ExportResult:
    """Lane B's ``RetroSessionEnded`` -> export_session, for both call sites. ``dispatch_id`` (the
    admitted leisure dispatch) wins over the runner's echoed ``act_id``."""
    kw = session_end_inputs(ended)
    if dispatch_id:
        kw["dispatch_id"] = dispatch_id
    d = ended.as_dict() if hasattr(ended, "as_dict") else dict(ended)
    learned = [x for x in d.get("learned_lines") or () if isinstance(x, str)]
    return await export_session(pool=pool, bus=bus, enabled=enabled, learned=learned, **kw)


# The leisure listener's name for the same path.
export_leisure_outcome = export_session_end


def make_session_end_sink(*, pool_provider, bus_provider):
    """``app.state.retro_session_end_sink``: Lane B calls it with every co-play RetroSessionEnded."""
    async def sink(ended: Any) -> ExportResult:
        return await export_session_end(pool=pool_provider(), bus=bus_provider(), ended=ended)
    return sink


__all__ = ["CARD_SOURCE", "fetch_learned_lines", "learned_lines_from", "export_leisure_outcome", "export_session_end", "make_session_end_sink",
           "session_end_inputs", "DDL", "ExportResult", "INSERT_SQL", "TABLE", "build_session_summary", "export_session",
           "game_title", "item_words", "memory_card_for"]
