"""situation.update: Orion's running situation as a LangGraph thread (spec
docs/superpowers/specs/2026-10-07-situation-graph-design.md, step 2, SHADOW).

One invocation per (coalesced) event on the day's thread ``situation:juniper:<YYYY-MM-DD>``:

    ingest -> reduce -> expire -> cue -> prime_recall -> project -> done

* reduce: re-derives the current facts from ``episode_memory`` (via ``deps.load_facts``) on every
  event, so a step is idempotent: a crash or a dropped event loses nothing the next event cannot
  rebuild. That is why the driver checkpoints once per event (``durability="exit"``) rather than
  per node -- per-node checkpoints would only add rows to the runner's resume sweep.
* expire: facts that were current last revision and no longer are become ``lapsed``.
* cue: referent keys of the current facts, plus known names in the turn's own text.
* prime_recall: memories sharing a cue referent, ranked by strength decayed over their half-life
  -- recall by referent, not by resemblance. Skipped when the cues have not changed.
* project: revision +1 only when facts / primed / lapsed changed; ``deps.project`` writes Redis and
  the bus.

No LLM node and no GPU lease: the distiller already did the thinking.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from orion.memory.episode.validate import PARTICIPANT_REFERENTS, named_referents
from orion.schemas.situation_state import (
    SITUATION_WORKFLOW,
    SituationEventRefV1,
    SituationFactV1,
    SituationJuniperV1,
    SituationLapsedV1,
    SituationPrimedV1,
    SituationRecallV1,
    SituationStateV1,
)

logger = logging.getLogger("orion-durable-runs.situation_graph")

NODES = ("ingest", "reduce", "expire", "cue", "prime_recall", "project", "done")
MAX_DOING = 3
MAX_WAITING = 3
MAX_RECENT = 3
MAX_LAPSED = 3
MAX_PRIMED = 6
MAX_CUES = 12
GIST_CHARS = 240


class SituationGraphState(TypedDict, total=False):
    workflow: str            # always SITUATION_WORKFLOW; the runner's resume sweep skips it
    thread_id: str
    event: dict              # {event_id, kind, correlation_id, text, at}
    situation: dict          # SituationStateV1 json, carried across events by the checkpoint
    rows: list               # this event's fact rows (transient)
    known_keys: list         # referent keys in use (transient)
    facts: dict              # {"whereabouts": ..., "doing": [...], "waiting_on": [...]} (transient)
    lapsed: list
    cues: list
    primed: list
    primed_at: Optional[str]
    primed_cues: list
    primed_revision: int
    prime_ms: Optional[float]
    prime_error: Optional[str]
    changed: bool
    skipped: bool


@dataclass
class SituationDeps:
    load_facts: Callable[[datetime], Awaitable[dict]]            # -> {"rows": [...], "known_keys": [...]}
    prime: Callable[[list, list, int], Awaitable[list]]          # (cues, exclude_ids, limit) -> rows
    project: Callable[[SituationStateV1], Awaitable[None]]
    now: Callable[[], datetime]
    default_ttl: timedelta = timedelta(hours=48)
    prime_timeout_sec: float = 0.4
    reprime_after: timedelta = timedelta(hours=6)


# --- pure helpers (unit-tested directly) ------------------------------------------------------


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


def _gist(text: str) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= GIST_CHARS else t[: GIST_CHARS - 1].rstrip() + "…"


def _is_whereabouts(referents: list[dict]) -> bool:
    return any(str(r.get("key", "")).startswith("place:") and r.get("role") == "location" for r in referents)


def facts_from_rows(rows: list[dict], now: datetime, default_ttl: timedelta) -> dict:
    """Facts by slot. A row counts when its window covers ``now``:

    * follow_up: until its ``expires_at`` (the distiller's follow-up window) -> waiting_on;
    * any other purpose with ``expires_at`` (kept by the validator only on Juniper's own words)
      -> current state: whereabouts when it has a place referent in the ``location`` role (the
      newest wins; older ones stay as doing), otherwise doing;
    * ``happened`` without one -> recent, for ``default_ttl`` after it was told. Never current
      state: ``occurred_at`` is when Juniper told Orion, not when it happened;
    * ``about_juniper`` / ``orion_view`` without one: long-term, not situation -> skipped.
    """
    current: list[SituationFactV1] = []
    for r in rows:
        valid_from = _dt(r.get("occurred_at")) or _dt(r.get("created_at"))
        if valid_from is None:
            continue
        purpose = r.get("purpose")
        expires = _dt(r.get("expires_at"))
        if purpose == "follow_up":
            if expires is None:
                continue
            until, source, slot = expires, "follow_up", "waiting_on"
        elif expires is not None:
            until, source, slot = expires, "juniper_words", None
        elif purpose == "happened":
            until, source, slot = valid_from + default_ttl, "default_ttl", "recent"
        else:
            continue
        if until <= now or valid_from > now:
            continue
        refs = list(r.get("referents") or [])
        if slot is None:
            slot = "whereabouts" if _is_whereabouts(refs) else "doing"
        current.append(SituationFactV1(
            memory_id=str(r["memory_id"]), slot=slot, gist=_gist(r.get("statement", "")),
            valid_from=valid_from, valid_until=until, until_source=source,
            voice=str(r.get("voice") or ""), confirmation=str(r.get("confirmation_state") or ""),
            referents=sorted({str(x.get("key")) for x in refs if x.get("key")}),
        ))
    newest_first = sorted(current, key=lambda f: f.valid_from, reverse=True)
    places = [f for f in newest_first if f.slot == "whereabouts"]
    whereabouts = places[0] if places else None
    doing = [f.model_copy(update={"slot": "doing"}) for f in places[1:]] + [f for f in newest_first if f.slot == "doing"]
    doing.sort(key=lambda f: f.valid_from, reverse=True)
    waiting = sorted((f for f in current if f.slot == "waiting_on"), key=lambda f: f.valid_until)
    recent = [f for f in newest_first if f.slot == "recent"]
    return {
        # Every current memory id, before the per-slot display caps: lapsing is judged against
        # this, so a fact pushed out of a full slot is not mistaken for one that stopped being true.
        "current_ids": sorted(f.memory_id for f in current),
        "whereabouts": whereabouts.model_dump(mode="json") if whereabouts else None,
        "doing": [f.model_dump(mode="json") for f in doing[:MAX_DOING]],
        "waiting_on": [f.model_dump(mode="json") for f in waiting[:MAX_WAITING]],
        "recent": [f.model_dump(mode="json") for f in recent[:MAX_RECENT]],
    }


def _all_facts(juniper: dict) -> list[dict]:
    out = [juniper["whereabouts"]] if juniper.get("whereabouts") else []
    return out + [f for slot in ("doing", "waiting_on", "recent") for f in (juniper.get(slot) or [])]


def lapsed_from(previous: dict, current_ids: list[str], prior_lapsed: list, now: datetime) -> list[dict]:
    """Facts shown last revision that are no longer current at all, newest first, prepended to
    the prior list. ``current_ids`` is every current fact, not just the displayed ones."""
    still = set(current_ids)
    gone = [
        SituationLapsedV1(memory_id=f["memory_id"], slot=f["slot"], gist=f["gist"], lapsed_at=now).model_dump(mode="json")
        for f in _all_facts(previous) if f["memory_id"] not in still
    ]
    kept = [x for x in prior_lapsed if x["memory_id"] not in still and x["memory_id"] not in {g["memory_id"] for g in gone}]
    return (gone + kept)[:MAX_LAPSED]


def cues_for(facts: dict, turn_text: str, known_keys: list[str]) -> list[str]:
    keys = {k for f in _all_facts(facts) for k in f.get("referents", []) if k not in PARTICIPANT_REFERENTS}
    if turn_text:
        keys.update(named_referents(turn_text, known_keys))
    return sorted(keys)[:MAX_CUES]


def rank_primed(rows: list[dict], cues: list[str], now: datetime, limit: int = MAX_PRIMED) -> list[dict]:
    """strength x 0.5^(age / half_life): recall by shared referent, ordered by how alive it is."""
    cue_set = set(cues)
    scored = []
    for r in rows:
        why = sorted(cue_set.intersection(r.get("referent_keys") or []))
        if not why:
            continue
        at = _dt(r.get("last_reinforced_at")) or _dt(r.get("created_at")) or now
        age_days = max((now - at).total_seconds() / 86400.0, 0.0)
        hl = r.get("half_life_days")
        decay = math.pow(0.5, age_days / float(hl)) if hl else 1.0
        scored.append(SituationPrimedV1(
            memory_id=str(r["memory_id"]), gist=_gist(r.get("statement", "")), voice=str(r.get("voice") or ""),
            confirmation=str(r.get("confirmation_state") or ""), why=why[0],
            score=round(float(r.get("strength") or 0.0) * decay, 4),
        ))
    scored.sort(key=lambda p: (-p.score, p.memory_id))
    return [p.model_dump(mode="json") for p in scored[:limit]]


def content_hash(juniper: dict, primed: list, lapsed: list) -> str:
    def strip(fs: list) -> list:
        return [{k: v for k, v in f.items() if k != "lapsed_at"} for f in fs]
    blob = json.dumps({"j": juniper, "p": [p["memory_id"] for p in primed], "l": strip(lapsed)}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def empty_situation(thread_id: str, now: datetime) -> dict:
    return SituationStateV1(thread_id=thread_id, updated_at=now).model_dump(mode="json")


# --- graph ------------------------------------------------------------------------------------


def build_situation_graph(deps: SituationDeps, checkpointer: Any):
    async def ingest(state: SituationGraphState) -> dict:
        ev = dict(state.get("event") or {})
        sit = state.get("situation") or empty_situation(state["thread_id"], deps.now())
        last = (sit.get("last_event") or {}).get("event_id")
        # Step flags are per step: reset them so a skipped step never reports the last one's.
        return {"workflow": SITUATION_WORKFLOW, "situation": sit, "changed": False, "prime_error": None,
                "prime_ms": None, "skipped": bool(ev.get("event_id")) and ev.get("event_id") == last}

    def after_ingest(state: SituationGraphState) -> str:
        return "done" if state.get("skipped") else "reduce"

    async def reduce(state: SituationGraphState) -> dict:
        loaded = await deps.load_facts(deps.now())
        return {"facts": facts_from_rows(loaded.get("rows") or [], deps.now(), deps.default_ttl),
                "known_keys": list(loaded.get("known_keys") or [])}

    async def expire(state: SituationGraphState) -> dict:
        sit = state["situation"]
        return {"lapsed": lapsed_from(sit.get("juniper") or {}, state["facts"].get("current_ids") or [],
                                      list(sit.get("lapsed") or []), deps.now())}

    async def cue(state: SituationGraphState) -> dict:
        text = str((state.get("event") or {}).get("text") or "")
        return {"cues": cues_for(state["facts"], text, state.get("known_keys") or [])}

    async def prime_recall(state: SituationGraphState) -> dict:
        sit = state["situation"]
        rec = sit.get("recall") or {}
        cues = state["cues"]
        primed_at = _dt(rec.get("primed_at"))
        fresh = primed_at is not None and deps.now() - primed_at < deps.reprime_after
        keep = {"primed": list(rec.get("primed") or []), "primed_at": rec.get("primed_at"),
                "primed_cues": list(rec.get("primed_cues") or []),
                "primed_revision": int(rec.get("primed_revision") or 0), "prime_ms": None, "prime_error": None}
        if cues == keep["primed_cues"] and fresh:
            return keep
        if not cues:
            return {**keep, "primed": [], "primed_cues": [], "primed_at": deps.now().isoformat(),
                    "primed_revision": int(sit.get("revision") or 0) + 1}
        exclude = [f["memory_id"] for f in _all_facts(state["facts"])]
        t0 = time.monotonic()
        try:
            rows = await asyncio.wait_for(deps.prime(cues, exclude, MAX_PRIMED * 4), timeout=deps.prime_timeout_sec)
        except Exception as exc:  # noqa: BLE001 - priming is best-effort; the facts still project
            return {**keep, "prime_ms": round((time.monotonic() - t0) * 1000, 1), "prime_error": type(exc).__name__}
        return {"primed": rank_primed(rows, cues, deps.now()), "primed_at": deps.now().isoformat(), "primed_cues": list(cues),
                "primed_revision": int(sit.get("revision") or 0) + 1,
                "prime_ms": round((time.monotonic() - t0) * 1000, 1), "prime_error": None}

    async def project(state: SituationGraphState) -> dict:
        sit = state["situation"]
        now = deps.now()
        before = content_hash(sit.get("juniper") or {}, (sit.get("recall") or {}).get("primed") or [], sit.get("lapsed") or [])
        shown = {k: v for k, v in state["facts"].items() if k != "current_ids"}
        after = content_hash(shown, state["primed"], state["lapsed"])
        changed = before != after or not sit.get("revision")
        revision = int(sit.get("revision") or 0) + (1 if changed else 0)
        ev = state.get("event") or {}
        model = SituationStateV1(
            thread_id=state["thread_id"], revision=revision, updated_at=now,
            last_event=SituationEventRefV1(event_id=str(ev.get("event_id") or ""), kind=ev.get("kind") or "tick",
                                           correlation_id=ev.get("correlation_id")),
            juniper=SituationJuniperV1.model_validate(shown),
            recall=SituationRecallV1(cues=state["cues"], primed_cues=list(state.get("primed_cues") or []),
                                     primed=state["primed"],
                                     primed_at=_dt(state.get("primed_at")),
                                     primed_revision=min(int(state.get("primed_revision") or 0), revision)),
            lapsed=state["lapsed"],
        )
        if changed:
            await deps.project(model)
        return {"situation": model.model_dump(mode="json"), "changed": changed,
                "rows": [], "known_keys": [], "facts": {}}

    async def done(state: SituationGraphState) -> dict:
        return {}

    g = StateGraph(SituationGraphState)
    for name, fn in (("ingest", ingest), ("reduce", reduce), ("expire", expire), ("cue", cue),
                     ("prime_recall", prime_recall), ("project", project), ("done", done)):
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_conditional_edges("ingest", after_ingest, {"reduce": "reduce", "done": "done"})
    for a, b in (("reduce", "expire"), ("expire", "cue"), ("cue", "prime_recall"), ("prime_recall", "project"), ("project", "done")):
        g.add_edge(a, b)
    g.add_edge("done", END)
    return g.compile(checkpointer=checkpointer)
