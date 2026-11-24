"""Reveries that ask: in a quiet stretch, one piece of real material, one question for curiosity.

Spec: docs/superpowers/specs/2026-11-18-reverie-questions-design.md (Patch 1). Replaces the old
coalition-narration reverie (reverie_narrate, the 4-step chain, refractory, EMA, resonance monitor,
room-camera expectations), which is retired: a loud gauge no longer starts a reverie.

One poll (every ORION_REVERIE_INTERVAL_SEC):

1. **Quiet gate** (``reverie_seeds.quiet_gate``, pure): arousal ``idle`` AND no attention winner for
   ORION_REVERIE_QUIET_MIN_CALM_SEC, one reverie per quiet stretch, ORION_REVERIE_MAX_PER_DAY a day.
2. **Seed** (pure builders over read rows): the least recently used kind that has material, among
   absence, lost bet and far memory. No material -> no reverie (nothing written).
3. **Question**: one ``reverie_question`` call on the agent lane (curiosity's lane). The model sees the
   seed and Orion's now (#3499) and nothing else.
4. **Check** (``check_question``, pure): refs must be the seed's; no near-duplicate of an open focus, an
   open self-question or a reverie question of the lapse window; no machinery words. A refusal is stored
   with its reason and routed nowhere.
5. **Persist** one ``reverie_question`` row (this spends the stretch), then **route** an accepted question
   to the curiosity focus slot (origin self, never replacing a pending focus, ORION_REVERIE_FOCUS_MAX_PER_DAY),
   then post the question's cites of the now.

The chronicle reads the table as the ``reverie_question`` experience (orion-durable-runs
temporal_self_sources.py); nothing is published on the bus but the focus request.

A refused lane (``ReverieLlmUnavailable``: shed, deadline, outage, timeout) writes no row; the next poll in
the same stretch tries again, and after 3 refusals the stretch is given up (``lane_refused_stretch``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Sequence
from uuid import uuid4

from orion.cognition.plan_loader import build_plan_for_verb
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef
from orion.gpu_pool.shed import is_gpu_pool_refusal, shed_reason_from_error
from orion.regulation.arousal_read import read_arousal_async
from orion.reverie.semantic_lift import infra_vocabulary_hit
from orion.schemas.cortex.schemas import PlanExecutionArgs, PlanExecutionRequest
from orion.schemas.reverie import (
    MAX_EXPERIENCE_CITES,
    MAX_REVERIE_NONE_REASON_CHARS,
    MAX_REVERIE_QUESTION_CHARS,
    MAX_REVERIE_WHY_CHARS,
    ReverieQuestionV1,
    ReverieSeedV1,
)

from .bus_listener import exec_failure_reason, extract_stance_react_payload
from .cortex_client import CortexExecClient
from .reverie_seeds import (
    MAX_LANE_REFUSALS,
    LivedDoc,
    QuietGateInputs,
    absence_seeds,
    far_memory_seed,
    is_near_duplicate,
    kinds_by_least_recent,
    lost_bet_seeds,
    quiet_gate,
)
from .rpc_health import fold_bus
from .settings import settings

logger = logging.getLogger("orion-thought.reverie")

VERB = "reverie_question"
# The lane curiosity's runs use (services/orion-hub/scripts/curiosity_investigation.py: route "agent").
REVERIE_LLM_ROUTE = "agent"
FOCUS_RPC_TIMEOUT_SEC = 10.0
FOCUS_INVOCATION_CONTEXT = "reverie"
LIVED_MEMORY_COLLECTION = "orion_lived_memory"


class ReverieLlmUnavailable(RuntimeError):
    """The agent lane refused the question call (``gpu_pool_unavailable[:<reason>]``, or a timeout).

    The model was never asked (or never answered), so this says nothing about the seed: no row is written
    and the stretch is not spent (spec "A refused lane does not spend the stretch")."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.shed_reason = shed_reason_from_error(reason)


def _source() -> ServiceRef:
    return ServiceRef(name=settings.service_name, node=settings.node_name, version=settings.service_version)


# Why a poll wrote nothing, counted per process and logged with each skip so the rate is readable from
# the log alone (the retired chain's pattern; Orion checks lane_refused_stretch live).
REVERIE_SKIP_COUNTS: Counter[str] = Counter()


def _count_skip(reason: str, **fields: Any) -> None:
    REVERIE_SKIP_COUNTS[reason] += 1
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info("reverie_skipped reason=%s count=%d%s", reason, REVERIE_SKIP_COUNTS[reason],
                f" {extra}" if extra else "")


# Per stretch (keyed by its start): lane refusals so far, and "no seed had material" (so a stretch with
# nothing to think about is not re-read every poll). Bounded: only the last few stretches are kept.
_STRETCH_REFUSALS: dict[str, int] = {}
_STRETCH_NO_MATERIAL: dict[str, bool] = {}
_STRETCH_MEMORY = 8


def _remember(mapping: dict[str, Any], key: str, value: Any) -> None:
    mapping[key] = value
    while len(mapping) > _STRETCH_MEMORY:
        mapping.pop(next(iter(mapping)))


# --- Orion's now (#3499 read-back) -------------------------------------------------------------------


@dataclass(frozen=True)
class NowCue:
    now_block: str = ""
    shown_id: str | None = None
    shown_event_ids: tuple[str, ...] = ()
    reason: str | None = None


async def build_reverie_now_cue(correlation_id: str, *, run_ref: str) -> NowCue:
    """GET /temporal-self/now?organ=reverie, rendered. Off, unreachable or slow -> the seed alone, logged
    ``reverie_now_unavailable``. Never raises."""
    if not settings.reverie_now_enabled:
        return NowCue(reason="disabled")
    try:
        from .temporal_self_cue import REVERIE_ORGAN, fetch_now_bounded, render_now_block

        read = await fetch_now_bounded(settings.reverie_now_fetch_timeout_sec, organ=REVERIE_ORGAN,
                                       run_ref=run_ref, correlation_id=correlation_id)
        if read.ok and read.now is not None:
            cue = NowCue(now_block=render_now_block(read.now), shown_id=read.now.shown_id,
                         shown_event_ids=tuple(i.event_id for i in read.now.items))
            logger.info("reverie_now corr=%s shown_id=%s items=%d", correlation_id, cue.shown_id,
                        len(cue.shown_event_ids))
            return cue
        reason = read.reason
    except Exception as exc:  # noqa: BLE001 -- a reverie must never fail on this read
        reason = f"error:{type(exc).__name__}"
    logger.info("reverie_now_unavailable corr=%s reason=%s", correlation_id, reason)
    return NowCue(reason=reason)


async def record_reverie_now_use(cue: NowCue, question: ReverieQuestionV1) -> None:
    """At the question's close: post its cites against the now it was shown. Never raises."""
    if not cue.shown_id or not question.cites:
        return
    try:
        from .temporal_self_cue import post_now_use

        result = await post_now_use(cue.shown_id, list(question.cites), f"reverie_question:{question.question_id}",
                                    settings.reverie_now_fetch_timeout_sec, correlation_id=question.correlation_id or "")
    except Exception as exc:  # noqa: BLE001
        logger.warning("reverie_now_use_failed q=%s err=%s", question.question_id, type(exc).__name__)
        return
    if result is not None:
        logger.info("reverie_now_use q=%s shown_id=%s cited=%d stored=%d", question.question_id, cue.shown_id,
                    len(question.cites), len(result.get("uses") or []))


# --- the question call --------------------------------------------------------------------------------


def build_question_plan_request(seed: ReverieSeedV1, *, correlation_id: str, orion_now: str = "") -> PlanExecutionRequest:
    """One ``reverie_question`` call on the agent lane. The context is the seed and the now, nothing else."""
    plan = build_plan_for_verb(VERB, mode="brain")
    context: dict[str, Any] = {
        "seed_kind": seed.kind,
        "seed_material": seed.material,
        "seed_refs": list(seed.refs),
        "seed_evidence": seed.evidence,
        "user_message": None,
        "mode": "reverie",
    }
    if orion_now:
        context["orion_now"] = orion_now
    return PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(
            request_id=correlation_id,
            trigger_source=settings.service_name,
            extra={"llm_route": REVERIE_LLM_ROUTE, "execution_lane": "agent", "mode": "reverie"},
        ),
        context=context,
    )


def parse_model_answer(raw: Any) -> dict[str, Any] | None:
    """The model's JSON object, or None when it is not one (text around the object is ignored)."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(raw[start:end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    s = " ".join(value.split())
    return s if s and s.lower() not in {"null", "none", "n/a"} else None


def check_question(
    answer: Mapping[str, Any] | None,
    *,
    seed: ReverieSeedV1,
    now_event_ids: Sequence[str] = (),
    open_texts: Sequence[str] = (),
) -> dict[str, Any]:
    """Code's check of the model's answer (pure). Returns the ReverieQuestionV1 fields it decides:
    question, why, refs, cites, none_reason, refused_reason, refused_detail.

    Refusals, in order: unparseable; too_long; no_refs / ref_not_in_seed; machinery_words (the structural
    vocabulary check orion/reverie/semantic_lift.py infra_vocabulary_hit); near_duplicate (word-set Jaccard
    >= 0.6 against any open text). A refused question keeps its (clipped) text so the refusal is readable."""
    out: dict[str, Any] = {"question": None, "why": None, "refs": [], "cites": [], "none_reason": None,
                           "refused_reason": None, "refused_detail": None}
    if answer is None:
        out["refused_reason"] = "unparseable"
        return out
    from .temporal_self_cue import shown_cites

    raw_cites = answer.get("cites")
    out["cites"] = shown_cites([str(c) for c in raw_cites] if isinstance(raw_cites, list) else [],
                               list(now_event_ids), cap=MAX_EXPERIENCE_CITES) if now_event_ids else []
    question, why = _text(answer.get("question")), _text(answer.get("why"))
    none_reason = _text(answer.get("none_reason"))
    if question is None:
        if none_reason is None:
            out["refused_reason"], out["refused_detail"] = "unparseable", "neither a question nor a none_reason"
        else:
            out["none_reason"] = none_reason[:MAX_REVERIE_NONE_REASON_CHARS]
        return out
    out["question"] = question[:MAX_REVERIE_QUESTION_CHARS]
    out["why"] = why[:MAX_REVERIE_WHY_CHARS] if why else None
    raw_refs = answer.get("refs")
    refs = [str(r).strip() for r in raw_refs if str(r).strip()] if isinstance(raw_refs, list) else []
    if len(question) > MAX_REVERIE_QUESTION_CHARS or (why and len(why) > MAX_REVERIE_WHY_CHARS):
        out["refused_reason"] = "too_long"
        out["refused_detail"] = f"question {len(question)} chars, why {len(why or '')} chars"
        return out
    if not refs:
        out["refused_reason"] = "no_refs"
        return out
    stray = [r for r in refs if r not in seed.refs]
    if stray:
        out["refused_reason"], out["refused_detail"] = "ref_not_in_seed", ", ".join(stray)[:300]
        return out
    out["refs"] = list(dict.fromkeys(refs))
    if infra_vocabulary_hit(f"{question} {why or ''}"):
        out["refused_reason"] = "machinery_words"
        return out
    dup = next((t for t in open_texts if is_near_duplicate(question, t)), None)
    if dup is not None:
        out["refused_reason"], out["refused_detail"] = "near_duplicate", dup[:300]
        return out
    return out


async def ask_model(
    bus: OrionBusAsync, seed: ReverieSeedV1, *, correlation_id: str, orion_now: str,
    client: CortexExecClient | None = None,
) -> tuple[Any, str | None]:
    """(raw payload, model_id). Raises ReverieLlmUnavailable on a lane refusal or a timeout."""
    client = client or CortexExecClient(bus, request_channel=settings.channel_cortex_exec_request)
    request = build_question_plan_request(seed, correlation_id=correlation_id, orion_now=orion_now)
    try:
        result = await client.execute_plan(source=_source(), req=request, correlation_id=correlation_id,
                                           timeout_sec=settings.stance_react_timeout_sec)
    except asyncio.TimeoutError as exc:
        raise ReverieLlmUnavailable("timeout") from exc
    ok = isinstance(result, dict) and result.get("status") == "success"
    failure = None if ok else exec_failure_reason(result)
    if failure is not None and is_gpu_pool_refusal(failure):
        raise ReverieLlmUnavailable(failure)
    payload = extract_stance_react_payload(result)
    model_id = None
    if isinstance(payload, dict) and payload.get("model_id"):
        model_id = str(payload["model_id"])
    return payload, model_id


# --- routing ------------------------------------------------------------------------------------------


def focus_why(question: ReverieQuestionV1) -> str:
    """The focus row's why: the reverie's own why plus where the question came from (seed and refs), so
    the run can open the material. Bounded by the focus tool's text cap."""
    from orion.schemas.introspect import FOCUS_TEXT_CAP

    base = question.why or "a question from a quiet reverie"
    tail = f" (reverie {question.seed_kind}; reverie_question:{question.question_id}; refs: {', '.join(question.refs)})"
    return (base + tail)[:FOCUS_TEXT_CAP]


async def route_to_focus(bus: OrionBusAsync, question: ReverieQuestionV1) -> tuple[str, str]:
    """Ask Hub to queue the question as a curiosity focus: origin self, keep_pending (never replaces a
    pending focus), Hub's own daily cap. Returns (routed_to, note). Never raises."""
    from orion.core.bus.bus_schemas import BaseEnvelope
    from orion.introspect.transport import (
        QUEUE_CURIOSITY_FOCUS_REQUEST_CHANNEL,
        REQUEST_KIND,
        RESULT_KIND,
        RESULT_PREFIX,
    )
    from orion.schemas.introspect import (
        IntrospectRequestV1,
        IntrospectResultV1,
        IntrospectToolBindingV1,
        QueueCuriosityFocusArguments,
    )

    corr = uuid4()
    reply = f"{RESULT_PREFIX}{corr}"
    try:
        args = QueueCuriosityFocusArguments(question=question.question or "", why=focus_why(question),
                                            origin="self").model_dump(mode="json")
        args.pop("refs", None)  # reverie refs are not orion/refs.py kinds; they ride in the why
        request = IntrospectRequestV1(
            operation="queue_curiosity_focus",
            binding=IntrospectToolBindingV1(invocation_context=FOCUS_INVOCATION_CONTEXT,
                                            parent_run_id=question.question_id,
                                            parent_trace_id=f"reverie:{question.question_id}",
                                            memory_allowed=False),
            args=args,
        )
        raw = await bus.rpc_request(
            QUEUE_CURIOSITY_FOCUS_REQUEST_CHANNEL,
            BaseEnvelope(kind=REQUEST_KIND, correlation_id=corr, reply_to=reply, source=_source(),
                         payload=request.model_dump(mode="json")),
            reply_channel=reply, timeout_sec=FOCUS_RPC_TIMEOUT_SEC,
        )
        decoded = bus.codec.decode(raw.get("data"))
        if not decoded.ok or decoded.envelope.kind != RESULT_KIND:
            return "none", "focus: malformed reply (answer unknown)"
        result = IntrospectResultV1.model_validate(decoded.envelope.payload)
    except Exception as exc:  # noqa: BLE001
        return "none", f"focus: no answer ({type(exc).__name__}; answer unknown)"
    if result.ok and result.items:
        return f"focus:{result.items[0].id}", "queued"
    return "none", f"focus refused: {(result.error or 'refused')[:240]}"


# --- one poll -----------------------------------------------------------------------------------------

SeedReader = Callable[[set[str]], Awaitable[list[ReverieSeedV1]]]


@dataclass
class ReverieDeps:
    """Every read and write a poll makes, injectable for tests."""

    read_arousal: Callable[[], Awaitable[str | None]]
    gate_rows: Callable[[], Awaitable[dict[str, Any]]]
    seed_state: Callable[[], Awaitable[dict[str, Any]]]
    seed_readers: Mapping[str, SeedReader]
    open_texts: Callable[[], Awaitable[list[str]]]
    now_cue: Callable[[str, str], Awaitable[NowCue]]
    ask: Callable[[ReverieSeedV1, str, str], Awaitable[tuple[Any, str | None]]]
    persist: Callable[[ReverieQuestionV1], Awaitable[bool]]
    focus_routes_24h: Callable[[], Awaitable[int]]
    route: Callable[[ReverieQuestionV1], Awaitable[tuple[str, str]]]
    update_route: Callable[[str, str, str | None], Awaitable[bool]]
    record_use: Callable[[NowCue, ReverieQuestionV1], Awaitable[None]]
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)


async def pick_seed_lazily(
    last_used: Mapping[str, datetime | None], readers: Mapping[str, SeedReader], spent_refs: set[str],
) -> ReverieSeedV1 | None:
    """The first seed of the least recently used kind with material. A kind's reader runs only when every
    less-recent kind came back empty (far memory reads the whole Chroma index). A reader that fails is a
    kind with no material this time, logged."""
    for kind in kinds_by_least_recent(last_used):
        reader = readers.get(kind)
        if reader is None:
            continue
        try:
            seeds = await reader(spent_refs)
        except Exception as exc:  # noqa: BLE001 -- one unreadable source never blocks the others
            logger.warning("reverie_seed_read_failed kind=%s err=%s: %s", kind, type(exc).__name__, exc)
            continue
        if seeds:
            return seeds[0]
    return None


async def run_reverie_once(deps: ReverieDeps) -> ReverieQuestionV1 | None:
    """One poll of the quiet gate and, when it is open, one reverie. Returns the stored question row (a
    question, a none_reason or a refusal), or None when nothing was written. Never raises."""
    try:
        return await _run(deps)
    except Exception:
        logger.exception("reverie poll failed")
        return None


async def _run(deps: ReverieDeps) -> ReverieQuestionV1 | None:
    level = await deps.read_arousal()
    try:
        rows = await deps.gate_rows()
    except Exception as exc:  # noqa: BLE001 -- an unreadable gate is a shut gate
        _count_skip("gate_unreadable", err=type(exc).__name__)
        return None
    inputs = QuietGateInputs(now=deps.now(), arousal_level=level, last_winner_at=rows.get("last_winner_at"),
                             latest_tick_at=rows.get("latest_tick_at"),
                             rows_since_stretch=int(rows.get("rows_since_stretch") or 0),
                             rows_last_24h=int(rows.get("rows_last_24h") or 0))
    gate = dict(min_calm_sec=settings.reverie_quiet_min_calm_sec, max_per_day=settings.reverie_max_per_day)
    probe = quiet_gate(inputs, **gate)
    key = probe.stretch_start.isoformat() if probe.stretch_start else "-"
    decision = quiet_gate(replace(inputs, lane_refusals_this_stretch=_STRETCH_REFUSALS.get(key, 0)), **gate)
    if not decision.open:
        _count_skip(decision.reason, calm_sec=int(decision.calm_sec), arousal=level or "unknown")
        return None
    if _STRETCH_NO_MATERIAL.get(key):
        _count_skip("no_material_this_stretch")
        return None

    try:
        state = await deps.seed_state()
    except Exception as exc:  # noqa: BLE001
        _count_skip("seed_state_unreadable", err=type(exc).__name__)
        return None
    seed = await pick_seed_lazily(state.get("last_used") or {}, deps.seed_readers, set(state.get("spent_refs") or ()))
    if seed is None:
        _remember(_STRETCH_NO_MATERIAL, key, True)
        _count_skip("no_material")
        return None
    thread_id = (state.get("threads") or {}).get(seed.thread_key) or str(uuid4())
    seed = seed.model_copy(update={"thread_id": thread_id})

    question_id, correlation_id = str(uuid4()), str(uuid4())
    cue = await deps.now_cue(correlation_id, f"reverie_question:{question_id}")
    try:
        raw, model_id = await deps.ask(seed, correlation_id, cue.now_block)
    except ReverieLlmUnavailable as exc:
        refusals = _STRETCH_REFUSALS.get(key, 0) + 1
        _remember(_STRETCH_REFUSALS, key, refusals)
        reason = "lane_refused_stretch" if refusals >= MAX_LANE_REFUSALS else "llm_unavailable"
        _count_skip(reason, shed=exc.shed_reason or "-", refusals=refusals, detail=exc.reason[:120])
        return None

    try:
        open_texts: list[str] | None = await deps.open_texts()
    except Exception as exc:  # noqa: BLE001 -- without the open questions, duplicates cannot be checked
        logger.warning("reverie_open_texts_unreadable err=%s; question refused as unchecked", type(exc).__name__)
        open_texts = None
    fields = check_question(parse_model_answer(raw), seed=seed, now_event_ids=cue.shown_event_ids,
                            open_texts=open_texts or [])
    if open_texts is None and fields.get("question") and not fields.get("refused_reason"):
        fields["refused_reason"], fields["refused_detail"] = "near_duplicate", "open questions unreadable; unchecked"
    question = ReverieQuestionV1(
        question_id=question_id, thread_id=thread_id, thread_key=seed.thread_key, seed_kind=seed.kind,
        seed_refs=list(seed.refs), seed_material=seed.material, correlation_id=correlation_id,
        model_id=model_id, created_at=deps.now(), **fields,
    )
    if not await deps.persist(question):
        _count_skip("persist_failed", q=question_id)
        return None

    if question.accepted:
        cap = int(settings.reverie_focus_max_per_day)
        try:
            routed = await deps.focus_routes_24h()
        except Exception:  # noqa: BLE001 -- an unknown count does not route: the cap must hold
            routed = cap
        if cap <= 0:
            routed_to, note = "none", "focus routing off (ORION_REVERIE_FOCUS_MAX_PER_DAY=0)"
        elif routed >= cap:
            routed_to, note = "none", f"focus cap reached ({routed} of {cap} in 24 h)"
        else:
            routed_to, note = await deps.route(question)
        await deps.update_route(question_id, routed_to, note)
        question = question.model_copy(update={"routed_to": routed_to, "route_note": note})
    await deps.record_use(cue, question)
    logger.info(
        "reverie_question q=%s kind=%s thread=%s accepted=%s refused=%s none=%s routed=%s",
        question_id, seed.kind, thread_id, question.accepted, question.refused_reason,
        bool(question.none_reason), question.routed_to,
    )
    return question


# --- live deps ----------------------------------------------------------------------------------------


async def far_memory_reader(spent: set[str]) -> list[ReverieSeedV1]:
    """Live lived-memory records (the serving gate) + every vector in orion_lived_memory -> one seed."""
    import httpx

    from orion.introspect.semantic_index import SearchConfig, all_vectors

    from .store import load_live_lived_memory

    records = await load_live_lived_memory()
    docs = {r.doc_id: LivedDoc(r.doc_id, r.text, r.occurred_at, r.created_at) for r in records}
    cfg = SearchConfig(chroma_url=settings.reverie_far_memory_chroma_url, embed_url="-",
                       collection=LIVED_MEMORY_COLLECTION, min_similarity=0.0)
    async with httpx.AsyncClient(timeout=30.0) as client:
        vectors = dict(await all_vectors(client, cfg))
    seed = await asyncio.to_thread(far_memory_seed, docs, vectors, band=settings.reverie_far_memory_band_ranks,
                                   spent_refs=spent)
    return [seed] if seed is not None else []


async def absence_reader(spent: set[str]) -> list[ReverieSeedV1]:
    from .store import load_absence_rows

    rows = await asyncio.to_thread(load_absence_rows)
    return absence_seeds(rows, tz_name=settings.reverie_timezone, spent_refs=spent)


async def lost_bet_reader(spent: set[str]) -> list[ReverieSeedV1]:
    from .store import load_lost_bet_rows

    rows = await asyncio.to_thread(load_lost_bet_rows)
    return lost_bet_seeds(rows, spent_refs=spent)


SEED_READERS: dict[str, SeedReader] = {
    "absence": absence_reader,
    "lost_bet": lost_bet_reader,
    "far_memory": far_memory_reader,
}


def live_deps(bus: OrionBusAsync) -> ReverieDeps:
    from . import store

    async def arousal() -> str | None:
        view = await read_arousal_async(bus, enabled=True, reader="reverie_quiet_gate",
                                        max_age_sec=float(settings.regulation_reader_max_age_sec))
        return view.level if view.known else None

    async def ask(seed: ReverieSeedV1, corr: str, now_block: str) -> tuple[Any, str | None]:
        return await ask_model(bus, seed, correlation_id=corr, orion_now=now_block)

    lapse = float(settings.reverie_thread_lapse_days)
    return ReverieDeps(
        read_arousal=arousal,
        gate_rows=lambda: asyncio.to_thread(store.load_quiet_gate_rows, None),
        seed_state=lambda: asyncio.to_thread(store.load_reverie_seed_state, lapse),
        seed_readers=SEED_READERS,
        open_texts=lambda: asyncio.to_thread(store.load_open_question_texts, lapse),
        now_cue=lambda corr, run_ref: build_reverie_now_cue(corr, run_ref=run_ref),
        ask=ask,
        persist=lambda q: asyncio.to_thread(store.persist_reverie_question, q),
        focus_routes_24h=lambda: asyncio.to_thread(store.count_reverie_focus_routes_24h),
        route=lambda q: route_to_focus(bus, q),
        update_route=lambda qid, r, n: asyncio.to_thread(store.update_reverie_question_route, qid, r, n),
        record_use=record_reverie_now_use,
    )


async def run_reverie_worker(stop_event: asyncio.Event | None = None) -> None:
    """The reverie loop: polls the quiet gate every ORION_REVERIE_INTERVAL_SEC. Off with ORION_REVERIE_ENABLED=false."""
    if not settings.reverie_enabled:
        logger.info("reverie disabled; worker not started")
        return
    if not settings.orion_bus_enabled:
        logger.info("bus disabled; reverie worker not started")
        return
    bus = OrionBusAsync(url=settings.orion_bus_url)
    await bus.connect()
    deps = live_deps(bus)
    logger.info("reverie worker started interval=%ss quiet_min_calm=%ss max_per_day=%s focus_max_per_day=%s",
                settings.reverie_interval_sec, settings.reverie_quiet_min_calm_sec, settings.reverie_max_per_day,
                settings.reverie_focus_max_per_day)
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            await run_reverie_once(deps)
            fold_bus(bus)  # per poll: this worker bus lives for the process (app/rpc_health.py)
            try:
                if stop_event is not None:
                    await asyncio.wait_for(stop_event.wait(), timeout=settings.reverie_interval_sec)
                    break
                await asyncio.sleep(settings.reverie_interval_sec)
            except asyncio.TimeoutError:
                continue
    except asyncio.CancelledError:
        raise
    finally:
        with suppress(Exception):
            await bus.close()
