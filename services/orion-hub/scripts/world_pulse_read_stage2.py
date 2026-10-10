"""World-pulse Stage 2 loop: chew a Stage 1 handoff, journal, optional Stage 1 re-entry.

Sibling of Stage 1 / curiosity_investigation. Debits Wallet B only for the
FCC pass. Never writes orion:curiosity:*. Stage 1 re-entry is enqueue-only
in v1 (finding-style seed) with a hard round-trip ceiling; Wallet A is
checked before enqueue so a depleted Stage 1 wallet stops the loop. The
Stage 1 Hub loop later reads those seeds and debits Wallet A itself.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, NamedTuple, Optional
from uuid import uuid4, uuid5, NAMESPACE_URL
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from orion.core.bus.bus_schemas import ServiceRef
from orion.core.llm_json import parse_json_object
from orion.world_pulse_read.journal import publish_journal
from orion.llm.routes import fcc_model_for_route
from orion.schemas.reading import ReadingRequestedV1
from orion.schemas.reading_turn import ReadingRunBriefV1
from orion.world_pulse_read.durable import (
    ReadingCancelled,
    ReadingPending,
    bind_turn,
    cancel_claim,
    poll_turn,
    reading_retrieval_query,
    release_claim,
)
from orion.world_pulse_read.events import publish_lifecycle
from orion.world_pulse_read.urls import validate_source_url
from orion.schemas.world_pulse_read import (
    WorldPulseReadHandoffV1,
    WorldPulseReadSeedV1,
    WorldPulseReadStage2ResultV1,
    strip_model_claim_receipts,
)
from orion.world_pulse_read.assertions import (
    READING_ACTOR,
    ClaimContextV1,
    build_claim_context,
    claim_prompt_section,
    journal_claims,
    plan_claims,
    stored_object_lookup,
)
from orion.substrate.graph_journal import SubstrateGraphJournal
from orion.world_pulse_read.fetch_text import load_retained_texts
from orion.world_pulse_read.timestamps import stamp_server_created_at
from orion.world_pulse_read.queue import (
    ALREADY_READ,
    READ_URL_SQL,
    RECLAIM_REASON_PROCESS_RESTART,
    RECLAIM_REASON_STALE_TIMEOUT,
    claim_next_stage2_seed,
    enqueue_reading,
    confirm_landings,
    pending_journal_landings,
    _json_object,
    request_for_seed,
    mark_stage2_done,
    mark_stage2_skipped,
    mark_stage2_failed,
    reclaim_stale_stage2_claimed,
    skip_already_read_stage2,
)
from orion.world_pulse_read.retry import is_refused_before_work
from orion.world_pulse_read.read_evidence import NO_READ_EVIDENCE
from orion.world_pulse_read.wallet_a import (
    WalletAInputs,
    read_wallet_a_retry_wait,
    wallet_a_block_reason,
)
from orion.world_pulse_read.wallet_b import (
    WalletBInputs,
    debit_wallet_b,
    read_wallet_b_retry_wait,
    refund_wallet_b,
    settle_wallet_b,
    wallet_b_block_reason,
    window_is_configured,
)

logger = logging.getLogger("orion-hub.world_pulse_read_stage2")

JOURNAL_WRITE_CHANNEL = "orion:journal:write"
PIPELINE_TAG = "world_pulse_read_stage2"
_AUTHOR = "orion"
_FORCE_OVERRIDE = frozenset({"outside_window", "refund_backoff"})
# Refund backoff (a turn refused before reading) doubles from this base per
# consecutive refusal up to _REFUND_BACKOFF_CAP_MULTIPLIER x base: 0.5h, 1h, 2h,
# 4h, 4h... so a full-day capacity outage costs ~8 stance calls instead of one
# per tick. The only pacing left after daily caps and cooldowns were removed.
_REFUND_BACKOFF_BASE_SEC = 1800.0
_REFUND_BACKOFF_CAP_MULTIPLIER = 8
_REFUND_BACKOFF_CAP_SEC = _REFUND_BACKOFF_CAP_MULTIPLIER * _REFUND_BACKOFF_BASE_SEC
# Cap on how much of a raw exception message / non-final-frame error string
# lands in `fail_reason` -- keep it grep-friendly (short label + a hint of
# context), not a full stack trace stuffed into the `stage2_error` column.
_FAIL_REASON_DETAIL_MAX_LEN = 200


class GenerateOutcome(NamedTuple):
    """Result of `_generate`: generated text (empty on any failure) plus a
    specific, short, machine-greppable reason for that failure.

    Before this, every one of the six ways `_generate` can produce no text
    collapsed to a bare `""`, so `_stage2_pass` always raised the identical
    `ValueError("empty_generation")` no matter which of six very different
    situations actually happened (bus never wired up, the whole-turn timeout,
    an exception from the turn itself, no final frame in the response, a
    blank final response, or error-shaped text). `fail_reason` is `None` on
    success (non-empty `text`).
    """

    text: str
    fail_reason: Optional[str] = None
    trace_id: str | None = None


def _reason_from_non_final_frame(frames: list[Any]) -> str:
    """Pull the real failure reason off the last frame when there is no
    `type == "final"` frame in the response, instead of a generic label.

    See `orion/hub/turn_orchestrator.py` for how these frames get built:
    `_harness_error_frame` (`type == "turn_error"`) carries the real cause on
    `error`/`error_code`; `_thought_deferred_frame` (`type == "turn_deferred"`)
    carries it on `reason`. Falls back to `no_final_frame` only when the last
    frame itself carries nothing useful (empty frame list, or a frame shape
    with none of the known reason fields).
    """
    if not frames:
        return "no_final_frame"
    last = frames[-1]
    if not isinstance(last, dict):
        return "no_final_frame"
    frame_type = last.get("type")
    if frame_type == "turn_error":
        code = last.get("error_code")
        if code:
            return f"turn_error:{code}"
        detail = str(last.get("error") or "").strip()
        if detail:
            return f"turn_error:{detail[:_FAIL_REASON_DETAIL_MAX_LEN]}"
        return "turn_error"
    if frame_type == "turn_deferred":
        reason = str(last.get("reason") or "").strip()
        if reason:
            return f"turn_deferred:{reason[:_FAIL_REASON_DETAIL_MAX_LEN]}"
        return "turn_deferred"
    if frame_type:
        return f"non_final_frame:{frame_type}"
    return "no_final_frame"


def _turn_payload(source: str, fcc_model_label: Optional[str]) -> dict:
    payload: dict = {"no_write": True, "source": source}
    if fcc_model_label:
        payload["fcc_model_label"] = fcc_model_label
    return payload


def _build_stage2_prompt(
    handoff: WorldPulseReadHandoffV1, trace_id: str, claim_context: ClaimContextV1 | None = None
) -> str:
    """Prompt and schema must agree, the way Stage 1's prompt already does.

    Before this the prompt said "form/test priors, note hops" but listed only
    summary/need_stage1_urls, so the model returned its priors, tests and
    hops under keys the ``extra="forbid"`` schema then rejected -- 3 of the
    last 8 live Stage 2 completions (2026-09-14/15) were thrown away for
    obeying the instruction. The skeleton below lists exactly the fields
    ``WorldPulseReadStage2ResultV1`` accepts.
    """
    payload = handoff.model_dump(mode="json")
    return (
        "You are continuing a deliberate source reading. Start from this Stage 1 handoff "
        "JSON (do not ignore it). Treat source claims as attributed candidates, not settled truth. "
        "Do not write RDF, execute graph queries, or call Graphiti. Form/test priors, note hops, and return ONE "
        "JSON object — no prose outside a fenced JSON block.\n"
        f"handoff={payload}\n"
        "Required JSON shape (use exactly these top-level keys; anything else is dropped):\n"
        "{\n"
        '  "summary": "non-empty prose",\n'
        '  "priors_tested": [{"claim_ref": "a Stage 1 claim", '
        '"verdict": "supported|revised|refuted|untested", "why": "string"}],\n'
        '  "candidate_priors": [{"claim": "new or revised claim", "confidence": 0.5}],\n'
        '  "concept_candidates": [{"label": "string", "definition": "optional", "link_hints": ["string"]}],\n'
        '  "open_threads": ["string"],\n'
        '  "hops": ["what you fetched/searched and what came back"],\n'
        '  "need_stage1_urls": ["http(s) URLs that still need a heavy Stage 1 read, or empty"],\n'
        f'  "trace_id": {trace_id!r},\n'
        f'  "seed_id": {handoff.seed_ref.seed_id!r}\n'
        "}\n"
        "candidate_priors MUST be objects with claim (not bare strings). "
        "priors_tested MUST be objects with claim_ref. producer_hint and created_at are forced server-side."
        # Only when this read has retained text, stored concepts and existing candidates
        # (orion/world_pulse_read/assertions.py); otherwise the prompt is unchanged.
        + claim_prompt_section(claim_context or ClaimContextV1())
    )


_STAGE2_RESULT_FIELDS = frozenset(WorldPulseReadStage2ResultV1.model_fields)


def _split_unknown_top_level_keys(parsed: dict) -> tuple[dict, list[str]]:
    """Separate the model's known fields from anything it invented at the top
    level. Pure; the caller decides what to log. Nested shapes are left to
    the schema's own coercers/validators."""
    unknown = sorted(k for k in parsed if k not in _STAGE2_RESULT_FIELDS)
    known = {k: v for k, v in parsed.items() if k in _STAGE2_RESULT_FIELDS}
    return known, unknown


def _as_stage2_result(
    raw: Any,
    *,
    fallback_trace: str,
    seed_id: str,
    on_dropped: Optional[Callable[[list[str]], None]] = None,
) -> WorldPulseReadStage2ResultV1:
    """Validate the model's JSON into the Stage 2 result.

    Unknown TOP-LEVEL keys are dropped with a single WARNING naming them (and
    ``on_dropped`` is called so the loop can count) instead of failing the
    turn. This is an explicit, logged drop -- not ``extra="ignore"`` -- so
    prompt/schema drift stays visible in the logs rather than silently
    vanishing (see the repo's history on silent ignores).
    """
    if isinstance(raw, WorldPulseReadStage2ResultV1):
        return raw
    if not isinstance(raw, dict):
        raise ValueError("stage2_result_not_object")
    parsed, unknown = _split_unknown_top_level_keys(strip_model_claim_receipts(dict(raw)))
    if unknown:
        logger.warning(
            "world_pulse_read_stage2_unknown_keys_dropped seed=%s n=%s keys=%s",
            seed_id,
            len(unknown),
            ",".join(unknown),
        )
        if on_dropped is not None:
            on_dropped(unknown)
    parsed.setdefault("trace_id", fallback_trace)
    # Never trust a model-written time, whichever caller handed us raw JSON.
    stamp_server_created_at(parsed, seed_id=seed_id, stage="stage2")
    parsed.setdefault("seed_id", seed_id)
    parsed["producer_hint"] = "world_pulse_read_stage2"
    return WorldPulseReadStage2ResultV1.model_validate(parsed)


class WorldPulseReadStage2Pipeline:
    def __init__(
        self,
        *,
        enabled: bool,
        tick_interval_sec: float,
        window_start_hour: int = 0,
        window_end_hour: int = 0,
        timeout_sec: float,
        session_id: str,
        llm_route: str = "",
        timezone_name: str = "UTC",
        max_round_trips: int = 5,
        max_attempts: int = 1,
        wallet_a_window_start_hour: int = 0,
        wallet_a_window_end_hour: int = 0,
        pool_provider: Callable[[], Any],
        source_ref: ServiceRef,
        step_relay_provider: Optional[Callable[[], Any]] = None,
        store_provider: Optional[Callable[[], Any]] = None,
        durable_url: str = "http://127.0.0.1:8124",
        assertions_enabled: bool = True,
        projector_readiness: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.enabled = enabled
        # HUB_WORLD_PULSE_READ_ASSERTIONS_ENABLED: kill switch for relationship claims
        # (prompt block, journal writes, and Hub's reading assertion projector).
        self.assertions_enabled = bool(assertions_enabled)
        self._projector_readiness = projector_readiness
        self._assertion_projector: Any = None
        self._claim_context: ClaimContextV1 = ClaimContextV1()
        self.durable_url = durable_url
        self.tick_interval_sec = tick_interval_sec
        self.window_start_hour = int(window_start_hour)
        self.window_end_hour = int(window_end_hour)
        self.timeout_sec = timeout_sec
        self.session_id = session_id
        self.llm_route = str(llm_route or "").strip()
        self._fcc_model_label = fcc_model_for_route(self.llm_route) if self.llm_route else None
        self.timezone_name = timezone_name
        self.max_round_trips = max(0, int(max_round_trips))
        # Bounded retry for transient turn failures (orion/world_pulse_read/retry.py).
        # 1 == legacy terminal-on-first-failure.
        self.max_attempts = max(1, int(max_attempts))
        self.wallet_a_window_start_hour = int(wallet_a_window_start_hour)
        self.wallet_a_window_end_hour = int(wallet_a_window_end_hour)
        try:
            self._tz = ZoneInfo(timezone_name)
            self._tz_loaded = True
        except (ZoneInfoNotFoundError, KeyError, Exception):  # noqa: BLE001
            logger.warning("world_pulse_read_stage2_bad_timezone name=%s falling back to UTC", timezone_name)
            self._tz = timezone.utc
            self._tz_loaded = False
        self._pool_provider = pool_provider
        self._source_ref = source_ref
        self._step_relay_provider = step_relay_provider
        self._store_provider = store_provider
        self._startup_reclaim_done = False
        self._bus: Any = None
        self._harness_rpc_bus: Any = None
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self.last_round_trips: int = 0
        # Running count of unknown top-level keys dropped from model output
        # (each is also a WARNING log line naming the keys).
        self.unknown_keys_dropped_total: int = 0

    def _redis(self) -> Any:
        bus = self._bus
        if bus is None:
            return None
        return getattr(bus, "redis", None) or getattr(bus, "_redis", None)

    async def start(self, bus: Any, harness_rpc_bus: Any = None) -> None:
        self._bus = bus
        self._harness_rpc_bus = harness_rpc_bus or bus
        if not self.enabled:
            logger.info("world_pulse_read_stage2 disabled")
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run())
        logger.info(
            "world_pulse_read_stage2 started tick=%ss round_trips=%s max_attempts=%s",
            self.tick_interval_sec,
            self.max_round_trips,
            self.max_attempts,
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.warning("world_pulse_read_stage2_tick_failed", exc_info=True)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.tick_interval_sec)
            except (TimeoutError, asyncio.TimeoutError):
                continue

    async def _with_conn(self, fn):
        pool = self._pool_provider() if self._pool_provider else None
        if pool is None:
            return None
        async with pool.acquire() as conn:
            return await fn(conn)

    async def tick(self, *, force: bool = False) -> str | None:
        now = datetime.now(timezone.utc)
        self.last_round_trips = 0
        await self._reclaim_stale_claimed()
        await self._repair_journal_landings()
        # Every tick, not only after a read: retries decisions whose endpoints or
        # readers were not ready yet.
        await self._project_assertions()
        for landed in (await self._with_conn(confirm_landings) or []):
            await publish_lifecycle(self._bus, landed, "landing_completed", source=self._source_ref)
        try:
            skipped = await self._with_conn(skip_already_read_stage2)
        except Exception:  # noqa: BLE001
            logger.warning("world_pulse_read_stage2_skip_already_read_failed", exc_info=True)
        else:
            if skipped:
                logger.info("world_pulse_read_stage2_skipped_already_read n=%s", skipped)

        redis = self._redis()
        retry_wait = None
        if redis is not None:
            retry_wait = await read_wallet_b_retry_wait(redis, now=now)

        local_hour = None
        if window_is_configured(self.window_start_hour, self.window_end_hour) and self._tz_loaded:
            local_hour = now.astimezone(self._tz).hour
        reason = wallet_b_block_reason(
            WalletBInputs(
                enabled=self.enabled,
                now_hour=local_hour,
                window_start_hour=self.window_start_hour,
                window_end_hour=self.window_end_hour,
                seconds_until_retry=retry_wait,
            )
        )
        if force and reason in _FORCE_OVERRIDE:
            logger.warning(
                "world_pulse_read_stage2_forced overriding=%s", reason,
            )
            reason = None
        if reason == "disabled":
            logger.info("world_pulse_read_stage2_blocked reason=%s", reason)
            return reason

        # Admission gates must not strand an already-submitted durable result.
        claim = await self._with_conn(
            lambda conn: claim_next_stage2_seed(conn, active_only=reason is not None)
        )
        if claim is None:
            if reason is not None:
                logger.info("world_pulse_read_stage2_blocked reason=%s", reason)
            return reason or "empty_queue"

        await publish_lifecycle(self._bus, claim.seed, "stage2_started", source=self._source_ref)

        try:
            handoff = WorldPulseReadHandoffV1.model_validate(claim.handoff_json)
            handoff.seed_ref = claim.seed.model_copy(update={"request": request_for_seed(claim.seed)})
        except Exception as exc:  # noqa: BLE001
            await self._fail_stage2(claim.seed.seed_id, f"handoff_invalid:{exc}")
            await publish_lifecycle(self._bus, claim.seed, "stage2_failed", source=self._source_ref, error="handoff_invalid")
            return "handoff_invalid"

        if not handoff.read_evidence:
            # A Stage 1 handoff with no tool-trace read of its source (every
            # row written before 2026-09-25, and live the hollow networkworld
            # `finding:60d59b10...` read) has nothing for a second pass to
            # build on. Skip before the debit; not a failure of Stage 2.
            await self._with_conn(
                lambda conn: mark_stage2_skipped(
                    conn, claim.seed.seed_id, reason=NO_READ_EVIDENCE
                )
            )
            logger.info(
                "world_pulse_read_stage2_skipped_unread seed=%s", claim.seed.seed_id
            )
            # `stage2_started` was already published at claim; close it.
            await publish_lifecycle(
                self._bus, claim.seed, "stage2_failed", source=self._source_ref,
                error=NO_READ_EVIDENCE,
            )
            return NO_READ_EVIDENCE

        # Debit only once a turn is actually about to run: an invalid stored
        # handoff never reaches the model and must not spend a Wallet B slot.
        receipt = None
        self._settlement_run_id = None
        self._claim_context = await self._build_claim_context(handoff)

        try:
            result = _as_stage2_result(
                await self._stage2_pass(handoff),
                fallback_trace=str(uuid4()),
                seed_id=claim.seed.seed_id,
                on_dropped=self._note_dropped_keys,
            )
        except ReadingCancelled:
            await self._with_conn(lambda conn: cancel_claim(conn, claim.seed.seed_id, 2))
            return "cancelled"
        except ReadingPending as exc:
            await self._with_conn(lambda conn: release_claim(conn, claim.seed.seed_id, 2))
            logger.info("reading_stage2_waiting seed=%s detail=%s", claim.seed.seed_id, exc)
            return "waiting_resource"
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "world_pulse_read_stage2_failed seed=%s err=%s", claim.seed.seed_id, exc
            )
            await self._settle_wallet(redis, receipt, str(exc), claim.seed.seed_id)
            await self._fail_stage2(claim.seed.seed_id, str(exc) or "parse_failed")
            await publish_lifecycle(self._bus, claim.seed, "stage2_failed", source=self._source_ref, error=str(exc))
            return "parse_failed"
        await self._settle_wallet(redis, receipt, None, claim.seed.seed_id)

        result.request = request_for_seed(claim.seed)
        round_trips = 0
        for url in result.need_stage1_urls:
            if round_trips >= self.max_round_trips:
                logger.info(
                    "world_pulse_read_stage2_round_trip_ceiling seed=%s cap=%s",
                    claim.seed.seed_id,
                    self.max_round_trips,
                )
                break
            reentry_reason = await self._reenter_stage1(url, parent_seed=claim.seed)
            if reentry_reason == ALREADY_READ:
                logger.info(
                    "world_pulse_read_stage2_reentry_already_read seed=%s url=%s",
                    claim.seed.seed_id,
                    url,
                )
                continue
            if reentry_reason is not None:
                logger.info(
                    "world_pulse_read_stage2_reentry_stopped reason=%s seed=%s",
                    reentry_reason,
                    claim.seed.seed_id,
                )
                break
            round_trips += 1
        self.last_round_trips = round_trips
        result.round_trips = round_trips
        result = await self._record_claims(claim.seed.seed_id, result)

        try:
            await self._journal(handoff, result)
            await self._with_conn(
                lambda conn: mark_stage2_done(
                    conn, claim.seed.seed_id, stage2_trace_id=result.trace_id, result=result
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "world_pulse_read_stage2_post_failed seed=%s err=%s",
                claim.seed.seed_id,
                exc,
            )
            await self._fail_stage2(claim.seed.seed_id, str(exc) or "post_failed", consume=False)
            await publish_lifecycle(self._bus, claim.seed, "stage2_failed", source=self._source_ref, error=str(exc))
            return "post_failed"
        await publish_lifecycle(self._bus, claim.seed, "stage2_completed", source=self._source_ref, trace_id=result.trace_id)
        await self._project_assertions()
        return None

    async def _build_claim_context(self, handoff: WorldPulseReadHandoffV1) -> ClaimContextV1:
        """Subjects, existing candidates and retained text for this read's claims. Empty
        (no prompt block, no claims) when disabled, or anything needed is missing."""
        if not self.assertions_enabled:
            return ClaimContextV1()
        store = self._store_provider() if self._store_provider else None
        if store is None:
            return ClaimContextV1()
        unretained = [e for e in handoff.read_evidence if not e.content_sha256]
        if unretained:
            # A web read with no retained text: an old read (expected) or a governor /
            # durable-runs still on code that drops SourceFetchEvidenceV1.content_text.
            logger.info("reading_claim_text_missing seed=%s fetches=%d tools=%s",
                        handoff.seed_ref.seed_id, len(unretained),
                        ",".join(sorted({e.tool_name for e in unretained})))
        try:
            texts = await self._with_conn(lambda conn: load_retained_texts(conn, handoff.read_evidence)) or []
            ctx = await asyncio.to_thread(build_claim_context, store, handoff, texts)
        except Exception:  # noqa: BLE001 - a read without claims is still a read
            logger.warning("reading_claim_context_failed seed=%s", handoff.seed_ref.seed_id, exc_info=True)
            return ClaimContextV1()
        logger.info(
            "reading_claim_context seed=%s texts=%d subjects=%d objects=%d usable=%s",
            handoff.seed_ref.seed_id, len(ctx.texts), len(ctx.subjects), len(ctx.objects), ctx.usable,
        )
        return ctx

    async def _record_claims(
        self, seed_id: str, result: WorldPulseReadStage2ResultV1
    ) -> WorldPulseReadStage2ResultV1:
        """Validate the model's claims, journal them, and put a receipt on each."""
        if not self.assertions_enabled:
            return result.model_copy(update={"relationship_claims": []})
        if not result.relationship_claims:
            return result
        store = self._store_provider() if self._store_provider else None
        ctx = self._claim_context
        # Not ctx.usable: the offered object list may have emptied since the prompt was bound.
        lookup = stored_object_lookup(store, ctx) if store is not None and ctx.subjects and ctx.texts else None
        plans = await asyncio.to_thread(
            lambda: plan_claims(result.relationship_claims, ctx, seed_id=seed_id,
                                recorded_at=datetime.now(timezone.utc), object_lookup=lookup))
        pool = self._pool_provider() if self._pool_provider else None
        report = await journal_claims(SubstrateGraphJournal(pool) if pool is not None else None, plans)
        outcomes: dict[str, int] = {}
        for c in report.claims:
            outcomes[c.receipt.reason] = outcomes.get(c.receipt.reason, 0) + 1
        logger.info("reading_claims_recorded seed=%s claims=%d proposals=%d accepted=%d reasons=%s",
                    seed_id, len(report.claims), report.proposals, report.accepted,
                    ",".join(f"{k}:{v}" for k, v in sorted(outcomes.items())))
        return result.model_copy(update={"relationship_claims": report.claims})

    def _build_projector(self) -> Any:
        store = self._store_provider() if self._store_provider else None
        pool = self._pool_provider() if self._pool_provider else None
        if store is None or pool is None:
            return None
        from orion.substrate.assertion_projector import AssertionProjector
        from orion.substrate.materializer import SubstrateGraphMaterializer

        readiness = self._projector_readiness
        if readiness is None:
            import os
            from orion.substrate.reader_capability import readiness as reader_readiness, required_readers_from_env

            uri = str(os.getenv("FALKORDB_URI", "") or "").strip()
            required = required_readers_from_env()
            readiness = lambda: reader_readiness(uri, required)  # noqa: E731
        return AssertionProjector(
            journal=SubstrateGraphJournal(pool), materializer=SubstrateGraphMaterializer(store=store),
            readiness=readiness, proposal_actors=(READING_ACTOR,),
        )

    async def _project_assertions(self) -> None:
        """Apply pending reading decisions to the graph Hub reads and writes (the store the
        reading concepts live in). Memory's projector applies only memory's own claims."""
        if not self.assertions_enabled:
            return
        try:
            if self._assertion_projector is None:
                self._assertion_projector = self._build_projector()
            if self._assertion_projector is None:
                return
            report = await self._assertion_projector.run_once()
        except Exception:  # noqa: BLE001 - one bad pass must not stop reading
            logger.warning("reading_assertion_projection_failed", exc_info=True)
            return
        if report.applied or report.failed:
            logger.info("reading_assertions_projected applied=%d failed=%s waiting=%d",
                        len(report.applied), dict(report.failed), len(report.waiting))

    async def _reclaim_stale_claimed(self) -> None:
        older = 0.0 if not self._startup_reclaim_done else float(self.timeout_sec)
        reason = (
            RECLAIM_REASON_PROCESS_RESTART
            if not self._startup_reclaim_done
            else RECLAIM_REASON_STALE_TIMEOUT
        )
        try:
            reclaimed = await self._with_conn(
                lambda conn: reclaim_stale_stage2_claimed(
                    conn, older_than_sec=older, reason=reason
                )
            )
        except Exception:  # noqa: BLE001
            logger.warning("world_pulse_read_stage2_reclaim_failed", exc_info=True)
            return
        if reclaimed is None:
            return
        self._startup_reclaim_done = True
        if reclaimed:
            logger.info(
                "world_pulse_read_stage2_reclaimed n=%s older_than_sec=%s",
                reclaimed,
                older,
            )

    def _note_dropped_keys(self, keys: list[str]) -> None:
        self.unknown_keys_dropped_total += len(keys)

    async def _settle_wallet(
        self, redis: Any, receipt: Any, reason: Optional[str], seed_id: str
    ) -> None:
        """Decide what the turn's Wallet B debit cost. A turn refused before any
        reading (stance refusal / GPU capacity -- see
        ``orion.world_pulse_read.retry.is_refused_before_work``) gets its slot
        back plus a backed-off retry time; anything that reached the reader
        (``reason`` None on success, or a real failure) keeps the charge and
        resets the refusal streak. Best-effort: a Redis error here must not
        stop the seed from being marked done/failed."""
        try:
            if receipt is None and getattr(self, "_settlement_run_id", None):
                from orion.world_pulse_read.wallet_b import settle_durable_turn

                await settle_durable_turn(redis, run_id=self._settlement_run_id,
                    now=datetime.now(timezone.utc), timezone_name=self.timezone_name,
                    refused=is_refused_before_work(reason),
                    backoff_base_sec=_REFUND_BACKOFF_BASE_SEC,
                    backoff_cap_sec=_REFUND_BACKOFF_CAP_SEC)
                return
            if receipt is None:
                receipt = await debit_wallet_b(
                    redis, now=datetime.now(timezone.utc), timezone_name=self.timezone_name
                )
            if not is_refused_before_work(reason):
                await settle_wallet_b(redis, receipt)
                return
            refunded = await refund_wallet_b(
                redis,
                receipt,
                now=datetime.now(timezone.utc),
                backoff_base_sec=_REFUND_BACKOFF_BASE_SEC,
                backoff_cap_sec=_REFUND_BACKOFF_CAP_SEC,
            )
        except Exception:  # noqa: BLE001
            logger.warning("world_pulse_read_stage2_wallet_settle_failed seed=%s", seed_id, exc_info=True)
            return
        if refunded:
            logger.info(
                "world_pulse_read_stage2_wallet_refunded seed=%s day_key=%s reason=%s",
                seed_id,
                receipt.count_key,
                str(reason)[:_FAIL_REASON_DETAIL_MAX_LEN],
            )

    async def _fail_stage2(self, seed_id: str, error: str, *, consume: bool = True) -> None:
        async def fail(conn):
            async def mark():
                return await mark_stage2_failed(conn, seed_id, error=error, max_attempts=self.max_attempts)
            if consume and getattr(self, "_settlement_run_id", None) and getattr(self, "_settlement_seed_id", None) == seed_id:
                from orion.world_pulse_read.durable import consume_turn
                async with conn.transaction():
                    outcome = await mark()
                    await consume_turn(conn, self._settlement_run_id)
                    return outcome
            return await mark()
        outcome = await self._with_conn(fail)
        if outcome is not None and outcome.retry_scheduled:
            logger.warning(
                "world_pulse_read_stage2_retry_scheduled seed=%s attempts=%s max=%s reason=%s",
                seed_id,
                outcome.attempts,
                self.max_attempts,
                error[:_FAIL_REASON_DETAIL_MAX_LEN],
            )

    async def _journal(self, handoff, result) -> None:
        await publish_journal(self._bus, self._source_ref, handoff, result, round_trips=self.last_round_trips)

    async def _repair_journal_landings(self) -> None:
        for row in (await self._with_conn(pending_journal_landings) or []):
            try:
                handoff = WorldPulseReadHandoffV1.model_validate(_json_object(row["handoff_json"]))
                if row["missing_stage1"]:
                    await publish_journal(self._bus, self._source_ref, handoff)
                if row["missing_stage2"] and row["stage2_status"] == "done" and row["stage2_result_json"]:
                    result = WorldPulseReadStage2ResultV1.model_validate(_json_object(row["stage2_result_json"]))
                    await publish_journal(self._bus, self._source_ref, handoff, result)
            except Exception:
                logger.warning("reading_journal_replay_failed seed=%s", row["seed_id"], exc_info=True)

    async def _reenter_stage1(
        self, url: str, *, parent_seed: WorldPulseReadSeedV1
    ) -> str | None:
        """Enqueue a finding-style seed for Stage 1. Tests replace this.

        v1 does not call Stage 1 FCC inline (avoids a claim race with the
        Stage 1 loop). Wallet A is gated here; the Stage 1 loop debits when
        it actually reads.
        """
        try:
            url = await validate_source_url(url)
        except ValueError:
            return "bad_url"
        # Before the Wallet A gate: a closed gate must not stop the loop on a
        # URL that would be passed on anyway.
        if await self._with_conn(lambda conn: conn.fetchval(READ_URL_SQL, url, "")):
            return ALREADY_READ
        redis = self._redis()
        now = datetime.now(timezone.utc)
        retry_wait = None
        if redis is not None:
            retry_wait = await read_wallet_a_retry_wait(redis, now=now)
        local_hour = None
        if (
            window_is_configured(self.wallet_a_window_start_hour, self.wallet_a_window_end_hour)
            and self._tz_loaded
        ):
            local_hour = now.astimezone(self._tz).hour
        blocked = wallet_a_block_reason(
            WalletAInputs(
                enabled=True,
                now_hour=local_hour,
                window_start_hour=self.wallet_a_window_start_hour,
                window_end_hour=self.wallet_a_window_end_hour,
                seconds_until_retry=retry_wait,
            )
        )
        if blocked is not None:
            return blocked
        parent = request_for_seed(parent_seed)
        request = ReadingRequestedV1(
            request_id=uuid5(NAMESPACE_URL, f"reading-reentry:{parent.request_id}:{url}"),
            url=url, requested_by=parent.requested_by,
            invocation_context=parent.invocation_context, why_now=parent.why_now,
            parent_run_id=parent.parent_run_id, parent_trace_id=parent.parent_trace_id,
            root_request_id=parent.root_request_id or parent.request_id,
            parent_request_id=parent.request_id,
        )
        try:
            receipt = await self._with_conn(lambda conn: enqueue_reading(
                conn, request, bus=self._bus, source=self._source_ref,
                max_round_trips=self.max_round_trips,
            ))
            if receipt is None:
                return "queue_unavailable"
        except ValueError as exc:
            return str(exc)
        if receipt.get("duplicate") == ALREADY_READ:
            return ALREADY_READ
        return None

    async def _stage2_pass(self, handoff: WorldPulseReadHandoffV1) -> WorldPulseReadStage2ResultV1:
        """Production path: unified turn + fenced JSON. Tests replace this."""
        trace_id = str(uuid4())
        outcome = await self._generate(
            _build_stage2_prompt(handoff, trace_id, self._claim_context), trace_id,
            seed_id=handoff.seed_ref.seed_id,
            # Recall searches the source and what stage 1 concluded, not the
            # stage-2 prompt (which embeds the whole stage-1 handoff JSON).
            retrieval_query=reading_retrieval_query(handoff.seed_ref, handoff.what_i_learned),
        )
        trace_id = outcome.trace_id or trace_id
        if not outcome.text:
            raise ValueError(outcome.fail_reason or "empty_generation")
        parsed = parse_json_object(outcome.text)
        parsed["trace_id"] = trace_id
        # Server clock at receipt, never the model's text (see stamp_server_created_at).
        stamp_server_created_at(parsed, seed_id=handoff.seed_ref.seed_id, stage="stage2")
        parsed["seed_id"] = handoff.seed_ref.seed_id
        parsed["request"] = request_for_seed(handoff.seed_ref).model_dump(mode="json")
        return _as_stage2_result(
            parsed,
            fallback_trace=trace_id,
            seed_id=handoff.seed_ref.seed_id,
            on_dropped=self._note_dropped_keys,
        )

    async def _generate(
        self, prompt: str, correlation_id: str, *, seed_id: str, retrieval_query: str | None = None,
    ) -> GenerateOutcome:
        brief = ReadingRunBriefV1(seed_id=seed_id, stage=2, prompt=prompt,
            session_id=self.session_id, timeout_sec=self.timeout_sec,
            fcc_model_label=self._fcc_model_label,
            retrieval_query=retrieval_query)
        try:
            request = await self._with_conn(lambda conn: bind_turn(conn, brief, correlation_id))
            if request is None:
                raise RuntimeError("reading_queue_unavailable")
        except Exception as exc:
            raise ReadingPending(f"reading_binding_unavailable:{type(exc).__name__}") from exc
        self._settlement_run_id = request.run_id
        self._settlement_seed_id = seed_id
        result = await poll_turn(request, self.durable_url)
        return GenerateOutcome(result.text, result.error, result.correlation_id)
