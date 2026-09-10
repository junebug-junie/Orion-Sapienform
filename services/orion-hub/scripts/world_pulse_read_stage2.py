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
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.llm_json import parse_json_object
from orion.journaler.schemas import JournalEntryWriteV1
from orion.llm.routes import fcc_model_for_route
from orion.schemas.world_pulse_read import (
    WorldPulseReadHandoffV1,
    WorldPulseReadSeedV1,
    WorldPulseReadStage2ResultV1,
)
from orion.world_pulse_read.queue import (
    RECLAIM_REASON_PROCESS_RESTART,
    RECLAIM_REASON_STALE_TIMEOUT,
    claim_next_stage2_seed,
    enqueue_seeds,
    mark_stage2_done,
    mark_stage2_failed,
    reclaim_stale_stage2_claimed,
)
from orion.world_pulse_read.seeds import make_seed_id
from orion.world_pulse_read.wallet_a import (
    WalletAInputs,
    read_wallet_a_state,
    wallet_a_block_reason,
)
from orion.world_pulse_read.wallet_b import (
    WalletBInputs,
    debit_wallet_b,
    paced_cooldown_sec,
    read_wallet_b_state,
    wallet_b_block_reason,
    window_is_configured,
)

logger = logging.getLogger("orion-hub.world_pulse_read_stage2")

JOURNAL_WRITE_CHANNEL = "orion:journal:write"
PIPELINE_TAG = "world_pulse_read_stage2"
_AUTHOR = "orion"
_FORCE_OVERRIDE = frozenset({"cooldown", "daily_cap", "outside_window"})
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


def _build_stage2_prompt(handoff: WorldPulseReadHandoffV1, trace_id: str) -> str:
    payload = handoff.model_dump(mode="json")
    return (
        "You are continuing a world-pulse read. Start from this Stage 1 handoff "
        "JSON (do not ignore it). Form/test priors, note hops, and return ONE "
        "JSON object — no prose outside a fenced JSON block.\n"
        f"handoff={payload}\n"
        "Fields: summary (non-empty), need_stage1_urls (list of http(s) URLs "
        "that still need a heavy Stage 1 read, or empty), "
        f"trace_id={trace_id!r}, created_at (ISO-8601 UTC), "
        f"seed_id={handoff.seed_ref.seed_id!r}, "
        "producer_hint=world_pulse_read_stage2."
    )


def _as_stage2_result(raw: Any, *, fallback_trace: str, seed_id: str) -> WorldPulseReadStage2ResultV1:
    if isinstance(raw, WorldPulseReadStage2ResultV1):
        return raw
    if not isinstance(raw, dict):
        raise ValueError("stage2_result_not_object")
    parsed = dict(raw)
    parsed.setdefault("trace_id", fallback_trace)
    parsed.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    parsed.setdefault("seed_id", seed_id)
    parsed["producer_hint"] = "world_pulse_read_stage2"
    return WorldPulseReadStage2ResultV1.model_validate(parsed)


class WorldPulseReadStage2Pipeline:
    def __init__(
        self,
        *,
        enabled: bool,
        tick_interval_sec: float,
        min_cooldown_sec: float,
        daily_cap: int,
        window_start_hour: int = 0,
        window_end_hour: int = 0,
        timeout_sec: float,
        session_id: str,
        llm_route: str = "",
        timezone_name: str = "UTC",
        max_round_trips: int = 5,
        wallet_a_daily_cap: int = 6,
        wallet_a_min_cooldown_sec: float = 1800.0,
        wallet_a_window_start_hour: int = 0,
        wallet_a_window_end_hour: int = 0,
        pool_provider: Callable[[], Any],
        source_ref: ServiceRef,
        step_relay_provider: Optional[Callable[[], Any]] = None,
        store_provider: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.enabled = enabled
        self.tick_interval_sec = tick_interval_sec
        self.min_cooldown_sec = min_cooldown_sec
        self.daily_cap = daily_cap
        self.window_start_hour = int(window_start_hour)
        self.window_end_hour = int(window_end_hour)
        self.timeout_sec = timeout_sec
        self.session_id = session_id
        self.llm_route = str(llm_route or "").strip()
        self._fcc_model_label = fcc_model_for_route(self.llm_route) if self.llm_route else None
        self.timezone_name = timezone_name
        self.max_round_trips = max(0, int(max_round_trips))
        self.wallet_a_daily_cap = int(wallet_a_daily_cap)
        self.wallet_a_min_cooldown_sec = float(wallet_a_min_cooldown_sec)
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

    @property
    def effective_cooldown_sec(self) -> float:
        return paced_cooldown_sec(
            min_cooldown_sec=self.min_cooldown_sec,
            daily_cap=self.daily_cap,
            start_hour=self.window_start_hour,
            end_hour=self.window_end_hour,
        )

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
            "world_pulse_read_stage2 started tick=%ss cooldown=%ss cap=%s round_trips=%s",
            self.tick_interval_sec,
            round(self.effective_cooldown_sec),
            self.daily_cap,
            self.max_round_trips,
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

        redis = self._redis()
        if redis is None:
            since, done_today = None, 0
        else:
            since, done_today = await read_wallet_b_state(
                redis, now=now, timezone_name=self.timezone_name
            )

        local_hour = None
        if window_is_configured(self.window_start_hour, self.window_end_hour) and self._tz_loaded:
            local_hour = now.astimezone(self._tz).hour
        reason = wallet_b_block_reason(
            WalletBInputs(
                enabled=self.enabled,
                done_today=done_today,
                daily_cap=self.daily_cap,
                seconds_since_last=since,
                min_cooldown_sec=self.effective_cooldown_sec,
                now_hour=local_hour,
                window_start_hour=self.window_start_hour,
                window_end_hour=self.window_end_hour,
            )
        )
        if force and reason in _FORCE_OVERRIDE:
            logger.warning(
                "world_pulse_read_stage2_forced overriding=%s done_today=%s cap=%s",
                reason,
                done_today,
                self.daily_cap,
            )
            reason = None
        if reason is not None:
            logger.info("world_pulse_read_stage2_blocked reason=%s", reason)
            return reason

        claim = await self._with_conn(claim_next_stage2_seed)
        if claim is None:
            return "empty_queue"

        if redis is not None:
            await debit_wallet_b(redis, now=now, timezone_name=self.timezone_name)

        try:
            handoff = WorldPulseReadHandoffV1.model_validate(claim.handoff_json)
        except Exception as exc:  # noqa: BLE001
            await self._fail_stage2(claim.seed.seed_id, f"handoff_invalid:{exc}")
            return "handoff_invalid"

        try:
            result = _as_stage2_result(
                await self._stage2_pass(handoff),
                fallback_trace=str(uuid4()),
                seed_id=claim.seed.seed_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "world_pulse_read_stage2_failed seed=%s err=%s", claim.seed.seed_id, exc
            )
            await self._fail_stage2(claim.seed.seed_id, str(exc) or "parse_failed")
            return "parse_failed"

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
            if reentry_reason is not None:
                logger.info(
                    "world_pulse_read_stage2_reentry_stopped reason=%s seed=%s",
                    reentry_reason,
                    claim.seed.seed_id,
                )
                break
            round_trips += 1
        self.last_round_trips = round_trips

        try:
            await self._journal(handoff, result)
            await self._with_conn(
                lambda conn: mark_stage2_done(
                    conn, claim.seed.seed_id, stage2_trace_id=result.trace_id
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "world_pulse_read_stage2_post_failed seed=%s err=%s",
                claim.seed.seed_id,
                exc,
            )
            await self._fail_stage2(claim.seed.seed_id, str(exc) or "post_failed")
            return "post_failed"
        return None

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

    async def _fail_stage2(self, seed_id: str, error: str) -> None:
        await self._with_conn(lambda conn: mark_stage2_failed(conn, seed_id, error=error))

    async def _journal(
        self, handoff: WorldPulseReadHandoffV1, result: WorldPulseReadStage2ResultV1
    ) -> None:
        if self._bus is None:
            return
        seed = handoff.seed_ref
        body = (
            f"{result.summary.strip()}\n\n"
            f"{seed.url}\n"
            f"stage1_trace_id={handoff.trace_id}\n"
            f"stage2_trace_id={result.trace_id}\n"
            f"round_trips={self.last_round_trips}"
        )
        entry = JournalEntryWriteV1(
            author=_AUTHOR,
            mode="manual",
            title=seed.title or "World pulse stage 2",
            body=body,
            source_kind="world_pulse",
            source_ref=f"world_pulse_read_stage2:{result.trace_id}",
            correlation_id=result.trace_id,
        )
        try:
            await self._bus.publish(
                JOURNAL_WRITE_CHANNEL,
                BaseEnvelope(
                    kind="journal.entry.write.v1",
                    source=self._source_ref,
                    payload=entry.model_dump(mode="json"),
                ),
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "world_pulse_read_stage2_journal_failed trace=%s",
                result.trace_id,
                exc_info=True,
            )

    async def _reenter_stage1(
        self, url: str, *, parent_seed: WorldPulseReadSeedV1
    ) -> str | None:
        """Enqueue a finding-style seed for Stage 1. Tests replace this.

        v1 does not call Stage 1 FCC inline (avoids a claim race with the
        Stage 1 loop). Wallet A is gated here; the Stage 1 loop debits when
        it actually reads.
        """
        url = (url or "").strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            return "bad_url"
        redis = self._redis()
        now = datetime.now(timezone.utc)
        since, done_today = None, 0
        if redis is not None:
            since, done_today = await read_wallet_a_state(
                redis, now=now, timezone_name=self.timezone_name
            )
        local_hour = None
        if (
            window_is_configured(self.wallet_a_window_start_hour, self.wallet_a_window_end_hour)
            and self._tz_loaded
        ):
            local_hour = now.astimezone(self._tz).hour
        blocked = wallet_a_block_reason(
            WalletAInputs(
                enabled=True,
                done_today=done_today,
                daily_cap=self.wallet_a_daily_cap,
                seconds_since_last=since,
                min_cooldown_sec=self.wallet_a_min_cooldown_sec,
                now_hour=local_hour,
                window_start_hour=self.wallet_a_window_start_hour,
                window_end_hour=self.wallet_a_window_end_hour,
            )
        )
        if blocked is not None:
            return blocked
        seed = WorldPulseReadSeedV1(
            seed_id=make_seed_id(kind="finding", run_id=parent_seed.run_id, url=url),
            kind="finding",
            run_id=parent_seed.run_id,
            url=url,
            title="",
            section=parent_seed.section,
        )
        await self._with_conn(lambda conn: enqueue_seeds(conn, [seed]))
        return None

    async def _stage2_pass(self, handoff: WorldPulseReadHandoffV1) -> WorldPulseReadStage2ResultV1:
        """Production path: unified turn + fenced JSON. Tests replace this."""
        trace_id = str(uuid4())
        created_at = datetime.now(timezone.utc)
        outcome = await self._generate(_build_stage2_prompt(handoff, trace_id), trace_id)
        if not outcome.text:
            raise ValueError(outcome.fail_reason or "empty_generation")
        parsed = parse_json_object(outcome.text)
        parsed["trace_id"] = trace_id
        parsed.setdefault("created_at", created_at.isoformat())
        parsed["seed_id"] = handoff.seed_ref.seed_id
        parsed["producer_hint"] = "world_pulse_read_stage2"
        return WorldPulseReadStage2ResultV1.model_validate(parsed)

    async def _generate(self, prompt: str, correlation_id: str) -> GenerateOutcome:
        """Real unified-turn generation. Every failure path returns a distinct,
        short `fail_reason` instead of collapsing to a bare empty string --
        see `GenerateOutcome` for why that used to make root-causing a stall
        indistinguishable from five other, very different failures."""
        if self._bus is None:
            return GenerateOutcome("", "bus_unavailable")
        from orion.cognition.cortex_payload_extract import looks_like_error_text
        from orion.hub.turn_orchestrator import execute_unified_turn

        try:
            frames = await asyncio.wait_for(
                execute_unified_turn(
                    bus=self._bus,
                    correlation_id=correlation_id,
                    session_id=self.session_id,
                    user_message=prompt,
                    payload=_turn_payload(PIPELINE_TAG, self._fcc_model_label),
                    continuity_messages=None,
                    harness_rpc_bus=self._harness_rpc_bus or self._bus,
                    harness_step_relay=(
                        self._step_relay_provider() if self._step_relay_provider else None
                    ),
                    harness_step_queue=None,
                ),
                timeout=self.timeout_sec,
            )
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("world_pulse_read_stage2_generate_timeout corr=%s", correlation_id)
            return GenerateOutcome("", "stage2_turn_timeout")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "world_pulse_read_stage2_generate_failed corr=%s err=%s", correlation_id, exc
            )
            detail = str(exc)[:_FAIL_REASON_DETAIL_MAX_LEN]
            return GenerateOutcome("", f"turn_exception:{detail}" if detail else "turn_exception")

        final = next(
            (f for f in frames if isinstance(f, dict) and f.get("type") == "final"), None
        )
        if final is None:
            return GenerateOutcome("", _reason_from_non_final_frame(frames))
        text = str(final.get("llm_response") or "").strip()
        if not text:
            return GenerateOutcome("", "blank_final_response")
        if looks_like_error_text(text):
            return GenerateOutcome("", "looks_like_error_text")
        return GenerateOutcome(text, None)
