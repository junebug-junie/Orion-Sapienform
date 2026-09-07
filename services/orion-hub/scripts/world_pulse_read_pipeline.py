"""World-pulse Stage 1 loop: dequeue a seed, read, land concepts, journal.

Sibling of curiosity_investigation — same tick / Wallet / unified-turn
lifecycle, different Redis keys and a Postgres seed queue. Never writes
orion:curiosity:* and never calls curiosity debit APIs.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.llm_json import parse_json_object
from orion.journaler.schemas import JournalEntryWriteV1
from orion.llm.routes import fcc_model_for_route
from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1, WorldPulseReadSeedV1
from orion.substrate.adapters.world_pulse_read import map_world_pulse_read_handoff_to_substrate
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.world_pulse_read.queue import (
    claim_next_seed,
    enqueue_from_recent_digests,
    mark_seed_done,
    mark_seed_failed,
    reclaim_stale_claimed,
)
from orion.world_pulse_read.wallet_a import (
    WalletAInputs,
    debit_wallet_a,
    paced_cooldown_sec,
    read_wallet_a_state,
    wallet_a_block_reason,
    window_is_configured,
)

logger = logging.getLogger("orion-hub.world_pulse_read_pipeline")

JOURNAL_WRITE_CHANNEL = "orion:journal:write"
PIPELINE_TAG = "world_pulse_read"
_AUTHOR = "orion"
_FORCE_OVERRIDE = frozenset({"cooldown", "daily_cap", "outside_window"})


def _turn_payload(source: str, fcc_model_label: Optional[str]) -> dict:
    payload: dict = {"no_write": True, "source": source}
    if fcc_model_label:
        payload["fcc_model_label"] = fcc_model_label
    return payload


def _build_stage1_prompt(seed: WorldPulseReadSeedV1, trace_id: str) -> str:
    return (
        "Read this world-pulse article and return ONE JSON object — no prose "
        "outside a fenced JSON block.\n"
        f"seed_id={seed.seed_id} kind={seed.kind} run_id={seed.run_id}\n"
        f"url={seed.url}\ntitle={seed.title}\nsection={seed.section}\n"
        "Fields: seed_ref (echo the seed), what_i_learned (non-empty), "
        "candidate_priors, concept_candidates (label + optional definition), "
        f"open_threads, trace_id={trace_id!r}, created_at (ISO-8601 UTC), "
        "producer_hint=world_pulse_read_pipeline."
    )


class WorldPulseReadPipeline:
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
        try:
            self._tz = ZoneInfo(timezone_name)
            self._tz_loaded = True
        except (ZoneInfoNotFoundError, KeyError, Exception):  # noqa: BLE001
            logger.warning("world_pulse_read_bad_timezone name=%s falling back to UTC", timezone_name)
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
            logger.info("world_pulse_read_pipeline disabled")
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run())
        logger.info(
            "world_pulse_read_pipeline started tick=%ss cooldown=%ss cap=%s",
            self.tick_interval_sec,
            round(self.effective_cooldown_sec),
            self.daily_cap,
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
                logger.warning("world_pulse_read_tick_failed", exc_info=True)
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
        await self._reclaim_stale_claimed()
        await self._maybe_enqueue_recent()

        redis = self._redis()
        if redis is None:
            since, done_today = None, 0
        else:
            since, done_today = await read_wallet_a_state(
                redis, now=now, timezone_name=self.timezone_name
            )

        local_hour = None
        if window_is_configured(self.window_start_hour, self.window_end_hour) and self._tz_loaded:
            local_hour = now.astimezone(self._tz).hour
        reason = wallet_a_block_reason(
            WalletAInputs(
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
                "world_pulse_read_forced overriding=%s done_today=%s cap=%s",
                reason,
                done_today,
                self.daily_cap,
            )
            reason = None
        if reason is not None:
            logger.info("world_pulse_read_blocked reason=%s", reason)
            return reason

        seed = await self._with_conn(claim_next_seed)
        if seed is None:
            return "empty_queue"

        if redis is not None:
            await debit_wallet_a(redis, now=now, timezone_name=self.timezone_name)

        try:
            handoff = await self._stage1_read(seed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("world_pulse_read_stage1_failed seed=%s err=%s", seed.seed_id, exc)
            await self._fail_seed(seed.seed_id, str(exc) or "parse_failed")
            return "parse_failed"
        if handoff is None:
            await self._fail_seed(seed.seed_id, "empty_generation")
            return "empty_generation"

        try:
            record = map_world_pulse_read_handoff_to_substrate(handoff)
            if record.nodes:
                store = self._store_provider() if self._store_provider else None
                if store is None:
                    raise RuntimeError("concept_atlas_store_unavailable")
                SubstrateGraphMaterializer(store=store).apply_record(record)

            await self._journal(handoff)
            await self._with_conn(
                lambda conn: mark_seed_done(conn, seed.seed_id, trace_id=handoff.trace_id)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "world_pulse_read_post_read_failed seed=%s err=%s", seed.seed_id, exc
            )
            await self._fail_seed(seed.seed_id, str(exc) or "post_read_failed")
            return "post_read_failed"
        return None

    async def _reclaim_stale_claimed(self) -> None:
        # First tick after start: free every leftover claimed row (Hub
        # just came up; the previous process is gone). Later ticks only
        # reclaim claims older than timeout_sec so a live FCC turn is
        # not stolen. Pool is created after pipeline.start() in Hub
        # startup, so start() itself cannot do this.
        older = 0.0 if not self._startup_reclaim_done else float(self.timeout_sec)
        try:
            reclaimed = await self._with_conn(
                lambda conn: reclaim_stale_claimed(conn, older_than_sec=older)
            )
        except Exception:  # noqa: BLE001
            logger.warning("world_pulse_read_reclaim_failed", exc_info=True)
            return
        if reclaimed is None:
            return
        self._startup_reclaim_done = True
        if reclaimed:
            logger.info(
                "world_pulse_read_reclaimed n=%s older_than_sec=%s",
                reclaimed,
                older,
            )

    async def _maybe_enqueue_recent(self) -> None:
        try:
            await self._with_conn(lambda conn: enqueue_from_recent_digests(conn, limit_digests=3))
        except Exception:  # noqa: BLE001
            logger.warning("world_pulse_read_enqueue_failed", exc_info=True)

    async def _fail_seed(self, seed_id: str, error: str) -> None:
        await self._with_conn(lambda conn: mark_seed_failed(conn, seed_id, error=error))

    async def _journal(self, handoff: WorldPulseReadHandoffV1) -> None:
        if self._bus is None:
            return
        seed = handoff.seed_ref
        body = (
            f"{handoff.what_i_learned.strip()}\n\n"
            f"{seed.url}\n"
            f"trace_id={handoff.trace_id}"
        )
        entry = JournalEntryWriteV1(
            author=_AUTHOR,
            mode="manual",
            title=seed.title or "World pulse read",
            body=body,
            source_kind="world_pulse",
            source_ref=f"world_pulse_read:{handoff.trace_id}",
            correlation_id=handoff.trace_id,
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
                "world_pulse_read_journal_failed trace=%s", handoff.trace_id, exc_info=True
            )

    async def _stage1_read(self, seed: WorldPulseReadSeedV1) -> WorldPulseReadHandoffV1:
        """Production path: unified turn + fenced JSON. Tests replace this."""
        trace_id = str(uuid4())
        created_at = datetime.now(timezone.utc)
        text = await self._generate(_build_stage1_prompt(seed, trace_id), trace_id)
        if not text:
            raise ValueError("empty_generation")
        parsed = parse_json_object(text)
        parsed["trace_id"] = trace_id
        parsed["seed_ref"] = seed.model_dump(mode="json")
        parsed.setdefault("created_at", created_at.isoformat())
        parsed["producer_hint"] = "world_pulse_read_pipeline"
        return WorldPulseReadHandoffV1.model_validate(parsed)

    async def _generate(self, prompt: str, correlation_id: str) -> str:
        if self._bus is None:
            return ""
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
            logger.warning("world_pulse_read_generate_timeout corr=%s", correlation_id)
            return ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("world_pulse_read_generate_failed corr=%s err=%s", correlation_id, exc)
            return ""

        final = next(
            (f for f in frames if isinstance(f, dict) and f.get("type") == "final"), None
        )
        if final is None:
            return ""
        text = str(final.get("llm_response") or "").strip()
        if looks_like_error_text(text):
            return ""
        return text
