"""Phase C — reverie chain (a train of thought).

Successive reverie steps that climb the ladder, terminating when the spawning
pressure discharges (or a step-cap / low-salience floor). Continuity is the
last-n verbatim thought ids; the wide-n memory is a *lossy* EMA low-pass, never
a growing verbatim window. A discharged theme is put in a refractory table so it
cannot immediately re-ignite (ouroboros habituation).

Everything safety-relevant here is deterministic (§4) and injectable, so
termination / EMA / refractory are unit-testable without the live mesh. The LLM
only writes each step's thought text (via the injected step_fn). This module
never reads a dream — no process reads its own output kind.

Default-off: `run_reverie_chain_worker` is a no-op unless ORION_REVERIE_CHAIN_ENABLED.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Protocol
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
from orion.schemas.reverie import (
    MAX_CHAIN_THOUGHTS,
    CompactionRequestV1,
    ReverieChainV1,
    SpontaneousThoughtV1,
    TerminalReason,
)

from .reverie import _default_broadcast_reader, _source, run_reverie_once
from .settings import settings
from .store import (
    persist_compaction_request,
    persist_reverie_chain,
    reverie_refractory_is_suppressed,
    reverie_refractory_suppress,
)

logger = logging.getLogger("orion-thought.chain")

# Pressure at/below this is considered discharged → the chain has done its job.
PRESSURE_DISCHARGE_THRESHOLD = 0.15

StepFn = Callable[[str, int], Awaitable[SpontaneousThoughtV1 | None]]


class RefractoryStore(Protocol):
    def is_suppressed(self, theme_key: str, now: datetime) -> bool: ...
    def suppress(self, theme_key: str, until: datetime) -> None: ...


class InMemoryRefractoryStore:
    """Refractory store with no DB — the default for tests and bus-less runs."""

    def __init__(self) -> None:
        self._until: dict[str, datetime] = {}

    def is_suppressed(self, theme_key: str, now: datetime) -> bool:
        until = self._until.get(theme_key)
        return until is not None and until > now

    def suppress(self, theme_key: str, until: datetime) -> None:
        self._until[theme_key] = until


class DbRefractoryStore:
    """Postgres-backed refractory store (best-effort — degrades, never raises)."""

    def is_suppressed(self, theme_key: str, now: datetime) -> bool:
        return reverie_refractory_is_suppressed(theme_key, now)

    def suppress(self, theme_key: str, until: datetime) -> None:
        reverie_refractory_suppress(theme_key, until)


def theme_key_for(coalition: Any) -> str:
    """Deterministic theme key for a coalition — the unit refractory acts on."""
    if coalition is None:
        return "unknown"
    if getattr(coalition, "selected_open_loop_id", None):
        return f"loop:{coalition.selected_open_loop_id}"
    attended = sorted(getattr(coalition, "attended_node_ids", []) or [])
    if attended:
        return "nodes:" + ",".join(attended)
    return "unknown"


def update_ema(prev: float, salience: float, *, alpha: float) -> float:
    """Lossy low-pass. alpha in (0,1]; higher = more weight on the latest step."""
    a = max(0.0, min(1.0, alpha))
    return a * float(salience) + (1.0 - a) * float(prev)


def build_compaction_request(chain: ReverieChainV1) -> CompactionRequestV1 | None:
    """Deterministic ask from a *settled* chain (Phase E). None if not settled.

    A discharged / capped chain means its theme feels resolved → hint the offline
    dream to consolidate it. Never a downscale/prune from the awake path — those
    are the dream's call. Applied by nothing; this only queues an ask.
    """
    if chain.terminal_reason not in ("pressure_discharged", "max_steps"):
        return None
    if not chain.theme_key or chain.theme_key == "unknown":
        return None
    return CompactionRequestV1(
        request_id=f"compaction-request:{chain.chain_id}",
        theme=chain.theme_key,
        reason=f"reverie_chain_{chain.terminal_reason}",
        op_hint="consolidate",
        evidence_refs=list(chain.thought_ids)[:200],
        origin_chain_id=chain.chain_id,
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def run_reverie_chain(
    bus: OrionBusAsync,
    *,
    step_fn: StepFn,
    refractory_store: RefractoryStore,
    broadcast_reader: Callable[[], AttentionBroadcastProjectionV1 | None] | None = None,
    pressure_reader: Callable[[], float | None] | None = None,
    max_steps: int = 4,
    refractory_sec: float = 900.0,
    alpha: float = 0.5,
    min_ema_salience: float = 0.0,
    now_fn: Callable[[], datetime] = _now,
    publish: bool = True,
) -> ReverieChainV1 | None:
    """Run one train of thought. Returns the chain readout, or None.

    Returns None when there is no coalition to think about or the theme is in
    refractory (suppressed re-trigger). Never raises.
    """
    reader = broadcast_reader or _default_broadcast_reader
    coalition = reader()
    if coalition is None:
        logger.info("reverie chain skipped: no coalition")
        return None

    theme_key = theme_key_for(coalition)
    now = now_fn()
    if refractory_store.is_suppressed(theme_key, now):
        logger.info("reverie chain suppressed by refractory theme=%s", theme_key)
        return None

    chain_id = str(uuid4())
    thought_ids: list[str] = []
    ema = 0.0
    terminal: TerminalReason = "max_steps"

    for index in range(max(1, max_steps)):
        try:
            thought = await step_fn(chain_id, index)
        except Exception as exc:
            logger.warning("reverie chain step failed chain=%s idx=%s err=%s", chain_id, index, exc)
            thought = None
        if thought is None:
            terminal = "no_coalition" if index == 0 else "pressure_discharged"
            break

        thought_ids.append(thought.thought_id)
        ema = update_ema(ema, thought.salience, alpha=alpha)

        if pressure_reader is not None:
            pressure = pressure_reader()
            if pressure is not None and pressure <= PRESSURE_DISCHARGE_THRESHOLD:
                terminal = "pressure_discharged"
                break
        if ema < min_ema_salience:
            terminal = "low_salience"
            break
    else:
        terminal = "max_steps"

    # Habituate a resolved theme so a discharged loop can't immediately re-ignite.
    if terminal in ("pressure_discharged", "max_steps", "low_salience"):
        with suppress(Exception):
            refractory_store.suppress(theme_key, now + timedelta(seconds=refractory_sec))

    chain = ReverieChainV1(
        chain_id=chain_id,
        theme_key=theme_key,
        thought_ids=thought_ids[:MAX_CHAIN_THOUGHTS],
        ema_salience=max(0.0, min(1.0, ema)),
        ema_summary=f"{len(thought_ids)} steps on {theme_key}; ema_salience={ema:.3f}",
        terminal_reason=terminal,
    )

    if publish:
        with suppress(Exception):
            envelope = BaseEnvelope(
                kind="reverie.chain.v1",
                source=_source(),
                payload=chain.model_dump(mode="json"),
            )
            await bus.publish(settings.channel_reverie_chain, envelope)
        persist_reverie_chain(chain)

        # Phase E: a settled chain queues a compaction *request* (no consumer).
        if settings.reverie_compaction_request_enabled:
            request = build_compaction_request(chain)
            if request is not None:
                with suppress(Exception):
                    await bus.publish(
                        settings.channel_dream_compaction_request,
                        BaseEnvelope(
                            kind="dream.compaction.request.v1",
                            source=_source(),
                            payload=request.model_dump(mode="json"),
                        ),
                    )
                persist_compaction_request(request)

    logger.info(
        "reverie chain complete chain=%s steps=%d terminal=%s theme=%s",
        chain_id,
        len(thought_ids),
        terminal,
        theme_key,
    )
    return chain


async def run_reverie_chain_worker(stop_event: asyncio.Event | None = None) -> None:
    """Self-driven chain loop. Default-off; no-op unless ORION_REVERIE_CHAIN_ENABLED."""
    if not settings.reverie_chain_enabled:
        logger.info("reverie chain disabled; worker not started")
        return
    if not settings.orion_bus_enabled:
        logger.info("bus disabled; reverie chain worker not started")
        return

    bus = OrionBusAsync(url=settings.orion_bus_url)
    await bus.connect()
    refractory = DbRefractoryStore()

    async def _step(chain_id: str, index: int) -> SpontaneousThoughtV1 | None:
        return await run_reverie_once(bus, chain_context=(chain_id, index))

    logger.info("reverie chain worker started interval=%ss", settings.reverie_interval_sec)
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            try:
                await run_reverie_chain(
                    bus,
                    step_fn=_step,
                    refractory_store=refractory,
                    max_steps=settings.reverie_chain_max_steps,
                    refractory_sec=settings.reverie_refractory_sec,
                )
            except Exception:
                logger.exception("unhandled reverie chain error")
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
