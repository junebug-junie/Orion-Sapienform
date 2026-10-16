"""Phase C — reverie chain (a train of thought).

Successive reverie steps. Each step sees the thoughts before it and the
forward pointer (`next_focus`) the last step wrote, so a chain is a train of
thought rather than one prompt sampled N times. It ends when a step has nothing
further (`settled`), a later step yields nothing, the step cap is hit, or the
EMA salience floor is crossed. A calm tick (no attention winner) starts no chain. Continuity is the
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
import re
from collections import Counter
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Protocol
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.resilience import publish_with_reconnect
from orion.regulation.arousal_read import read_arousal_async
from orion.reverie.attention_schema import to_attention_schema
from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
from orion.schemas.attention_schema import ATTENTION_SCHEMA_CHANNEL, ATTENTION_SCHEMA_KIND, bind_correlation
from orion.schemas.reverie import (
    MAX_CHAIN_THOUGHTS,
    MAX_EVIDENCE_REFS,
    CompactionRequestV1,
    ReverieChainV1,
    SpontaneousThoughtV1,
    TerminalReason,
)

from .resonance_monitor import check_resonance_worsening
from .reverie import ChainContinuity, _default_broadcast_reader, _source, run_reverie_once
from .rpc_health import fold_bus
from .settings import settings
from .store import (
    load_latest_no_coalition_streak,
    load_recent_chain_theme_events,
    persist_compaction_request,
    persist_resonance_alert,
    persist_reverie_chain,
    reverie_refractory_is_suppressed,
    reverie_refractory_suppress,
)

logger = logging.getLogger("orion-thought.chain")

# A step whose words overlap the previous step's this much says nothing new.
# Live 2026-10-16 (2,963 consecutive pairs from the old same-prompt chains):
# word-set Jaccard p50 0.36, p90 0.53; 4.4% were >= 0.6. 0.6 therefore only
# catches near-restatements; the main stop signal is the model's own null
# next_focus.
REPEAT_JACCARD_THRESHOLD = 0.6

StepFn = Callable[[str, int, ChainContinuity], Awaitable[SpontaneousThoughtV1 | None]]

# Why chains did not start, counted per process and logged with each skip so
# the rate is readable from the log alone.
CHAIN_SKIP_COUNTS: Counter[str] = Counter()


def _count_skip(reason: str, **fields: Any) -> None:
    CHAIN_SKIP_COUNTS[reason] += 1
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info(
        "reverie_chain_skipped reason=%s count=%d%s",
        reason, CHAIN_SKIP_COUNTS[reason], f" {extra}" if extra else "",
    )


def has_attention_winner(broadcast: Any) -> bool:
    """True when the broadcast attended something (a node or a selected loop).

    The same calm rule `run_reverie_once` applies per tick; checked here before
    a chain exists so a calm tick writes no chain row and no refractory entry.
    """
    if broadcast is None:
        return False
    return bool(getattr(broadcast, "attended_node_ids", None)) or bool(
        getattr(broadcast, "selected_open_loop_id", None)
    )


_WORD = re.compile(r"[a-z0-9']+")


def is_repeat_thought(current: str, previous: str, *, threshold: float = REPEAT_JACCARD_THRESHOLD) -> bool:
    """Deterministic "nothing new": word-set Jaccard of two interpretations >= threshold."""
    a = set(_WORD.findall((current or "").lower()))
    b = set(_WORD.findall((previous or "").lower()))
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= threshold


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
    """Deterministic theme key for a coalition — the unit refractory acts on.

    A loop-selected theme key is the *bare* loop id — deliberately matching
    `attention_salience_trace.theme_key` and `attention_loops_store.suppress_loop`
    (both orion-hub), which already write/read this same
    `substrate_reverie_refractory` table keyed by bare loop id. A chain must see
    the refractory suppression a human's Resolve/Dismiss action in the Hub just
    wrote, or a closed loop keeps re-igniting chains indefinitely. (Previously
    prefixed `loop:...` — a format only this function ever wrote or read, so it
    silently never matched the Hub's suppression writes.)
    """
    if coalition is None:
        return "unknown"
    if getattr(coalition, "selected_open_loop_id", None):
        return str(coalition.selected_open_loop_id)
    attended = sorted(getattr(coalition, "attended_node_ids", []) or [])
    if attended:
        return "nodes:" + ",".join(attended)
    return "unknown"


def resolve_reverie_chain_stuck_loop(
    terminal: TerminalReason, streak: int, max_repeats: int
) -> tuple[bool, int]:
    """Mirrors `visual_chain.py::resolve_visual_chain_continuity`'s shape
    (that module's Patch 4): a deterministic, testable cap on a repeated
    same-cause failure, not a heuristic guess.

    `terminal` is THIS chain's own terminal reason. `streak` is
    `store.load_latest_no_coalition_streak(theme_key)`'s return -- how many
    CONSECUTIVE prior chains for this theme also ended `"no_coalition"`
    (never got a first grounded thought). `max_repeats` is
    `settings.reverie_no_coalition_max_repeats`.

    `"no_coalition"` is deliberately excluded from `run_reverie_chain`'s own
    refractory-arming terminal set: a single hollow first attempt shouldn't
    cost a theme a full `refractory_sec` cooldown when the very next tick
    might ground fine. But a theme repeating that exact failure
    `max_repeats` times running is not a fluke -- `run_reverie_once` cannot
    produce a first thought for it right now, and refusing to arm cooldown
    just lets it re-fire every tick indefinitely. Confirmed live 2026-09-10:
    47 of 63 resonance violations logged for one loop directly followed a
    `no_coalition` row, spanning 13 days / 678 chains, because nothing ever
    broke that cycle. This gives that theme the same mechanical guarantee
    `resolve_visual_chain_continuity` already gives the image side against
    "still doing the same images of Roman aqueducts": the loop cannot run
    unbounded.

    Returns `(force_suppress, next_streak)`:
      - `force_suppress`: True once `next_streak >= max_repeats` -- the
        caller should arm the refractory cooldown even though `terminal`
        isn't in the normal arm-list.
      - `next_streak`: the value to persist in this chain's own
        `no_coalition_streak`, for the next chain's decision. 0 for any
        non-`no_coalition` terminal (a real step happened -- the "can't
        even start" streak is broken) and after a forced suppression (the
        cooldown itself is the reset).
    """
    if terminal != "no_coalition":
        return False, 0
    next_streak = streak + 1
    if next_streak >= max(1, max_repeats):
        return True, 0
    return False, next_streak


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
    if chain.terminal_reason not in ("pressure_discharged", "max_steps", "settled"):
        return None
    if not chain.theme_key or chain.theme_key == "unknown":
        return None
    return CompactionRequestV1(
        request_id=f"compaction-request:{chain.chain_id}",
        theme=chain.theme_key,
        reason=f"reverie_chain_{chain.terminal_reason}",
        op_hint="consolidate",
        evidence_refs=list(chain.thought_ids)[:MAX_EVIDENCE_REFS],  # track schema cap
        origin_chain_id=chain.chain_id,
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _maybe_emit_resonance_alert(bus: OrionBusAsync) -> None:
    """Phase H tripwire: scan recent chains for a theme re-igniting inside its
    refractory bound and, if found, publish + persist a ResonanceAlertV1.

    Observation only — never mutates cognition. Default-off; best-effort; the
    whole body is guarded so a tripwire pass never breaks the chain worker.
    """
    if not settings.reverie_resonance_alert_enabled:
        return
    try:
        from orion.reverie.resonance import ThemeEvent, detect_resonance

        rows = load_recent_chain_theme_events(settings.reverie_resonance_window)
        events: list[ThemeEvent] = []
        for theme_key, created_at in rows:
            at = created_at
            if isinstance(at, datetime) and at.tzinfo is None:
                at = at.replace(tzinfo=timezone.utc)
            if isinstance(at, datetime):
                events.append(ThemeEvent(theme_key, at))
        alert = detect_resonance(events, refractory_sec=settings.reverie_refractory_sec)
        if alert is None:
            # No theme is currently resonant. Still let the health monitor
            # re-check any theme it previously flagged worsening, so a
            # recovery note fires once that theme's trend has calmed rather
            # than staying silently "open" forever. Offloaded to a thread:
            # this makes a blocking DB query + possible orion-notify HTTP
            # call, and must not stall this coroutine's shared event loop.
            with suppress(Exception):
                await asyncio.to_thread(check_resonance_worsening, None)
            return
        with suppress(Exception):
            await bus.publish(
                settings.channel_reverie_resonance_alert,
                BaseEnvelope(
                    kind="reverie.resonance.alert.v1",
                    source=_source(),
                    payload=alert.model_dump(mode="json"),
                ),
            )
        persist_resonance_alert(alert)
        logger.warning(
            "resonance alert theme=%s violations=%d min_gap=%.1fs refractory=%.0fs",
            alert.theme_key, alert.violation_count, alert.min_gap_sec, alert.refractory_sec,
        )
        # Phase H+: proactive paging when this theme's violation_count is
        # actually climbing across its last 2 persisted samples (not merely
        # "an alert exists" -- see resonance_monitor.py for why that distinction
        # matters). Observation only; never mutates the alert or the chain.
        # Offloaded to a thread (blocking DB + orion-notify HTTP calls).
        with suppress(Exception):
            await asyncio.to_thread(check_resonance_worsening, alert)
    except Exception as exc:
        logger.debug("resonance tripwire pass failed: %s", exc)


async def run_reverie_chain(
    bus: OrionBusAsync,
    *,
    step_fn: StepFn,
    refractory_store: RefractoryStore,
    broadcast_reader: Callable[[], AttentionBroadcastProjectionV1 | None] | None = None,
    max_steps: int = 4,
    refractory_sec: float = 900.0,
    alpha: float = 0.5,
    min_ema_salience: float = 0.0,
    now_fn: Callable[[], datetime] = _now,
    publish: bool = True,
    no_coalition_max_repeats: int = 3,
    no_coalition_streak_loader: Callable[[str], int] = load_latest_no_coalition_streak,
) -> ReverieChainV1 | None:
    """Run one train of thought. Returns the chain readout, or None.

    Returns None when there is no broadcast, the broadcast has no attention
    winner (a calm tick), or the theme is in refractory. Never raises.
    """
    reader = broadcast_reader or _default_broadcast_reader
    try:
        coalition = reader()
    except Exception as exc:
        logger.warning("reverie chain coalition read failed: %s", exc)
        coalition = None
    if coalition is None:
        _count_skip("no_broadcast")
        return None
    if not has_attention_winner(coalition):
        # Calm: nothing won attention. No chain row, no "unknown" refractory
        # entry (live 2026-10-09..15: ~190 such hollow chains a day).
        _count_skip("calm_no_winner")
        return None

    theme_key = theme_key_for(coalition)
    now = now_fn()
    if refractory_store.is_suppressed(theme_key, now):
        _count_skip("refractory", theme=theme_key)
        return None

    chain_id = str(uuid4())
    thought_ids: list[str] = []
    # The thoughts themselves, kept only for the attention-surface projection
    # below (it needs the first coalition + interpretation and the last
    # next_focus); the chain readout still carries ids alone, unchanged.
    thoughts: list[SpontaneousThoughtV1] = []
    ema = 0.0
    terminal: TerminalReason = "max_steps"
    steps = max(1, max_steps)
    next_focus_hint: str | None = None

    for index in range(steps):
        continuity = ChainContinuity(
            index=index,
            is_final=index == steps - 1,
            prior_thoughts=[t.interpretation for t in thoughts],
            next_focus_hint=next_focus_hint,
        )
        try:
            thought = await step_fn(chain_id, index, continuity)
        except Exception as exc:
            logger.warning("reverie chain step failed chain=%s idx=%s err=%s", chain_id, index, exc)
            thought = None
        if thought is None:
            # Step 0: nothing grounded to say. Later step: it yielded nothing
            # publishable (hollow, below the salience floor, or failed).
            terminal = "no_coalition" if index == 0 else "pressure_discharged"
            break

        previous = thoughts[-1].interpretation if thoughts else None
        thought_ids.append(thought.thought_id)
        thoughts.append(thought)
        ema = update_ema(ema, thought.salience, alpha=alpha)

        if previous is not None and is_repeat_thought(thought.interpretation, previous):
            logger.info("reverie_chain_settled chain=%s idx=%d reason=repeat", chain_id, index)
            terminal = "settled"
            break
        if not continuity.is_final and not thought.next_focus:
            logger.info("reverie_chain_settled chain=%s idx=%d reason=no_next_focus", chain_id, index)
            terminal = "settled"
            break
        next_focus_hint = thought.next_focus
        if ema < min_ema_salience:
            terminal = "low_salience"
            break
    else:
        terminal = "max_steps"

    # A theme repeatedly unable to produce even a first thought is not a
    # fluke -- force the cooldown anyway once that streak is proven, so it
    # can't re-fire every tick indefinitely. See
    # resolve_reverie_chain_stuck_loop's own docstring for why "no_coalition"
    # alone doesn't already arm the cooldown below.
    no_coalition_streak = 0
    force_suppress = False
    if terminal == "no_coalition":
        try:
            prior_streak = no_coalition_streak_loader(theme_key)
        except Exception as exc:
            logger.debug("no_coalition streak lookup failed theme=%s err=%s", theme_key, exc)
            prior_streak = 0
        force_suppress, no_coalition_streak = resolve_reverie_chain_stuck_loop(
            terminal, prior_streak, no_coalition_max_repeats
        )

    # Habituate a resolved theme so a discharged loop can't immediately re-ignite.
    if terminal in ("pressure_discharged", "max_steps", "low_salience", "settled") or force_suppress:
        with suppress(Exception):
            refractory_store.suppress(theme_key, now + timedelta(seconds=refractory_sec))

    chain = ReverieChainV1(
        chain_id=chain_id,
        theme_key=theme_key,
        thought_ids=thought_ids[:MAX_CHAIN_THOUGHTS],
        ema_salience=max(0.0, min(1.0, ema)),
        ema_summary=f"{len(thought_ids)} steps on {theme_key}; ema_salience={ema:.3f}",
        terminal_reason=terminal,
        no_coalition_streak=no_coalition_streak,
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

        # Attention schema surface: adapt, do not migrate. The chain and its
        # table/consumers above are untouched; this emits the shared shape
        # alongside (orion/reverie/attention_schema.py).
        with suppress(Exception):
            row, corr = bind_correlation(to_attention_schema(chain, thoughts[:MAX_CHAIN_THOUGHTS]))
            await publish_with_reconnect(
                bus,
                ATTENTION_SCHEMA_CHANNEL,
                BaseEnvelope(
                    kind=ATTENTION_SCHEMA_KIND,
                    source=_source(),
                    correlation_id=corr,
                    payload=row.model_dump(mode="json"),
                ),
                log_label="reverie_attention_schema_publish",
            )

        # Phase E: a settled chain queues a compaction *request* (no consumer).
        if settings.reverie_compaction_request_enabled:
            with suppress(Exception):
                request = build_compaction_request(chain)
                if request is not None:
                    await bus.publish(
                        settings.channel_dream_compaction_request,
                        BaseEnvelope(
                            kind="dream.compaction.request.v1",
                            source=_source(),
                            payload=request.model_dump(mode="json"),
                        ),
                    )
                    persist_compaction_request(request)

        # Phase H: after persisting this chain, run the ouroboros tripwire over
        # recent chains (default-off, observation only).
        await _maybe_emit_resonance_alert(bus)

    logger.info(
        "reverie chain complete chain=%s steps=%d terminal=%s theme=%s",
        chain_id,
        len(thought_ids),
        terminal,
        theme_key,
    )
    return chain


def reverie_chain_sleep_sec(base_sec: float, level: str | None, multiplier: float) -> float:
    """R3 reader 5: the pause before the next chain. Strained stretches it; nothing shortens it.

    ``level`` is Orion's arousal at the moment the pause starts (None = unknown / flag off /
    Redis down = today's interval). A multiplier at or below 1 changes nothing."""
    base = float(base_sec)
    m = float(multiplier)
    if level == "strained" and m > 1.0:
        return base * m
    return base


async def _next_chain_sleep_sec(bus: OrionBusAsync) -> float:
    """Read arousal right before the pause (the decision point), never at loop start."""
    view = await read_arousal_async(
        bus,
        enabled=bool(settings.reverie_chain_arousal_enabled),
        reader="reverie_chain",
        max_age_sec=float(settings.regulation_reader_max_age_sec),
    )
    level = view.level if view.known else None
    sleep_sec = reverie_chain_sleep_sec(
        settings.reverie_interval_sec, level, settings.reverie_chain_arousal_strained_multiplier
    )
    if sleep_sec != float(settings.reverie_interval_sec):
        logger.info(
            "reverie_chain_paced reason=arousal_strained sleep_sec=%.0f base_sec=%.0f since=%s",
            sleep_sec, float(settings.reverie_interval_sec), view.since,
        )
    return sleep_sec


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

    async def _step(
        chain_id: str, index: int, continuity: ChainContinuity
    ) -> SpontaneousThoughtV1 | None:
        return await run_reverie_once(bus, chain_context=(chain_id, index), continuity=continuity)

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
                    no_coalition_max_repeats=settings.reverie_no_coalition_max_repeats,
                )
            except Exception:
                logger.exception("unhandled reverie chain error")
            fold_bus(bus)  # per chain run: this worker bus lives for the process (app/rpc_health.py)
            sleep_sec = await _next_chain_sleep_sec(bus)
            try:
                if stop_event is not None:
                    await asyncio.wait_for(stop_event.wait(), timeout=sleep_sec)
                    break
                await asyncio.sleep(sleep_sec)
            except asyncio.TimeoutError:
                continue
    except asyncio.CancelledError:
        raise
    finally:
        with suppress(Exception):
            await bus.close()
