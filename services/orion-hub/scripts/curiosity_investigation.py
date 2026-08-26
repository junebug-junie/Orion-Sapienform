"""Orion notices what Juniper has been talking about, and goes and finds out why.

The loop Juniper asked for: *analyze, find something interesting, then go wild
non-deterministically.*

    detect (deterministic)  ->  a real unified turn  ->  journal  ->  [outreach]
    orion/curiosity/            Orion's own voice        existing     existing,
    term_surfacing.py           and own recall           path         flag-off

WHY IT LIVES IN HUB AND NOT IN A NEW SERVICE. The investigation is a real
`orion.hub.turn_orchestrator.execute_unified_turn` -- the SAME function a
browser chat turn calls, the same one `endogenous_outreach.py` already drives
unprompted. Measured 2026-08-26: calling it from a standalone process inside the
Hub container times out at 300s and logs `session_turn_phase_read_bus_unbound` /
`juniper_affect_state_read_bus_unbound`, because the harness RPC worker and
those modules' bus binds live in Hub's own event loop. Meanwhile outreach --
running inside that loop -- sent 12 real turns in the preceding 48h. So this is
a sibling loop, not a service.

WHAT IT DOES NOT NEED, and this is the point. No FCC/`claude -p` spawn, no
read-only Postgres role, no new SQL capability, no new collection surface:

  * `execute_unified_turn` hard-codes every turn to read-only permissions
    (`write_*`/`mutate_runtime`/`network_enabled`/`shell_enabled` all False), so
    "let Orion look things up" needs no new boundary drawn.
  * `read_recall` is already on, and recall's own chat table is
    `chat_history_log` (`services/orion-recall/app/settings.py`), so "go see
    what else is out there" is a real lookup with a real tool behind it.
  * the corpus is `orion.dev_economics.claude_code_ingest.iter_all_human_messages`,
    already in production feeding the Juniper affective-state signal, under its
    own spec's framing: "no new data source, no new collection surface".

NO "IS A TURN IN FLIGHT" TRACKING, deliberately. `execute_unified_turn` begins
with a real `ThoughtClient.react()` stance evaluation that can defer or refuse,
and `endogenous_outreach.py`'s docstring already establishes that as the honest
"something else is happening, don't interrupt" signal for an unsolicited turn.
Duplicating connection bookkeeping here would add a second, weaker answer to a
question the pipeline already answers well.

SILENCE OVER A FALSE POSITIVE. Every failure path -- no signal, thin corpus,
deferred turn, empty text, error-shaped text -- writes nothing and says why in
the decision log. An investigation loop that always finds something interesting
manufactures significance daily, and the journal fills with cognition-shaped
output that no lookup supports.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional, Sequence, Tuple

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.curiosity.investigation_prompt import build_investigation_prompt
from orion.curiosity.term_surfacing import (
    SurfacedTerm,
    SurfacingReport,
    build_surfacing_report,
)
from orion.journaler.schemas import JournalEntryWriteV1

logger = logging.getLogger("orion-hub.curiosity_investigation")

INVESTIGATION_TAG = "curiosity_investigation"
JOURNAL_WRITE_CHANNEL = "orion:journal:write"
_JOURNAL_SOURCE_KIND = "self_study"
_AUTHOR = "orion"

# Redis mark so the same word is not investigated twice in a row. Same shape as
# the self-study analysis run marks (SETEX + ISO timestamp), and for the same
# reason: rotation/dedup state is ephemeral, and losing it should mean "look
# again", which is the safe direction to fail.
_TERM_MARK_PREFIX = "orion:curiosity:investigated:"


@dataclass(frozen=True)
class SchedulingGateInputs:
    """The CHEAP gates -- everything decidable without reading the corpus."""

    enabled: bool
    seconds_since_last: Optional[float]
    min_cooldown_sec: float
    done_today: int
    daily_cap: int


@dataclass(frozen=True)
class SignalGateInputs:
    """The gates that need the corpus, checked only after the cheap ones pass."""

    has_signal: bool
    underpowered: bool
    term_recently_investigated: bool


def scheduling_block_reason(inp: SchedulingGateInputs) -> Optional[str]:
    """First scheduling reason this tick must not investigate, or None.

    Checked BEFORE the corpus is read, and that ordering is not cosmetic:
    parsing the Claude Code transcripts takes ~7.8s of blocking IO over ~1.1 GB
    (measured 2026-08-26). Reading it on every tick would burn that repeatedly
    to answer a question the cooldown alone already settles, and would stall
    Hub's event loop while doing it. With the cheap gates first, the parse
    happens at most once per cooldown window instead of once per tick."""
    if not inp.enabled:
        return "disabled"
    if inp.daily_cap >= 0 and inp.done_today >= inp.daily_cap:
        return "daily_cap"
    if (
        inp.seconds_since_last is not None
        and inp.seconds_since_last < inp.min_cooldown_sec
    ):
        return "cooldown"
    return None


def signal_block_reason(inp: SignalGateInputs) -> Optional[str]:
    """First corpus-derived reason this tick must not investigate, or None."""
    if inp.underpowered:
        # NOT the same as "nothing surfaced" -- too little was said for the
        # comparison to mean anything. Collapsing these two would make a broken
        # transcript reader indistinguishable from a quiet day.
        return "corpus_underpowered"
    if not inp.has_signal:
        return "no_surfaced_term"
    if inp.term_recently_investigated:
        return "term_already_investigated"
    return None


def build_investigation_journal_entry(
    *,
    report: SurfacingReport,
    target: SurfacedTerm,
    body_text: str,
    correlation_id: str,
    created_at: Optional[datetime] = None,
) -> JournalEntryWriteV1:
    """Orion's own written result, with the finding that prompted it attached.

    The detection line is kept verbatim in the entry so a reader can tell what
    Orion was reacting to, and can check the claim -- an entry that only carries
    the conclusion is not inspectable."""
    stamp = created_at or datetime.now(timezone.utc)
    body = "\n".join(
        [
            "What I noticed:",
            f"  {target.describe()}",
            "",
            body_text.strip(),
        ]
    )
    return JournalEntryWriteV1(
        created_at=stamp,
        author=_AUTHOR,
        mode="manual",
        title=f"Curiosity: {target.term}",
        body=body,
        source_kind=_JOURNAL_SOURCE_KIND,
        # Namespaced so this never collides with the four self-study analysis
        # sources, whose own cooldown matches on a `<source>:` prefix.
        source_ref=f"curiosity:{target.term}",
        correlation_id=correlation_id,
    )


class CuriosityInvestigation:
    """Tick loop. Mirrors `EndogenousOutreach`'s lifecycle exactly."""

    def __init__(
        self,
        *,
        enabled: bool,
        tick_interval_sec: float,
        min_cooldown_sec: float,
        daily_cap: int,
        timeout_sec: float,
        session_id: str,
        term_mark_ttl_sec: int,
        recent_hours: float,
        baseline_days: float,
        message_source: Callable[[], Sequence[Tuple[datetime, str]]],
        source_ref: ServiceRef,
    ) -> None:
        self.enabled = enabled
        self.tick_interval_sec = tick_interval_sec
        self.min_cooldown_sec = min_cooldown_sec
        self.daily_cap = daily_cap
        self.timeout_sec = timeout_sec
        self.session_id = session_id
        self.term_mark_ttl_sec = term_mark_ttl_sec
        self.recent_hours = recent_hours
        self.baseline_days = baseline_days
        self._message_source = message_source
        self._source_ref = source_ref
        self._bus: Any = None
        self._harness_rpc_bus: Any = None
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._last_investigation_monotonic: Optional[float] = None
        self._done_today = 0
        self._done_today_date: Optional[str] = None

    # --- lifecycle ---------------------------------------------------------

    async def start(self, bus: Any, harness_rpc_bus: Any = None) -> None:
        self._bus = bus
        self._harness_rpc_bus = harness_rpc_bus or bus
        if not self.enabled:
            logger.info("curiosity_investigation disabled")
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run())
        logger.info(
            "curiosity_investigation started tick=%ss cooldown=%ss cap=%s window=%sh/%sd",
            self.tick_interval_sec,
            self.min_cooldown_sec,
            self.daily_cap,
            self.recent_hours,
            self.baseline_days,
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
            except Exception:  # noqa: BLE001 -- a bad tick must never kill the loop
                logger.warning("curiosity_investigation_tick_failed", exc_info=True)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.tick_interval_sec)
            except (TimeoutError, asyncio.TimeoutError):
                continue

    # --- the tick ----------------------------------------------------------

    def _roll_daily_counter(self, now: datetime) -> None:
        today = now.date().isoformat()
        if self._done_today_date != today:
            self._done_today_date = today
            self._done_today = 0

    def _seconds_since_last(self) -> Optional[float]:
        if self._last_investigation_monotonic is None:
            return None
        return time.monotonic() - self._last_investigation_monotonic

    async def _term_recently_investigated(self, term: str) -> bool:
        """Fail-open toward looking: an unreadable mark reads as "not yet
        investigated", which permits the turn rather than silently suppressing
        Orion's curiosity on a Redis hiccup."""
        redis = getattr(self._bus, "redis", None)
        if redis is None:
            return False
        try:
            return await redis.get(_TERM_MARK_PREFIX + term) is not None
        except Exception:  # noqa: BLE001
            logger.warning("curiosity_term_mark_read_failed term=%s", term, exc_info=True)
            return False

    async def _mark_term(self, term: str, *, now: datetime) -> None:
        redis = getattr(self._bus, "redis", None)
        if redis is None:
            return
        try:
            await redis.setex(_TERM_MARK_PREFIX + term, self.term_mark_ttl_sec, now.isoformat())
        except Exception:  # noqa: BLE001
            logger.warning("curiosity_term_mark_write_failed term=%s", term, exc_info=True)

    async def tick(self) -> Optional[str]:
        """One decision. Returns the block reason, or None if it investigated."""
        now = datetime.now(timezone.utc)
        self._roll_daily_counter(now)

        reason = scheduling_block_reason(
            SchedulingGateInputs(
                enabled=self.enabled,
                seconds_since_last=self._seconds_since_last(),
                min_cooldown_sec=self.min_cooldown_sec,
                done_today=self._done_today,
                daily_cap=self.daily_cap,
            )
        )
        if reason is not None:
            logger.debug("curiosity_investigation_blocked reason=%s", reason)
            return reason

        # Only now is the corpus worth reading. Off the event loop: this is
        # ~7.8s of blocking file IO and Hub is serving real turns on this loop.
        messages = await asyncio.to_thread(lambda: list(self._message_source()))
        report = build_surfacing_report(
            messages,
            now=now,
            recent_hours=self.recent_hours,
            baseline_days=self.baseline_days,
        )
        target = report.terms[0] if report.terms else None
        term_seen = (
            await self._term_recently_investigated(target.term) if target else False
        )
        reason = signal_block_reason(
            SignalGateInputs(
                has_signal=report.has_signal,
                underpowered=report.underpowered,
                term_recently_investigated=term_seen,
            )
        )
        if reason is not None or target is None:
            logger.debug("curiosity_investigation_blocked reason=%s", reason)
            return reason or "no_surfaced_term"

        # Marked and counted BEFORE the turn, so a turn that errors, times out,
        # or gets deferred by Thought still consumes its slot. Otherwise a term
        # that reliably fails would be retried every tick forever.
        self._last_investigation_monotonic = time.monotonic()
        self._done_today += 1
        await self._mark_term(target.term, now=now)

        correlation_id = f"{INVESTIGATION_TAG}:{target.term}:{int(now.timestamp())}"
        logger.info(
            "curiosity_investigation_starting term=%s recent=%s msgs=%s corr=%s",
            target.term,
            target.recent_count,
            target.recent_messages,
            correlation_id,
        )
        text, debug = await self._generate(
            build_investigation_prompt(report, target), correlation_id
        )
        if not text:
            logger.info(
                "curiosity_investigation_no_text term=%s debug=%s", target.term, debug
            )
            return "empty_generation"

        await self._journal(report=report, target=target, text=text, correlation_id=correlation_id)
        logger.info(
            "curiosity_investigation_journaled term=%s chars=%s corr=%s",
            target.term,
            len(text),
            correlation_id,
        )
        return None

    # --- the turn ----------------------------------------------------------

    async def _generate(self, prompt: str, correlation_id: str) -> Tuple[str, dict]:
        """Real unified-turn generation. Returns ("", debug) on any failure,
        defer, or degraded run -- same "never fabricate, silence over a false
        positive" contract `endogenous_outreach._generate` uses."""
        if self._bus is None:
            return "", {"error": "no_bus"}
        from orion.cognition.cortex_payload_extract import looks_like_error_text
        from orion.hub.turn_orchestrator import execute_unified_turn

        started = time.monotonic()
        try:
            frames = await asyncio.wait_for(
                execute_unified_turn(
                    bus=self._bus,
                    correlation_id=correlation_id,
                    session_id=self.session_id,
                    user_message=prompt,
                    # no_write: the journal entry below is the sole persistence
                    # path, so this does not also land as an untagged chat row.
                    payload={"no_write": True, "source": INVESTIGATION_TAG},
                    continuity_messages=None,
                    harness_rpc_bus=self._harness_rpc_bus or self._bus,
                    harness_step_relay=None,
                    harness_step_queue=None,
                ),
                timeout=self.timeout_sec,
            )
        except (TimeoutError, asyncio.TimeoutError):
            return "", {"error": "timeout", "timeout_sec": self.timeout_sec}
        except Exception as exc:  # noqa: BLE001
            logger.warning("curiosity_generate_failed corr=%s err=%s", correlation_id, exc)
            return "", {"error": type(exc).__name__, "detail": str(exc)[:240]}

        elapsed = round(time.monotonic() - started, 3)
        final = next(
            (f for f in frames if isinstance(f, dict) and f.get("type") == "final"), None
        )
        if final is None:
            # turn_deferred / turn_error / turn_degraded. Thought declining an
            # unsolicited turn is a legitimate outcome, not an error to alarm on.
            other = frames[-1] if frames else {}
            return "", {
                "error": "no_final_frame",
                "frame_type": other.get("type") if isinstance(other, dict) else None,
                "elapsed_sec": elapsed,
            }
        if final.get("context_overflow"):
            return "", {"error": "context_overflow", "elapsed_sec": elapsed}
        text = str(final.get("llm_response") or "").strip()
        if looks_like_error_text(text):
            logger.warning("curiosity_error_shaped_text corr=%s", correlation_id)
            return "", {"error": "error_shaped_text", "elapsed_sec": elapsed}
        return text, {"elapsed_sec": elapsed}

    async def _journal(
        self,
        *,
        report: SurfacingReport,
        target: SurfacedTerm,
        text: str,
        correlation_id: str,
    ) -> None:
        entry = build_investigation_journal_entry(
            report=report, target=target, body_text=text, correlation_id=correlation_id
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
            logger.warning("curiosity_journal_publish_failed corr=%s", correlation_id, exc_info=True)
