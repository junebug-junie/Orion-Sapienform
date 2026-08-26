"""Orion's own time: it looks at what it has been forming, and follows what it wants.

    present (no choosing)  ->  a real unified turn  ->  journal
    approved crystallizations   Orion picks its own      what it chose
    + concept induction         subject and digs         and what it found

WHAT THIS REPLACED, AND WHY IT MATTERS. The first version ran a term-frequency
detector over Juniper's typed words and handed Orion the highest-lift word to
investigate. Juniper's verdict: "this isnt supposed to be determinstic and it
shouldn't be words... this is just turdy keyword cathedrals masquerading as
autonomy and substance." Both halves of that were right. A word is not a
concept, and being told what to be curious about is not curiosity. Nothing in
this loop now chooses a subject: code decides only WHEN Orion gets time, and
Orion decides what to do with it.

THE MATERIAL IS ORION'S OWN COGNITION, AND ONLY THE APPROVED PART. Of 1,282
crystallizations, 636 are unapproved -- and those 636 are exactly the ones
whose `subject` is byte-identical to their `summary`, i.e. a chat turn with a
label on it rather than an induced concept. Filtering to Juniper-approved
(`status='active'`) is what keeps the replacement from being the same mistake
one layer down. See `orion/curiosity/study_material.py`.

WHY IT LIVES IN HUB. The investigation is a real
`orion.hub.turn_orchestrator.execute_unified_turn` -- the same function a
browser chat turn calls, already driven unprompted by `endogenous_outreach.py`.
Measured 2026-08-26: calling it from a standalone process inside the Hub
container times out at 300s with `session_turn_phase_read_bus_unbound`, because
the harness RPC worker and several module bus binds live in Hub's own event
loop. It also reads `app.state.memory_pg_pool`, the same asyncpg pool
`crystallization_routes.py` uses. So: a sibling loop, not a service.

NO NEW CAPABILITY SURFACE. `execute_unified_turn` hard-codes every turn to
read-only (`write_*`/`mutate_runtime`/`network_enabled`/`shell_enabled` all
False) and `read_recall`/`read_memory`/`read_graph` are already on -- so "go
pull more" needs no new boundary drawn, no SQL capability, and no Postgres role.

SILENCE OVER A FALSE POSITIVE. Every failure path -- no material, unreadable
store, deferred turn, empty text, a turn that looked nothing up -- writes
nothing and says why at INFO or WARNING. A loop that always finds something
worth writing up manufactures significance daily.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Callable, Optional, Sequence, Tuple
from uuid import NAMESPACE_URL, uuid4, uuid5

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.curiosity.kickoff_prompt import build_kickoff_prompt
from orion.curiosity.study_material import (
    APPROVED_COUNT_SQL,
    APPROVED_SAMPLE_SQL,
    DEFAULT_CRYSTALLIZATION_SAMPLE,
    DEFAULT_RECENT_STUDY_SAMPLE,
    DEFAULT_RELATION_SAMPLE,
    RECENT_STUDY_SQL,
    RELATION_COUNT_SQL,
    RELATION_RESOLVABLE_SQL,
    RELATION_SAMPLE_SQL,
    StudyMaterial,
    assemble_study_material,
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
# Cooldown/daily-count state lives in Redis, not in the process. Review finding
# 2026-08-26: both were plain instance fields, so every Hub restart reset the
# cooldown to "never" and the daily counter to 0 -- demonstrated live, six
# consecutive restarts produced six journal entries against a configured cap of
# 3/day with 4h between. A redeploy is not a licence to investigate again.
_COOLDOWN_KEY = "orion:curiosity:last_investigation_at"
_DAILY_COUNT_KEY_PREFIX = "orion:curiosity:count:"

# The turn has to show evidence it actually went and looked. `harness_step_count`
# is already on the final frame (orion/hub/turn_orchestrator.py) and costs
# nothing to read.
#
# THIS IS THE LOAD-BEARING GATE OF THE WHOLE FEATURE. The prompt asks Orion to
# only say what a lookup supports, but a prompt is an instruction, not a
# mechanism: a turn that called no tools and simply wrote four fluent paragraphs
# about "foveal" from parametric knowledge produces a perfectly well-formed
# `llm_response` and, without this check, lands in the journal byte-for-byte
# indistinguishable from a real investigation. That is CLAUDE.md 0A's
# no-empty-shell-cognition clause verbatim -- "if Orion says it remembered or
# reflected, there must be inspectable evidence for that claim."
#
# 3 rather than 1: a turn that merely answers takes a step or two on its own.
# The first real live investigation reached 29 steps with Read and ToolSearch,
# so this bar is far below a genuine run and only excludes the degenerate case.
MIN_HARNESS_STEPS = 3


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
    """The gates that need the stores, checked only after the cheap ones pass.

    Note what is NOT here any more: there is no "already investigated this"
    gate, because there is no longer a subject for code to compare. Orion is
    shown what it recently looked into and may repeat itself if it wants to --
    that is its call, not a lock."""

    has_material: bool
    # Could not read the stores at all. Distinct from "nothing there" on
    # purpose -- see `signal_block_reason`.
    stores_unavailable: bool = False


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
    if inp.stores_unavailable:
        # A BROKEN QUERY OR AN ABSENT POOL, not an empty mind. These must never
        # be the same state: an unreadable store and a mind with nothing in it
        # are indistinguishable otherwise, and the only symptom of the former
        # would be an absence of journal entries -- which is also what a quiet
        # stretch looks like. Same shape as the 21h vision blackout.
        return "stores_unavailable"
    if not inp.has_material:
        return "no_approved_material"
    return None


def build_investigation_journal_entry(
    *,
    material: StudyMaterial,
    body_text: str,
    correlation_id: str,
    run_id: str,
    harness_step_count: Optional[int] = None,
    harness_grounding_status: Optional[str] = None,
    created_at: Optional[datetime] = None,
) -> JournalEntryWriteV1:
    """Orion's own written result.

    The title is deliberately NOT derived from a subject, because code no
    longer knows the subject -- Orion chose it inside the turn and it lives in
    the prose. Deriving one here would mean re-inferring Orion's choice with a
    heuristic, which is the exact move this rewrite exists to delete.

    What IS recorded is the shape of what Orion was offered, so a reader can
    tell what was on the table when it chose."""
    stamp = created_at or datetime.now(timezone.utc)
    offered = ", ".join(
        f"{kind} {count}" for kind, count in sorted(material.approved_by_kind.items())
    )
    lines = [
        body_text.strip(),
        "",
        f"(Offered {len(material.crystallizations)} of "
        f"{material.approved_total} approved concepts [{offered}] and "
        f"{len(material.relations)} of {material.relation_total} relation "
        "judgements, all sampled at random.",
    ]
    if harness_step_count is not None:
        # The evidence that this was a lookup and not a recollection, kept in
        # the artifact so the claim stays checkable after the fact.
        lines[-1] += (
            f" Investigated over {harness_step_count} harness steps"
            + (f", grounding: {harness_grounding_status}" if harness_grounding_status else "")
        )
    lines[-1] += ".)"
    return JournalEntryWriteV1(
        created_at=stamp,
        author=_AUTHOR,
        mode="manual",
        title="Curiosity",
        body="\n".join(lines),
        source_kind=_JOURNAL_SOURCE_KIND,
        # Namespaced away from the four self-study analysis sources, whose own
        # cooldown matches on a `<source>:` prefix. Keyed on the run rather
        # than on a subject, since there is no code-known subject any more.
        source_ref=f"curiosity:{run_id}",
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
        crystallization_sample: int = DEFAULT_CRYSTALLIZATION_SAMPLE,
        relation_sample: int = DEFAULT_RELATION_SAMPLE,
        timezone_name: str = "UTC",
        pool_provider: Callable[[], Any],
        source_ref: ServiceRef,
    ) -> None:
        self.enabled = enabled
        self.tick_interval_sec = tick_interval_sec
        self.min_cooldown_sec = min_cooldown_sec
        self.daily_cap = daily_cap
        self.timeout_sec = timeout_sec
        self.session_id = session_id
        self.crystallization_sample = crystallization_sample
        self.relation_sample = relation_sample
        self.timezone_name = timezone_name
        try:
            self._tz = ZoneInfo(timezone_name)
        except Exception:  # noqa: BLE001 -- a bad zone must not stop the loop
            logger.warning("curiosity_bad_timezone name=%s falling back to UTC", timezone_name)
            self._tz = timezone.utc
        self._pool_provider = pool_provider
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
            "curiosity_investigation started tick=%ss cooldown=%ss cap=%s sample=%s+%s",
            self.tick_interval_sec,
            self.min_cooldown_sec,
            self.daily_cap,
            self.crystallization_sample,
            self.relation_sample,
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

    def _seconds_since_last_in_process(self) -> Optional[float]:
        if self._last_investigation_monotonic is None:
            return None
        return time.monotonic() - self._last_investigation_monotonic

    def _daily_key(self, now: datetime) -> str:
        # Keyed on the operator's LOCAL date, not UTC. "3 per day" meaning
        # 18:00-to-18:00 for someone in MDT is not what the setting says.
        local = now.astimezone(self._tz) if self._tz else now
        return f"{_DAILY_COUNT_KEY_PREFIX}{local.date().isoformat()}"

    async def _read_persisted_state(self, now: datetime) -> tuple[Optional[float], int]:
        """(seconds since last investigation, count so far today) from Redis.

        Fail-open to (None, 0) -- an unreadable store must not silently freeze
        Orion's curiosity. The per-term mark is the backstop in that case."""
        redis = getattr(self._bus, "redis", None)
        if redis is None:
            return self._seconds_since_last_in_process(), self._done_today
        since: Optional[float] = None
        count = 0
        try:
            raw = await redis.get(_COOLDOWN_KEY)
            if raw is not None:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                last = datetime.fromisoformat(str(raw))
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                since = max(0.0, (now - last).total_seconds())
        except Exception:  # noqa: BLE001
            logger.warning("curiosity_cooldown_read_failed", exc_info=True)
        try:
            raw_count = await redis.get(self._daily_key(now))
            if raw_count is not None:
                count = int(raw_count)
        except Exception:  # noqa: BLE001
            logger.warning("curiosity_daily_count_read_failed", exc_info=True)
        return since, count

    async def _record_investigation(self, now: datetime) -> None:
        """Persist the cooldown stamp and bump today's counter."""
        redis = getattr(self._bus, "redis", None)
        if redis is None:
            return
        try:
            # Two days of TTL so the counter cannot outlive its own date key.
            await redis.setex(_COOLDOWN_KEY, 172800, now.isoformat())
            key = self._daily_key(now)
            await redis.incr(key)
            await redis.expire(key, 172800)
        except Exception:  # noqa: BLE001
            logger.warning("curiosity_state_write_failed", exc_info=True)

    async def _read_study_material(self, now: datetime) -> StudyMaterial:
        """Read both stores. Never raises -- an unreadable store is reported as
        `unavailable`, which is a DIFFERENT state from an empty one."""
        pool = self._pool_provider()
        if pool is None:
            return StudyMaterial(generated_at=now, unavailable_reason="no_pool")
        try:
            async with pool.acquire() as conn:
                approved_counts = await conn.fetch(APPROVED_COUNT_SQL)
                approved_rows = await conn.fetch(
                    APPROVED_SAMPLE_SQL, self.crystallization_sample
                )
                relation_counts = await conn.fetch(RELATION_COUNT_SQL)
                relation_rows = await conn.fetch(RELATION_SAMPLE_SQL, self.relation_sample)
                resolvable = await conn.fetchval(RELATION_RESOLVABLE_SQL)
                recent_titles = await conn.fetch(
                    RECENT_STUDY_SQL, DEFAULT_RECENT_STUDY_SAMPLE
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("curiosity_study_material_read_failed err=%s", exc, exc_info=True)
            return StudyMaterial(
                generated_at=now, unavailable_reason=f"query_failed:{type(exc).__name__}"
            )
        return assemble_study_material(
            now=now,
            approved_counts=approved_counts,
            approved_rows=approved_rows,
            relation_counts=relation_counts,
            relation_rows=relation_rows,
            relation_resolvable=int(resolvable or 0),
            recent_titles=recent_titles,
        )

    async def tick(self) -> Optional[str]:
        """One decision. Returns the block reason, or None if it investigated."""
        now = datetime.now(timezone.utc)
        self._roll_daily_counter(now)

        # Read from Redis, not from instance fields. Review finding 2026-08-26:
        # both gates were pure in-process state, so a Hub restart reset the
        # cooldown to "never" and the daily counter to 0 -- six consecutive
        # restarts produced six journal entries against a configured cap of 3.
        # `.env_example` states the cap as a guarantee; it has to be one.
        since_last, done_today = await self._read_persisted_state(now)
        reason = scheduling_block_reason(
            SchedulingGateInputs(
                enabled=self.enabled,
                seconds_since_last=since_last,
                min_cooldown_sec=self.min_cooldown_sec,
                done_today=done_today,
                daily_cap=self.daily_cap,
            )
        )
        if reason is not None:
            # INFO, not DEBUG. These fire at most once per tick (5 min), so the
            # volume is trivial, and a loop whose every refusal is invisible is
            # indistinguishable from a loop that is dead.
            logger.info("curiosity_investigation_blocked reason=%s", reason)
            return reason

        # Only now are the stores worth reading. Two small indexed reads plus
        # two random samples -- cheap, but there is no reason to run them on a
        # tick that the cooldown was going to refuse anyway.
        material = await self._read_study_material(now)
        reason = signal_block_reason(
            SignalGateInputs(
                has_material=material.has_material,
                stores_unavailable=material.is_unavailable,
            )
        )
        if reason is not None:
            if reason == "stores_unavailable":
                logger.warning(
                    "curiosity_investigation_blocked reason=stores_unavailable detail=%s "
                    "-- check app.state.memory_pg_pool and the two memory tables",
                    material.unavailable_reason,
                )
            else:
                logger.info(
                    "curiosity_investigation_blocked reason=%s approved=%s relations=%s",
                    reason,
                    material.approved_total,
                    material.relation_total,
                )
            return reason

        # Counted BEFORE the turn, so a turn that errors, times out, or gets
        # deferred by Thought still consumes its slot -- otherwise a reliably
        # failing turn would be retried every tick forever.
        self._last_investigation_monotonic = time.monotonic()
        self._done_today = done_today + 1
        await self._record_investigation(now)

        # UUID-shaped, because `BaseEnvelope.correlation_id` validates as one --
        # and `execute_unified_turn` builds envelopes internally, so a readable
        # `tag:term:ts` string fails the whole turn before it starts. Confirmed
        # live 2026-08-26 on the first deploy: the loop detected `foveal`
        # correctly and then died on `uuid_parsing`. uuid5 rather than uuid4 so
        # the id is still deterministic from the same (term, tick) and can be
        # traced back to a readable seed.
        run_id = uuid4().hex[:12]
        seed = f"{INVESTIGATION_TAG}:{run_id}"
        correlation_id = str(uuid5(NAMESPACE_URL, seed))
        logger.info(
            "curiosity_investigation_starting run=%s offered=%s/%s concepts "
            "%s/%s relations corr=%s shown=%s",
            run_id,
            len(material.crystallizations),
            material.approved_total,
            len(material.relations),
            material.relation_total,
            correlation_id,
            ",".join(i[:8] for i in material.shown_ids()),
        )
        text, debug = await self._generate(build_kickoff_prompt(material), correlation_id)
        if not text:
            logger.info("curiosity_investigation_no_text run=%s debug=%s", run_id, debug)
            return "empty_generation"

        await self._journal(
            material=material,
            text=text,
            correlation_id=correlation_id,
            run_id=run_id,
            harness_step_count=debug.get("harness_step_count"),
            harness_grounding_status=debug.get("harness_grounding_status"),
        )
        logger.info(
            "curiosity_investigation_journaled run=%s chars=%s corr=%s",
            run_id,
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

        # Did it actually look? See MIN_HARNESS_STEPS.
        steps = final.get("harness_step_count")
        grounding = final.get("harness_grounding_status")
        try:
            step_count = int(steps) if steps is not None else 0
        except (TypeError, ValueError):
            step_count = 0
        if step_count < MIN_HARNESS_STEPS:
            logger.warning(
                "curiosity_no_lookup corr=%s steps=%s grounding=%s chars=%s -- "
                "refusing to journal a turn that did not look anything up",
                correlation_id,
                step_count,
                grounding,
                len(text),
            )
            return "", {
                "error": "no_lookup",
                "harness_step_count": step_count,
                "harness_grounding_status": grounding,
                "elapsed_sec": elapsed,
            }
        return text, {
            "elapsed_sec": elapsed,
            "harness_step_count": step_count,
            "harness_grounding_status": grounding,
        }

    async def _journal(
        self,
        *,
        material: StudyMaterial,
        text: str,
        correlation_id: str,
        run_id: str,
        harness_step_count: Optional[int] = None,
        harness_grounding_status: Optional[str] = None,
    ) -> None:
        entry = build_investigation_journal_entry(
            material=material,
            run_id=run_id,
            body_text=text,
            correlation_id=correlation_id,
            harness_step_count=harness_step_count,
            harness_grounding_status=harness_grounding_status,
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
