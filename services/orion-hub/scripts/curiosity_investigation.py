"""Orion's own time: it looks at what it has been forming, and follows what it wants.

    open priors  --select-->  a real unified turn  -->  what it worked out
         ^                    Orion researches one         |
         |                    using its own credentials    |
         |                                                 v
         +---------  updated priors + new priors  <--  journal (prose, for Juniper)
                              |                       graph  (structure, for Orion)
                              v
                      world-view graph

WHAT THIS REPLACED, AND WHY IT MATTERS. The first version ran a term-frequency
detector over Juniper's typed words and handed Orion the highest-lift word to
investigate. Juniper's verdict: "this isnt supposed to be determinstic and it
shouldn't be words... this is just turdy keyword cathedrals masquerading as
autonomy and substance." Both halves of that were right. A word is not a
concept, and being told what to be curious about is not curiosity. Nothing in
this loop chooses a subject: code decides only WHEN Orion gets time, and Orion
decides what to do with it.

WHAT THE LOOP COULD NOT DO BEFORE THIS PATCH: LEARN. It showed Orion a random
12 of 646 approved concepts every four hours, forever. Nothing accumulated
between runs -- the only state carried forward was "when did I last run", so
run 40 was exactly as ignorant as run 1. Orion could not become less uncertain
about anything, because it never recorded what it was uncertain about.

WHAT CARRIES FORWARD NOW, and where each piece lives:

  priors           `:Prior` nodes in `orion_worldview`, Orion's own FalkorDB
                   graph. A claim it holds that could turn out to be wrong,
                   with a confidence and a status. Orion writes them; Hub only
                   reads. See `orion/curiosity/worldview.py`.
  continuation     a `:TurnOutcome` node this run's turn may write to itself.
                   The next run opens on that note instead of a cold menu.
  hop notes        `:Hop` nodes, written as the turn goes rather than
                   reconstructed at the end, so the journal can recount the
                   path actually taken instead of a conclusion with the working
                   thrown away.

THE ASYMMETRY IS DELIBERATE AND SHOULD BE ON PURPOSE. Orion is fully autonomous
inside its own graph -- it writes there directly, in-turn, with real Cypher, and
nothing it adds needs approval. Everything that leaves the turn is mediated: the
journal entry is published by this loop, and any message to Juniper goes through
a SECOND turn with its own stance gate and the existing outreach gates. The
graph is private; the journal and the outreach are shared.

WHY IT LIVES IN HUB. The investigation is a real
`orion.hub.turn_orchestrator.execute_unified_turn` -- the same function a
browser chat turn calls, already driven unprompted by `endogenous_outreach.py`.
Measured 2026-08-26: calling it from a standalone process inside the Hub
container times out at 300s with `session_turn_phase_read_bus_unbound`, because
the harness RPC worker and several module bus binds live in Hub's own event
loop. It also reads `app.state.memory_pg_pool`, the same asyncpg pool
`crystallization_routes.py` uses. So: a sibling loop, not a service.

THE CREDENTIALS ARE NOT A NEW CAPABILITY SURFACE. `execute_unified_turn` still
hard-codes every turn read-only at the Orion capability layer. What changed is
that the `claude -p` subprocess underneath it now finds a Postgres DSN and
FalkorDB credentials in its environment -- both already reachable from that
sandbox through a mounted file, and both bounded by the database itself: a role
with SELECT on four tables and nothing else, and an ACL user that is read-only
on the Juniper-curated Atlas and write-capable only on Orion's own graph. See
`orion/curiosity/sandbox_env.py`.

SILENCE OVER A FALSE POSITIVE. Every failure path -- no material, unreadable
store, unreachable graph, deferred turn, empty text, a turn that looked nothing
up -- writes nothing and says why at INFO or WARNING. A loop that always finds
something worth writing up manufactures significance daily.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Callable, Optional, Tuple
from uuid import NAMESPACE_URL, uuid4, uuid5

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.curiosity.acl import assert_orion_acl, ensure_graph_exists
from orion.curiosity.kickoff_prompt import DEFAULT_MAX_HOPS, build_kickoff_prompt
from orion.curiosity.outreach_prompt import build_outreach_composition_prompt
from orion.curiosity.study_material import (
    APPROVED_COUNT_SQL,
    APPROVED_SAMPLE_SQL,
    DEFAULT_CRYSTALLIZATION_SAMPLE,
    DEFAULT_RELATION_SAMPLE,
    RELATION_COUNT_SQL,
    RELATION_RESOLVABLE_SQL,
    RELATION_SAMPLE_SQL,
    StudyMaterial,
    assemble_study_material,
)
from orion.curiosity.worldview import (
    TurnOutcome,
    WorldviewReader,
    WorldviewSnapshot,
    read_hop_notes,
    read_run_footprint,
    read_snapshot,
    read_turn_outcome,
)
from orion.journaler.schemas import JournalEntryWriteV1

logger = logging.getLogger("orion-hub.curiosity_investigation")

INVESTIGATION_TAG = "curiosity_investigation"
OUTREACH_TAG = "curiosity_outreach"
JOURNAL_WRITE_CHANNEL = "orion:journal:write"
_JOURNAL_SOURCE_KIND = "self_study"
_AUTHOR = "orion"

# Cooldown/daily-count state lives in Redis, not in the process. Review finding
# 2026-08-26: both were plain instance fields, so every Hub restart reset the
# cooldown to "never" and the daily counter to 0 -- demonstrated live, six
# consecutive restarts produced six journal entries against a configured cap of
# 3/day with 4h between. A redeploy is not a licence to investigate again.
_COOLDOWN_KEY = "orion:curiosity:last_investigation_at"
_DAILY_COUNT_KEY_PREFIX = "orion:curiosity:count:"

# The id of the run whose `:TurnOutcome` the NEXT run should open on.
#
# Keyed on the run rather than read as "the newest TurnOutcome in the graph",
# and that distinction is load-bearing: reading the newest would silently
# re-open a note from two runs ago every time a turn died before writing its
# own, which is exactly the case whose safe default (a cold menu) this design
# depends on. Redis, not process state, for the same restart reason as above.
_LAST_RUN_KEY = "orion:curiosity:last_run_id"
_STATE_TTL_SEC = 172800

# The turn has to show evidence it actually went and looked. `harness_step_count`
# is already on the final frame (orion/hub/turn_orchestrator.py) and costs
# nothing to read.
#
# THIS IS THE LOAD-BEARING GATE OF THE WHOLE FEATURE. The prompt asks Orion to
# only say what a lookup supports, but a prompt is an instruction, not a
# mechanism: a turn that called no tools and simply wrote four fluent paragraphs
# from parametric knowledge produces a perfectly well-formed `llm_response` and,
# without this check, lands in the journal byte-for-byte indistinguishable from
# a real investigation. That is AGENTS.md 0A's no-empty-shell-cognition clause
# verbatim -- "if Orion says it remembered or reflected, there must be
# inspectable evidence for that claim."
#
# 3 rather than 1: a turn that merely answers takes a step or two on its own.
# The first real live investigation reached 29 steps with Read and ToolSearch,
# so this bar is far below a genuine run and only excludes the degenerate case.
MIN_HARNESS_STEPS = 3

# Does the read-only role the FCC sandbox connects as actually exist? Checked
# through Hub's OWN pool, which connects as a different (privileged) user, so
# this is a real independent check and not the credential validating itself.
PG_ROLE_EXISTS_SQL = "SELECT 1 FROM pg_roles WHERE rolname = $1"

_RUN_ID_RE = re.compile(r"^[0-9a-f]{6,32}$")


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

    Note what is NOT here: there is no "already investigated this" gate,
    because there is no longer a subject for code to compare. Orion is shown
    what it has settled and may repeat itself if it wants to -- that is its
    call, not a lock."""

    has_material: bool
    # Could not read the stores at all. Distinct from "nothing there" on
    # purpose -- see `signal_block_reason`.
    stores_unavailable: bool = False


def scheduling_block_reason(inp: SchedulingGateInputs) -> Optional[str]:
    """First scheduling reason this tick must not investigate, or None.

    Checked BEFORE anything else is read. With the cheap gates first, the
    Postgres samples, the graph reads and the ACL round trip happen at most
    once per cooldown window instead of once per tick."""
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


def format_footprint(footprint: dict[str, int]) -> str:
    """`{'Prior': 2, 'Hop': 5}` -> `"Hop 5, Prior 2"`. Empty string for {}."""
    return ", ".join(f"{label} {n}" for label, n in sorted(footprint.items()))


def build_investigation_journal_entry(
    *,
    material: StudyMaterial,
    body_text: str,
    correlation_id: str,
    run_id: str,
    harness_step_count: Optional[int] = None,
    harness_grounding_status: Optional[str] = None,
    graph_footprint: Optional[dict[str, int]] = None,
    hop_notes: Optional[list[tuple[int, str]]] = None,
    created_at: Optional[datetime] = None,
) -> JournalEntryWriteV1:
    """Orion's own written result.

    The title is deliberately NOT derived from a subject, because code no
    longer knows the subject -- Orion chose it inside the turn and it lives in
    the prose. Deriving one here would mean re-inferring Orion's choice with a
    heuristic, which is the exact move this rewrite exists to delete.

    THE FOOTPRINT IS THE EVIDENCE, and it is reported whether or not it is
    flattering. `graph_footprint` counts what Orion actually created in its own
    graph during THIS run; a run that wrote nothing says so in plain words
    rather than letting fluent prose imply that structure was formed. Same
    contract as the harness step count next to it: if Orion says it worked
    something out, there is an inspectable artifact behind the claim.

    `None` means the footprint could not be read (no graph configured, or the
    graph did not answer) and prints NOTHING, which is different from `{}`
    meaning Orion genuinely wrote nothing and saying so.
    """
    stamp = created_at or datetime.now(timezone.utc)
    offered = ", ".join(
        f"{kind} {count}" for kind, count in sorted(material.approved_by_kind.items())
    )
    lines = [body_text.strip()]

    if hop_notes:
        lines += ["", "---", "", "The path, as it was recorded at each stop:", ""]
        lines += [f"{n}. {note}" for n, note in hop_notes]

    lines += [
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
    if graph_footprint is not None:
        # `{}` and `None` are DIFFERENT here, and the distinction lands in the
        # one artifact Juniper actually reads: `{}` is "Orion wrote nothing",
        # `None` is "the graph could not answer", and printing the former for
        # the latter would put a false claim about Orion's own work in its
        # journal. `read_run_footprint` keeps them apart for this reason.
        lines[-1] += (
            f" Wrote to its own graph: {format_footprint(graph_footprint)}"
            if graph_footprint
            else " Wrote nothing to its own graph this run"
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
        # --- Orion's own graph -------------------------------------------
        graph_host: str = "",
        graph_port: int = 6379,
        graph_own: str = "orion_worldview",
        graph_atlas: str = "orion_substrate",
        graph_user: str = "",
        graph_password: str = "",
        hub_url: str = "http://127.0.0.1:8080",
        prior_sample: int = 8,
        stale_prior_tests: int = 3,
        max_hops: int = DEFAULT_MAX_HOPS,
        pg_readonly_role: str = "orion_readonly",
        # --- second turn --------------------------------------------------
        outreach_enabled: bool = False,
        outreach_provider: Optional[Callable[[], Any]] = None,
        reader: Optional[WorldviewReader] = None,
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

        self.graph_host = graph_host
        self.graph_port = int(graph_port)
        self.graph_own = graph_own
        self.graph_atlas = graph_atlas
        self.graph_user = graph_user
        self.graph_password = graph_password
        self.hub_url = hub_url
        self.prior_sample = int(prior_sample)
        self.stale_prior_tests = int(stale_prior_tests)
        self.max_hops = int(max_hops)
        self.pg_readonly_role = pg_readonly_role
        self.outreach_enabled = outreach_enabled
        self._outreach_provider = outreach_provider

        # Injectable so the whole loop is testable without a FalkorDB. When it
        # is None AND no host is configured, the graph half is off and the
        # prompt degrades to material-only rather than naming a store Orion
        # cannot reach.
        self._reader = reader
        if self._reader is None and graph_host and graph_user and graph_password:
            self._reader = WorldviewReader(
                host=graph_host, port=self.graph_port, graph_name=graph_own
            )
        elif self._reader is None and graph_host:
            # CONFIGURED HOST, NO CREDENTIAL -> the graph half is OFF, and the
            # rest of the loop still runs. A review finding, and a sharp one:
            # `.env_example` ships HUB_CURIOSITY_GRAPH_ORION_PASSWORD blank (it
            # is a secret) while the host default is a real address, so an
            # operator following the template verbatim would have had
            # `graph_enabled` True, `acl_setuser_argv` raising on the empty
            # password, and EVERY tick returning `graph_unavailable` -- killing
            # even the Postgres-only half that worked before this patch, behind
            # one WARNING.
            #
            # A missing optional secret is an opt-out, not a breakage. A
            # credential that IS set and then fails still hard-blocks, because
            # that one is a real fault.
            logger.warning(
                "curiosity_graph_credential_missing host=%s user=%s -- Orion's "
                "own graph is DISABLED for this process (priors, hops and "
                "continuation notes are all off); the Postgres half still runs. "
                "Set HUB_CURIOSITY_GRAPH_ORION_USER/PASSWORD to enable it.",
                graph_host,
                graph_user or "<unset>",
            )

        self._bus: Any = None
        self._harness_rpc_bus: Any = None
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._last_investigation_monotonic: Optional[float] = None
        self._done_today = 0
        self._done_today_date: Optional[str] = None
        self._acl_error: Optional[str] = "not_asserted_yet"

    @property
    def graph_enabled(self) -> bool:
        return self._reader is not None

    # --- lifecycle ---------------------------------------------------------

    async def start(self, bus: Any, harness_rpc_bus: Any = None) -> None:
        self._bus = bus
        self._harness_rpc_bus = harness_rpc_bus or bus
        if not self.enabled:
            logger.info("curiosity_investigation disabled")
            return
        if self.graph_enabled:
            await self._assert_acl()
        self._stop.clear()
        self._task = asyncio.create_task(self._run())
        logger.info(
            "curiosity_investigation started tick=%ss cooldown=%ss cap=%s sample=%s+%s "
            "graph=%s hops=%s outreach=%s",
            self.tick_interval_sec,
            self.min_cooldown_sec,
            self.daily_cap,
            self.crystallization_sample,
            self.relation_sample,
            self.graph_own if self.graph_enabled else "off",
            self.max_hops,
            self.outreach_enabled,
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

    # --- the graph ---------------------------------------------------------

    async def _assert_acl(self) -> Optional[str]:
        """Re-apply Orion's FalkorDB grant. Stores and returns the failure, if any.

        Run at startup AND immediately before every real run, not just at
        startup: `aclfile` is unset and immutable on this FalkorDB, so the rule
        lives only in the running process's memory and a restart at any hour
        silently removes Orion's access. A startup-only assert would leave the
        loop degraded until the next Hub deploy. One round trip per four hours
        is not a cost worth optimising against that. See orion/curiosity/acl.py.
        """
        if self._reader is None:
            self._acl_error = "no_graph_configured"
            return self._acl_error

        def _apply() -> Optional[str]:
            client = self._reader.client()
            # Order matters: the grant is useless against a key FalkorDB has
            # never seen, and `GRAPH.RO_QUERY` on one errors rather than
            # returning empty -- see `ensure_graph_exists` for the deadlock
            # that produces on a fresh deployment.
            created = ensure_graph_exists(client=client, graph_name=self.graph_own)
            if created:
                return created
            return assert_orion_acl(
                client=client,
                username=self.graph_user,
                password=self.graph_password,
                atlas_graph=self.graph_atlas,
                own_graph=self.graph_own,
            )

        try:
            self._acl_error = await asyncio.to_thread(_apply)
        except Exception as exc:  # noqa: BLE001
            self._acl_error = f"{type(exc).__name__}: {str(exc)[:160]}"
        if self._acl_error:
            logger.warning(
                "curiosity_acl_assert_failed err=%s -- Orion cannot reach its own "
                "graph; the run will be blocked rather than degraded silently",
                self._acl_error,
            )
        return self._acl_error

    async def _read_worldview(self, run_id_of_last: Optional[str]) -> WorldviewSnapshot:
        """Orion's own graph, plus the note the previous run left itself."""
        if self._reader is None:
            return WorldviewSnapshot(unavailable_reason="no_graph_configured")
        reader = self._reader

        def _read() -> WorldviewSnapshot:
            view = read_snapshot(
                reader,
                sample=self.prior_sample,
                stale_after=self.stale_prior_tests,
            )
            if view.is_unavailable or not run_id_of_last:
                return view
            return replace(view, continuation=read_turn_outcome(reader, run_id_of_last))

        try:
            return await asyncio.to_thread(_read)
        except Exception as exc:  # noqa: BLE001
            return WorldviewSnapshot(
                unavailable_reason=f"{type(exc).__name__}: {str(exc)[:160]}"
            )

    async def _read_turn_result(
        self, run_id: str
    ) -> Tuple[Optional[TurnOutcome], Optional[dict[str, int]], list[tuple[int, str]]]:
        """What the turn left behind in Orion's own graph.

        A `None` footprint means the graph could not answer, which is NOT the
        same as Orion having written nothing -- see `read_run_footprint`."""
        if self._reader is None:
            return None, None, []
        reader = self._reader

        def _read() -> Tuple[
            Optional[TurnOutcome], Optional[dict[str, int]], list[tuple[int, str]]
        ]:
            return (
                read_turn_outcome(reader, run_id),
                read_run_footprint(reader, run_id),
                read_hop_notes(reader, run_id),
            )

        try:
            return await asyncio.to_thread(_read)
        except Exception as exc:  # noqa: BLE001
            logger.warning("curiosity_turn_result_read_failed run=%s err=%s", run_id, exc)
            return None, None, []

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

        Fail-open to in-process state -- an unreadable Redis must not silently
        freeze Orion's curiosity."""
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

    async def _read_last_run_id(self) -> Optional[str]:
        redis = getattr(self._bus, "redis", None)
        if redis is None:
            return None
        try:
            raw = await redis.get(_LAST_RUN_KEY)
        except Exception:  # noqa: BLE001
            logger.warning("curiosity_last_run_read_failed", exc_info=True)
            return None
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        value = str(raw).strip()
        # Validated here as well as at the Cypher builder: a junk value in
        # Redis must read as "no previous run", never as a query fragment.
        return value if _RUN_ID_RE.match(value) else None

    async def _record_investigation(self, now: datetime, run_id: str) -> None:
        """Persist the cooldown stamp, today's counter, and this run's id."""
        redis = getattr(self._bus, "redis", None)
        if redis is None:
            return
        try:
            # Two days of TTL so the counter cannot outlive its own date key.
            await redis.setex(_COOLDOWN_KEY, _STATE_TTL_SEC, now.isoformat())
            key = self._daily_key(now)
            await redis.incr(key)
            await redis.expire(key, _STATE_TTL_SEC)
            # Longer than the others on purpose: a continuation note should
            # survive a quiet weekend. Seven days.
            await redis.setex(_LAST_RUN_KEY, 604800, run_id)
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
        )

    async def _pg_role_missing(self) -> bool:
        """Does the role the FCC sandbox connects as exist? True if it does NOT.

        A DETERMINISTIC GATE, not a courtesy check. Without it, a dropped or
        renamed `orion_readonly` produces a turn that spends its whole budget
        discovering it cannot authenticate, then writes an articulate paragraph
        about being unable to reach its memory -- which lands in the journal
        looking like a finding. AGENTS.md: the right fix for a silent
        dependency is a failing gate, not a hopeful prompt.

        Unreadable answers as "present": Hub's own pool being unavailable is
        already caught by `stores_unavailable` one step later, and guessing
        `missing` here would block on the wrong evidence.
        """
        pool = self._pool_provider()
        if pool is None:
            return False
        try:
            async with pool.acquire() as conn:
                found = await conn.fetchval(PG_ROLE_EXISTS_SQL, self.pg_readonly_role)
        except Exception as exc:  # noqa: BLE001
            logger.warning("curiosity_pg_role_check_failed err=%s", exc)
            return False
        return found is None

    async def tick(self) -> Optional[str]:
        """One decision. Returns the block reason, or None if it investigated."""
        now = datetime.now(timezone.utc)
        self._roll_daily_counter(now)

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

        if self.pg_readonly_role and await self._pg_role_missing():
            logger.warning(
                "curiosity_investigation_blocked reason=pg_role_missing role=%s -- "
                "the FCC sandbox has a DSN for a role Postgres does not have; "
                "recreate it before this loop can read its own material",
                self.pg_readonly_role,
            )
            return "pg_role_missing"

        # Re-asserted here, not only at startup -- see `_assert_acl`.
        if self.graph_enabled:
            acl_error = await self._assert_acl()
            if acl_error:
                logger.warning(
                    "curiosity_investigation_blocked reason=graph_unavailable detail=%s",
                    acl_error,
                )
                return "graph_unavailable"

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

        # UUID-shaped, because `BaseEnvelope.correlation_id` validates as one --
        # and `execute_unified_turn` builds envelopes internally, so a readable
        # `tag:term:ts` string fails the whole turn before it starts. Confirmed
        # live 2026-08-26 on the first deploy: the loop detected its subject
        # correctly and then died on `uuid_parsing`. uuid5 rather than uuid4 so
        # the id is still deterministic from the same seed and can be traced
        # back to a readable one.
        run_id = uuid4().hex[:12]
        correlation_id = str(uuid5(NAMESPACE_URL, f"{INVESTIGATION_TAG}:{run_id}"))

        last_run_id = await self._read_last_run_id()
        view = await self._read_worldview(last_run_id)
        if view.is_unavailable and self.graph_enabled:
            # The ACL assert above succeeded, so this is a query-level failure
            # rather than a missing grant. Reported and NOT fatal: Orion can
            # still investigate its Postgres material, and the prompt says the
            # graph could not be read rather than implying an empty mind.
            logger.warning(
                "curiosity_worldview_degraded run=%s detail=%s", run_id, view.unavailable_reason
            )

        # Counted BEFORE the turn, so a turn that errors, times out, or gets
        # deferred by Thought still consumes its slot -- otherwise a reliably
        # failing turn would be retried every tick forever.
        self._last_investigation_monotonic = time.monotonic()
        self._done_today = done_today + 1
        await self._record_investigation(now, run_id)

        logger.info(
            "curiosity_investigation_starting run=%s offered=%s/%s concepts "
            "%s/%s relations priors=%s/%s continuing=%s corr=%s",
            run_id,
            len(material.crystallizations),
            material.approved_total,
            len(material.relations),
            material.relation_total,
            len(view.open_priors),
            view.open_total,
            bool(view.continuation and view.continuation.continue_line),
            correlation_id,
        )

        prompt = build_kickoff_prompt(
            material,
            view=view,
            run_id=run_id,
            own_graph=self.graph_own,
            atlas_graph=self.graph_atlas,
            hub_url=self.hub_url,
            max_hops=self.max_hops,
            stale_after=self.stale_prior_tests,
            # `graph_enabled` says a graph is CONFIGURED, not that it answered.
            # The prompt splits those apart itself: an unreadable graph is
            # disclosed to Orion and the write sections are dropped. Collapsing
            # them here would silence the disclosure -- see build_kickoff_prompt.
            graph_enabled=self.graph_enabled,
        )
        text, debug = await self._generate(prompt, correlation_id)
        if not text:
            logger.info("curiosity_investigation_no_text run=%s debug=%s", run_id, debug)
            return "empty_generation"

        outcome, footprint, hops = await self._read_turn_result(run_id)

        await self._journal(
            material=material,
            text=text,
            correlation_id=correlation_id,
            run_id=run_id,
            harness_step_count=debug.get("harness_step_count"),
            harness_grounding_status=debug.get("harness_grounding_status"),
            graph_footprint=footprint,
            hop_notes=hops,
        )
        logger.info(
            "curiosity_investigation_journaled run=%s chars=%s wrote=%s hops=%s "
            "continue=%s reach_out=%s corr=%s",
            run_id,
            len(text),
            "unreadable" if footprint is None else (format_footprint(footprint) or "nothing"),
            len(hops),
            bool(outcome and outcome.continue_line),
            bool(outcome and outcome.reach_out),
            correlation_id,
        )

        if outcome is not None and outcome.reach_out:
            await self._maybe_reach_out(outcome=outcome, finding_text=text, run_id=run_id)
        return None

    # --- the turn ----------------------------------------------------------

    async def _generate(
        self,
        prompt: str,
        correlation_id: str,
        source: str = INVESTIGATION_TAG,
        require_lookup: bool = True,
    ) -> Tuple[str, dict]:
        """Real unified-turn generation. Returns ("", debug) on any failure,
        defer, or degraded run -- same "never fabricate, silence over a false
        positive" contract `endogenous_outreach._generate` uses.

        `require_lookup=False` for the COMPOSITION turn, and that is not a
        loosening of the gate -- it is the gate being applied to the turn it was
        written for. `MIN_HARNESS_STEPS` proves an INVESTIGATION went and
        looked; the composition turn is deliberately the opposite (see
        `orion/curiosity/outreach_prompt.py`: no material, no priors, no
        schema, nothing to look up). A pure writing turn produces about three
        stream-json lines, so it would have cleared the bar by a margin of zero
        -- this gate's own comment estimates "a turn that merely answers takes a
        step or two", i.e. BELOW it. Any change to the stream shape would then
        have killed outreach silently, reported as `empty_generation`, which is
        indistinguishable from a real generation failure. A review finding."""
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
                    payload={"no_write": True, "source": source},
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
        if require_lookup and step_count < MIN_HARNESS_STEPS:
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
            "fcc_model_label": final.get("fcc_model_label"),
        }

    # --- the second turn ---------------------------------------------------

    async def _maybe_reach_out(
        self, *, outcome: TurnOutcome, finding_text: str, run_id: str
    ) -> Optional[str]:
        """Orion decided a finding is worth telling Juniper about. Compose it.

        A SECOND TURN, not a reuse of the first one's text, and the extra cost
        buys one specific thing: the second turn gets its OWN
        `ThoughtClient.react()` stance check. So Orion can find something
        genuinely worth saying and the system can still independently decide
        "not now, she is in the middle of something". One turn would collapse
        "this is interesting" and "this is worth interrupting her for" into a
        single judgement made at the wrong moment.

        The outreach gates are checked BEFORE the turn as well as inside the
        delivery: quiet hours can span eight hours, and spending a full
        unified turn to compose a message that cannot be delivered for another
        six is a waste of Orion's own compute, not a safety issue.
        """
        if not self.outreach_enabled:
            logger.info("curiosity_outreach_disabled run=%s", run_id)
            return "disabled"
        outreach = self._outreach_provider() if self._outreach_provider else None
        if outreach is None:
            logger.warning(
                "curiosity_outreach_unavailable run=%s -- Orion asked to reach out "
                "and there is no outreach loop to deliver through",
                run_id,
            )
            return "no_outreach_loop"

        blocked = outreach.blocked_reason()
        if blocked:
            logger.info(
                "curiosity_outreach_blocked run=%s reason=%s why=%s",
                run_id,
                blocked,
                outcome.reach_out_why[:120],
            )
            return blocked

        correlation_id = str(uuid5(NAMESPACE_URL, f"{OUTREACH_TAG}:{run_id}"))
        prompt = build_outreach_composition_prompt(
            finding_text=finding_text, reach_out_why=outcome.reach_out_why
        )
        text, debug = await self._generate(
            prompt, correlation_id, source=OUTREACH_TAG, require_lookup=False
        )
        if not text:
            logger.info("curiosity_outreach_no_text run=%s debug=%s", run_id, debug)
            return "empty_generation"

        result = await outreach.offer_message(
            text=text,
            correlation_id=correlation_id,
            tag=OUTREACH_TAG,
            model=debug.get("fcc_model_label"),
        )
        logger.info(
            "curiosity_outreach_result run=%s sent=%s reason=%s",
            run_id,
            result.get("outreach"),
            result.get("reason"),
        )
        return None if result.get("outreach") else str(result.get("reason") or "not_sent")

    async def _journal(
        self,
        *,
        material: StudyMaterial,
        text: str,
        correlation_id: str,
        run_id: str,
        harness_step_count: Optional[int] = None,
        harness_grounding_status: Optional[str] = None,
        graph_footprint: Optional[dict[str, int]] = None,
        hop_notes: Optional[list[tuple[int, str]]] = None,
    ) -> None:
        entry = build_investigation_journal_entry(
            material=material,
            run_id=run_id,
            body_text=text,
            correlation_id=correlation_id,
            harness_step_count=harness_step_count,
            harness_grounding_status=harness_grounding_status,
            graph_footprint=graph_footprint,
            hop_notes=hop_notes,
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
