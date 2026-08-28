"""Read side of `orion_worldview` -- Orion's own graph, which Hub never writes.

THE ASYMMETRY IS THE DESIGN, not an accident of who got there first. Orion
writes this graph directly, in-turn, with real Cypher over a FalkorDB ACL that
grants it `GRAPH.QUERY` on `orion_worldview` and only `GRAPH.RO_QUERY` on the
Juniper-curated Atlas (`orion_substrate`). Hub reads it back and never writes a
node: every query in this module goes out as `GRAPH.RO_QUERY`, so a bug here
cannot corrupt Orion's space even though Hub connects as the unrestricted
`default` user. The one write Hub does make is `ACL SETUSER`, which grants that
access rather than using it -- see `assert_orion_acl`.

WHY HUB CONNECTS AS `default` AND ORION DOES NOT. FalkorDB's `default` user is
`nopass ~* &* +@all` on this host, so Hub needs no credential and Orion's
credential is the whole boundary. Confirmed live 2026-08-26 as `orion_curiosity`:
`GRAPH.QUERY orion_substrate "CREATE (:Tmp)"` -> `NOPERM No permissions to
access a key`, and `GRAPH.RO_QUERY orion_substrate "CREATE (:Tmp)"` -> refused
as a read-only command. Two independent refusals on the one graph Orion must
not corrupt.

WHAT HUB READS, AND WHY EACH ONE EARNS ITS QUERY.

  open priors      what orders the next run's presentation. Uncertainty
                   orders it; Orion still chooses. See `read_priors`.
  last TurnOutcome how a decision made INSIDE the turn crosses the boundary.
                   Absence is a safe default: no node means no continuation
                   and no outreach. See `read_turn_outcome`.
  recently settled priors Orion has CLOSED. This replaces a hint that used to
                   read the journal's title column and was dead from the day it
                   shipped -- see `study_material.py` where it used to live for
                   the full account of why a prose heuristic was the wrong
                   source and structure Orion authored is the right one.
  run footprint    what Orion actually wrote this run, counted by label. This
                   is the inspectable evidence behind the journal's claim that
                   a run formed structure -- AGENTS.md 0A's no-empty-shell
                   clause applied to a graph instead of to prose. See
                   `read_run_footprint`.

NO PARAMETERS, DELIBERATELY. FalkorDB takes query parameters through a `CYPHER
k=v ` prefix string, which is string interpolation with extra steps. Every
query here is static except for a `run_id`, which is validated against
`_RUN_ID_RE` before it is ever put in a query string -- so there is no
injection surface to route around rather than a sanitiser to trust.

FLOATS COME BACK AS STRINGS. With `decode_responses=True`, FalkorDB returns
doubles as decimal strings (`'0.55'`, not `0.55`) -- confirmed live against
this deployment. Every numeric read goes through `_as_float`/`_as_int`, which
is why a prior whose confidence Orion wrote as a string still reads correctly.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

logger = logging.getLogger("orion.curiosity.worldview")

# Orion's runs are named by `uuid4().hex[:12]` in the Hub loop. Anything else
# never reaches a query string.
_RUN_ID_RE = re.compile(r"^[0-9a-f]{6,32}$")

# Node labels Orion is asked to use. Named here, in the prompt, and in the
# reader -- three places, one contract, stated once in `PROMPT_SCHEMA` below so
# the prompt cannot drift away from what this module can actually read back.
LABEL_PRIOR = "Prior"
LABEL_CONCEPT = "Concept"
LABEL_FINDING = "Finding"
LABEL_HOP = "Hop"
LABEL_TURN_OUTCOME = "TurnOutcome"

STATUS_OPEN = "open"
STATUS_SUPPORTED = "supported"
STATUS_REVISED = "revised"
STATUS_REFUTED = "refuted"
STATUS_RETIRED = "retired_unresolvable"

# A PRIOR IS LIVE UNTIL ORION EXPLICITLY CLOSES IT.
#
# `supported` and `revised` are what a TEST returned, not a decision to stop
# holding the belief -- a claim at confidence 0.85 after one test is not
# settled, and `revised` means the claim itself just changed, which is the most
# live a prior can be. Only `refuted` ("this is wrong") and
# `retired_unresolvable` ("I cannot answer this with what I can reach") are
# Orion saying it is done.
#
# This was a real outage of the accumulation loop, not a hypothetical: on
# 2026-08-27 run `7736d5271d97` tested its one inherited prior and revised it,
# and run `0a14e9531089` four hours later was offered `priors=0/0` because the
# reader asked for `status = 'open'` only. Every prior Orion had ever formed
# was invisible to it, and its own new prior was written `supported` on
# formation -- born already deleted from its future. Confidence could not move
# a second time because nothing came back to be tested twice.
#
# Defined as the COMPLEMENT of closed, not as a list of live statuses, so a
# status Orion typos reads as live. Losing a belief to a spelling mistake is
# the worse failure: re-litigation is already bounded by `stale_after`, and a
# stale prior is still shown with an explicit retire option.
CLOSED_STATUSES = (STATUS_REFUTED, STATUS_RETIRED)


class WorldviewUnavailable(RuntimeError):
    """The graph could not be reached or answered. Distinct from empty."""


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value)) if value is not None else default
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    """FalkorDB booleans survive as real bools; strings are Orion's mistake.

    Accepting `"true"` here is not laxity -- Orion writes this Cypher by hand
    and a quoted boolean is the single likeliest typo. What is NOT accepted is
    anything unrecognised: it reads as False, so a malformed decision fails
    closed (no continuation, no outreach) rather than opening a turn or
    interrupting Juniper on a value nobody can interpret.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _clip(text: Any, limit: int = 240) -> str:
    value = str(text or "").strip().replace("\n", " ")
    return value[:limit].rstrip() + "…" if len(value) > limit else value


@dataclass(frozen=True)
class Prior:
    """A claim Orion holds about its world that could turn out to be wrong."""

    prior_id: str
    claim: str
    confidence: Optional[float]
    status: str
    times_tested: int
    formed_from: str = ""
    last_tested_at: str = ""

    @property
    def uncertainty(self) -> float:
        """0.0 = maximally uncertain. Used only to ORDER the presentation.

        A prior with no confidence recorded sorts as maximally uncertain,
        which is the honest reading of "Orion never said how sure it was".
        """
        if self.confidence is None:
            return 0.0
        return abs(self.confidence - 0.5)

    def preview(self) -> str:
        confidence = (
            f"{self.confidence:.2f}" if self.confidence is not None else "no confidence recorded"
        )
        tested = (
            "never tested" if self.times_tested <= 0
            else f"tested {self.times_tested}x"
        )
        line = f"[{self.prior_id[:8]} confidence={confidence}, {tested}] {_clip(self.claim)}"
        if self.formed_from:
            line += f"\n      formed from: {_clip(self.formed_from, 120)}"
        return line


@dataclass(frozen=True)
class TurnOutcome:
    """The decision a turn made inside itself, written where Orion can write.

    Rejected alternative (recorded because it is the obvious one): a fenced
    JSON block in the prose, parsed by the loop. That makes Orion's decision an
    artifact of formatting, loses a real finding to a malformed fence, and puts
    a regex between a model and a decision it already had a place to record.
    """

    run_id: str
    continue_line: bool
    continue_note: str
    reach_out: bool
    reach_out_why: str
    written_at: Optional[int] = None


@dataclass(frozen=True)
class WorldviewSnapshot:
    """Everything Hub read from Orion's graph for one run's presentation."""

    # `live_*`, not `open_*`: these hold every prior Orion has not explicitly
    # closed, which includes `supported` and `revised` ones. A field named for
    # one status while holding several is how the next reader reintroduces the
    # bug CLOSED_STATUSES documents.
    live_priors: list[Prior] = field(default_factory=list)
    stale_priors: list[Prior] = field(default_factory=list)
    recently_settled: list[tuple[str, str]] = field(default_factory=list)
    live_total: int = 0
    closed_total: int = 0
    concept_total: int = 0
    continuation: Optional[TurnOutcome] = None
    unavailable_reason: Optional[str] = None

    @property
    def is_unavailable(self) -> bool:
        return self.unavailable_reason is not None


class WorldviewReader:
    """Read-only `GRAPH.RO_QUERY` access to Orion's own graph, from Hub.

    Synchronous on purpose: `redis-py` is what Hub already ships and the
    queries are single-digit-millisecond reads over localhost. Every caller in
    Hub wraps this in `asyncio.to_thread`, the same rule
    `endogenous_outreach._should_roll` already applies to its Postgres round
    trip -- Hub runs one uvicorn worker, so a blocking call straight in a
    coroutine stalls every connected websocket.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        graph_name: str,
        socket_timeout: float = 5.0,
        client: Any = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.graph_name = graph_name
        self._socket_timeout = float(socket_timeout)
        self._client = client

    def client(self) -> Any:
        """The underlying Redis connection.

        Public because `orion/curiosity/acl.py` needs a connection to issue
        `ACL SETUSER` on -- a WRITE, and the only one Hub makes anywhere near
        this graph. It is deliberately not routed through `query()`: that method
        is `GRAPH.RO_QUERY` by construction, which is what keeps a bug in this
        module from corrupting Orion's space.
        """
        return self._redis()

    def _redis(self) -> Any:
        if self._client is not None:
            return self._client
        import redis  # local import: keeps this module importable in unit tests

        self._client = redis.Redis(
            host=self.host,
            port=self.port,
            decode_responses=True,
            socket_timeout=self._socket_timeout,
            socket_connect_timeout=self._socket_timeout,
        )
        return self._client

    def query(self, cypher: str) -> list[dict[str, Any]]:
        """Run one read-only Cypher and return rows as name-keyed dicts."""
        try:
            reply = self._redis().execute_command(
                "GRAPH.RO_QUERY", self.graph_name, cypher
            )
        except Exception as exc:  # noqa: BLE001 -- surfaced as a typed failure
            raise WorldviewUnavailable(f"{type(exc).__name__}: {exc}") from exc
        return rows_from_reply(reply)


def rows_from_reply(reply: Any) -> list[dict[str, Any]]:
    """`[header, rows, stats]` -> `[{col: value}]`. Empty on any other shape.

    Kept a module-level function rather than a method so the reply parsing is
    testable without a Redis at all -- the shape is the contract with FalkorDB,
    and it is the part most likely to change under us.
    """
    if not isinstance(reply, (list, tuple)) or len(reply) < 2:
        return []
    header = reply[0] if isinstance(reply[0], (list, tuple)) else []
    names = [str(h[-1]) if isinstance(h, (list, tuple)) and h else str(h) for h in header]
    out: list[dict[str, Any]] = []
    for row in reply[1] or []:
        if not isinstance(row, (list, tuple)):
            continue
        out.append({names[i]: row[i] for i in range(min(len(names), len(row)))})
    return out


# --- Cypher. Static except for a validated run_id. --------------------------

_PRIOR_FIELDS = (
    "p.prior_id AS prior_id, p.claim AS claim, p.confidence AS confidence, "
    "p.status AS status, p.times_tested AS times_tested, "
    "p.formed_from AS formed_from, p.last_tested_at AS last_tested_at"
)

# `p.status IS NULL OR NOT ... IN` rather than a positive `IN [live]` list:
# in Cypher `NOT null IN [...]` evaluates to null, which filters the row OUT,
# so a prior written with no status at all would vanish without the explicit
# null arm. See CLOSED_STATUSES for why unknown must read as live.
_LIVE_WHERE = (
    "(p.status IS NULL OR NOT p.status IN "
    f"[{', '.join(repr(s) for s in CLOSED_STATUSES)}])"
)
_CLOSED_WHERE = f"(p.status IN [{', '.join(repr(s) for s in CLOSED_STATUSES)}])"

# The live set does NOT drain the way the old `open`-only set did -- a prior
# now leaves it only when Orion refutes or retires it -- so a silent truncation
# here is a new failure mode: rows past the limit would never be shown, never
# accumulate `times_tested`, and so never reach `stale_after` to be retired.
#
# NOT fixed with a server-side `ORDER BY abs(p.confidence - 0.5)`, which is the
# obvious repair and is WORSE. Orion writes this graph by hand and sometimes
# quotes a number; FalkorDB rejects the whole query on the first one
# ("Type mismatch: expected ... but was String", reproduced live 2026-08-27),
# which costs Orion its entire world view rather than merely mis-ordering it.
# `_as_float` in Python tolerates the same value. So: a bound high enough not
# to bite, and loud when it is reached.
LIVE_PRIORS_LIMIT = 2000

LIVE_PRIORS_CYPHER = (
    f"MATCH (p:{LABEL_PRIOR}) WHERE {_LIVE_WHERE} "
    f"RETURN {_PRIOR_FIELDS} LIMIT {LIVE_PRIORS_LIMIT}"
)

COUNTS_CYPHER = (
    f"MATCH (p:{LABEL_PRIOR}) "
    f"RETURN sum(CASE WHEN {_LIVE_WHERE} THEN 1 ELSE 0 END) AS live_total, "
    f"sum(CASE WHEN {_CLOSED_WHERE} THEN 1 ELSE 0 END) AS closed_total"
)

CONCEPT_COUNT_CYPHER = f"MATCH (c:{LABEL_CONCEPT}) RETURN count(c) AS n"

# Priors Orion has closed, newest first. `last_tested_at` is a string Orion
# writes by hand, so this ordering is only as good as what it wrote -- which is
# exactly why it is a HINT in the prompt and never a gate on anything.
#
# CLOSED, not merely "not open": a `supported` prior is still offered for
# testing above, and listing it here as well would show Orion the same claim
# twice in one prompt under two contradictory headings.
RECENT_SETTLED_CYPHER = (
    f"MATCH (p:{LABEL_PRIOR}) WHERE {_CLOSED_WHERE} "
    "RETURN p.claim AS claim, p.status AS status, "
    "p.last_tested_at AS last_tested_at "
    "ORDER BY p.last_tested_at DESC LIMIT 8"
)

def run_footprint_cypher(run_id: str) -> str:
    """Count what Orion wrote during ONE run, grouped by node label.

    This is the run's inspectable evidence. `labels(n)[0]` rather than the
    whole list because Orion writes single-labelled nodes; a multi-labelled one
    would be counted under its first label, which is a reporting inaccuracy and
    not a correctness problem.
    """
    if not _RUN_ID_RE.match(str(run_id or "")):
        raise ValueError(f"refusing to build Cypher for a non-hex run_id: {run_id!r}")
    return (
        f"MATCH (n) WHERE n.run_id = '{run_id}' "
        "RETURN labels(n)[0] AS label, count(n) AS n"
    )


def outcome_for_run_cypher(run_id: str) -> str:
    """The `:TurnOutcome` THIS run wrote, not merely the newest one.

    Reading the newest instead would silently attribute a previous run's
    decision to this one whenever a turn died before writing its own -- and
    "died before writing" is exactly the case whose safe default (no
    continuation, no outreach) this design depends on.
    """
    if not _RUN_ID_RE.match(str(run_id or "")):
        raise ValueError(f"refusing to build Cypher for a non-hex run_id: {run_id!r}")
    return (
        f"MATCH (t:{LABEL_TURN_OUTCOME}) WHERE t.run_id = '{run_id}' "
        "RETURN t.run_id AS run_id, t.continue_line AS continue_line, "
        "t.continue_note AS continue_note, t.reach_out AS reach_out, "
        "t.reach_out_why AS reach_out_why, t.written_at AS written_at "
        "ORDER BY t.written_at DESC LIMIT 1"
    )


def hops_for_run_cypher(run_id: str) -> str:
    if not _RUN_ID_RE.match(str(run_id or "")):
        raise ValueError(f"refusing to build Cypher for a non-hex run_id: {run_id!r}")
    return (
        f"MATCH (h:{LABEL_HOP}) WHERE h.run_id = '{run_id}' "
        "RETURN h.n AS n, h.note AS note ORDER BY h.n ASC LIMIT 20"
    )


# --- Row -> dataclass -------------------------------------------------------


def build_prior(row: dict[str, Any]) -> Optional[Prior]:
    """None for a row with no usable identity or claim.

    A prior Orion wrote without a `prior_id` or without a `claim` cannot be
    presented back to it as a testable claim, and inventing either would be the
    heuristic re-inference this whole arc exists to delete. Callers count the
    drops and log them, so a schema drift is visible rather than silent.
    """
    prior_id = str(row.get("prior_id") or "").strip()
    claim = str(row.get("claim") or "").strip()
    if not prior_id or not claim:
        return None
    return Prior(
        prior_id=prior_id,
        claim=claim,
        confidence=_as_float(row.get("confidence")),
        status=str(row.get("status") or STATUS_OPEN).strip() or STATUS_OPEN,
        times_tested=_as_int(row.get("times_tested"), 0),
        formed_from=str(row.get("formed_from") or "").strip(),
        last_tested_at=str(row.get("last_tested_at") or "").strip(),
    )


def build_turn_outcome(row: dict[str, Any]) -> Optional[TurnOutcome]:
    run_id = str(row.get("run_id") or "").strip()
    if not run_id:
        return None
    return TurnOutcome(
        run_id=run_id,
        continue_line=_as_bool(row.get("continue_line")),
        continue_note=str(row.get("continue_note") or "").strip(),
        reach_out=_as_bool(row.get("reach_out")),
        reach_out_why=str(row.get("reach_out_why") or "").strip(),
        written_at=_as_int(row.get("written_at"), 0) or None,
    )


def _rotation_key(prior_id: str, seed: str) -> str:
    """A per-run ordering for priors that are otherwise exactly tied.

    Every prior Orion forms from the prompt's template starts at confidence
    0.55 with `times_tested: 0`, so `(uncertainty, times_tested)` ties across
    the whole fresh pool and the last term decides everything. A plain
    `prior_id` tiebreak is stable across runs, which means the same
    lexicographically-lowest `sample` priors are shown every run and the rest
    are never presented, never tested, and never retirable. That did not matter
    while a prior left the pool on its first test; it does now.

    Seeded by run so a single run is reproducible -- the same run always builds
    the same prompt -- while the window moves between runs.
    """
    return hashlib.sha256(f"{seed}:{prior_id}".encode()).hexdigest()


def select_priors(
    rows: Sequence[dict[str, Any]],
    *,
    sample: int,
    stale_after: int,
    rotate_seed: str = "",
) -> tuple[list[Prior], list[Prior], int]:
    """Split LIVE priors into (offered, stale, dropped_count).

    UNCERTAINTY ORDERS THE PRESENTATION; ORION STILL CHOOSES. That distinction
    is the whole difference from the keyword detector this arc deleted: the
    code is not naming a subject, it is showing Orion where its own map is
    thin. Most-uncertain first, then least-tested, then a PER-RUN rotation among
    exact ties -- see `_rotation_key`. It was a stable id tiebreak until the
    pool stopped draining on first test, at which point stable meant the same
    `sample` priors every run forever and the rest never presented at all.

    STALE PRIORS ARE SEPARATED, NOT HIDDEN. A prior tested `stale_after` times
    without being closed is exactly the "finds a favourite and
    re-litigates it" failure in a new costume, so it leaves the uncertainty
    list -- but it is still shown, in its own bucket, with the explicit option
    to retire it. Dropping it silently would leave it live in the graph
    forever with nothing able to close it, since Hub never writes.
    """
    priors: list[Prior] = []
    dropped = 0
    for row in rows:
        prior = build_prior(row)
        if prior is None:
            dropped += 1
            continue
        priors.append(prior)

    stale = [p for p in priors if stale_after > 0 and p.times_tested >= stale_after]
    stale_ids = {p.prior_id for p in stale}
    fresh = [p for p in priors if p.prior_id not in stale_ids]
    fresh.sort(
        key=lambda p: (
            p.uncertainty,
            p.times_tested,
            _rotation_key(p.prior_id, rotate_seed),
        )
    )
    stale.sort(
        key=lambda p: (-p.times_tested, _rotation_key(p.prior_id, rotate_seed))
    )
    return fresh[: max(0, sample)], stale[: max(0, sample)], dropped


def read_snapshot(
    reader: WorldviewReader,
    *,
    sample: int,
    stale_after: int,
    rotate_seed: str = "",
) -> WorldviewSnapshot:
    """One read of everything the next prompt needs. Never raises.

    An unreachable graph is reported as `unavailable_reason`, NOT as an empty
    world view -- those must never be the same state, or a broken ACL after a
    FalkorDB restart looks identical to a mind that has not formed a prior yet.
    Same shape as `StudyMaterial.is_unavailable` next door, and for the same
    reason: the only symptom of the former would otherwise be an absence.
    """
    try:
        prior_rows = reader.query(LIVE_PRIORS_CYPHER)
        count_rows = reader.query(COUNTS_CYPHER)
        concept_rows = reader.query(CONCEPT_COUNT_CYPHER)
        settled_rows = reader.query(RECENT_SETTLED_CYPHER)
    except WorldviewUnavailable as exc:
        return WorldviewSnapshot(unavailable_reason=str(exc)[:200])
    if len(prior_rows) >= LIVE_PRIORS_LIMIT:
        logger.warning(
            "curiosity_worldview_priors_truncated limit=%s -- the live pool has "
            "outgrown one read, so priors past the limit are never shown and "
            "cannot accumulate times_tested or be retired",
            LIVE_PRIORS_LIMIT,
        )
    offered, stale, dropped = select_priors(
        prior_rows,
        sample=sample,
        stale_after=stale_after,
        rotate_seed=rotate_seed,
    )
    if dropped:
        logger.warning(
            "curiosity_worldview_unreadable_priors dropped=%s -- rows with no "
            "prior_id or no claim; Orion's Cypher may have drifted from the "
            "schema the prompt states",
            dropped,
        )
    counts = count_rows[0] if count_rows else {}
    live_total = _as_int(counts.get("live_total"), len(offered))
    closed_total = _as_int(counts.get("closed_total"), 0)
    if live_total == 0 and closed_total > 0:
        # Orion has formed priors and closed every one of them. Legitimate in
        # principle; in practice this is what the 2026-08-27 status-filter bug
        # looked like from the outside, and the only symptom was a run quietly
        # starting from nothing. Loud here so a recurrence is not invisible.
        logger.warning(
            "curiosity_worldview_pool_dead closed=%s -- no live priors left, so "
            "the next run inherits nothing to test; expected only if Orion has "
            "genuinely refuted or retired all of them",
            closed_total,
        )
    return WorldviewSnapshot(
        live_priors=offered,
        stale_priors=stale,
        recently_settled=[
            (str(r.get("claim") or "").strip(), str(r.get("status") or "").strip())
            for r in settled_rows
            if str(r.get("claim") or "").strip()
        ],
        live_total=live_total,
        closed_total=closed_total,
        concept_total=_as_int((concept_rows[0] if concept_rows else {}).get("n"), 0),
    )


def read_turn_outcome(reader: WorldviewReader, run_id: str) -> Optional[TurnOutcome]:
    """This run's decision, or None. Never raises -- None is the safe default."""
    try:
        rows = reader.query(outcome_for_run_cypher(run_id))
    except (WorldviewUnavailable, ValueError) as exc:
        logger.warning("curiosity_turn_outcome_read_failed run=%s err=%s", run_id, exc)
        return None
    return build_turn_outcome(rows[0]) if rows else None


def read_run_footprint(
    reader: WorldviewReader, run_id: str
) -> Optional[dict[str, int]]:
    """What Orion wrote this run, by label.

    `{}` means Orion wrote nothing; `None` means the question could not be
    answered. Collapsing those two would put "wrote nothing to its own graph"
    in the journal for a run whose graph was simply unreachable -- the same
    unreadable-vs-empty conflation this module refuses everywhere else, landing
    in the one artifact Juniper actually reads.
    """
    try:
        rows = reader.query(run_footprint_cypher(run_id))
    except (WorldviewUnavailable, ValueError) as exc:
        logger.warning("curiosity_run_footprint_read_failed run=%s err=%s", run_id, exc)
        return None
    return {
        str(r.get("label") or "unknown"): _as_int(r.get("n"), 0)
        for r in rows
        if _as_int(r.get("n"), 0) > 0
    }


def read_hop_notes(reader: WorldviewReader, run_id: str) -> list[tuple[int, str]]:
    """The reflections Orion recorded as it went, in order. `[]` on failure."""
    try:
        rows = reader.query(hops_for_run_cypher(run_id))
    except (WorldviewUnavailable, ValueError) as exc:
        logger.warning("curiosity_hop_notes_read_failed run=%s err=%s", run_id, exc)
        return []
    return [
        (_as_int(r.get("n"), 0), str(r.get("note") or "").strip())
        for r in rows
        if str(r.get("note") or "").strip()
    ]
