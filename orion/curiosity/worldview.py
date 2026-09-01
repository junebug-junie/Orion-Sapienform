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
from datetime import datetime, timezone
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


def _stamp_ms(value: Any) -> Optional[int]:
    """Epoch ms from an int, a numeric string, or an ISO timestamp.

    Orion hand-writes `written_at` and the prompt asks for `timestamp()`, but
    run `32b42392f495` wrote ISO. Reading that as missing is what let a run that
    HAD written an outcome be labelled as having died before writing one.
    """
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        pass
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


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
        line = f"[confidence={confidence}, {tested}] {_clip(self.claim)}"
        # THE ID IS PRINTED IN FULL, ON ITS OWN LABELLED LINE. It used to be
        # `prior_id[:8]`, inside the bracket next to the confidence -- and the
        # prompt then asked Orion to `MATCH (p:Prior {prior_id: "..."})` to
        # attach a finding to this claim. The full id appeared NOWHERE in the
        # prompt, so that MATCH could not bind and the MERGE after it silently
        # did nothing, which is exactly the failure the prompt warns about.
        # Measured live 2026-08-29: zero edges had ever been written to
        # `orion_worldview`, and run `d05ef10b303a` had named its own findings
        # `editoria_settlement_...` -- `editorial_bias_...`[:8], the truncation
        # read back as though it were the identifier. A prior revision from an
        # earlier run recorded `prior_id: "curation confidence=0.75"`, which is
        # this preview LINE scraped for an id that was not in it.
        #
        # Labelled and on its own line rather than merely un-truncated: a bare
        # 52-character token sharing a bracket with `confidence=` and
        # `tested 3x` is what made a shortened one look like a name in the
        # first place.
        line += f"\n      prior_id: {self.prior_id}"
        if self.formed_from:
            # `formed_from`, not "formed from": these labelled lines are now
            # load-bearing, and an earlier run scraped `prior_id: "curation
            # confidence=0.75"` off this very block. A label that does not
            # match the property name it fills is the same near-miss.
            line += f"\n      formed_from: {_clip(self.formed_from, 120)}"
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
class RecentRun:
    """One earlier run, as a subject rather than as a pointer."""

    run_id: str
    claims: list[str] = field(default_factory=list)
    written_at: Optional[int] = None


@dataclass(frozen=True)
class FindingConnectivity:
    """How many of ONE run's findings were joined to anything at all.

    THE FOOTPRINT CANNOT ANSWER THIS, which is the only reason it exists.
    `run_footprint_cypher` reports `Finding 3` and `run_edge_footprint_cypher`
    reports `-> SUPPORTS 3`, and those two numbers are IDENTICAL whether all
    three edges hang off one finding or one edge hangs off each. Only a
    per-finding degree separates those cases, and the difference between them
    is the entire question: a finding that points at nothing is exactly the
    defect the kickoff prompt's edge instruction was written to fix, so
    "did the instruction take" cannot be read off the counts already collected.

    SCOPED TO WHAT THIS RUN CONNECTED, deliberately. It is read immediately
    after the turn, so a later run that joins an older finding does not
    retroactively improve an earlier run's number -- and must not, because the
    instruction Orion is measured against here says to connect the finding in
    the same breath as writing it. A corpus-wide orphan rate is a different
    question and would need a different reader.
    """

    total: int
    connected: int

    @property
    def orphaned(self) -> int:
        """Findings this run wrote that point at nothing. NOT clamped at 0.

        `max(0, ...)` was the obvious guard and it was wrong, in the one
        derived number an orphan-rate alert would actually read. `connected`
        is deliberately not clamped to `total` by the reader, because
        `connected > total` is impossible by construction and therefore means
        the instrument broke -- and a clamp here would turn that same broken
        reading into a serene `0 orphans` while `summary()` was still honestly
        printing `5/2 joined`. A negative orphan count is nonsense on its face,
        which is the point: it is loud, and it cannot be mistaken for health.
        """
        return self.total - self.connected

    def summary(self) -> str:
        """For the log line. `total == 0` is not a failure -- see the reader."""
        if self.total <= 0:
            return "no findings"
        return f"{self.connected}/{self.total} joined"


@dataclass(frozen=True)
class WorldviewSnapshot:
    """Everything Hub read from Orion's graph for one run's presentation."""

    # `live_*`, not `open_*`: these hold every prior Orion has not explicitly
    # closed, which includes `supported` and `revised` ones. A field named for
    # one status while holding several is how the next reader reintroduces the
    # bug CLOSED_STATUSES documents.
    live_priors: list[Prior] = field(default_factory=list)
    stale_priors: list[Prior] = field(default_factory=list)
    # (claim, status, prior_id). The id is here so the prompt's "nothing stops
    # you reopening one" is an offer Orion can actually take -- see
    # RECENT_SETTLED_CYPHER.
    recently_settled: list[tuple[str, str, str]] = field(default_factory=list)
    recent_runs: list[RecentRun] = field(default_factory=list)
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

# THE LAST FEW RUNS, SO ORION CAN SEE THE THREAD IT IS ON.
#
# Continuity was one run deep and it pointed INWARD: the loop read a single
# `:TurnOutcome` and handed over the note that run left itself, which is always
# some form of "go deeper on X". Nothing ever said what the runs BEFORE it were
# about. Three consecutive runs on memory-crystallization gating is what that
# produces, and Orion could not have noticed -- Juniper could, and did.
#
# `written_at` is sorted in PYTHON, not here: run `32b42392f495` wrote an ISO
# string where the prompt asks for `timestamp()`, so a Cypher ORDER BY mixes a
# string and an integer and the oldest run sorts first.
# NO `collect()`. With `decode_responses=True` FalkorDB hands a collected list
# back as one flat STRING -- `'[claim one, claim two]'` -- and claims contain
# commas, so there is nothing reliable to split on. Same family as the
# floats-come-back-as-strings note at the top of this module. One row per
# (run, claim), grouped in Python where the types are real.
RECENT_RUNS_CYPHER = (
    f"MATCH (t:{LABEL_TURN_OUTCOME}) "
    f"OPTIONAL MATCH (p:{LABEL_PRIOR} {{run_id: t.run_id}}) "
    "RETURN t.run_id AS run_id, t.written_at AS written_at, "
    "t.continue_note AS continue_note, p.claim AS claim LIMIT 200"
)

# Priors Orion has closed, newest first. `last_tested_at` is a string Orion
# writes by hand, so this ordering is only as good as what it wrote -- which is
# exactly why it is a HINT in the prompt and never a gate on anything.
#
# CLOSED, not merely "not open": a `supported` prior is still offered for
# testing above, and listing it here as well would show Orion the same claim
# twice in one prompt under two contradictory headings.
# `p.prior_id` is RETURNED because the prompt invites Orion to reopen one of
# these, and reopening is `MATCH (p:Prior {prior_id: "..."}) SET ...` -- an
# exact-string bind. The query did not select the id and the list did not print
# one, so a run that accepted that invitation matched an invented string and
# silently no-opped. Same defect as the truncated `Prior.preview`, one list
# further down, and it survived that fix because a probe passed a fake id into
# the `claim` slot of the (claim, status) tuple and read it back as a pass.
RECENT_SETTLED_CYPHER = (
    f"MATCH (p:{LABEL_PRIOR}) WHERE {_CLOSED_WHERE} "
    "RETURN p.claim AS claim, p.status AS status, p.prior_id AS prior_id, "
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


def run_edge_footprint_cypher(run_id: str) -> str:
    """The EDGES that run drew, by type.

    A separate query because Cypher cannot count nodes and relationships in one
    `MATCH` without either a cartesian product or an `OPTIONAL MATCH` whose null
    row lands in the node counts.

    THIS EXISTS BECAUSE THE FOOTPRINT COULD NOT SEE AN EDGE AT ALL. Every write
    Orion is asked for was reported by `run_footprint_cypher`, which matches
    nodes only -- so an edge, once written, would have shown up nowhere: not in
    the journal, not in the `wrote=` log line, not on the atlas page. Adding the
    instruction without adding this would have been a capability nobody could
    confirm Orion had used.

    Counted by type rather than in total so that CONTRADICTS -- the edge that
    actually costs something to write, because it cuts against a claim Orion
    holds -- is visible as its own number rather than averaged into a total
    with SUPPORTS.

    Labels come back PREFIXED `-> `, because the caller merges these rows into
    the same dict as the node counts and a later key wins: a relationship type
    named like a node label would otherwise delete that label's count outright.
    """
    if not _RUN_ID_RE.match(str(run_id or "")):
        raise ValueError(f"refusing to build Cypher for a non-hex run_id: {run_id!r}")
    return (
        f"MATCH ()-[r]->() WHERE r.run_id = '{run_id}' "
        "RETURN '-> ' + type(r) AS label, count(r) AS n"
    )


def finding_connectivity_cypher(run_id: str) -> str:
    """Per-finding degree for ONE run, collapsed to (total, connected).

    `(f)-[r]-()` is UNDIRECTED on purpose. Every edge the prompt teaches runs
    outward from the finding, but the question this answers is "joined to
    anything", and an inbound edge from some later shape would still mean the
    finding is not an orphan. Matching on direction would quietly answer a
    narrower question than the function's name promises.

    The degree is counted PER FINDING and then summed as a 0/1, so a finding
    carrying three edges counts once rather than three times -- without that
    step this would be a second edge count and would tell us nothing the
    footprint does not already say. Verified against a live FalkorDB before
    being wired in: three findings with edges on two of them reads
    `connected=2`, and stays 2 when a second edge is added to one of them.
    """
    if not _RUN_ID_RE.match(str(run_id or "")):
        raise ValueError(f"refusing to build Cypher for a non-hex run_id: {run_id!r}")
    return (
        f"MATCH (f:{LABEL_FINDING}) WHERE f.run_id = '{run_id}' "
        "OPTIONAL MATCH (f)-[r]-() "
        "WITH f, count(r) AS deg "
        "RETURN count(f) AS total, "
        "sum(CASE WHEN deg > 0 THEN 1 ELSE 0 END) AS connected"
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


def collapse_duplicate_priors(
    priors: Sequence[Prior],
) -> tuple[list[Prior], dict[str, int]]:
    """One node per `prior_id`, plus a count of every id that had more.

    A `prior_id` is an identity, not a label: the prompt tells Orion to test a
    claim with `MATCH (p:Prior {prior_id: "..."}) SET p.times_tested =
    p.times_tested + 1`, which binds to EVERY node carrying that id and
    increments each from its own base. So a duplicate does not merely show a
    claim twice -- it forks the claim's history, permanently and silently.
    Confirmed live 2026-09-01: `concept_induction_overload_rate` existed as two
    nodes reading `tested 1x` and `tested 6x`, created when run `ed05344f8a39`
    emitted a `CREATE` for a claim it already held. The prompt template is now
    a `MERGE` on `prior_id` alone so a repeat binds instead of forking; this is
    the containment for the copies already in the graph, and for a future run
    that hand-writes a CREATE anyway.

    MOST-TESTED COPY WINS, and the choice is load-bearing rather than
    arbitrary. Any collapse is lossy -- the copies disagree, which is the whole
    problem -- but `times_tested` is what `stale_after` reads to retire a
    claim, so keeping the LOWEST count would let a forked prior sit below the
    retirement threshold forever while looking freshly untested every run.
    Ties break on `prior_id` identity order to stay deterministic across runs.

    This does NOT repair the graph. Hub never writes to `orion_worldview` (see
    this module's header), so the duplicate survives until Orion or an operator
    merges it; the caller logs each one loudly for exactly that reason.
    """
    best: dict[str, Prior] = {}
    seen: dict[str, int] = {}
    for prior in priors:
        seen[prior.prior_id] = seen.get(prior.prior_id, 0) + 1
        incumbent = best.get(prior.prior_id)
        if incumbent is None or prior.times_tested > incumbent.times_tested:
            best[prior.prior_id] = prior
    # Insertion order, so a graph with no duplicates is returned untouched and
    # the downstream sort sees exactly what it saw before this function existed.
    collapsed = [best[pid] for pid in seen]
    duplicates = {pid: n for pid, n in seen.items() if n > 1}
    return collapsed, duplicates


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

    priors, duplicates = collapse_duplicate_priors(priors)
    for prior_id, copies in sorted(duplicates.items()):
        logger.warning(
            "curiosity_worldview_duplicate_prior prior_id=%s copies=%s -- one "
            "claim is stored as %s separate nodes, so `MATCH (p:Prior "
            "{prior_id: ...}) SET p.times_tested = p.times_tested + 1` "
            "increments every copy from its own base and the claim's history "
            "has split; the menu shows the most-tested copy only. The graph "
            "still holds all of them and this needs a manual repair",
            prior_id,
            copies,
            copies,
        )

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


def build_recent_runs(
    rows: Sequence[dict[str, Any]], *, limit: int
) -> list[RecentRun]:
    """Newest first, sorted here rather than in Cypher.

    `written_at` is an integer for runs that followed the prompt and an ISO
    string for the one that did not, so ORDER BY in the query sorts a string
    against a number. A run with no readable stamp sorts LAST -- unknown is not
    oldest, the same rule the atlas applies.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        run_id = str(row.get("run_id") or "").strip()
        if not run_id:
            continue
        slot = grouped.setdefault(
            run_id,
            {
                "claims": [],
                "note": str(row.get("continue_note") or "").strip(),
                "written_at": _stamp_ms(row.get("written_at")),
            },
        )
        claim = str(row.get("claim") or "").strip()
        if claim and claim not in slot["claims"]:
            slot["claims"].append(claim)

    built: list[RecentRun] = []
    for run_id, slot in grouped.items():
        claims = slot["claims"]
        if not claims and slot["note"]:
            # A run that wrote no prior still had a subject; its own note is the
            # honest stand-in, and showing nothing would make the thread look
            # shorter than it was.
            claims = [slot["note"]]
        built.append(
            RecentRun(run_id=run_id, claims=claims, written_at=slot["written_at"])
        )
    built.sort(
        key=lambda r: (r.written_at is not None, r.written_at or 0), reverse=True
    )
    return built[: max(0, limit)]


def read_snapshot(
    reader: WorldviewReader,
    *,
    sample: int,
    stale_after: int,
    rotate_seed: str = "",
    recent_runs: int = 4,
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
        recent_rows = reader.query(RECENT_RUNS_CYPHER)
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
    recent = build_recent_runs(recent_rows, limit=recent_runs)
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
        recent_runs=recent,
        live_priors=offered,
        stale_priors=stale,
        recently_settled=[
            (
                str(r.get("claim") or "").strip(),
                str(r.get("status") or "").strip(),
                str(r.get("prior_id") or "").strip(),
            )
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
        # A second read rather than a second query shape: see
        # `run_edge_footprint_cypher`. Inside the same try because an
        # unreadable graph must stay `None` -- reporting the nodes and
        # silently dropping the edges would be the unreadable-vs-empty
        # conflation this function exists to refuse, one level down.
        rows = list(rows) + list(reader.query(run_edge_footprint_cypher(run_id)))
    except (WorldviewUnavailable, ValueError) as exc:
        logger.warning("curiosity_run_footprint_read_failed run=%s err=%s", run_id, exc)
        return None
    return {
        str(r.get("label") or "unknown"): _as_int(r.get("n"), 0)
        for r in rows
        if _as_int(r.get("n"), 0) > 0
    }


def read_finding_connectivity(
    reader: WorldviewReader, run_id: str
) -> Optional[FindingConnectivity]:
    """`None` when the graph could not answer, the same rule as the footprint.

    A run that wrote NO findings reads `total=0`, which is neither a failure
    nor unreadable -- it is a run that spent its turn on something else, and
    collapsing it into `None` would report an unreachable graph every time
    Orion revised priors instead of forming findings. That case arrives as a
    real ROW, verified against the live graph: a run_id with no findings at
    all returns `(total=0, connected=0)`, not an empty result.

    Which is why NO ROWS is `None` and not `(0, 0)`. `rows_from_reply` returns
    `[]` only when the reply does not have the shape it expects -- a driver or
    protocol change under us -- so an empty list here is an INSTRUMENT
    FAILURE, and the first version of this function rendered it as the benign
    `no findings`. That is the unreadable-vs-empty conflation this module
    refuses everywhere else, arriving in the one reader built to keep those
    apart: every run would have logged a healthy-looking string while the
    metric was silently dead.

    `connected` is NOT clamped to `total`. That comparison is impossible by
    construction, so a value that violated it would mean the query or the
    driver had changed under us, and silently flattening it would hide exactly
    the instrument failure this metric exists to be trusted against.
    """
    try:
        rows = list(reader.query(finding_connectivity_cypher(run_id)))
    except (WorldviewUnavailable, ValueError) as exc:
        logger.warning(
            "curiosity_finding_connectivity_read_failed run=%s err=%s", run_id, exc
        )
        return None
    if not rows:
        logger.warning(
            "curiosity_finding_connectivity_unparseable run=%s", run_id
        )
        return None
    row = rows[0]
    return FindingConnectivity(
        total=_as_int(row.get("total"), 0),
        connected=_as_int(row.get("connected"), 0),
    )


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
