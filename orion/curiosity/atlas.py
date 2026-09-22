"""An operator's read of Orion's world view — what changed, run by run.

Separate module from `worldview.py` on purpose. That one answers "what does the
next PROMPT need to show Orion"; this one answers "what does a human need to see
to know the loop is working". They read the same graph and share
`WorldviewReader`, but they are different projections with different failure
modes: a gap in the prompt's read costs Orion its continuity, a gap here costs
Juniper a dashboard panel.

WHY THIS EXISTS. On 2026-08-27 the accumulation loop went to zero — a reader
asking for `status = 'open'` stopped returning any prior Orion had ever tested —
and the only symptom was one log line, `priors=0/0`, that nobody read for four
hours. Everything needed to see it was already in FalkorDB, Postgres, Redis and
docker logs; nothing put it on one screen. That is the gap.

WHAT IS AND IS NOT RECOVERABLE. Every node Orion writes carries `run_id`, and a
tested prior also carries `last_run_id`, so **which run created or last touched
each node is real history** and the run-by-run growth of the graph can be
reconstructed exactly. The confidence a prior held BEFORE a given test is not:
the graph stores only the current value, and a `SET p.confidence = 0.72`
overwrites what was there.

So `:PriorRevision` was added — written by ORION, in the same statement it
already writes when testing a prior, not by Hub. Hub never writes to this graph
and that invariant is worth more than a backfilled chart. The consequence is
honest and must stay visible in the UI: **the confidence trajectory starts when
Orion first writes a revision, and is empty for everything before that.** An
empty trajectory here means "not recorded yet", never "confidence did not move".
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from orion.curiosity.peer_briefs import LABEL_PEER_BRIEF
from orion.curiosity.worldview import (
    CLOSED_STATUSES,
    LABEL_CONCEPT,
    LABEL_FINDING,
    LABEL_HOP,
    LABEL_INVESTIGATION_ROLE,
    LABEL_PRIOR,
    LABEL_TURN_OUTCOME,
    WorldviewReader,
    WorldviewUnavailable,
    _as_float,
    _as_int,
)

logger = logging.getLogger("orion.curiosity.atlas")

LABEL_PRIOR_REVISION = "PriorRevision"

# --- Cypher. Fully static: this module takes no caller input into a query. ---

ATLAS_PRIORS_LIMIT = 2000

ATLAS_PRIORS_CYPHER = (
    f"MATCH (p:{LABEL_PRIOR}) RETURN p.prior_id AS prior_id, p.claim AS claim, "
    "p.confidence AS confidence, p.status AS status, "
    "p.times_tested AS times_tested, p.formed_from AS formed_from, "
    "p.last_tested_at AS last_tested_at, p.run_id AS run_id, "
    "p.last_run_id AS last_run_id, p.why AS why "
    f"LIMIT {ATLAS_PRIORS_LIMIT}"
)

ATLAS_REVISIONS_CYPHER = (
    f"MATCH (r:{LABEL_PRIOR_REVISION}) RETURN r.prior_id AS prior_id, "
    "r.run_id AS run_id, r.from_confidence AS from_confidence, "
    "r.to_confidence AS to_confidence, r.from_status AS from_status, "
    "r.to_status AS to_status, r.written_at AS written_at LIMIT 5000"
)

ATLAS_UNUSED_CYPHER = (
    f"MATCH (c:{LABEL_CONCEPT}) RETURN count(c) AS n"
)

ATLAS_EDGES_CYPHER = "MATCH ()-[r]->() RETURN count(r) AS n"

# --- per-run graph nodes, bounded by a window rather than a row cap --------
#
# These feed `orion/curiosity/run_story.py`. The earlier atlas read pulled
# every Hop/Finding/TurnOutcome ever written under a LIMIT (5000 hops) and
# the page rendered all of it every 60s; the strip is 14 days by
# construction, so the read is too. Values are injected through FalkorDB's
# `CYPHER k=v` parameter prefix, never spliced into the pattern: the window is
# an int by construction and every run id passes `_RUN_ID_RE` first.
#
# `written_at` IS SELECTED on hops now. `worldview.hop_order_key` has sorted
# a resumed run's hops by clock since 2026-09-19, and the atlas read never
# adopted it -- a retried run rendered its two attempts interleaved 1,1,2,2.

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_ID_LIST_MAX = 2000

# Self-inquiry runs write one of these instead of a Finding. Both carry
# `run_id` and `written_at`; without them a self-inquiry run that wrote a
# lived answer read as "wrote nothing" (run `3dc94088912b`, live 2026-09-22).
LABEL_SELF_DEFINITION = "SelfDefinition"
LABEL_LIVED_ANSWER = "LivedAnswer"
LABEL_HELP_REQUEST = "HelpRequest"

RUN_NODE_FIELDS: dict[str, str] = {
    LABEL_SELF_DEFINITION: "n.run_id AS run_id, n.text AS text, n.evidence AS evidence, n.revises AS revises, n.written_at AS written_at",
    LABEL_LIVED_ANSWER: (
        "n.run_id AS run_id, n.question_id AS question_id, n.family AS family, n.text AS text, "
        "n.evidence AS evidence, n.revises AS revises, n.written_at AS written_at"
    ),
    LABEL_INVESTIGATION_ROLE: "n.run_id AS run_id, n.choice AS choice, n.why AS why, n.written_at AS written_at",
    LABEL_HOP: "n.run_id AS run_id, n.n AS n, n.note AS note, n.written_at AS written_at",
    LABEL_FINDING: "n.run_id AS run_id, n.finding_id AS finding_id, n.text AS text, n.evidence AS evidence, n.written_at AS written_at",
    LABEL_PRIOR_REVISION: (
        "n.run_id AS run_id, n.prior_id AS prior_id, n.from_confidence AS from_confidence, "
        "n.to_confidence AS to_confidence, n.from_status AS from_status, n.to_status AS to_status, "
        "n.written_at AS written_at"
    ),
    LABEL_TURN_OUTCOME: (
        "n.run_id AS run_id, n.continue_line AS continue_line, n.continue_note AS continue_note, "
        "n.reach_out AS reach_out, n.reach_out_why AS reach_out_why, n.written_at AS written_at"
    ),
}


def valid_run_id(value: Any) -> Optional[str]:
    """The only shape a run id may take before it reaches a query."""
    text = str(value or "").strip()
    return text if _RUN_ID_RE.match(text) else None


def _id_list(ids: list[str]) -> str:
    clean = sorted({i for i in (valid_run_id(x) for x in ids) if i})[:_ID_LIST_MAX]
    return "[" + ",".join(f"'{i}'" for i in clean) + "]"


def run_ids_since_cypher(since_ms: int) -> str:
    """Every run id that wrote a dated node at or after `since_ms`."""
    return (
        f"CYPHER since={int(since_ms)} MATCH (n) WHERE n.run_id IS NOT NULL "
        "AND n.written_at IS NOT NULL AND n.written_at >= $since "
        "RETURN DISTINCT n.run_id AS run_id"
    )


def run_nodes_cypher(label: str, run_ids: list[str]) -> str:
    fields = RUN_NODE_FIELDS[label]
    return f"CYPHER ids={_id_list(run_ids)} MATCH (n:{label}) WHERE n.run_id IN $ids RETURN {fields}"


def prior_claims_cypher(prior_ids: list[str]) -> str:
    """Claim text for the priors a set of revisions / HelpRequests touched.

    Prior ids are Orion-authored free text, so they are parameterised as a
    JSON string list rather than trusted against `_RUN_ID_RE`.
    """
    import json as _json

    clean = sorted({str(p)[:200] for p in prior_ids if p})[:_ID_LIST_MAX]
    return (
        f"CYPHER ids={_json.dumps(clean)} MATCH (p:{LABEL_PRIOR}) WHERE p.prior_id IN $ids "
        "RETURN p.prior_id AS prior_id, p.claim AS claim, p.line AS line, "
        "p.status AS status, p.confidence AS confidence"
    )


def help_requests_with_prior_cypher(run_ids: list[str]) -> str:
    """HelpRequests for runs, with ABOUT prior claim when linked."""
    return (
        f"CYPHER ids={_id_list(run_ids)} MATCH (h:{LABEL_HELP_REQUEST}) WHERE h.run_id IN $ids "
        f"OPTIONAL MATCH (h)-[:ABOUT]->(p:{LABEL_PRIOR}) "
        "RETURN h.help_id AS help_id, h.run_id AS run_id, "
        "coalesce(h.prior_id, p.prior_id) AS prior_id, "
        "h.question AS question, h.tried_summary AS tried_summary, "
        "h.success_criteria AS success_criteria, h.written_at AS written_at, "
        "p.claim AS prior_claim, p.status AS prior_status, "
        "p.confidence AS prior_confidence, p.line AS prior_line"
    )


def peer_briefs_for_runs_cypher(run_ids: list[str]) -> str:
    return (
        f"CYPHER ids={_id_list(run_ids)} MATCH (b:{LABEL_PEER_BRIEF}) WHERE b.run_id IN $ids "
        "RETURN b.brief_id AS brief_id, b.help_id AS help_id, b.run_id AS run_id, "
        "b.peer AS peer, b.status AS status, b.summary AS summary, "
        "b.refusal_reason AS refusal_reason, b.written_at AS written_at"
    )


@dataclass(frozen=True)
class RunNodeRows:
    """One read of every node kind a run story is built from."""

    roles: list[dict[str, Any]] = field(default_factory=list)
    hops: list[dict[str, Any]] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    revisions: list[dict[str, Any]] = field(default_factory=list)
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    priors: list[dict[str, Any]] = field(default_factory=list)
    # `:SelfDefinition` / `:LivedAnswer`, each row tagged `kind`.
    self_writes: list[dict[str, Any]] = field(default_factory=list)
    help_requests: list[dict[str, Any]] = field(default_factory=list)
    peer_briefs: list[dict[str, Any]] = field(default_factory=list)


def read_run_ids_since(reader: WorldviewReader, since_ms: int) -> list[str]:
    """Raises `WorldviewUnavailable`; the caller decides what a dead graph
    means for its payload."""
    rows = reader.query(run_ids_since_cypher(since_ms))
    return [i for i in (valid_run_id(r.get("run_id")) for r in rows) if i]


def read_run_nodes(reader: WorldviewReader, run_ids: list[str]) -> RunNodeRows:
    """Every node kind a run story is built from, for the given runs, plus
    the claim text of every prior those runs revised or hired about. Raises
    `WorldviewUnavailable`."""
    ids = [i for i in (valid_run_id(x) for x in run_ids) if i]
    if not ids:
        return RunNodeRows()
    roles = reader.query(run_nodes_cypher(LABEL_INVESTIGATION_ROLE, ids))
    hops = reader.query(run_nodes_cypher(LABEL_HOP, ids))
    findings = reader.query(run_nodes_cypher(LABEL_FINDING, ids))
    revisions = reader.query(run_nodes_cypher(LABEL_PRIOR_REVISION, ids))
    outcomes = reader.query(run_nodes_cypher(LABEL_TURN_OUTCOME, ids))
    self_writes = [
        {**r, "kind": "self_definition"} for r in reader.query(run_nodes_cypher(LABEL_SELF_DEFINITION, ids))
    ] + [
        {**r, "kind": "lived_answer"} for r in reader.query(run_nodes_cypher(LABEL_LIVED_ANSWER, ids))
    ]
    # Help/peer are best-effort: missing labels must not kill the core story.
    help_requests: list[dict[str, Any]] = []
    peer_briefs: list[dict[str, Any]] = []
    try:
        help_requests = list(reader.query(help_requests_with_prior_cypher(ids)) or [])
    except Exception:  # noqa: BLE001
        logger.debug("run_story_help_request_read_failed", exc_info=True)
    try:
        peer_briefs = list(reader.query(peer_briefs_for_runs_cypher(ids)) or [])
    except Exception:  # noqa: BLE001
        logger.debug("run_story_peer_brief_read_failed", exc_info=True)

    prior_ids = [str(r.get("prior_id") or "") for r in revisions]
    prior_ids.extend(str(h.get("prior_id") or "") for h in help_requests)
    prior_ids = [p for p in prior_ids if p]
    priors = reader.query(prior_claims_cypher(prior_ids)) if prior_ids else []
    return RunNodeRows(
        roles=list(roles), hops=list(hops), findings=list(findings),
        revisions=list(revisions), outcomes=list(outcomes), priors=list(priors),
        self_writes=self_writes,
        help_requests=help_requests,
        peer_briefs=peer_briefs,
    )


# Contractor PeerBriefs. Queried best-effort and separately from the core atlas
# reads: a missing :PeerBrief label (or any brief-query failure) must empty this
# list, never mark the whole atlas unavailable.
ATLAS_PEER_BRIEFS_CYPHER = (
    f"MATCH (b:{LABEL_PEER_BRIEF}) "
    "RETURN b.brief_id AS brief_id, b.help_id AS help_id, b.run_id AS run_id, "
    "b.peer AS peer, b.status AS status, b.summary AS summary, "
    "b.refusal_reason AS refusal_reason "
    "ORDER BY b.written_at DESC LIMIT 100"
)


def _stamp_ms(value: Any) -> Optional[int]:
    """Epoch milliseconds from what Orion actually wrote, ISO string included.

    The prompt asks for `written_at: timestamp()` and recent runs comply, but
    run `32b42392f495` wrote an ISO string and `_as_int` mapped that to 0 ->
    None. Two visible consequences, both wrong: the ledger labelled a run that
    HAD written a `:TurnOutcome` as "died before writing an outcome", and the
    undated run then counted as accounted-for on any day, masking the genuinely
    traceless run the banner exists to catch. Juniper spotted the first one in
    the rendered page.
    """
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _text(value: Any, limit: int = 4000) -> str:
    """Bounded, newline-preserving. Orion writes prose into these properties and
    a runaway value should shorten a panel, not a browser tab."""
    out = str(value or "").strip()
    return out[:limit]


@dataclass(frozen=True)
class AtlasPrior:
    prior_id: str
    claim: str
    confidence: Optional[float]
    status: str
    times_tested: int
    formed_from: str
    last_tested_at: str
    created_by_run: str
    last_run_id: str
    why: str

    @property
    def is_closed(self) -> bool:
        return self.status in CLOSED_STATUSES


@dataclass(frozen=True)
class AtlasPeerBrief:
    """One contractor PeerBrief for the operator page (ok or refused_budget)."""

    brief_id: str
    help_id: str
    run_id: str
    peer: str
    status: str
    summary: str
    refusal_reason: Optional[str] = None


@dataclass(frozen=True)
class AtlasRevision:
    """One recorded movement of a prior. Orion writes these; Hub only reads."""

    prior_id: str
    run_id: str
    from_confidence: Optional[float]
    to_confidence: Optional[float]
    from_status: str
    to_status: str
    written_at: Optional[int]

    @property
    def delta(self) -> Optional[float]:
        if self.from_confidence is None or self.to_confidence is None:
            return None
        return self.to_confidence - self.from_confidence


@dataclass(frozen=True)
class AtlasView:
    priors: list[AtlasPrior] = field(default_factory=list)
    revisions: list[AtlasRevision] = field(default_factory=list)
    peer_briefs: list[AtlasPeerBrief] = field(default_factory=list)
    concept_total: int = 0
    edge_total: int = 0
    unavailable_reason: Optional[str] = None
    _by_prior: Optional[dict[str, list["AtlasRevision"]]] = field(
        default=None, repr=False, compare=False
    )

    @property
    def is_unavailable(self) -> bool:
        return self.unavailable_reason is not None

    @property
    def live_total(self) -> int:
        return sum(1 for p in self.priors if not p.is_closed)

    @property
    def closed_total(self) -> int:
        return sum(1 for p in self.priors if p.is_closed)

    @property
    def revisions_by_prior(self) -> dict[str, list["AtlasRevision"]]:
        """Built once per view rather than rescanned per prior.

        `to_payload` calls `trajectory_for` for every prior, and a linear scan
        each time is O(priors x revisions) -- 10M iterations at this module's
        own query limits, on Hub's single uvicorn worker.
        """
        if self._by_prior is None:
            index: dict[str, list[AtlasRevision]] = {}
            for rev in self.revisions:
                index.setdefault(rev.prior_id, []).append(rev)
            object.__setattr__(self, "_by_prior", index)
        return self._by_prior  # type: ignore[return-value]

    @property
    def pool_is_dead(self) -> bool:
        """Every prior closed. Legal, and also the shape of the outage."""
        return self.live_total == 0 and self.closed_total > 0


def _build_prior(row: dict[str, Any]) -> Optional[AtlasPrior]:
    prior_id = _text(row.get("prior_id"), 200)
    if not prior_id:
        return None
    return AtlasPrior(
        prior_id=prior_id,
        claim=_text(row.get("claim")),
        confidence=_as_float(row.get("confidence")),
        status=_text(row.get("status"), 60),
        times_tested=_as_int(row.get("times_tested"), 0),
        formed_from=_text(row.get("formed_from"), 500),
        last_tested_at=_text(row.get("last_tested_at"), 60),
        created_by_run=_text(row.get("run_id"), 40),
        last_run_id=_text(row.get("last_run_id"), 40),
        why=_text(row.get("why")),
    )


def _build_revision(row: dict[str, Any]) -> Optional[AtlasRevision]:
    prior_id = _text(row.get("prior_id"), 200)
    run_id = _text(row.get("run_id"), 40)
    if not prior_id or not run_id:
        return None
    return AtlasRevision(
        prior_id=prior_id,
        run_id=run_id,
        from_confidence=_as_float(row.get("from_confidence")),
        to_confidence=_as_float(row.get("to_confidence")),
        from_status=_text(row.get("from_status"), 60),
        to_status=_text(row.get("to_status"), 60),
        written_at=_stamp_ms(row.get("written_at")),
    )


def _build_peer_brief(row: dict[str, Any]) -> Optional[AtlasPeerBrief]:
    brief_id = _text(row.get("brief_id"), 200)
    if not brief_id:
        return None
    refusal = row.get("refusal_reason")
    return AtlasPeerBrief(
        brief_id=brief_id,
        help_id=_text(row.get("help_id"), 200),
        run_id=_text(row.get("run_id"), 40),
        peer=_text(row.get("peer"), 60),
        status=_text(row.get("status"), 60),
        summary=_text(row.get("summary"), 800),
        refusal_reason=_text(refusal, 200) if refusal not in (None, "") else None,
    )


def _read_peer_briefs_best_effort(reader: WorldviewReader) -> list[AtlasPeerBrief]:
    """PeerBriefs are additive. Failure here must not blank the atlas."""
    try:
        rows = reader.query(ATLAS_PEER_BRIEFS_CYPHER)
    except WorldviewUnavailable as exc:
        logger.warning(
            "curiosity_atlas_peer_briefs_unavailable err=%s -- continuing without briefs",
            exc,
        )
        return []
    except Exception as exc:  # noqa: BLE001 — operator page must stay up
        logger.warning(
            "curiosity_atlas_peer_briefs_failed err=%s -- continuing without briefs",
            exc,
        )
        return []
    return [b for b in (_build_peer_brief(r) for r in rows) if b is not None]


def read_atlas(reader: WorldviewReader) -> AtlasView:
    """One read of everything the operator page shows. Never raises.

    Same contract as `worldview.read_snapshot`: an unreachable graph is reported
    as `unavailable_reason` and never as an empty view, so a broken ACL cannot
    render as "Orion has not thought anything yet".
    """
    try:
        prior_rows = reader.query(ATLAS_PRIORS_CYPHER)
        revision_rows = reader.query(ATLAS_REVISIONS_CYPHER)
        concept_rows = reader.query(ATLAS_UNUSED_CYPHER)
        edge_rows = reader.query(ATLAS_EDGES_CYPHER)
    except WorldviewUnavailable as exc:
        return AtlasView(unavailable_reason=str(exc)[:200])

    if len(prior_rows) >= ATLAS_PRIORS_LIMIT:
        logger.warning(
            "curiosity_atlas_priors_truncated limit=%s -- the page's pool counts "
            "are computed from the rows it read, so a truncated read makes them "
            "disagree with the loop's own server-side COUNTS_CYPHER",
            ATLAS_PRIORS_LIMIT,
        )
    priors = [p for p in (_build_prior(r) for r in prior_rows) if p is not None]
    revisions = [r for r in (_build_revision(x) for x in revision_rows) if r is not None]
    # A missing `written_at` is UNKNOWN, not oldest. Orion writes these by
    # hand; omit `timestamp()` once
    # and sorting it first would seed the trajectory's origin from its
    # `from_confidence` and draw that prior's whole chart backwards.
    revisions.sort(
        key=lambda r: (r.written_at is not None, r.written_at or 0, r.prior_id)
    )

    return AtlasView(
        concept_total=_as_int(concept_rows[0].get("n"), 0) if concept_rows else 0,
        edge_total=_as_int(edge_rows[0].get("n"), 0) if edge_rows else 0,
        priors=sorted(priors, key=lambda p: (p.is_closed, -p.times_tested, p.prior_id)),
        revisions=revisions,
        peer_briefs=_read_peer_briefs_best_effort(reader),
    )


def trajectory_for(view: AtlasView, prior_id: str) -> list[dict[str, Any]]:
    """Confidence over time for one prior, oldest first.

    The current value is appended as the last point ONLY when no revision
    already reports it, so a prior with recorded history does not get a
    duplicated endpoint and a prior with none still plots as a single dot at
    where it stands now. An empty list means no revision was ever recorded, not
    that confidence never moved -- the caller must say which.
    """
    points: list[dict[str, Any]] = []
    for rev in view.revisions_by_prior.get(prior_id, ()):
        if not points and rev.from_confidence is not None:
            points.append(
                {
                    "run_id": "",
                    "confidence": rev.from_confidence,
                    "status": rev.from_status,
                    "recorded": True,
                    # The origin has no clock of its own: it is the value
                    # BEFORE the first recorded revision. None, not the
                    # revision's stamp, so the sparkline's time axis does
                    # not draw a zero-length first segment.
                    "written_at": None,
                }
            )
        points.append(
            {
                "run_id": rev.run_id,
                "confidence": rev.to_confidence,
                "status": rev.to_status,
                "recorded": True,
                "written_at": rev.written_at,
            }
        )
    current = next((p for p in view.priors if p.prior_id == prior_id), None)
    if current is None:
        return points
    if not points or points[-1]["confidence"] != current.confidence:
        points.append(
            {
                "run_id": current.last_run_id or current.created_by_run,
                "confidence": current.confidence,
                "status": current.status,
                # ALWAYS False: this point is the prior's current state, never a
                # `:PriorRevision`. Tagging it True whenever some other revision
                # exists would invert the one distinction this flag carries.
                "recorded": False,
                "written_at": _stamp_ms(current.last_tested_at),
            }
        )
    return points


def to_payload(view: AtlasView) -> dict[str, Any]:
    """JSON for the page. Flat and boring on purpose — the template does no
    reshaping, so what the panel shows and what this module read cannot drift."""
    if view.is_unavailable:
        return {"available": False, "reason": view.unavailable_reason}
    return {
        "available": True,
        "live_total": view.live_total,
        "closed_total": view.closed_total,
        "pool_is_dead": view.pool_is_dead,
        "history_recorded": bool(view.revisions),
        "concept_total": view.concept_total,
        "edge_total": view.edge_total,
        "priors": [
            {
                **asdict(p),
                "is_closed": p.is_closed,
                # Orion writes `last_tested_at` by hand (ISO or ms); the
                # page sorts on this parsed epoch, not the raw text.
                "last_tested_at_ms": _stamp_ms(p.last_tested_at),
                "trajectory": trajectory_for(view, p.prior_id),
            }
            for p in view.priors
        ],
        "revisions": [{**asdict(r), "delta": r.delta} for r in view.revisions],
        "peer_briefs": [
            {
                "brief_id": b.brief_id,
                "help_id": b.help_id,
                "run_id": b.run_id,
                "peer": b.peer,
                "status": b.status,
                "summary": b.summary,
                "refusal_reason": b.refusal_reason,
            }
            for b in view.peer_briefs
        ],
    }
