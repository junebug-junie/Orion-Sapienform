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
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from orion.curiosity.worldview import (
    CLOSED_STATUSES,
    LABEL_FINDING,
    LABEL_HOP,
    LABEL_PRIOR,
    LABEL_TURN_OUTCOME,
    WorldviewReader,
    WorldviewUnavailable,
    _as_bool,
    _as_float,
    _as_int,
)

logger = logging.getLogger("orion.curiosity.atlas")

LABEL_PRIOR_REVISION = "PriorRevision"

# --- Cypher. Fully static: this module takes no caller input into a query. ---

ATLAS_PRIORS_CYPHER = (
    f"MATCH (p:{LABEL_PRIOR}) RETURN p.prior_id AS prior_id, p.claim AS claim, "
    "p.confidence AS confidence, p.status AS status, "
    "p.times_tested AS times_tested, p.formed_from AS formed_from, "
    "p.last_tested_at AS last_tested_at, p.run_id AS run_id, "
    "p.last_run_id AS last_run_id, p.why AS why LIMIT 2000"
)

ATLAS_REVISIONS_CYPHER = (
    f"MATCH (r:{LABEL_PRIOR_REVISION}) RETURN r.prior_id AS prior_id, "
    "r.run_id AS run_id, r.from_confidence AS from_confidence, "
    "r.to_confidence AS to_confidence, r.from_status AS from_status, "
    "r.to_status AS to_status, r.written_at AS written_at LIMIT 5000"
)

ATLAS_FINDINGS_CYPHER = (
    f"MATCH (f:{LABEL_FINDING}) RETURN f.finding_id AS finding_id, "
    "f.text AS text, f.evidence AS evidence, f.run_id AS run_id, "
    "f.written_at AS written_at LIMIT 2000"
)

ATLAS_HOPS_CYPHER = (
    f"MATCH (h:{LABEL_HOP}) RETURN h.run_id AS run_id, h.n AS n, "
    "h.note AS note LIMIT 5000"
)

ATLAS_OUTCOMES_CYPHER = (
    f"MATCH (t:{LABEL_TURN_OUTCOME}) RETURN t.run_id AS run_id, "
    "t.continue_line AS continue_line, t.continue_note AS continue_note, "
    "t.reach_out AS reach_out, t.reach_out_why AS reach_out_why, "
    "t.written_at AS written_at LIMIT 2000"
)

# Every node, by label and run, so graph growth is counted from the same source
# the other reads use rather than by summing them (a label nobody has a reader
# for yet would silently vanish from the totals).
ATLAS_GROWTH_CYPHER = (
    "MATCH (n) WHERE n.run_id IS NOT NULL "
    "RETURN labels(n)[0] AS label, n.run_id AS run_id, count(n) AS n"
)


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
class AtlasRun:
    """One turn, assembled from every node that carries its run_id."""

    run_id: str
    written_at: Optional[int] = None
    hops: int = 0
    hop_notes: list[dict[str, Any]] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    added: dict[str, int] = field(default_factory=dict)
    priors_created: list[str] = field(default_factory=list)
    priors_touched: list[str] = field(default_factory=list)
    continue_line: bool = False
    continue_note: str = ""
    reach_out: bool = False
    reach_out_why: str = ""

    @property
    def total_added(self) -> int:
        return sum(self.added.values())


@dataclass(frozen=True)
class AtlasView:
    priors: list[AtlasPrior] = field(default_factory=list)
    revisions: list[AtlasRevision] = field(default_factory=list)
    runs: list[AtlasRun] = field(default_factory=list)
    unavailable_reason: Optional[str] = None

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
        written_at=_as_int(row.get("written_at"), 0) or None,
    )


def assemble_runs(
    *,
    growth_rows: list[dict[str, Any]],
    outcome_rows: list[dict[str, Any]],
    hop_rows: list[dict[str, Any]],
    finding_rows: list[dict[str, Any]],
    priors: list[AtlasPrior],
) -> list[AtlasRun]:
    """One row per run, newest first, from every node carrying that run_id.

    A run appears here if ANY node carries its id — not only if it wrote a
    `:TurnOutcome`. A turn killed mid-write (which happened on 2026-08-27 at
    06:17 when both containers were recreated) leaves hops and no outcome, and
    showing nothing for it would hide exactly the runs worth looking at.
    """
    runs: dict[str, dict[str, Any]] = {}

    def _slot(run_id: str) -> dict[str, Any]:
        return runs.setdefault(
            run_id,
            {
                "added": {},
                "hop_notes": [],
                "findings": [],
                "priors_created": [],
                "priors_touched": [],
                "written_at": None,
                "continue_line": False,
                "continue_note": "",
                "reach_out": False,
                "reach_out_why": "",
            },
        )

    for row in growth_rows:
        run_id = _text(row.get("run_id"), 40)
        label = _text(row.get("label"), 60)
        if not run_id or not label:
            continue
        _slot(run_id)["added"][label] = _as_int(row.get("n"), 0)

    for row in hop_rows:
        run_id = _text(row.get("run_id"), 40)
        if not run_id:
            continue
        _slot(run_id)["hop_notes"].append(
            {"n": _as_int(row.get("n"), 0), "note": _text(row.get("note"), 2000)}
        )

    for row in finding_rows:
        run_id = _text(row.get("run_id"), 40)
        if not run_id:
            continue
        _slot(run_id)["findings"].append(
            {
                "finding_id": _text(row.get("finding_id"), 200),
                "text": _text(row.get("text")),
                "evidence": _text(row.get("evidence")),
            }
        )

    for row in outcome_rows:
        run_id = _text(row.get("run_id"), 40)
        if not run_id:
            continue
        slot = _slot(run_id)
        slot["continue_line"] = _as_bool(row.get("continue_line"))
        slot["continue_note"] = _text(row.get("continue_note"), 2000)
        slot["reach_out"] = _as_bool(row.get("reach_out"))
        slot["reach_out_why"] = _text(row.get("reach_out_why"), 2000)
        slot["written_at"] = _as_int(row.get("written_at"), 0) or None

    for prior in priors:
        if prior.created_by_run:
            _slot(prior.created_by_run)["priors_created"].append(prior.prior_id)
        if prior.last_run_id and prior.last_run_id != prior.created_by_run:
            _slot(prior.last_run_id)["priors_touched"].append(prior.prior_id)

    built = [
        AtlasRun(
            run_id=run_id,
            written_at=slot["written_at"],
            hops=len(slot["hop_notes"]),
            hop_notes=sorted(slot["hop_notes"], key=lambda h: h["n"]),
            findings=slot["findings"],
            added=slot["added"],
            priors_created=sorted(slot["priors_created"]),
            priors_touched=sorted(slot["priors_touched"]),
            continue_line=slot["continue_line"],
            continue_note=slot["continue_note"],
            reach_out=slot["reach_out"],
            reach_out_why=slot["reach_out_why"],
        )
        for run_id, slot in runs.items()
    ]
    # `written_at` is absent on a run killed before it wrote its outcome, and
    # those sort last rather than first -- a missing timestamp is not "oldest".
    built.sort(key=lambda r: (r.written_at is not None, r.written_at or 0), reverse=True)
    return built


def read_atlas(reader: WorldviewReader) -> AtlasView:
    """One read of everything the operator page shows. Never raises.

    Same contract as `worldview.read_snapshot`: an unreachable graph is reported
    as `unavailable_reason` and never as an empty view, so a broken ACL cannot
    render as "Orion has not thought anything yet".
    """
    try:
        prior_rows = reader.query(ATLAS_PRIORS_CYPHER)
        revision_rows = reader.query(ATLAS_REVISIONS_CYPHER)
        finding_rows = reader.query(ATLAS_FINDINGS_CYPHER)
        hop_rows = reader.query(ATLAS_HOPS_CYPHER)
        outcome_rows = reader.query(ATLAS_OUTCOMES_CYPHER)
        growth_rows = reader.query(ATLAS_GROWTH_CYPHER)
    except WorldviewUnavailable as exc:
        return AtlasView(unavailable_reason=str(exc)[:200])

    priors = [p for p in (_build_prior(r) for r in prior_rows) if p is not None]
    revisions = [r for r in (_build_revision(x) for x in revision_rows) if r is not None]
    revisions.sort(key=lambda r: (r.written_at or 0, r.prior_id))

    return AtlasView(
        priors=sorted(priors, key=lambda p: (p.is_closed, -p.times_tested, p.prior_id)),
        revisions=revisions,
        runs=assemble_runs(
            growth_rows=growth_rows,
            outcome_rows=outcome_rows,
            hop_rows=hop_rows,
            finding_rows=finding_rows,
            priors=priors,
        ),
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
    for rev in view.revisions:
        if rev.prior_id != prior_id:
            continue
        if not points and rev.from_confidence is not None:
            points.append(
                {
                    "run_id": "",
                    "confidence": rev.from_confidence,
                    "status": rev.from_status,
                    "recorded": True,
                }
            )
        points.append(
            {
                "run_id": rev.run_id,
                "confidence": rev.to_confidence,
                "status": rev.to_status,
                "recorded": True,
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
                "recorded": bool(points),
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
        "priors": [
            {
                **asdict(p),
                "is_closed": p.is_closed,
                "trajectory": trajectory_for(view, p.prior_id),
            }
            for p in view.priors
        ],
        "runs": [{**asdict(r), "total_added": r.total_added} for r in view.runs],
        "revisions": [{**asdict(r), "delta": r.delta} for r in view.revisions],
    }
