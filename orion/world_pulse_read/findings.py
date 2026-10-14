"""What each completed read changed: accepted links and moved beliefs, from stored records only.

Shared by the Orion's Day letter (orion/orion_day/gather.py, "What I read") and Hub's reading
outreach (services/orion-hub/scripts/reading_finding_outreach.py). Read-only: Postgres SELECTs
plus one parameterised ``GRAPH.RO_QUERY`` through a caller-supplied lookup.

link_made: a ``relationship_claims[*]`` entry in ``world_pulse_read_seed.stage2_result_json``
whose receipt is ``accepted_provisional`` AND whose latest ``substrate_graph_journal`` decision is
still accepted (a link Juniper rejected later is not reported). ``projected`` says whether the
AssertionProjector applied that decision.

belief_moved: readings never write priors (PR #2199). A curiosity run does, and records what it
did in ``curiosity_run_outcomes.per_prior`` (before/after confidence). The move counts for a read
only when the text that same run wrote on the prior names the read: ``why`` when the prior's
``last_run_id`` is the run, ``formed_from`` when its ``run_id`` is. The match is string evidence
(the read's URL path, seed id or request id appears verbatim), the same standard as the letter's
claim check; a run that read something and moved an unrelated prior is not credited.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence

from orion.schemas.reading_finding import ReadingFindingV1
from orion.substrate.neighborhood import ACCEPTED_ASSERTION_STATES

logger = logging.getLogger(__name__)

STATEMENT_CHARS = 300
CLAIM_CHARS = 200
EVIDENCE_CHARS = 400
MIN_URL_TOKEN_CHARS = 12

# Accepted claims of the given reads, each with its assertion's latest decision state and
# whether that decision was applied to the graph. Older rows without claims return nothing.
READ_LINKS_SQL = """
SELECT s.seed_id, s.stage2_completed_at, c.claim,
       d.state AS latest_state, m.applied_at AS projected_at
FROM world_pulse_read_seed s
CROSS JOIN LATERAL jsonb_array_elements(
    CASE WHEN jsonb_typeof(s.stage2_result_json->'relationship_claims') = 'array'
         THEN s.stage2_result_json->'relationship_claims' ELSE '[]'::jsonb END) AS c(claim)
LEFT JOIN LATERAL (
    SELECT j.decision_id, j.payload->>'resulting_state' AS state
    FROM substrate_graph_journal j
    WHERE j.event_kind = 'decision' AND j.target_id = c.claim->'receipt'->>'assertion_id'
    ORDER BY j.revision DESC NULLS LAST, j.recorded_at DESC
    LIMIT 1
) d ON true
LEFT JOIN LATERAL (
    SELECT max(j.recorded_at) AS applied_at
    FROM substrate_graph_journal j
    WHERE j.event_kind = 'materialization' AND j.outcome = 'applied'
      AND j.target_id = c.claim->'receipt'->>'assertion_id' AND j.decision_id = d.decision_id
) m ON true
WHERE s.seed_id = ANY($1::text[])
  AND c.claim->'receipt'->>'outcome' = 'accepted_provisional'
ORDER BY s.seed_id, c.claim->'receipt'->>'assertion_id'
"""

# Curiosity runs that finished in [$1, $2) and recorded per-prior outcomes.
RUN_PRIOR_MOVES_SQL = """
SELECT run_id, completed_at, per_prior
FROM curiosity_run_outcomes
WHERE completed_at >= $1 AND completed_at < $2
  AND jsonb_typeof(per_prior) = 'array' AND jsonb_array_length(per_prior) > 0
ORDER BY completed_at, run_id
"""


def prior_texts_cypher(prior_ids: Iterable[str]) -> str:
    """The provenance text of the named priors. Ids are Orion-authored free text, so they are a
    JSON parameter list, never spliced (orion/curiosity/atlas.py ``prior_claims_cypher``)."""
    clean = sorted({str(p)[:200] for p in prior_ids if p})
    if len(clean) > 200:
        logger.warning("reading_findings_prior_ids_truncated asked=%d kept=200", len(clean))
        clean = clean[:200]
    return (
        f"CYPHER ids={json.dumps(clean)} MATCH (p:Prior) WHERE p.prior_id IN $ids "
        "RETURN p.prior_id AS prior_id, p.claim AS claim, p.why AS why, p.formed_from AS formed_from, "
        "p.run_id AS run_id, p.last_run_id AS last_run_id, p.status AS status, p.confidence AS confidence"
    )


def read_prior_texts(reader: Any, prior_ids: Sequence[str]) -> Optional[dict[str, list[dict[str, Any]]]]:
    """prior_id -> every stored copy (a forked prior is two nodes). None when unreadable.
    Synchronous (``WorldviewReader.query``); callers use ``asyncio.to_thread``."""
    if not prior_ids:
        return {}
    if reader is None:
        return None
    try:
        rows = reader.query(prior_texts_cypher(prior_ids))
    except Exception as exc:  # noqa: BLE001 -- WorldviewUnavailable or a client error
        logger.warning("reading_findings_priors_unreadable err=%s", str(exc)[:200])
        return None
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        pid = str(row.get("prior_id") or "").strip()
        if pid:
            out.setdefault(pid, []).append(dict(row))
    return out


@dataclass(frozen=True)
class ReadRefV1:
    """One completed read, as the finders need it."""

    seed_id: str
    occurred_at: datetime
    url: str = ""
    request_id: str = ""
    title: str = ""
    learned: str = ""
    article_summary: str = ""

    def tokens(self) -> list[str]:
        """Strings whose verbatim presence in a prior's text names this read."""
        out = [t for t in (self.seed_id, self.request_id) if t and len(t) >= MIN_URL_TOKEN_CHARS]
        url = normalize_url_text(self.url).rstrip("/")
        # A bare domain ("coreweave.com") names a company, not this page.
        if "/" in url and len(url) >= MIN_URL_TOKEN_CHARS:
            out.append(url)
        return out


def normalize_url_text(text: str) -> str:
    return re.sub(r"https?://|www\.", "", str(text or "").lower())


def _names(token: str, hay: str) -> bool:
    """``token`` appears in ``hay`` as a whole: not as the prefix of a longer path or id
    ("nasa.gov/maxpower" does not name "nasa.gov/maxpower-x" or "nasa.gov/maxpower/sub")."""
    return re.search(re.escape(token) + r"/?(?![a-z0-9_\-/])", hay) is not None


def _obj(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return raw


def _aware(value: Any) -> Optional[datetime]:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _clip(text: Any, cap: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= cap else s[: cap - 1].rstrip() + "…"


def _float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN is not a confidence


# --- link_made -------------------------------------------------------------------------------


def link_findings(rows: Iterable[Mapping[str, Any]]) -> list[ReadingFindingV1]:
    """One finding per accepted assertion still accepted in the journal. Pure."""
    out: dict[tuple[str, str], ReadingFindingV1] = {}
    for row in rows:
        claim = _obj(row.get("claim"))
        if not isinstance(claim, dict):
            continue
        receipt = claim.get("receipt") or {}
        assertion_id = str(receipt.get("assertion_id") or "").strip()
        state = row.get("latest_state")
        if not assertion_id or state not in ACCEPTED_ASSERTION_STATES:
            # No journal decision, or a later decision (Juniper's review) un-accepted it.
            continue
        projected_at = _aware(row.get("projected_at"))
        # When the read accepted it, so a link projected after midnight still belongs to its day.
        observed = _aware(row.get("stage2_completed_at")) or projected_at or datetime.now(timezone.utc)
        statement = _clip(claim.get("statement_text"), STATEMENT_CHARS)
        seed_id = str(row["seed_id"])
        refs = [f"reading:{seed_id}", f"assertion:{assertion_id}"]
        if receipt.get("decision_id"):
            refs.append(f"decision:{receipt['decision_id']}")
        if receipt.get("content_sha256") and receipt.get("span_start") is not None:
            refs.append(f"reading_text:{receipt['content_sha256']}#bytes={receipt['span_start']}-{receipt.get('span_end')}")
        out.setdefault((seed_id, assertion_id), ReadingFindingV1(
            finding_id=f"link:{assertion_id}", kind="link_made", seed_id=seed_id, observed_at=observed,
            headline=f"Link made: {statement}" + ("" if projected_at else " (accepted, not yet in the graph)"),
            evidence_text=_clip(claim.get("quote"), EVIDENCE_CHARS), evidence_refs=refs,
            assertion_id=assertion_id, projected=projected_at is not None,
        ))
    return list(out.values())


# --- belief_moved ----------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorMoveV1:
    run_id: str
    completed_at: datetime
    prior_id: str
    kind: str  # per_prior.kind: "tested" | "formed"
    before: Optional[float]
    after: Optional[float]


def prior_moves(rows: Iterable[Mapping[str, Any]]) -> list[PriorMoveV1]:
    """Per-prior entries whose confidence changed (a formed prior counts: none -> after). Pure."""
    out: list[PriorMoveV1] = []
    for row in rows:
        completed = _aware(row.get("completed_at"))
        per_prior = _obj(row.get("per_prior"))
        if completed is None or not isinstance(per_prior, list):
            continue
        for entry in per_prior:
            if not isinstance(entry, dict) or not entry.get("prior_id"):
                continue
            before, after = _float(entry.get("before")), _float(entry.get("after"))
            if after is None or before == after:
                continue
            out.append(PriorMoveV1(run_id=str(row["run_id"]), completed_at=completed,
                                   prior_id=str(entry["prior_id"]), kind=str(entry.get("kind") or ""),
                                   before=before, after=after))
    return out


def _run_texts(move: PriorMoveV1, copies: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, Mapping[str, Any]]]:
    """(field, text, copy) this run itself wrote on the prior."""
    out = []
    for copy in copies:
        if str(copy.get("last_run_id") or "") == move.run_id and copy.get("why"):
            out.append(("why", str(copy["why"]), copy))
        if str(copy.get("run_id") or "") == move.run_id and copy.get("formed_from"):
            out.append(("formed_from", str(copy["formed_from"]), copy))
    return out


def belief_findings(
    reads: Sequence[ReadRefV1],
    moves: Sequence[PriorMoveV1],
    prior_texts: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[ReadingFindingV1]:
    """A move is credited to every read its run's own text names, if the read came first. Pure."""
    out: dict[tuple[str, str], ReadingFindingV1] = {}
    for move in moves:
        texts = _run_texts(move, prior_texts.get(move.prior_id) or ())
        if not texts:
            continue
        for read in reads:
            if read.occurred_at > move.completed_at:
                continue
            tokens = read.tokens()
            for field_name, text, copy in texts:
                hay = normalize_url_text(text)
                hit = next((t for t in tokens if _names(normalize_url_text(t), hay)), None)
                if hit is None:
                    continue
                claim = _clip(copy.get("claim"), CLAIM_CHARS) or move.prior_id
                change = (f"formed at {move.after:.2f}" if move.before is None
                          else f"{move.before:.2f} -> {move.after:.2f}")
                key = (read.seed_id, f"belief:{move.prior_id}@{move.run_id}")
                out.setdefault(key, ReadingFindingV1(
                    finding_id=key[1], kind="belief_moved", seed_id=read.seed_id,
                    observed_at=move.completed_at,
                    headline=f"Belief moved ({change}): {claim}",
                    evidence_text=_clip(text, EVIDENCE_CHARS),
                    evidence_refs=[f"reading:{read.seed_id}", f"curiosity:{move.run_id}", f"prior:{move.prior_id}",
                                   f"cites:{field_name}={hit}"],
                    prior_id=move.prior_id, run_id=move.run_id,
                    confidence_before=move.before, confidence_after=move.after,
                ))
                break
    return list(out.values())


PriorLookup = Callable[[list[str]], Awaitable[Optional[dict[str, list[dict[str, Any]]]]]]


def prior_lookup_from_reader(reader_provider: Optional[Callable[[], Any]]) -> PriorLookup:
    """A ``PriorLookup`` over a WorldviewReader provider (None reader -> unreadable)."""
    import asyncio

    async def lookup(prior_ids: list[str]) -> Optional[dict[str, list[dict[str, Any]]]]:
        reader = reader_provider() if reader_provider is not None else None
        return await asyncio.to_thread(read_prior_texts, reader, prior_ids)

    return lookup


@dataclass
class FindingsResultV1:
    by_seed: dict[str, list[ReadingFindingV1]]
    # Why belief_moved could not be checked ("" when it was): "no_prior_lookup" | "priors_unreadable"
    beliefs_unchecked: str = ""


async def findings_for_reads(
    conn: Any,
    reads: Sequence[ReadRefV1],
    *,
    until: datetime,
    prior_lookup: Optional[PriorLookup] = None,
) -> FindingsResultV1:
    """Every finding of ``reads`` recorded before ``until``, grouped by seed id, oldest first.
    Raises on a Postgres error (callers record the source as failed); a graph that cannot be
    read leaves belief_moved unchecked and says so."""
    result = FindingsResultV1(by_seed={r.seed_id: [] for r in reads})
    if not reads:
        return result
    findings: list[ReadingFindingV1] = []
    rows = await conn.fetch(READ_LINKS_SQL, [r.seed_id for r in reads])
    findings += [f for f in link_findings(rows) if f.observed_at < until]
    since = min(r.occurred_at for r in reads)
    moves = prior_moves(await conn.fetch(RUN_PRIOR_MOVES_SQL, since, until))
    if moves:
        if prior_lookup is None:
            result.beliefs_unchecked = "no_prior_lookup"
        else:
            texts = await prior_lookup(sorted({m.prior_id for m in moves}))
            if texts is None:
                result.beliefs_unchecked = "priors_unreadable"
            else:
                findings += belief_findings(reads, moves, texts)
    for f in sorted(findings, key=lambda f: (f.observed_at, f.finding_id)):
        result.by_seed.setdefault(f.seed_id, []).append(f)
    return result


def first_sentences(text: str, n: int = 2, cap: int = 360) -> str:
    """The first ``n`` sentences of stored text, verbatim (whitespace squashed), capped."""
    s = " ".join(str(text or "").split())
    if not s:
        return ""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(“])", s)
    return _clip(" ".join(parts[:n]), cap)
