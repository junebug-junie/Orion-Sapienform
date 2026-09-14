"""PeerBrief / HelpRequest helpers for kickoff nudge + dual-write.

Hub's WorldviewReader stays RO. MERGE Cypher here is for the system writer
(peer service or fixture persist path), never belief labels.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Sequence

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.curiosity_peer import (
    HELP_REQUEST_CHANNEL,
    HELP_REQUEST_KIND,
    HelpRequestV1,
    PeerBriefV1,
    clip,
)

logger = logging.getLogger("orion.curiosity.peer_briefs")

LABEL_HELP_REQUEST = "HelpRequest"
LABEL_PEER_BRIEF = "PeerBrief"

# Same run_id contract as orion.curiosity.worldview (uuid4().hex[:12]).
_RUN_ID_RE = re.compile(r"^[0-9a-f]{6,32}$")

_SELF_DEF_MARKERS = re.compile(
    r"(?is)(?:\bselfdefinition\b|\bself-definition\b|"
    r"MERGE\s*\(\s*s\s*:\s*SelfDefinition\b|"
    r"I am a (?:digital )?mind\b|"
    r"here is (?:a |the )?SelfDefinition\b)"
)


def peer_brief_merge_cypher(brief: PeerBriefV1) -> str:
    """MERGE PeerBrief + ANSWERS edge. Escapes single quotes in strings."""

    def q(value: object) -> str:
        return "'" + str(value or "").replace("\\", "\\\\").replace("'", "\\'") + "'"

    pointers = ",".join(q(p) for p in brief.evidence_pointers)
    opens = ",".join(q(p) for p in brief.open_questions)
    looks = ",".join(q(p) for p in brief.suggested_next_looks)
    prior = (
        f", b.prior_id = {q(brief.prior_id)}" if brief.prior_id else ""
    )
    refusal = (
        f", b.refusal_reason = {q(brief.refusal_reason)}"
        if brief.refusal_reason
        else ""
    )
    return (
        f"MERGE (b:{LABEL_PEER_BRIEF} {{brief_id: {q(brief.brief_id)}}}) "
        f"SET b.help_id = {q(brief.help_id)}, b.run_id = {q(brief.run_id)}, "
        f"b.peer = {q(brief.peer)}, b.status = {q(brief.status)}, "
        f"b.summary = {q(brief.summary)}, "
        f"b.evidence_pointers = [{pointers}], "
        f"b.open_questions = [{opens}], "
        f"b.suggested_next_looks = [{looks}], "
        f"b.written_at = timestamp(), b.consumed = false"
        f"{prior}{refusal} "
        f"WITH b "
        f"OPTIONAL MATCH (h:{LABEL_HELP_REQUEST} {{help_id: {q(brief.help_id)}}}) "
        f"FOREACH (_ IN CASE WHEN h IS NULL THEN [] ELSE [1] END | "
        f"MERGE (b)-[:ANSWERS]->(h))"
    )


def help_request_about_prior_cypher(help_id: str, prior_id: str) -> str:
    def q(value: object) -> str:
        return "'" + str(value or "").replace("\\", "\\\\").replace("'", "\\'") + "'"

    return (
        f"MATCH (h:{LABEL_HELP_REQUEST} {{help_id: {q(help_id)}}}), "
        f"(p:Prior {{prior_id: {q(prior_id)}}}) "
        f"MERGE (h)-[:ABOUT]->(p)"
    )


def list_help_requests_for_run_cypher(run_id: str) -> str:
    """RO Cypher: HelpRequests Orion wrote during one run (optional ABOUT prior)."""
    rid = str(run_id or "").strip()
    if not _RUN_ID_RE.match(rid):
        raise ValueError(f"refusing to build Cypher for a non-hex run_id: {run_id!r}")
    return (
        f"MATCH (h:{LABEL_HELP_REQUEST}) WHERE h.run_id = '{rid}' "
        "OPTIONAL MATCH (h)-[:ABOUT]->(p:Prior) "
        "RETURN h.help_id AS help_id, h.run_id AS run_id, "
        "coalesce(h.prior_id, p.prior_id) AS prior_id, "
        "h.mode AS mode, h.question AS question, "
        "h.tried_summary AS tried_summary, "
        "h.success_criteria AS success_criteria, "
        "h.written_at AS written_at "
        "ORDER BY h.written_at ASC"
    )


def list_help_requests_from_rows(rows: Sequence[dict[str, Any]]) -> list[HelpRequestV1]:
    out: list[HelpRequestV1] = []
    for row in rows:
        try:
            prior = row.get("prior_id")
            out.append(
                HelpRequestV1(
                    help_id=str(row.get("help_id") or ""),
                    run_id=str(row.get("run_id") or ""),
                    prior_id=str(prior) if prior else None,
                    mode=row.get("mode") or "world_curiosity",
                    question=str(row.get("question") or ""),
                    tried_summary=str(row.get("tried_summary") or ""),
                    success_criteria=str(row.get("success_criteria") or ""),
                )
            )
        except Exception:
            continue
    return out


async def publish_help_requests_for_run(
    *,
    enabled: bool,
    run_id: str,
    reader: Any,
    bus: Any,
    source_ref: Any = None,
) -> int:
    """RO-query HelpRequests for run_id and publish each on HELP_REQUEST_CHANNEL.

    Acceptance 1: flag off or zero HelpRequests → zero publishes.
    Returns the number of envelopes published.
    """
    if not enabled:
        return 0
    if reader is None or bus is None:
        return 0

    def _read() -> list[HelpRequestV1]:
        try:
            rows = reader.query(list_help_requests_for_run_cypher(run_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "curiosity_help_request_read_failed run=%s err=%s", run_id, exc
            )
            return []
        return list_help_requests_from_rows(rows or [])

    try:
        helps = await asyncio.to_thread(_read)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "curiosity_help_request_read_failed run=%s err=%s", run_id, exc
        )
        return 0

    published = 0
    for help_req in helps:
        try:
            await bus.publish(
                HELP_REQUEST_CHANNEL,
                BaseEnvelope(
                    kind=HELP_REQUEST_KIND,
                    source=source_ref or {"name": "orion-hub"},
                    payload=help_req.model_dump(mode="json"),
                ),
            )
            published += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "curiosity_help_request_publish_failed help_id=%s err=%s",
                help_req.help_id,
                exc,
            )
    if published:
        logger.info(
            "curiosity_help_requests_enqueued run=%s n=%s", run_id, published
        )
    return published


UNUSED_OK_BRIEFS_CYPHER = (
    f"MATCH (b:{LABEL_PEER_BRIEF}) "
    "WHERE coalesce(b.consumed, false) = false AND b.status = 'ok' "
    "RETURN b.brief_id AS brief_id, b.help_id AS help_id, b.run_id AS run_id, "
    "b.prior_id AS prior_id, b.peer AS peer, b.status AS status, "
    "b.summary AS summary, b.evidence_pointers AS evidence_pointers, "
    "b.open_questions AS open_questions, "
    "b.suggested_next_looks AS suggested_next_looks "
    "ORDER BY b.written_at DESC LIMIT 8"
)

REFUSED_OR_FAILED_RECENT_CYPHER = (
    f"MATCH (b:{LABEL_PEER_BRIEF}) "
    "WHERE coalesce(b.consumed, false) = false "
    "AND b.status IN ['refused_budget', 'failed'] "
    "RETURN b.brief_id AS brief_id, b.help_id AS help_id, b.run_id AS run_id, "
    "b.prior_id AS prior_id, b.peer AS peer, b.status AS status, "
    "b.summary AS summary, b.refusal_reason AS refusal_reason "
    "ORDER BY b.written_at DESC LIMIT 4"
)


def list_unused_ok_briefs_from_rows(rows: Sequence[dict[str, Any]]) -> list[PeerBriefV1]:
    out: list[PeerBriefV1] = []
    for row in rows:
        try:
            out.append(
                PeerBriefV1(
                    brief_id=str(row.get("brief_id") or ""),
                    help_id=str(row.get("help_id") or ""),
                    run_id=str(row.get("run_id") or ""),
                    prior_id=row.get("prior_id") or None,
                    peer=row.get("peer") or "cursor_auto",
                    status=row.get("status") or "ok",
                    summary=row.get("summary") or "",
                    evidence_pointers=list(row.get("evidence_pointers") or []),
                    open_questions=list(row.get("open_questions") or []),
                    suggested_next_looks=list(row.get("suggested_next_looks") or []),
                    refusal_reason=row.get("refusal_reason"),
                )
            )
        except Exception:
            continue
    return out


def format_soft_nudge(briefs: Sequence[PeerBriefV1]) -> list[str]:
    """Invitational soft-nudge lines for kickoff. Never 'you must incorporate'."""
    ok = [b for b in briefs if b.status == "ok" and (b.summary or b.evidence_pointers)]
    refused = [b for b in briefs if b.status == "refused_budget"]
    failed = [b for b in briefs if b.status == "failed"]
    lines: list[str] = []
    if ok:
        lines += [
            "PEER LOOKED (optional). A contractor peer left notes on something you "
            "asked about. You decide whether any of it matters — cite, contradict, "
            "extend, or leave unused. Nothing here is required.",
            "",
        ]
        for b in ok:
            lines.append(f"  brief {b.brief_id} (peer={b.peer}): {clip(b.summary, 400)}")
            for ptr in b.evidence_pointers[:6]:
                lines.append(f"    evidence: {ptr}")
            for look in b.suggested_next_looks[:4]:
                lines.append(f"    maybe look: {look}")
            lines.append("")
        lines += [
            "Your move. Writing :Prior / :Finding / :SelfDefinition remains yours alone.",
            "",
        ]
    if refused or failed:
        lines += [
            "COULD NOT HIRE. A HelpRequest was opened but the peer did not return "
            "usable help "
            f"(refused_budget={len(refused)}, failed={len(failed)}). "
            "Continue alone; do not invent peer evidence.",
            "",
        ]
    return lines


def strip_self_definition_draft(text: str) -> tuple[str, bool]:
    """Remove identity-drafting prose from peer output in self_inquiry mode."""
    if not text:
        return "", False
    if not _SELF_DEF_MARKERS.search(text):
        return text, False
    cleaned_lines: list[str] = []
    stripped = False
    for line in text.splitlines():
        if _SELF_DEF_MARKERS.search(line):
            stripped = True
            continue
        if "MERGE (s:SelfDefinition" in line or "MERGE (s: SelfDefinition" in line:
            stripped = True
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned, stripped or cleaned != text.strip()
