"""PeerBrief / HelpRequest helpers for kickoff nudge + dual-write.

Hub's WorldviewReader stays RO. MERGE Cypher here is for the system writer
(peer service or fixture persist path), never belief labels.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Sequence

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.curiosity.worldview import read_hop_notes
from orion.schemas.curiosity_peer import (
    HELP_REQUEST_CHANNEL,
    HELP_REQUEST_KIND,
    MAX_SUMMARY_CHARS,
    PEER_BRIEF_CONSUMED_CHANNEL,
    PEER_BRIEF_CONSUMED_KIND,
    HelpRequestV1,
    PeerAskExpectationV1,
    PeerBriefConsumedV1,
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


def peer_brief_merge_cypher(brief: PeerBriefV1) -> tuple[str, dict[str, Any]]:
    """MERGE PeerBrief + ANSWERS edge. Parameterized (no string concat of prose)."""
    params: dict[str, Any] = {
        "brief_id": brief.brief_id,
        "help_id": brief.help_id,
        "run_id": brief.run_id,
        "peer": brief.peer,
        "status": brief.status,
        "summary": brief.summary or "",
        "evidence_pointers": list(brief.evidence_pointers or []),
        "open_questions": list(brief.open_questions or []),
        "suggested_next_looks": list(brief.suggested_next_looks or []),
        "prior_id": brief.prior_id,
        "refusal_reason": clip(brief.refusal_reason, MAX_SUMMARY_CHARS)
        if brief.refusal_reason
        else None,
    }
    cypher = (
        f"MERGE (b:{LABEL_PEER_BRIEF} {{brief_id: $brief_id}}) "
        "ON CREATE SET b.consumed = false, b.written_at = timestamp() "
        "SET b.help_id = $help_id, b.run_id = $run_id, "
        "b.peer = $peer, b.status = $status, "
        "b.summary = $summary, "
        "b.evidence_pointers = $evidence_pointers, "
        "b.open_questions = $open_questions, "
        "b.suggested_next_looks = $suggested_next_looks, "
        "b.prior_id = $prior_id, b.refusal_reason = $refusal_reason "
        "WITH b "
        f"OPTIONAL MATCH (h:{LABEL_HELP_REQUEST} {{help_id: $help_id}}) "
        "FOREACH (_ IN CASE WHEN h IS NULL THEN [] ELSE [1] END | "
        "MERGE (b)-[:ANSWERS]->(h)) "
        "WITH b OPTIONAL MATCH (c:PeerAskCommit {help_id:$help_id,run_id:$run_id}) "
        "FOREACH (_ IN CASE WHEN c IS NULL THEN [] ELSE [1] END | "
        "MERGE (c)-[:RETURNED]->(b) "
        "SET c.responded_at=coalesce(c.responded_at,timestamp()))"
    )
    return cypher, params


def peer_brief_consume_cypher(brief_ids: Sequence[str]) -> tuple[str, dict[str, Any]]:
    """Mark PeerBrief nodes consumed so UNUSED_* queries skip them."""
    ids = [str(b).strip() for b in brief_ids if str(b or "").strip()]
    cypher = (
        "UNWIND $brief_ids AS bid "
        f"MATCH (b:{LABEL_PEER_BRIEF} {{brief_id: bid}}) "
        "SET b.consumed = true"
    )
    return cypher, {"brief_ids": ids}


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
        "h.written_at AS written_at, h.expected_reply AS expected_reply, "
        "h.if_not_asked AS if_not_asked, h.alternatives_json AS alternatives_json, "
        "h.within_seconds AS within_seconds "
        "ORDER BY h.written_at ASC"
    )


def list_help_requests_from_rows(rows: Sequence[dict[str, Any]]) -> list[HelpRequestV1]:
    out: list[HelpRequestV1] = []
    for row in rows:
        try:
            prior = row.get("prior_id")
            expectation = None
            if any(row.get(k) is not None for k in ("expected_reply", "if_not_asked", "alternatives_json", "within_seconds")):
                expectation = PeerAskExpectationV1(
                    expected_reply=row.get("expected_reply"),
                    if_not_asked=row.get("if_not_asked"),
                    alternatives=json.loads(row.get("alternatives_json") or "null"),
                    within_seconds=row.get("within_seconds"),
                )
            timestamp_fields = {"written_at": row["written_at"]} if row.get("written_at") is not None else {}
            out.append(
                HelpRequestV1(
                    help_id=str(row.get("help_id") or ""),
                    run_id=str(row.get("run_id") or ""),
                    prior_id=str(prior) if prior else None,
                    mode=row.get("mode") or "world_curiosity",
                    question=str(row.get("question") or ""),
                    tried_summary=str(row.get("tried_summary") or ""),
                    success_criteria=str(row.get("success_criteria") or ""),
                    expectation=expectation,
                    **timestamp_fields,
                )
            )
        except Exception:
            logger.warning("curiosity_help_request_invalid help_id=%s", row.get("help_id"))
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

    try:
        hops = await asyncio.to_thread(read_hop_notes, reader, run_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "curiosity_hop_notes_read_failed run=%s err=%s", run_id, exc
        )
        hops = []
    hop_count = len(hops)

    published = 0
    for help_req in helps:
        if not (help_req.tried_summary or "").strip() and hop_count > 0:
            logger.warning(
                "curiosity_help_request_skipped_empty_tried_summary help_id=%s run=%s hops=%s",
                help_req.help_id,
                run_id,
                hop_count,
            )
            continue
        try:
            await bus.publish(
                HELP_REQUEST_CHANNEL,
                BaseEnvelope(
                    kind=HELP_REQUEST_KIND,
                    source=source_ref or {"name": "orion-hub"},
                    payload=help_req.model_dump(mode="json", exclude_none=True),
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


def brief_ids_for_consume(briefs: Sequence[PeerBriefV1]) -> list[str]:
    """Ids that soft-nudge would surface — mark consumed so they do not reappear."""
    out: list[str] = []
    for b in briefs:
        if not b.brief_id:
            continue
        if b.status == "ok" and (b.summary or b.evidence_pointers):
            out.append(b.brief_id)
        elif b.status in ("refused_budget", "failed"):
            out.append(b.brief_id)
    return out


async def publish_peer_briefs_consumed(
    *,
    bus: Any,
    brief_ids: Sequence[str],
    source_ref: Any = None,
    consumer_run_id: str | None = None,
    phase: str = "offered",
) -> int:
    """Hub (RO worldview) asks peer service to MERGE consumed=true."""
    ids = [str(b).strip() for b in brief_ids if str(b or "").strip()]
    if (not ids and phase != "completed") or bus is None:
        return 0
    try:
        await bus.publish(
            PEER_BRIEF_CONSUMED_CHANNEL,
            BaseEnvelope(
                kind=PEER_BRIEF_CONSUMED_KIND,
                source=source_ref or {"name": "orion-hub"},
                payload=PeerBriefConsumedV1(brief_ids=ids, consumer_run_id=consumer_run_id, phase=phase).model_dump(mode="json", exclude_none=True, exclude={"phase"} if phase == "offered" else set()),
            ),
        )
        return len(ids)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "curiosity_peer_brief_consumed_publish_failed n=%s err=%s", len(ids), exc
        )
        return 0


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


def format_soft_nudge(briefs: Sequence[PeerBriefV1], *, consumer_run_id: str | None = None) -> list[str]:
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
            if b.peer == "claude_room":
                # Room companion runs with --tools "" — conversation notes only.
                lines.append(
                    f"  brief {b.brief_id} (peer=claude_room, conversation-only): "
                    f"{clip(b.summary, 400)}"
                )
                lines.append(
                    "    note: this peer could not look at the repo; treat as "
                    "conversation notes, not file evidence."
                )
                for look in b.suggested_next_looks[:4]:
                    lines.append(f"    maybe look: {look}")
                lines.append("")
                continue
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
        for b in refused + failed:
            lines.append(f"  brief {b.brief_id} (peer={b.peer}, status={b.status}): no usable peer help returned.")
    if consumer_run_id and brief_ids_for_consume(briefs):
        from orion.curiosity.agency_episode import decision_prompt

        lines += decision_prompt(consumer_run_id)
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
