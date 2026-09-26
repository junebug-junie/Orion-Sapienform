"""Durable peer-ask commitment and source-checked later decision receipts.

The peer service owns commit/offer records. Orion authors decisions. A
validated decision is an attributed self-report, not experimental causality.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orion.schemas.curiosity_peer import HelpRequestV1, PeerBriefConsumedV1


def commit_ask(client, request: HelpRequestV1) -> bool:
    """Claim once, before any external invocation. Ambiguous crashes never resend.

    The graph acknowledgment is required. A duplicate or changed request under
    the same help_id cannot claim again; an operator must inspect uncertainty.
    """
    if request.expectation is None:
        raise ValueError("peer ask requires an explicit expectation")
    encoded = request.model_dump_json()
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    token = uuid.uuid4().hex
    rows = client.graph_query(
        "MATCH (h:HelpRequest {help_id:$help_id}) WHERE h.run_id = $run_id "
        "MERGE (c:PeerAskCommit {help_id:$help_id}) ON CREATE SET "
        "c.run_id=$run_id, c.request_digest=$digest, c.commit_token=$token, "
        "c.request_json=$request_json, c.committed_at=timestamp(), "
        "c.deadline_at=timestamp()+$within_ms "
        "MERGE (c)-[:COMMITS]->(h) "
        "RETURN c.commit_token AS commit_token, c.request_digest AS request_digest",
        {"help_id": request.help_id, "run_id": request.run_id, "digest": digest,
         "token": token, "request_json": encoded,
         "within_ms": request.expectation.within_seconds * 1000},
    )
    if len(rows) != 1:
        raise RuntimeError("peer ask commitment not acknowledged uniquely")
    if rows[0]["request_digest"] != digest:
        raise ValueError("help_id was already committed with different content")
    return rows[0]["commit_token"] == token


def offer_or_complete_cypher(receipt: PeerBriefConsumedV1) -> tuple[str, dict]:
    """Keep offering, completion and actual use distinct, including on replay."""
    if receipt.consumer_run_id is None:
        raise ValueError("run identity required for an episode receipt")
    params = {"run_id": receipt.consumer_run_id, "brief_ids": receipt.brief_ids}
    if receipt.phase == "completed":
        return (
            "MATCH (o:PeerBriefOffer {run_id:$run_id}) "
            "SET o.completed_at=coalesce(o.completed_at,timestamp()) RETURN o.brief_id AS brief_id",
            params,
        )
    return (
        "UNWIND $brief_ids AS bid MATCH (b:PeerBrief {brief_id:bid}) "
        "MERGE (o:PeerBriefOffer {run_id:$run_id,brief_id:bid}) "
        "ON CREATE SET o.help_id=b.help_id,o.offered_at=timestamp() "
        "MERGE (o)-[:OFFERS]->(b) SET b.consumed=true RETURN b.brief_id AS brief_id",
        params,
    )


def decision_query(run_id: str | None = None, *, help_ids: list[str] | None = None) -> str:
    if (run_id is None) == (help_ids is None):
        raise ValueError("select one run or a bounded help-id set")
    if run_id is not None:
        if not re.fullmatch(r"[0-9a-f]{6,32}", run_id):
            raise ValueError("invalid consumer run_id")
        where = "o.run_id='" + run_id + "'"
    else:
        if len(help_ids) > 50:
            raise ValueError("too many help ids")
        where = "b.help_id IN " + json.dumps(help_ids)
    return (
        "MATCH (o:PeerBriefOffer)-[:OFFERS]->(b:PeerBrief) WHERE " + where + " "
        "MATCH (d:PeerBriefDecision {run_id:o.run_id,brief_id:b.brief_id}) "
        "OPTIONAL MATCH (h:Hop {run_id:o.run_id,n:d.hop_n}) "
        "RETURN o.run_id AS run_id,b.brief_id AS brief_id,b.help_id AS help_id, "
        "o.offered_at AS offered_at,o.completed_at AS completed_at, "
        "d.disposition AS disposition,d.reason AS reason,d.decision AS decision, "
        "d.hop_n AS hop_n,d.written_at AS written_at,h.written_at AS hop_written_at, "
        "h.note AS hop_note ORDER BY b.brief_id LIMIT 33"
    )


def validate_decision(row: dict) -> dict:
    """Fail closed on missing or unordered evidence; never fabricate a decision."""
    reason = None
    if row.get("disposition") not in {"used", "not_used"}:
        reason = "invalid_disposition"
    elif not all(str(row.get(k) or "").strip() for k in ("run_id", "brief_id", "help_id", "reason", "decision")):
        reason = "empty_decision"
    else:
        try:
            offered, written, completed = (int(row[k]) for k in ("offered_at", "written_at", "completed_at"))
            if not 0 < offered <= written <= completed:
                reason = "unordered_or_incomplete_run"
            if row["disposition"] == "used":
                if not str(row.get("hop_note") or "").strip() or int(row["hop_n"]) < 1 or not offered <= int(row["hop_written_at"]) <= written:
                    reason = "missing_or_unordered_hop"
        except (TypeError, ValueError, KeyError):
            reason = "missing_timestamps_or_hop"
    digest = hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()
    return {"run_id": row.get("run_id"), "brief_id": row.get("brief_id"),
            "help_id": row.get("help_id"), "status": "unverified" if reason else "attributed_self_report",
            "disposition": row.get("disposition"), "hop_n": row.get("hop_n"),
            "reason_unverified": reason, "evidence_digest": digest}


def decision_prompt(run_id: str) -> list[str]:
    if not re.fullmatch(r"[0-9a-f]{6,32}", run_id):
        return []
    return [
        "PEER ANSWER FOLLOW-THROUGH. After your work, record a decision for each offered brief. "
        "A note in your prompt is not evidence that you used it. If it affected a real "
        "investigation step, use disposition='used' and cite that step's existing Hop.n. "
        "Otherwise use 'not_used' and explain why; do not invent a step to claim learning.",
        "Before choosing the next step, compare the reply with your pre-ask expectation: "
        "MATCH (c:PeerAskCommit)-[:RETURNED]->(b:PeerBrief {brief_id:'<offered brief id>'}) "
        "RETURN c.request_json. If no commitment exists, say the expectation is unknown. "
        "In your reason, say what the reply confirmed, contradicted or left unresolved, "
        "and how that affected your choice. Do not infer rejection from a late or missing reply.",
        "MERGE (d:PeerBriefDecision {run_id: '" + run_id + "', brief_id: '<offered brief id>'})",
        "ON CREATE SET d.disposition='<used|not_used>', d.reason='<why>', "
        "d.decision='<the choice you actually made>', d.hop_n=<existing Hop.n, or null when not_used>, "
        "d.written_at=timestamp()",
        "This records your account of using the answer, not proof that it caused the decision.",
        "",
    ]
