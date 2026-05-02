from __future__ import annotations

import re
from datetime import datetime

from .types import ClaimResolutionRecord, EpisodeRecord, TurnBlockRecord

CLAIM_RE = re.compile(
    r"(the issue is [^.]+\.|this means [^.]+\.|the fix is [^.]+\.|do [^.]+\.)",
    re.IGNORECASE,
)
CONFIRM_RE = re.compile(r"\b(yup|that worked|confirmed|fixed|resolved)\b", re.IGNORECASE)
REJECT_RE = re.compile(r"\b(still broken|didn't work|didnt work|no)\b", re.IGNORECASE)


def mine_claims_and_resolutions(
    episodes: list[EpisodeRecord],
    blocks: list[TurnBlockRecord],
) -> list[ClaimResolutionRecord]:
    block_by_turn = {item.turn_id: item for item in blocks}
    out: list[ClaimResolutionRecord] = []
    claim_idx = 0
    for episode in episodes:
        episode_blocks = [block_by_turn[turn_id] for turn_id in episode.turn_ids if turn_id in block_by_turn]
        for block_idx, block in enumerate(episode_blocks):
            claims = _candidate_claims(block.assistant_answer_block)
            if not claims:
                continue
            status, evidence_ids = _resolve_status(episode_blocks[block_idx:], episode.turn_ids[block_idx:])
            for claim in claims:
                claim_idx += 1
                out.append(
                    ClaimResolutionRecord(
                        claim_id=f"claim-{claim_idx:06d}",
                        episode_id=episode.episode_id,
                        claim_text=claim,
                        status=status,
                        resolution_text=_resolution_text(status, claim),
                        evidence_turn_ids=evidence_ids,
                        status_reason=f"Derived from {len(evidence_ids)} evidence turns",
                        last_status_at=_last_status_time(episode_blocks[block_idx:]),
                    )
                )
    return out


def _candidate_claims(answer: str) -> list[str]:
    return [m.group(0).strip() for m in CLAIM_RE.finditer(answer)][:4]


def _resolve_status(blocks: list[TurnBlockRecord], turn_ids: list[str]) -> tuple[str, list[str]]:
    evidence: list[str] = []
    status = "candidate"
    for block, turn_id in zip(blocks, turn_ids):
        text = f"{block.user_problem_block}\n{block.assistant_answer_block}\n{block.log_or_error_block}"
        if CONFIRM_RE.search(text):
            status = "confirmed"
            evidence.append(turn_id)
            break
        if REJECT_RE.search(text):
            status = "rejected"
            evidence.append(turn_id)
            break
        if "error" in text.lower():
            status = "unresolved"
            evidence.append(turn_id)
    if status == "candidate" and evidence:
        status = "accepted"
    return status, evidence


def _resolution_text(status: str, claim_text: str) -> str:
    if status == "confirmed":
        return f"Confirmed: {claim_text}"
    if status == "rejected":
        return f"Rejected by later evidence: {claim_text}"
    if status == "unresolved":
        return f"Still unresolved: {claim_text}"
    if status == "accepted":
        return f"Accepted pending explicit confirmation: {claim_text}"
    return ""


def _last_status_time(blocks: list[TurnBlockRecord]) -> str:
    if not blocks:
        return datetime.now().replace(microsecond=0).isoformat()
    return blocks[-1].created_at
