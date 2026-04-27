from __future__ import annotations

from collections import defaultdict

from orion.schemas.world_pulse import ClaimRecordV1


def apply_corroboration(claims: list[ClaimRecordV1]) -> list[ClaimRecordV1]:
    index: dict[str, set[str]] = defaultdict(set)
    for claim in claims:
        key = claim.claim_text.lower().strip()
        index[key].update(claim.source_ids)
    out: list[ClaimRecordV1] = []
    for claim in claims:
        key = claim.claim_text.lower().strip()
        source_count = len(index[key])
        requires_corroboration = "requires_corroboration" in claim.caveats
        claim_extraction_disabled = "claim_extraction_disabled" in claim.caveats
        if claim_extraction_disabled:
            out.append(claim.model_copy(update={"promotion_status": "rejected"}))
        elif source_count >= 2 and claim.source_trust_tier <= 2:
            status = "candidate" if requires_corroboration else "accepted_working_claim"
            out.append(claim.model_copy(update={"corroboration_status": "corroborated", "promotion_status": status}))
        elif requires_corroboration:
            out.append(claim.model_copy(update={"corroboration_status": "insufficient_sources", "promotion_status": "candidate"}))
        else:
            out.append(claim)
    return out
