from __future__ import annotations

import re
from dataclasses import dataclass, field

from orion.memory.crystallization.schemas import CrystallizationLinkV1, MemoryCrystallizationV1


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", _normalize_text(text))}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class DetectionResult:
    duplicates: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    suggested_links: list[CrystallizationLinkV1] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def detect_duplicates(
    candidate: MemoryCrystallizationV1,
    existing: list[MemoryCrystallizationV1],
    *,
    threshold: float = 0.72,
) -> DetectionResult:
    result = DetectionResult()
    cand_tokens = _token_set(f"{candidate.subject} {candidate.summary}")
    for other in existing:
        if other.crystallization_id == candidate.crystallization_id:
            continue
        if other.status in ("rejected", "archived", "quarantined"):
            continue
        other_tokens = _token_set(f"{other.subject} {other.summary}")
        score = _jaccard(cand_tokens, other_tokens)
        scope_overlap = bool(set(candidate.scope) & set(other.scope)) or (not candidate.scope or not other.scope)
        if score >= threshold and scope_overlap and candidate.kind == other.kind:
            result.duplicates.append(other.crystallization_id)
            result.suggested_links.append(
                CrystallizationLinkV1(
                    target_crystallization_id=other.crystallization_id,
                    relation="related_to",
                    confidence=round(score, 3),
                    note="duplicate_candidate",
                )
            )
            result.warnings.append(f"possible_duplicate:{other.crystallization_id}:score={score:.2f}")
    return result


def detect_contradictions(
    candidate: MemoryCrystallizationV1,
    existing: list[MemoryCrystallizationV1],
) -> DetectionResult:
    result = DetectionResult()
    if candidate.kind == "contradiction":
        return result

    neg_patterns = ("not ", "no ", "never ", "avoid ", "without ", "reject ")
    cand_norm = _normalize_text(candidate.summary)
    cand_negative = any(p in cand_norm for p in neg_patterns)

    for other in existing:
        if other.crystallization_id == candidate.crystallization_id:
            continue
        if other.status not in ("active", "proposed"):
            continue
        if not (set(candidate.scope) & set(other.scope) or not candidate.scope or not other.scope):
            continue
        other_norm = _normalize_text(other.summary)
        overlap = _jaccard(_token_set(cand_norm), _token_set(other_norm))
        other_negative = any(p in other_norm for p in neg_patterns)
        if overlap >= 0.45 and cand_negative != other_negative and candidate.kind == other.kind:
            result.contradictions.append(other.crystallization_id)
            result.suggested_links.append(
                CrystallizationLinkV1(
                    target_crystallization_id=other.crystallization_id,
                    relation="contradicts",
                    confidence=round(overlap, 3),
                    note="auto_contradiction_signal",
                )
            )
            result.warnings.append(f"possible_contradiction:{other.crystallization_id}")

    return result


def merge_detection(*results: DetectionResult) -> DetectionResult:
    merged = DetectionResult()
    seen_links: set[tuple[str, str]] = set()
    for r in results:
        merged.duplicates.extend(r.duplicates)
        merged.contradictions.extend(r.contradictions)
        merged.warnings.extend(r.warnings)
        for link in r.suggested_links:
            key = (link.target_crystallization_id, link.relation)
            if key not in seen_links:
                seen_links.add(key)
                merged.suggested_links.append(link)
    return merged
