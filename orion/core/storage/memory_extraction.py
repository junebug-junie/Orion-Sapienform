from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List

from orion.core.contracts.memory_cards import MemoryCardV1


@dataclass
class CandidateCard:
    """Lightweight Stage-1 extraction candidate (not yet persisted)."""

    summary: str
    types: tuple[str, ...]
    anchor_class: str | None = None


_PLACE = re.compile(
    r"\b(i live in|i'm from|i am from|based in|located in)\s+([A-Z][a-zA-Z\s,]+)$",
    re.I | re.M,
)


def extract_candidates(turn_text: str, *, speaker: str = "user") -> List[CandidateCard]:
    if speaker.lower() != "user":
        return []
    out: List[CandidateCard] = []
    for m in _PLACE.finditer(turn_text or ""):
        place = (m.group(2) or "").strip().strip(".")
        if len(place) < 2:
            continue
        out.append(CandidateCard(summary=f"Lives in {place}", types=("anchor", "place"), anchor_class="place"))
    return out


def fingerprint(card: MemoryCardV1) -> str:
    norm = " ".join((card.summary or "").lower().split())
    anchor = (card.anchor_class or "").lower()
    h = hashlib.sha256(f"{anchor}|{norm}".encode("utf-8")).hexdigest()
    return h
