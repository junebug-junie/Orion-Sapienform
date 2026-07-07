"""Deterministic semantic lift for reverie: coalition pointers → ConcernCardV1."""

from __future__ import annotations

import logging
import re
from typing import Literal

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
from orion.schemas.reverie import ConcernCardV1, SpontaneousThoughtV1

from .referent_loader import ReferentLoader, parse_harness_closure_ref

logger = logging.getLogger("orion.reverie.semantic_lift")

MAX_CARDS_PER_TICK = 3
MIN_CARD_HUMAN_TEXT = 40


def _collect_coalition_refs(broadcast: AttentionBroadcastProjectionV1) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()

    def _add(ref: str) -> None:
        r = (ref or "").strip()
        if r and r not in seen:
            seen.add(r)
            refs.append(r)

    if broadcast.selected_open_loop_id:
        loop = next(
            (l for l in broadcast.frame.open_loops if l.id == broadcast.selected_open_loop_id),
            None,
        )
        if loop is not None:
            for ref in loop.source_refs:
                _add(ref)
    for loop in broadcast.frame.open_loops:
        _add(loop.id)
        for ref in loop.source_refs:
            _add(ref)
    for node_id in broadcast.attended_node_ids:
        _add(node_id)
    return refs


def coalition_audit_refs(broadcast: AttentionBroadcastProjectionV1) -> list[str]:
    """Coalition ids the LLM may cite in evidence_refs (audit-only, not for narration)."""
    return _collect_coalition_refs(broadcast)


def resolve_concern_cards(
    broadcast: AttentionBroadcastProjectionV1,
    *,
    referent_loader: ReferentLoader,
) -> list[ConcernCardV1]:
    cards: list[ConcernCardV1] = []
    seen_corr: set[str] = set()

    for ref in _collect_coalition_refs(broadcast):
        corr = parse_harness_closure_ref(ref)
        if not corr:
            continue
        if corr in seen_corr:
            continue
        try:
            row = referent_loader.load_by_coalition_ref(ref)
        except Exception as exc:
            logger.warning("referent_loader_error ref=%s err=%s", ref, exc)
            continue
        if row is None:
            continue
        if corr:
            seen_corr.add(corr)
        try:
            card = row.to_concern_card()
        except Exception as exc:
            logger.warning("concern_card_build_failed ref=%s err=%s", ref, exc)
            continue
        if card is not None:
            cards.append(card)
        if len(cards) >= MAX_CARDS_PER_TICK:
            break
    return cards


def reverie_semantic_gate(cards: list[ConcernCardV1]) -> Literal["proceed", "skip"]:
    if not cards:
        return "skip"
    if any(len(c.human_text.strip()) >= MIN_CARD_HUMAN_TEXT for c in cards):
        return "proceed"
    return "skip"


# Structural mechanism terms only — not user-content keyword cathedral.
_INFRA_VOCAB_RE = re.compile(
    r"\b("
    r"coalition|open\s+loop|harness_closure|substrate|stability\s+score|"
    r"dwell\s+tick|attended\s+node|projection_id"
    r")\b",
    re.IGNORECASE,
)

_CONTENT_TOKEN_RE = re.compile(r"[a-z0-9]{4,}", re.IGNORECASE)


def infra_vocabulary_hit(text: str) -> bool:
    return bool(_INFRA_VOCAB_RE.search(text or ""))


def _content_tokens(text: str) -> set[str]:
    return set(_CONTENT_TOKEN_RE.findall((text or "").lower()))


def referent_overlap(interpretation: str, cards: list[ConcernCardV1]) -> bool:
    interp_tokens = _content_tokens(interpretation)
    if not interp_tokens:
        return False
    for card in cards:
        card_tokens = _content_tokens(card.human_text)
        if not card_tokens:
            continue
        shared = interp_tokens & card_tokens
        if len(shared) >= 2:
            return True
        if len(shared) / max(len(card_tokens), 1) >= 0.20:
            return True
    return False


def enforce_semantic_quality(
    thought: SpontaneousThoughtV1,
    cards: list[ConcernCardV1],
) -> SpontaneousThoughtV1:
    if infra_vocabulary_hit(thought.interpretation):
        return thought.model_copy(update={"hollow": True, "hollow_reason": "infra_vocabulary"})
    if not referent_overlap(thought.interpretation, cards):
        return thought.model_copy(
            update={"hollow": True, "hollow_reason": "unanchored_no_referent_overlap"}
        )
    return thought.marked_hollow()
