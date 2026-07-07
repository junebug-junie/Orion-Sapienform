"""Deterministic semantic lift for reverie: coalition pointers → ConcernCardV1."""

from __future__ import annotations

import logging
from typing import Literal

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
from orion.schemas.reverie import ConcernCardV1

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
        for ref in loop.source_refs:
            parse_harness_closure_ref(ref)
            _add(ref)
    for node_id in broadcast.attended_node_ids:
        _add(node_id)
    return refs


def resolve_concern_cards(
    broadcast: AttentionBroadcastProjectionV1,
    *,
    referent_loader: ReferentLoader,
) -> list[ConcernCardV1]:
    cards: list[ConcernCardV1] = []
    seen_corr: set[str] = set()

    for ref in _collect_coalition_refs(broadcast):
        corr = parse_harness_closure_ref(ref)
        if corr and corr in seen_corr:
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
        card = row.to_concern_card()
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
