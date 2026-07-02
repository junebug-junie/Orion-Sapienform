"""Episodic-continuity adapter — the remembered past enters beliefs.

Rung-4 consumer: maps the latest ``EpisodeSummaryV1`` (a proposal-marked
rollup of one time-window of reduction receipts — "what happened to me")
into a single belief node so the present self can read its own recent
history. The node stays proposal-marked: episodes inform stance, they never
silently become accepted truth (Knowledge Forge rule).

ctx-sourced, pure (no network, no DB): reads ``ctx['episode_summary']`` as an
``EpisodeSummaryV1``, dict, or JSON string, and degrades to ``None`` when
absent or unparseable — never raises.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)
from orion.core.schemas.substrate_episodes import EPISODE_RECEIPT_CAP, EpisodeSummaryV1
from orion.substrate.adapters._common import make_temporal

logger = logging.getLogger("orion.substrate.relational.adapters.episodes_ctx")

_TIER_RANK = 4  # snapshot_ephemeral: derived autobiographical rollup, proposal-marked
_ORGAN_COUNT_CAP = 5
_NOTE_CAP = 4


def _make_prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="episode_summary",
        source_channel="substrate.episodes",
        producer="episodes_adapter",
        tier_rank=_TIER_RANK,
    )


def _coerce(raw: Any) -> EpisodeSummaryV1 | None:
    try:
        if isinstance(raw, EpisodeSummaryV1):
            return raw
        if isinstance(raw, str) and raw.strip():
            return EpisodeSummaryV1.model_validate_json(raw)
        if isinstance(raw, dict):
            return EpisodeSummaryV1.model_validate(raw)
    except Exception as exc:
        logger.debug("episodes_adapter_parse_failed error=%s", exc)
    return None


def map_episode_ctx_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
    """Map ``ctx['episode_summary']`` → one proposal-marked ``episode:latest`` node."""
    ctx = ctx if isinstance(ctx, dict) else {}
    episode = _coerce(ctx.get("episode_summary"))
    if episode is None:
        return None

    top_organs = dict(
        sorted(episode.organ_counts.items(), key=lambda kv: kv[1], reverse=True)[
            :_ORGAN_COUNT_CAP
        ]
    )
    # Busier windows are more salient; EPISODE_RECEIPT_CAP receipts saturate.
    salience = max(0.0, min(1.0, episode.receipt_count_total / float(EPISODE_RECEIPT_CAP)))
    node = ConceptNodeV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        label="episode:latest",
        temporal=make_temporal(observed_at=datetime.now(timezone.utc)),
        provenance=_make_prov(),
        signals=SubstrateSignalBundleV1(confidence=0.5, salience=salience),
        metadata={
            "source_kind": "episode_summary",
            "status": episode.status,  # stays 'proposal' — never accepted truth
            "episode_id": episode.episode_id,
            "window_start": episode.window_start.isoformat(),
            "window_end": episode.window_end.isoformat(),
            "receipt_count_total": episode.receipt_count_total,
            "organ_counts": top_organs,
            "warning_count": episode.warning_count,
            "notes": list(episode.notes[:_NOTE_CAP]),
        },
    )
    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=[node])
