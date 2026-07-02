"""Curiosity-signal adapter — frontier gaps enter beliefs.

Maps frontier curiosity signals into a single belief node so the unified belief
set contains a belief about *unresolved gaps* that Orion has identified. This adapter
consumes the top-3 curiosity signals by strength and aggregates them into one node.

ctx-sourced, pure (no network, no DB): reads ``ctx['curiosity_signals']`` as a list of
``FrontierInvocationSignalV1``, dicts, or JSON strings, and degrades to ``None`` when
absent or unparseable — never raises.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1
from orion.substrate.adapters._common import make_temporal

logger = logging.getLogger("orion.substrate.relational.adapters.curiosity_ctx")

_TIER_RANK = 4  # snapshot_ephemeral: derived curiosity state, refreshed every tick
_TOP_SIGNALS = 3  # Take top 3 signals by strength
_FOCAL_REFS_PER_SIGNAL = 2  # Up to 2 focal refs per signal
_TOTAL_FOCAL_REFS_CAP = 6  # 6 total cap (3 signals × 2 refs each)


def _make_prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="curiosity",
        source_channel="substrate.curiosity",
        producer="curiosity_adapter",
        tier_rank=_TIER_RANK,
    )


def _coerce(raw: Any) -> list[FrontierInvocationSignalV1] | None:
    """Coerce raw input to list of FrontierInvocationSignalV1.

    Accepts:
    - list of FrontierInvocationSignalV1 instances
    - list of dicts
    - list of JSON strings
    - single JSON string representation of a list

    Returns None on any parse failure. Never raises.
    """
    try:
        # Handle None, empty, or non-dict/non-list ctx
        if raw is None or (isinstance(raw, list) and len(raw) == 0):
            return None

        # If it's a string, try to parse as JSON
        if isinstance(raw, str):
            if not raw.strip():
                return None
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    raw = parsed
                else:
                    return None
            except (json.JSONDecodeError, TypeError):
                return None

        # Now raw should be a list
        if not isinstance(raw, list):
            return None

        if len(raw) == 0:
            return None

        # Convert each item to FrontierInvocationSignalV1
        signals: list[FrontierInvocationSignalV1] = []
        for item in raw:
            try:
                if isinstance(item, FrontierInvocationSignalV1):
                    signals.append(item)
                elif isinstance(item, str) and item.strip():
                    sig = FrontierInvocationSignalV1.model_validate_json(item)
                    signals.append(sig)
                elif isinstance(item, dict):
                    sig = FrontierInvocationSignalV1.model_validate(item)
                    signals.append(sig)
            except Exception as exc:
                logger.debug("curiosity_adapter_coerce_item_failed error=%s", exc)
                continue

        return signals if signals else None
    except Exception as exc:
        logger.debug("curiosity_adapter_coerce_failed error=%s", exc)
        return None


def map_curiosity_ctx_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
    """Map ``ctx['curiosity_signals']`` → one ``curiosity:unresolved_gaps`` node.

    Reads curiosity signals from ctx, coerces to list, sorts by signal_strength
    descending, takes top 3, and emits one node aggregating focal refs and
    evidence summaries.
    """
    ctx = ctx if isinstance(ctx, dict) else {}
    raw_signals = ctx.get("curiosity_signals")

    signals = _coerce(raw_signals)
    if signals is None or len(signals) == 0:
        return None

    # Sort by signal_strength descending, take top 3
    sorted_signals = sorted(signals, key=lambda s: s.signal_strength, reverse=True)[
        :_TOP_SIGNALS
    ]

    if not sorted_signals:
        return None

    # Extract focal node refs (up to 2 per signal, 6 total cap)
    focal_node_refs: list[str] = []
    for signal in sorted_signals:
        refs = signal.focal_node_refs[: _FOCAL_REFS_PER_SIGNAL]
        focal_node_refs.extend(refs)
        if len(focal_node_refs) >= _TOTAL_FOCAL_REFS_CAP:
            focal_node_refs = focal_node_refs[:_TOTAL_FOCAL_REFS_CAP]
            break

    # Extract evidence summaries
    evidence_summaries = [s.evidence_summary for s in sorted_signals if s.evidence_summary]

    # Calculate average confidence
    confidences = [s.confidence for s in sorted_signals]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
    avg_confidence = max(0.0, min(1.0, avg_confidence))

    # Extract signal types
    signal_types = [s.signal_type for s in sorted_signals]

    # Emit the node
    node = ConceptNodeV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        label="curiosity:unresolved_gaps",
        temporal=make_temporal(observed_at=datetime.now(timezone.utc)),
        provenance=_make_prov(),
        signals=SubstrateSignalBundleV1(confidence=avg_confidence, salience=0.4),
        metadata={
            "gap_count": len(sorted_signals),
            "focal_node_refs": focal_node_refs,
            "evidence_summaries": evidence_summaries,
            "aggregate_confidence": avg_confidence,
            "signal_types": signal_types,
        },
    )

    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=[node])
