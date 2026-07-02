"""Episodic continuity — rung 4 of the self-modeling loop.

`GraphConsolidationEvaluator` consolidates graph *regions*; this evaluator
consolidates *time*. It rolls a bounded window of reduction receipts into one
proposal-marked ``EpisodeSummaryV1`` so the substrate has a remembered
"what happened to me, and why" instead of only a durable audit log.

Discipline (Knowledge Forge):
- output is proposal-marked, never a mutation of accepted truth;
- episode ids derive deterministically from inputs, so replaying the same
  window is idempotent;
- receipts per episode are hard-capped.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Sequence

from orion.core.schemas.substrate_episodes import EPISODE_RECEIPT_CAP, EpisodeSummaryV1
from orion.schemas.reduction_receipt import ReductionReceiptV1

DEFAULT_WINDOW_SECONDS = 3600
_MAX_SAMPLE_WARNINGS = 8


def _utc(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def derive_episode_id(
    *,
    receipt_ids: Sequence[str],
    window_start: datetime,
    window_end: datetime,
) -> str:
    digest = hashlib.sha256()
    digest.update(_utc(window_start).isoformat().encode("utf-8"))
    digest.update(b"|")
    digest.update(_utc(window_end).isoformat().encode("utf-8"))
    for receipt_id in sorted(receipt_ids):
        digest.update(b"|")
        digest.update(receipt_id.encode("utf-8"))
    return f"episode:{digest.hexdigest()[:24]}"


class EpisodicConsolidationEvaluator:
    """Windowed rollup alongside the region-based GraphConsolidationEvaluator."""

    def __init__(
        self,
        *,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        max_receipts_per_episode: int = EPISODE_RECEIPT_CAP,
    ) -> None:
        if window_seconds < 1:
            raise ValueError("window_seconds must be >= 1")
        self._window_seconds = int(window_seconds)
        self._max_receipts = max(1, min(int(max_receipts_per_episode), EPISODE_RECEIPT_CAP))

    def consolidate(
        self,
        *,
        receipts: Sequence[ReductionReceiptV1],
        window_end: datetime | None = None,
        now: datetime | None = None,
    ) -> EpisodeSummaryV1 | None:
        """Roll the receipts inside the window into one episode, or None if empty.

        ``window_end`` defaults to the newest receipt timestamp so the same
        receipt set always yields the same window (and the same episode id).
        ``created_at`` defaults to ``window_end`` for the same reason; pass
        ``now`` when a wall-clock stamp is wanted.
        """
        if not receipts:
            return None

        end = _utc(window_end) if window_end is not None else max(_utc(r.created_at) for r in receipts)
        start = end - timedelta(seconds=self._window_seconds)

        in_window = sorted(
            (r for r in receipts if start <= _utc(r.created_at) <= end),
            key=lambda r: (_utc(r.created_at), r.receipt_id),
        )
        if not in_window:
            return None

        total = len(in_window)
        kept = in_window[: self._max_receipts]

        organ_counts: Counter[str] = Counter()
        reducer_counts: Counter[str] = Counter()
        accepted = rejected = merged = noop = deltas = projections = warnings = 0
        sample_warnings: list[str] = []
        for receipt in kept:
            if receipt.organ_id:
                organ_counts[receipt.organ_id] += 1
            for delta in receipt.state_deltas:
                if delta.reducer_id:
                    reducer_counts[delta.reducer_id] += 1
            accepted += len(receipt.accepted_event_ids)
            rejected += len(receipt.rejected_event_ids)
            merged += len(receipt.merged_event_ids)
            noop += len(receipt.noop_event_ids)
            deltas += len(receipt.state_deltas)
            projections += len(receipt.projection_updates)
            warnings += len(receipt.warnings)
            for warning in receipt.warnings:
                if len(sample_warnings) < _MAX_SAMPLE_WARNINGS:
                    sample_warnings.append(warning)

        receipt_ids = [r.receipt_id for r in kept]
        return EpisodeSummaryV1(
            episode_id=derive_episode_id(receipt_ids=receipt_ids, window_start=start, window_end=end),
            window_start=start,
            window_end=end,
            window_seconds=self._window_seconds,
            receipt_refs=receipt_ids,
            receipt_count_total=total,
            receipt_count_capped=total > len(kept),
            organ_counts=dict(organ_counts),
            reducer_counts=dict(reducer_counts),
            accepted_event_count=accepted,
            rejected_event_count=rejected,
            merged_event_count=merged,
            noop_event_count=noop,
            state_delta_count=deltas,
            projection_update_count=projections,
            warning_count=warnings,
            sample_warnings=sample_warnings,
            notes=["episodic_consolidation_v1", f"cap:{self._max_receipts}"],
            created_at=_utc(now) if now is not None else end,
        )
