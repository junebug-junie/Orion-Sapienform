from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from orion.core.schemas.substrate_review_queue import GraphReviewQueueItemV1, GraphReviewQueueSnapshotV1


class GraphReviewQueue:
    def __init__(self, *, max_items: int = 200) -> None:
        self._max_items = max_items
        self._items: dict[str, GraphReviewQueueItemV1] = {}

    def upsert(self, item: GraphReviewQueueItemV1) -> None:
        region_key = self._region_key(item.focal_node_refs, item.target_zone)
        existing_id = None
        for item_id, existing in self._items.items():
            if self._region_key(existing.focal_node_refs, existing.target_zone) == region_key:
                existing_id = item_id
                break
        if existing_id is not None:
            prev = self._items[existing_id]
            item = item.model_copy(
                update={
                    "queue_item_id": existing_id,
                    "cycle_budget": prev.cycle_budget,
                    "suppression_state": prev.suppression_state,
                    "termination_state": prev.termination_state,
                }
            )
            self._items[existing_id] = item
            return

        if len(self._items) >= self._max_items:
            evict_id = sorted(self._items.items(), key=lambda kv: (kv[1].priority, kv[1].created_at))[0][0]
            del self._items[evict_id]
        self._items[item.queue_item_id] = item

    def mark_reviewed(self, queue_item_id: str, *, reviewed_at: datetime | None = None) -> GraphReviewQueueItemV1 | None:
        item = self._items.get(queue_item_id)
        if item is None:
            return None
        now = reviewed_at or datetime.now(timezone.utc)
        next_cycle_count = item.cycle_budget.cycle_count + 1
        updated_budget = item.cycle_budget.model_copy(
            update={
                "cycle_count": next_cycle_count,
                "remaining_cycles": max(0, item.cycle_budget.max_cycles - next_cycle_count),
            }
        )
        updated = item.model_copy(
            update={
                "cycle_budget": updated_budget,
                "last_review_at": now,
                "termination_state": item.termination_state or next_cycle_count >= item.cycle_budget.max_cycles,
            }
        )
        self._items[queue_item_id] = updated
        return updated

    def apply_cycle_feedback(self, queue_item_id: str, *, no_change: bool) -> GraphReviewQueueItemV1 | None:
        item = self._items.get(queue_item_id)
        if item is None:
            return None
        next_no_change = item.cycle_budget.no_change_cycles + 1 if no_change else 0
        updated_budget = item.cycle_budget.model_copy(update={"no_change_cycles": next_no_change})
        updated = item.model_copy(
            update={
                "cycle_budget": updated_budget,
                "suppression_state": item.suppression_state
                or next_no_change >= item.cycle_budget.suppress_after_low_value_cycles,
                "termination_state": item.termination_state or item.cycle_budget.remaining_cycles <= 0,
            }
        )
        self._items[queue_item_id] = updated
        return updated

    def list_eligible(self, *, now: datetime | None = None, limit: int = 10) -> list[GraphReviewQueueItemV1]:
        t = now or datetime.now(timezone.utc)
        eligible = [
            item
            for item in self._items.values()
            if not item.termination_state
            and not item.suppression_state
            and item.next_review_at <= t
            and item.cycle_budget.remaining_cycles > 0
        ]
        eligible.sort(key=lambda item: (-item.priority, item.next_review_at, item.created_at))
        return eligible[:limit]

    def snapshot(self, *, limit: int = 200) -> GraphReviewQueueSnapshotV1:
        items = sorted(self._items.values(), key=lambda item: (-item.priority, item.next_review_at, item.created_at))
        truncated = len(items) > limit
        items = items[:limit]
        by_zone = Counter(item.target_zone for item in items)
        by_state = Counter(
            "terminated" if item.termination_state else "suppressed" if item.suppression_state else "active"
            for item in items
        )
        return GraphReviewQueueSnapshotV1(
            queue_items=items,
            counts_by_zone=dict(by_zone),
            counts_by_state=dict(by_state),
            top_priorities=[item.priority for item in items[:10]],
            truncated=truncated,
        )

    @staticmethod
    def _region_key(node_refs: list[str], zone: str) -> str:
        return f"{zone}|{','.join(sorted(node_refs))}"
