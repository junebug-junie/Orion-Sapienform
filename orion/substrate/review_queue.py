from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import sqlite3

from orion.core.schemas.substrate_review_queue import GraphReviewQueueItemV1, GraphReviewQueueSnapshotV1


class GraphReviewQueue:
    def __init__(
        self,
        *,
        max_items: int = 200,
        sql_db_path: str | None = None,
        postgres_url: str | None = None,
    ) -> None:
        self._max_items = max_items
        self.sql_db_path = sql_db_path
        self.postgres_url = postgres_url
        self._items: dict[str, GraphReviewQueueItemV1] = {}
        self._source_kind = "memory"
        self._last_error: str | None = None

        if self.postgres_url:
            try:
                self._ensure_postgres_schema()
                self._load_from_postgres()
                self._source_kind = "postgres"
                return
            except Exception as exc:
                self._source_kind = "fallback"
                self._last_error = str(exc)
        if self.sql_db_path:
            self._ensure_sql_schema()
            self._load_from_sql()
            self._source_kind = "sqlite"

    def source_kind(self) -> str:
        return self._source_kind

    def degraded(self) -> bool:
        return self._source_kind == "fallback" or self._last_error is not None

    def last_error(self) -> str | None:
        return self._last_error

    def upsert(self, item: GraphReviewQueueItemV1) -> None:
        self.refresh_from_storage()
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
            self._persist()
            return

        if len(self._items) >= self._max_items:
            evict_id = sorted(self._items.items(), key=lambda kv: (kv[1].priority, kv[1].created_at))[0][0]
            del self._items[evict_id]
        self._items[item.queue_item_id] = item
        self._persist()

    def mark_reviewed(self, queue_item_id: str, *, reviewed_at: datetime | None = None) -> GraphReviewQueueItemV1 | None:
        self.refresh_from_storage()
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
        self._persist()
        return updated

    def apply_cycle_feedback(self, queue_item_id: str, *, no_change: bool) -> GraphReviewQueueItemV1 | None:
        self.refresh_from_storage()
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
        self._persist()
        return updated

    def list_eligible(self, *, now: datetime | None = None, limit: int = 10) -> list[GraphReviewQueueItemV1]:
        self.refresh_from_storage()
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
        self.refresh_from_storage()
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


    def refresh_from_storage(self) -> None:
        if self.postgres_url:
            try:
                self._load_from_postgres()
                self._source_kind = "postgres"
                self._last_error = None
                return
            except Exception as exc:
                self._source_kind = "fallback"
                self._last_error = str(exc)
        if self.sql_db_path:
            try:
                self._load_from_sql()
                self._source_kind = "sqlite"
                self._last_error = None
            except Exception as exc:
                self._source_kind = "fallback"
                self._last_error = str(exc)

    @staticmethod
    def _region_key(node_refs: list[str], zone: str) -> str:
        return f"{zone}|{','.join(sorted(node_refs))}"

    def _persist(self) -> None:
        if self.postgres_url:
            try:
                self._persist_to_postgres()
                self._source_kind = "postgres"
                self._last_error = None
                return
            except Exception as exc:
                self._source_kind = "fallback"
                self._last_error = str(exc)
        if self.sql_db_path:
            self._persist_to_sql()

    def _ensure_sql_schema(self) -> None:
        if not self.sql_db_path:
            return
        with sqlite3.connect(self.sql_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS substrate_review_queue_item (
                    queue_item_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _persist_to_sql(self) -> None:
        if not self.sql_db_path:
            return
        with sqlite3.connect(self.sql_db_path) as conn:
            conn.execute("DELETE FROM substrate_review_queue_item")
            for item in self._items.values():
                conn.execute(
                    """
                    INSERT INTO substrate_review_queue_item(queue_item_id, created_at, payload_json)
                    VALUES (?, ?, ?)
                    """,
                    (
                        item.queue_item_id,
                        item.created_at.isoformat(),
                        json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                    ),
                )
            conn.commit()

    def _load_from_sql(self) -> None:
        if not self.sql_db_path:
            return
        with sqlite3.connect(self.sql_db_path) as conn:
            rows = conn.execute("SELECT payload_json FROM substrate_review_queue_item ORDER BY created_at ASC").fetchall()
        self._items = {}
        for (payload_json,) in rows:
            item = GraphReviewQueueItemV1.model_validate(json.loads(payload_json))
            self._items[item.queue_item_id] = item

    def _ensure_postgres_schema(self) -> None:
        if not self.postgres_url:
            return
        from sqlalchemy import create_engine, text

        engine = create_engine(self.postgres_url)
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS substrate_review_queue_item (
                        queue_item_id TEXT PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL,
                        payload_json JSONB NOT NULL
                    )
                    """
                )
            )

    def _persist_to_postgres(self) -> None:
        if not self.postgres_url:
            return
        from sqlalchemy import create_engine, text

        engine = create_engine(self.postgres_url)
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM substrate_review_queue_item"))
            for item in self._items.values():
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_review_queue_item(queue_item_id, created_at, payload_json)
                        VALUES (:queue_item_id, :created_at, CAST(:payload_json AS JSONB))
                        """
                    ),
                    {
                        "queue_item_id": item.queue_item_id,
                        "created_at": item.created_at,
                        "payload_json": json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                    },
                )

    def _load_from_postgres(self) -> None:
        if not self.postgres_url:
            return
        from sqlalchemy import create_engine, text

        engine = create_engine(self.postgres_url)
        with engine.begin() as conn:
            rows = conn.execute(
                text("SELECT payload_json::text FROM substrate_review_queue_item ORDER BY created_at ASC")
            ).fetchall()
        self._items = {}
        for (payload_json,) in rows:
            item = GraphReviewQueueItemV1.model_validate(json.loads(payload_json))
            self._items[item.queue_item_id] = item
