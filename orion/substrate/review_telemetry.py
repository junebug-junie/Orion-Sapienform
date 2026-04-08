from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import json
import sqlite3

from orion.core.schemas.substrate_review_telemetry import (
    GraphReviewCalibrationRecommendationV1,
    GraphReviewCalibrationRequestV1,
    GraphReviewTelemetryQueryV1,
    GraphReviewTelemetryRecordV1,
    GraphReviewTelemetrySummaryV1,
)


@dataclass
class GraphReviewTelemetryRecorder:
    max_records: int = 2000
    sql_db_path: str | None = None
    postgres_url: str | None = None
    _records: list[GraphReviewTelemetryRecordV1] = field(default_factory=list)
    _source_kind: str = field(default="memory", init=False)
    _last_error: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.postgres_url:
            try:
                self._ensure_postgres_schema()
                self._load_from_postgres()
                self._source_kind = "postgres"
                return
            except Exception as exc:
                self._last_error = str(exc)
                self._source_kind = "fallback"
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

    def record(self, entry: GraphReviewTelemetryRecordV1) -> None:
        self._records.append(entry)
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records :]

        if self.postgres_url:
            try:
                self._insert_postgres(entry)
                self._trim_postgres()
                self._source_kind = "postgres"
                self._last_error = None
                return
            except Exception as exc:
                self._source_kind = "fallback"
                self._last_error = str(exc)
        if self.sql_db_path:
            self._insert_sql(entry)
            self._trim_sql()

    def query(self, query: GraphReviewTelemetryQueryV1) -> list[GraphReviewTelemetryRecordV1]:
        records = self._records
        if self.postgres_url:
            try:
                records = self._load_from_postgres()
                self._source_kind = "postgres"
                self._last_error = None
            except Exception as exc:
                self._source_kind = "fallback"
                self._last_error = str(exc)
        elif self.sql_db_path:
            records = self._load_from_sql()

        if query.since is not None:
            records = [r for r in records if r.selected_at >= query.since]
        if query.invocation_surface is not None:
            records = [r for r in records if r.invocation_surface == query.invocation_surface]
        if query.target_zone is not None:
            records = [r for r in records if r.target_zone == query.target_zone]
        if query.subject_ref is not None:
            records = [r for r in records if r.subject_ref == query.subject_ref]
        if query.outcome is not None:
            records = [r for r in records if r.execution_outcome == query.outcome]
        if query.frontier_followup_invoked is not None:
            records = [r for r in records if r.frontier_followup_invoked == query.frontier_followup_invoked]
        records = sorted(records, key=lambda r: r.selected_at, reverse=True)
        return records[: query.limit]

    def summary(self, query: GraphReviewTelemetryQueryV1) -> GraphReviewTelemetrySummaryV1:
        records = self.query(query)
        outcome_counts = Counter(r.execution_outcome for r in records)
        zone_counts = Counter((r.target_zone or "unknown") for r in records)
        surface_counts = Counter(r.invocation_surface for r in records)
        frontier_counts = Counter("invoked" if r.frontier_followup_invoked else "not_invoked" for r in records)

        resolution_cycles = [
            r.cycle_count_before
            for r in records
            if r.cycle_count_before is not None and r.execution_outcome in {"executed", "terminated", "suppressed"}
        ]
        runtime_ms = [r.runtime_duration_ms for r in records]

        notes = ["bounded_telemetry_summary_v1"]
        if self.degraded() and self._last_error:
            notes.append("degraded_source_fallback")

        return GraphReviewTelemetrySummaryV1(
            total_executions=outcome_counts.get("executed", 0),
            total_noops=outcome_counts.get("noop", 0),
            total_suppressed=outcome_counts.get("suppressed", 0),
            total_terminated=outcome_counts.get("terminated", 0),
            total_failed=outcome_counts.get("failed", 0),
            outcome_counts=dict(outcome_counts),
            zone_counts=dict(zone_counts),
            surface_counts=dict(surface_counts),
            frontier_followup_counts=dict(frontier_counts),
            avg_cycles_before_resolution=(sum(resolution_cycles) / len(resolution_cycles)) if resolution_cycles else 0.0,
            avg_runtime_duration_ms=(sum(runtime_ms) / len(runtime_ms)) if runtime_ms else 0.0,
            query_metadata={
                "limit": query.limit,
                "since": query.since.isoformat() if isinstance(query.since, datetime) else None,
                "source_kind": self.source_kind(),
            },
            notes=notes,
        )

    def _ensure_sql_schema(self) -> None:
        if not self.sql_db_path:
            return
        with sqlite3.connect(self.sql_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS substrate_review_telemetry (
                    telemetry_id TEXT PRIMARY KEY,
                    selected_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _insert_sql(self, entry: GraphReviewTelemetryRecordV1) -> None:
        if not self.sql_db_path:
            return
        with sqlite3.connect(self.sql_db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO substrate_review_telemetry(telemetry_id, selected_at, payload_json)
                VALUES (?, ?, ?)
                """,
                (
                    entry.telemetry_id,
                    entry.selected_at.isoformat(),
                    json.dumps(entry.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                ),
            )
            conn.commit()

    def _trim_sql(self) -> None:
        if not self.sql_db_path:
            return
        with sqlite3.connect(self.sql_db_path) as conn:
            conn.execute(
                """
                DELETE FROM substrate_review_telemetry
                WHERE telemetry_id IN (
                    SELECT telemetry_id FROM substrate_review_telemetry
                    ORDER BY selected_at DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (self.max_records,),
            )
            conn.commit()

    def _load_from_sql(self) -> list[GraphReviewTelemetryRecordV1]:
        if not self.sql_db_path:
            return list(self._records)
        with sqlite3.connect(self.sql_db_path) as conn:
            rows = conn.execute("SELECT payload_json FROM substrate_review_telemetry ORDER BY selected_at ASC").fetchall()
        self._records = [
            GraphReviewTelemetryRecordV1.model_validate(json.loads(payload_json))
            for (payload_json,) in rows
        ][-self.max_records :]
        return list(self._records)

    def _ensure_postgres_schema(self) -> None:
        if not self.postgres_url:
            return
        from sqlalchemy import create_engine, text

        engine = create_engine(self.postgres_url)
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS substrate_review_telemetry (
                        telemetry_id TEXT PRIMARY KEY,
                        selected_at TIMESTAMPTZ NOT NULL,
                        payload_json JSONB NOT NULL
                    )
                    """
                )
            )

    def _insert_postgres(self, entry: GraphReviewTelemetryRecordV1) -> None:
        if not self.postgres_url:
            return
        from sqlalchemy import create_engine, text

        engine = create_engine(self.postgres_url)
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_review_telemetry(telemetry_id, selected_at, payload_json)
                    VALUES (:telemetry_id, :selected_at, CAST(:payload_json AS JSONB))
                    ON CONFLICT (telemetry_id) DO UPDATE SET
                      selected_at = EXCLUDED.selected_at,
                      payload_json = EXCLUDED.payload_json
                    """
                ),
                {
                    "telemetry_id": entry.telemetry_id,
                    "selected_at": entry.selected_at,
                    "payload_json": json.dumps(entry.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                },
            )

    def _trim_postgres(self) -> None:
        if not self.postgres_url:
            return
        from sqlalchemy import create_engine, text

        engine = create_engine(self.postgres_url)
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM substrate_review_telemetry
                    WHERE telemetry_id IN (
                        SELECT telemetry_id FROM substrate_review_telemetry
                        ORDER BY selected_at DESC
                        OFFSET :max_records
                    )
                    """
                ),
                {"max_records": self.max_records},
            )

    def _load_from_postgres(self) -> list[GraphReviewTelemetryRecordV1]:
        if not self.postgres_url:
            return list(self._records)
        from sqlalchemy import create_engine, text

        engine = create_engine(self.postgres_url)
        with engine.begin() as conn:
            rows = conn.execute(
                text("SELECT payload_json::text FROM substrate_review_telemetry ORDER BY selected_at ASC")
            ).fetchall()
        self._records = [
            GraphReviewTelemetryRecordV1.model_validate(json.loads(payload_json))
            for (payload_json,) in rows
        ][-self.max_records :]
        return list(self._records)


class GraphReviewCalibrationAnalyzer:
    def recommend(
        self,
        *,
        summary: GraphReviewTelemetrySummaryV1,
        request: GraphReviewCalibrationRequestV1 | None = None,
    ) -> list[GraphReviewCalibrationRecommendationV1]:
        policy = request or GraphReviewCalibrationRequestV1()
        total = sum(summary.outcome_counts.values())
        if total < policy.min_sample_size:
            return [
                GraphReviewCalibrationRecommendationV1(
                    recommendation_type="hold",
                    target_parameter="review_policy",
                    current_value="unchanged",
                    suggested_value="unchanged",
                    rationale="insufficient telemetry sample size",
                    sample_size=total,
                    confidence=0.2,
                    notes=["insufficient_data"],
                )
            ]

        recommendations: list[GraphReviewCalibrationRecommendationV1] = []
        suppression_ratio = summary.total_suppressed / total
        requeue_ratio = (summary.outcome_counts.get("executed", 0) / total) if total else 0.0
        failure_ratio = summary.total_failed / total

        if suppression_ratio > policy.high_suppression_ratio:
            recommendations.append(
                GraphReviewCalibrationRecommendationV1(
                    recommendation_type="increase_suppression_threshold",
                    target_parameter="suppress_after_low_value_cycles",
                    current_value="current",
                    suggested_value="+1",
                    rationale="suppression ratio indicates premature suppression churn",
                    sample_size=total,
                    confidence=0.72,
                )
            )

        if requeue_ratio > policy.high_requeue_ratio:
            recommendations.append(
                GraphReviewCalibrationRecommendationV1(
                    recommendation_type="increase_cadence_interval",
                    target_parameter="normal_revisit_seconds",
                    current_value="current",
                    suggested_value="+20%",
                    rationale="high repeated execution ratio suggests cadence may be too aggressive",
                    sample_size=total,
                    confidence=0.68,
                )
            )

        if failure_ratio > policy.high_failure_ratio:
            recommendations.append(
                GraphReviewCalibrationRecommendationV1(
                    recommendation_type="decrease_max_cycles",
                    target_parameter="max_cycles_*",
                    current_value="current",
                    suggested_value="-1",
                    rationale="high failure ratio suggests reducing repeated runtime attempts",
                    sample_size=total,
                    confidence=0.74,
                )
            )

        if not recommendations:
            recommendations.append(
                GraphReviewCalibrationRecommendationV1(
                    recommendation_type="hold",
                    target_parameter="review_policy",
                    current_value="unchanged",
                    suggested_value="unchanged",
                    rationale="telemetry indicates balanced loop behavior",
                    sample_size=total,
                    confidence=0.7,
                    notes=["no_change_recommended"],
                )
            )

        return recommendations
