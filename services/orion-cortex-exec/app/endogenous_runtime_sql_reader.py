from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import create_engine, text


@dataclass(frozen=True)
class SqlReadResult:
    source: str
    rows: list[dict[str, Any]]
    filters: dict[str, Any]
    error: str | None = None


class EndogenousRuntimeSqlReader:
    def __init__(self, *, enabled: bool, database_url: str) -> None:
        self._enabled = enabled
        self._database_url = database_url
        self._engine = create_engine(database_url, pool_pre_ping=True) if enabled else None

    def runtime_records(
        self,
        *,
        limit: int,
        invocation_surface: str | None = None,
        workflow_type: str | None = None,
        outcome: str | None = None,
        audit_status: str | None = None,
        subject_ref: str | None = None,
        anchor_scope: str | None = None,
        mentor_invoked: bool | None = None,
        execution_success: bool | None = None,
        calibration_profile_id: str | None = None,
        correlation_id: str | None = None,
        request_id: str | None = None,
        created_after: datetime | None = None,
    ) -> SqlReadResult:
        filters = {
            "limit": limit,
            "invocation_surface": invocation_surface,
            "workflow_type": workflow_type,
            "outcome": outcome,
            "audit_status": audit_status,
            "subject_ref": subject_ref,
            "anchor_scope": anchor_scope,
            "mentor_invoked": mentor_invoked,
            "execution_success": execution_success,
            "calibration_profile_id": calibration_profile_id,
            "correlation_id": correlation_id,
            "request_id": request_id,
            "created_after": created_after.isoformat() if created_after else None,
        }
        if not self._enabled or self._engine is None:
            return SqlReadResult(source="sql_disabled", rows=[], filters=filters)

        clauses = ["1=1"]
        params: dict[str, Any] = {"limit": max(1, min(limit, 5000))}
        if invocation_surface:
            clauses.append("r.invocation_surface = :invocation_surface")
            params["invocation_surface"] = invocation_surface
        if workflow_type:
            clauses.append("r.workflow_type = :workflow_type")
            params["workflow_type"] = workflow_type
        if outcome:
            clauses.append("r.trigger_outcome = :outcome")
            params["outcome"] = outcome
        if subject_ref:
            clauses.append("r.subject_ref = :subject_ref")
            params["subject_ref"] = subject_ref
        if anchor_scope:
            clauses.append("r.anchor_scope = :anchor_scope")
            params["anchor_scope"] = anchor_scope
        if mentor_invoked is not None:
            clauses.append("r.mentor_invoked = :mentor_invoked")
            params["mentor_invoked"] = mentor_invoked
        if execution_success is not None:
            clauses.append("r.execution_success = :execution_success")
            params["execution_success"] = execution_success
        if calibration_profile_id:
            clauses.append("r.calibration_profile_id = :calibration_profile_id")
            params["calibration_profile_id"] = calibration_profile_id
        if correlation_id:
            clauses.append("r.correlation_id = :correlation_id")
            params["correlation_id"] = correlation_id
        if request_id:
            clauses.append("r.request_id = :request_id")
            params["request_id"] = request_id
        if created_after:
            clauses.append("r.created_at >= :created_after")
            params["created_after"] = created_after
        if audit_status:
            clauses.append(
                "EXISTS (SELECT 1 FROM endogenous_runtime_audit a WHERE a.runtime_record_id = r.runtime_record_id AND a.status = :audit_status)"
            )
            params["audit_status"] = audit_status

        query = text(
            f"""
            SELECT r.payload
            FROM endogenous_runtime_records r
            WHERE {' AND '.join(clauses)}
            ORDER BY r.created_at DESC
            LIMIT :limit
            """
        )

        try:
            with self._engine.connect() as conn:
                rows = conn.execute(query, params).mappings().all()
            payloads = [dict(row.get("payload") or {}) for row in rows]
            return SqlReadResult(source="sql", rows=payloads, filters=filters)
        except Exception as exc:  # pragma: no cover
            return SqlReadResult(source="sql_error", rows=[], filters=filters, error=str(exc))

    def calibration_audit(
        self,
        *,
        limit: int,
        profile_id: str | None = None,
        event_type: str | None = None,
        previous_profile_id: str | None = None,
        operator_id: str | None = None,
        created_after: datetime | None = None,
    ) -> SqlReadResult:
        filters = {
            "limit": limit,
            "profile_id": profile_id,
            "event_type": event_type,
            "previous_profile_id": previous_profile_id,
            "operator_id": operator_id,
            "created_after": created_after.isoformat() if created_after else None,
        }
        if not self._enabled or self._engine is None:
            return SqlReadResult(source="sql_disabled", rows=[], filters=filters)

        clauses = ["1=1"]
        params: dict[str, Any] = {"limit": max(1, min(limit, 5000))}
        if profile_id:
            clauses.append("profile_id = :profile_id")
            params["profile_id"] = profile_id
        if event_type:
            clauses.append("event_type = :event_type")
            params["event_type"] = event_type
        if previous_profile_id:
            clauses.append("previous_profile_id = :previous_profile_id")
            params["previous_profile_id"] = previous_profile_id
        if operator_id:
            clauses.append("operator_id = :operator_id")
            params["operator_id"] = operator_id
        if created_after:
            clauses.append("recorded_at >= :created_after")
            params["created_after"] = created_after

        query = text(
            f"""
            SELECT payload
            FROM calibration_profile_audit
            WHERE {' AND '.join(clauses)}
            ORDER BY recorded_at DESC
            LIMIT :limit
            """
        )
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(query, params).mappings().all()
            payloads = [dict(row.get("payload") or {}) for row in rows]
            return SqlReadResult(source="sql", rows=payloads, filters=filters)
        except Exception as exc:  # pragma: no cover
            return SqlReadResult(source="sql_error", rows=[], filters=filters, error=str(exc))
