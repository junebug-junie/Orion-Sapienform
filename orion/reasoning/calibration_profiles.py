from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from sqlalchemy import create_engine, text

from orion.core.schemas.calibration_adoption import (
    CalibrationAdoptionRequestV1,
    CalibrationAdoptionResultV1,
    CalibrationProfileAuditV1,
    CalibrationProfileResolutionV1,
    CalibrationProfileV1,
    CalibrationRollbackRequestV1,
    CalibrationRollbackResultV1,
)

_ALLOWED_OVERRIDE_PREFIXES = (
    "trigger.",
    "cooldown.",
    "mentor.",
    "summary.",
    "promotion.",
)


@dataclass(frozen=True)
class CalibrationRuntimeContext:
    invocation_surface: str | None
    workflow_type: str | None = None
    mentor_enabled: bool = False
    operator_surface: bool = False
    seed: str = ""


class CalibrationProfileStore(Protocol):
    def apply(self, request: CalibrationAdoptionRequestV1) -> CalibrationAdoptionResultV1: ...
    def rollback(self, request: CalibrationRollbackRequestV1) -> CalibrationRollbackResultV1: ...
    def resolve(self, *, context: CalibrationRuntimeContext) -> CalibrationProfileResolutionV1: ...
    def active_profile(self) -> CalibrationProfileV1 | None: ...
    def list_profiles(self, *, include_rolled_back: bool = True) -> list[CalibrationProfileV1]: ...
    def list_audit(self, *, limit: int = 50) -> list[CalibrationProfileAuditV1]: ...


class InMemoryCalibrationProfileStore:
    """Operator-controlled manual calibration profile staging, activation, and rollback."""

    def __init__(self) -> None:
        self._profiles: dict[str, CalibrationProfileV1] = {}
        self._active_profile_id: str | None = None
        self._audits: list[CalibrationProfileAuditV1] = []

    def apply(self, request: CalibrationAdoptionRequestV1) -> CalibrationAdoptionResultV1:
        if request.action == "stage":
            assert request.profile is not None
            profile = self._validate_profile(request.profile.model_copy(update={"state": "staged", "created_by": request.operator_id, "rationale": request.rationale}))
            self._profiles[profile.profile_id] = profile
            self._record(
                CalibrationProfileAuditV1(
                    event_type="staged",
                    operator_id=request.operator_id,
                    rationale=request.rationale,
                    profile_id=profile.profile_id,
                )
            )
            return CalibrationAdoptionResultV1(
                accepted=True,
                action="stage",
                profile=profile,
                active_profile_id=self._active_profile_id,
            )

        profile_id = request.profile_id or ""
        profile = self._profiles.get(profile_id)
        if profile is None:
            return CalibrationAdoptionResultV1(
                accepted=False,
                action="activate",
                active_profile_id=self._active_profile_id,
                warnings=[f"profile_not_found:{profile_id}"],
            )

        previous = self._active_profile_id
        activated = profile.model_copy(
            update={
                "state": "active",
                "activated_at": datetime.now(timezone.utc),
                "previous_profile_id": previous,
            }
        )
        self._profiles[profile_id] = activated
        self._active_profile_id = profile_id
        self._record(
            CalibrationProfileAuditV1(
                event_type="activated",
                operator_id=request.operator_id,
                rationale=request.rationale,
                profile_id=profile_id,
                previous_profile_id=previous,
            )
        )
        return CalibrationAdoptionResultV1(
            accepted=True,
            action="activate",
            profile=activated,
            active_profile_id=profile_id,
            previous_profile_id=previous,
        )

    def rollback(self, request: CalibrationRollbackRequestV1) -> CalibrationRollbackResultV1:
        current_id = self._active_profile_id
        if current_id is None:
            return CalibrationRollbackResultV1(accepted=True, warnings=["already_at_baseline"])

        current = self._profiles[current_id]
        rolled = current.model_copy(update={"state": "rolled_back", "rolled_back_at": datetime.now(timezone.utc)})
        self._profiles[current_id] = rolled

        if request.rollback_mode == "to_baseline":
            self._active_profile_id = None
            self._record(
                CalibrationProfileAuditV1(
                    event_type="reverted_to_baseline",
                    operator_id=request.operator_id,
                    rationale=request.rationale,
                    profile_id=current_id,
                )
            )
            return CalibrationRollbackResultV1(
                accepted=True,
                active_profile_id=None,
                rolled_back_profile_id=current_id,
            )

        previous_id = current.previous_profile_id
        restored_id: str | None = None
        warnings: list[str] = []
        if previous_id and previous_id in self._profiles:
            restored = self._profiles[previous_id].model_copy(update={"state": "active", "activated_at": datetime.now(timezone.utc)})
            self._profiles[previous_id] = restored
            self._active_profile_id = previous_id
            restored_id = previous_id
        else:
            self._active_profile_id = None
            warnings.append("previous_profile_not_available_reverted_to_baseline")

        self._record(
            CalibrationProfileAuditV1(
                event_type="rolled_back",
                operator_id=request.operator_id,
                rationale=request.rationale,
                profile_id=current_id,
                previous_profile_id=restored_id,
            )
        )
        return CalibrationRollbackResultV1(
            accepted=True,
            active_profile_id=self._active_profile_id,
            rolled_back_profile_id=current_id,
            restored_profile_id=restored_id,
            warnings=warnings,
        )

    def resolve(self, *, context: CalibrationRuntimeContext) -> CalibrationProfileResolutionV1:
        profile = self.active_profile()
        if profile is None:
            return CalibrationProfileResolutionV1(mode="baseline", reason="no_active_profile")

        scope = profile.scope
        if scope.invocation_surfaces and (context.invocation_surface or "") not in set(scope.invocation_surfaces):
            return CalibrationProfileResolutionV1(mode="baseline", reason="surface_out_of_scope")
        if scope.workflow_types and context.workflow_type and context.workflow_type not in set(scope.workflow_types):
            return CalibrationProfileResolutionV1(mode="baseline", reason="workflow_out_of_scope")
        if scope.mentor_path == "mentor_enabled_only" and not context.mentor_enabled:
            return CalibrationProfileResolutionV1(mode="baseline", reason="mentor_path_out_of_scope")
        if scope.mentor_path == "mentor_disabled_only" and context.mentor_enabled:
            return CalibrationProfileResolutionV1(mode="baseline", reason="mentor_path_out_of_scope")
        if scope.operator_only and not context.operator_surface:
            return CalibrationProfileResolutionV1(mode="baseline", reason="operator_only_scope")
        if scope.sample_percent < 100:
            bucket = self._sampling_bucket(context.seed)
            if bucket >= scope.sample_percent:
                return CalibrationProfileResolutionV1(mode="baseline", reason="canary_sampled_out")

        return CalibrationProfileResolutionV1(
            mode="adopted",
            profile_id=profile.profile_id,
            overrides=dict(sorted(profile.overrides.items())),
            reason="active_profile_match",
        )

    def active_profile(self) -> CalibrationProfileV1 | None:
        if not self._active_profile_id:
            return None
        return self._profiles.get(self._active_profile_id)

    def list_profiles(self, *, include_rolled_back: bool = True) -> list[CalibrationProfileV1]:
        rows = list(self._profiles.values())
        if not include_rolled_back:
            rows = [row for row in rows if row.state != "rolled_back"]
        return sorted(rows, key=lambda item: item.created_at, reverse=True)

    def list_audit(self, *, limit: int = 50) -> list[CalibrationProfileAuditV1]:
        return list(reversed(self._audits[-max(1, limit) :]))

    @staticmethod
    def _sampling_bucket(seed: str) -> int:
        payload = seed or "calibration-default-seed"
        digest = hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()
        return int(digest[:8], 16) % 100

    @staticmethod
    def _validate_profile(profile: CalibrationProfileV1) -> CalibrationProfileV1:
        invalid = [key for key in profile.overrides if not key.startswith(_ALLOWED_OVERRIDE_PREFIXES)]
        if invalid:
            raise ValueError(f"invalid_override_keys:{','.join(sorted(invalid))}")
        return profile

    def _record(self, event: CalibrationProfileAuditV1) -> None:
        self._audits.append(event)


class SqlCalibrationProfileStore:
    """Durable SQL-backed calibration profile state store."""

    def __init__(self, *, database_url: str) -> None:
        self._engine = create_engine(database_url, pool_pre_ping=True)
        self._audits: list[CalibrationProfileAuditV1] = []
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS calibration_profiles (
                        profile_id TEXT PRIMARY KEY,
                        profile_version INT NOT NULL,
                        source_recommendation_ids TEXT NOT NULL,
                        source_evaluation_request_id TEXT NULL,
                        overrides TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        state TEXT NOT NULL,
                        previous_profile_id TEXT NULL,
                        created_by TEXT NOT NULL,
                        rationale TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        activated_at TIMESTAMP NULL,
                        rolled_back_at TIMESTAMP NULL,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
            )
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_calibration_profiles_state ON calibration_profiles (state)"))

    def apply(self, request: CalibrationAdoptionRequestV1) -> CalibrationAdoptionResultV1:
        if request.action == "stage":
            assert request.profile is not None
            profile = InMemoryCalibrationProfileStore._validate_profile(
                request.profile.model_copy(update={"state": "staged", "created_by": request.operator_id, "rationale": request.rationale})
            )
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO calibration_profiles (
                            profile_id, profile_version, source_recommendation_ids, source_evaluation_request_id,
                            overrides, scope, state, previous_profile_id, created_by, rationale,
                            created_at, activated_at, rolled_back_at, updated_at
                        ) VALUES (
                            :profile_id, :profile_version, :source_recommendation_ids, :source_evaluation_request_id,
                            :overrides, :scope, :state, :previous_profile_id, :created_by, :rationale,
                            :created_at, :activated_at, :rolled_back_at, CURRENT_TIMESTAMP
                        )
                        ON CONFLICT (profile_id) DO UPDATE SET
                            profile_version = EXCLUDED.profile_version,
                            source_recommendation_ids = EXCLUDED.source_recommendation_ids,
                            source_evaluation_request_id = EXCLUDED.source_evaluation_request_id,
                            overrides = EXCLUDED.overrides,
                            scope = EXCLUDED.scope,
                            state = EXCLUDED.state,
                            previous_profile_id = EXCLUDED.previous_profile_id,
                            created_by = EXCLUDED.created_by,
                            rationale = EXCLUDED.rationale,
                            created_at = EXCLUDED.created_at,
                            activated_at = EXCLUDED.activated_at,
                            rolled_back_at = EXCLUDED.rolled_back_at,
                            updated_at = CURRENT_TIMESTAMP
                        """
                    ),
                    self._profile_params(profile),
                )
            self._record(
                CalibrationProfileAuditV1(
                    event_type="staged",
                    operator_id=request.operator_id,
                    rationale=request.rationale,
                    profile_id=profile.profile_id,
                )
            )
            return CalibrationAdoptionResultV1(accepted=True, action="stage", profile=profile, active_profile_id=self._active_profile_id())

        profile_id = request.profile_id or ""
        profile = self.get_profile(profile_id)
        if profile is None:
            return CalibrationAdoptionResultV1(
                accepted=False,
                action="activate",
                active_profile_id=self._active_profile_id(),
                warnings=[f"profile_not_found:{profile_id}"],
            )
        previous = self._active_profile_id()
        activated = profile.model_copy(update={"state": "active", "activated_at": datetime.now(timezone.utc), "previous_profile_id": previous})
        with self._engine.begin() as conn:
            conn.execute(text("UPDATE calibration_profiles SET state='inactive', updated_at=CURRENT_TIMESTAMP WHERE state='active'"))
            conn.execute(
                text(
                    "UPDATE calibration_profiles SET state=:state, activated_at=:activated_at, previous_profile_id=:previous_profile_id, updated_at=CURRENT_TIMESTAMP WHERE profile_id=:profile_id"
                ),
                {
                    "state": activated.state,
                    "activated_at": activated.activated_at,
                    "previous_profile_id": activated.previous_profile_id,
                    "profile_id": activated.profile_id,
                },
            )
        self._record(
            CalibrationProfileAuditV1(
                event_type="activated",
                operator_id=request.operator_id,
                rationale=request.rationale,
                profile_id=activated.profile_id,
                previous_profile_id=previous,
            )
        )
        return CalibrationAdoptionResultV1(
            accepted=True,
            action="activate",
            profile=activated,
            active_profile_id=activated.profile_id,
            previous_profile_id=previous,
        )

    def rollback(self, request: CalibrationRollbackRequestV1) -> CalibrationRollbackResultV1:
        current = self.active_profile()
        if current is None:
            return CalibrationRollbackResultV1(accepted=True, warnings=["already_at_baseline"])
        current_id = current.profile_id
        with self._engine.begin() as conn:
            conn.execute(
                text("UPDATE calibration_profiles SET state='rolled_back', rolled_back_at=:rolled_back_at, updated_at=CURRENT_TIMESTAMP WHERE profile_id=:profile_id"),
                {"rolled_back_at": datetime.now(timezone.utc), "profile_id": current_id},
            )
            if request.rollback_mode == "to_baseline":
                self._record(
                    CalibrationProfileAuditV1(
                        event_type="reverted_to_baseline",
                        operator_id=request.operator_id,
                        rationale=request.rationale,
                        profile_id=current_id,
                    )
                )
                return CalibrationRollbackResultV1(accepted=True, active_profile_id=None, rolled_back_profile_id=current_id)

            restored_id = current.previous_profile_id
            warnings: list[str] = []
            if restored_id and self.get_profile(restored_id):
                conn.execute(
                    text("UPDATE calibration_profiles SET state='active', activated_at=:activated_at, updated_at=CURRENT_TIMESTAMP WHERE profile_id=:profile_id"),
                    {"activated_at": datetime.now(timezone.utc), "profile_id": restored_id},
                )
            else:
                restored_id = None
                warnings.append("previous_profile_not_available_reverted_to_baseline")
            self._record(
                CalibrationProfileAuditV1(
                    event_type="rolled_back",
                    operator_id=request.operator_id,
                    rationale=request.rationale,
                    profile_id=current_id,
                    previous_profile_id=restored_id,
                )
            )
            return CalibrationRollbackResultV1(
                accepted=True,
                active_profile_id=restored_id,
                rolled_back_profile_id=current_id,
                restored_profile_id=restored_id,
                warnings=warnings,
            )

    def resolve(self, *, context: CalibrationRuntimeContext) -> CalibrationProfileResolutionV1:
        profile = self.active_profile()
        if profile is None:
            return CalibrationProfileResolutionV1(mode="baseline", reason="no_active_profile")
        scope = profile.scope
        if scope.invocation_surfaces and (context.invocation_surface or "") not in set(scope.invocation_surfaces):
            return CalibrationProfileResolutionV1(mode="baseline", reason="surface_out_of_scope")
        if scope.workflow_types and context.workflow_type and context.workflow_type not in set(scope.workflow_types):
            return CalibrationProfileResolutionV1(mode="baseline", reason="workflow_out_of_scope")
        if scope.mentor_path == "mentor_enabled_only" and not context.mentor_enabled:
            return CalibrationProfileResolutionV1(mode="baseline", reason="mentor_path_out_of_scope")
        if scope.mentor_path == "mentor_disabled_only" and context.mentor_enabled:
            return CalibrationProfileResolutionV1(mode="baseline", reason="mentor_path_out_of_scope")
        if scope.operator_only and not context.operator_surface:
            return CalibrationProfileResolutionV1(mode="baseline", reason="operator_only_scope")
        if scope.sample_percent < 100:
            bucket = InMemoryCalibrationProfileStore._sampling_bucket(context.seed)
            if bucket >= scope.sample_percent:
                return CalibrationProfileResolutionV1(mode="baseline", reason="canary_sampled_out")
        return CalibrationProfileResolutionV1(
            mode="adopted",
            profile_id=profile.profile_id,
            overrides=dict(sorted(profile.overrides.items())),
            reason="active_profile_match",
        )

    def active_profile(self) -> CalibrationProfileV1 | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM calibration_profiles WHERE state='active' ORDER BY updated_at DESC LIMIT 1")
            ).mappings().first()
        return self._row_to_profile(row) if row else None

    def _active_profile_id(self) -> str | None:
        active = self.active_profile()
        return active.profile_id if active else None

    def get_profile(self, profile_id: str) -> CalibrationProfileV1 | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM calibration_profiles WHERE profile_id=:profile_id"),
                {"profile_id": profile_id},
            ).mappings().first()
        return self._row_to_profile(row) if row else None

    def list_profiles(self, *, include_rolled_back: bool = True) -> list[CalibrationProfileV1]:
        query = "SELECT * FROM calibration_profiles"
        params: dict[str, object] = {}
        if not include_rolled_back:
            query += " WHERE state <> 'rolled_back'"
        query += " ORDER BY created_at DESC"
        with self._engine.connect() as conn:
            rows = conn.execute(text(query), params).mappings().all()
        return [self._row_to_profile(row) for row in rows]

    def list_audit(self, *, limit: int = 50) -> list[CalibrationProfileAuditV1]:
        return list(reversed(self._audits[-max(1, limit) :]))

    @staticmethod
    def _profile_params(profile: CalibrationProfileV1) -> dict[str, object]:
        return {
            "profile_id": profile.profile_id,
            "profile_version": profile.profile_version,
            "source_recommendation_ids": json.dumps(profile.source_recommendation_ids),
            "source_evaluation_request_id": profile.source_evaluation_request_id,
            "overrides": json.dumps(profile.overrides),
            "scope": json.dumps(profile.scope.model_dump(mode="json")),
            "state": profile.state,
            "previous_profile_id": profile.previous_profile_id,
            "created_by": profile.created_by,
            "rationale": profile.rationale,
            "created_at": profile.created_at,
            "activated_at": profile.activated_at,
            "rolled_back_at": profile.rolled_back_at,
        }

    @staticmethod
    def _row_to_profile(row: dict[str, object]) -> CalibrationProfileV1:
        return CalibrationProfileV1(
            profile_id=str(row["profile_id"]),
            profile_version=int(row["profile_version"]),
            source_recommendation_ids=list(json.loads(row.get("source_recommendation_ids") or "[]")),
            source_evaluation_request_id=(str(row["source_evaluation_request_id"]) if row.get("source_evaluation_request_id") else None),
            overrides=dict(json.loads(row.get("overrides") or "{}")),
            scope=json.loads(row.get("scope") or "{}"),
            state=str(row.get("state") or "staged"),
            previous_profile_id=(str(row["previous_profile_id"]) if row.get("previous_profile_id") else None),
            created_by=str(row.get("created_by") or "operator"),
            rationale=str(row.get("rationale") or "unspecified"),
            created_at=row.get("created_at") or datetime.now(timezone.utc),
            activated_at=row.get("activated_at"),
            rolled_back_at=row.get("rolled_back_at"),
        )

    def _record(self, event: CalibrationProfileAuditV1) -> None:
        self._audits.append(event)
