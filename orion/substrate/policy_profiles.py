from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from orion.core.schemas.frontier_expansion import FrontierTargetZoneV1
from orion.core.schemas.substrate_policy_adoption import (
    SubstratePolicyAdoptionRequestV1,
    SubstratePolicyAdoptionResultV1,
    SubstratePolicyAuditEventV1,
    SubstratePolicyComparisonV1,
    SubstratePolicyInspectionV1,
    SubstratePolicyProfileV1,
    SubstratePolicyResolutionV1,
    SubstratePolicyRollbackRequestV1,
    SubstratePolicyRollbackResultV1,
    SubstratePolicyRolloutScopeV1,
)
from orion.core.schemas.substrate_review_runtime import GraphReviewRuntimeSurfaceV1


def _scope_key(scope: SubstratePolicyRolloutScopeV1) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    return (tuple(sorted(scope.invocation_surfaces)), tuple(sorted(scope.target_zones)), bool(scope.operator_only))


@dataclass
class SubstratePolicyProfileStore:
    """Manual, operator-controlled policy adoption store.

    This is an in-memory bounded implementation intended for deterministic runtime seams.
    """

    max_profiles: int = 200
    max_audit_events: int = 500
    sql_db_path: str | None = None
    persistence_path: str | None = None  # legacy JSON fallback / migration source
    _profiles: dict[str, SubstratePolicyProfileV1] = field(default_factory=dict)
    _active_by_scope: dict[tuple[tuple[str, ...], tuple[str, ...], bool], str] = field(default_factory=dict)
    _audit_events: list[SubstratePolicyAuditEventV1] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.sql_db_path:
            self._ensure_sql_schema()
            self._load_from_sql()
            self._migrate_from_legacy_json_if_needed()
        else:
            self._load_if_available()

    def adopt(self, request: SubstratePolicyAdoptionRequestV1) -> SubstratePolicyAdoptionResultV1:
        profile = SubstratePolicyProfileV1(
            source_recommendation_id=request.source_recommendation_id,
            source_summary_window=request.source_summary_window,
            rollout_scope=request.rollout_scope,
            policy_overrides=request.policy_overrides,
            activation_state="staged",
            operator_id=request.operator_id,
            rationale=request.rationale,
            notes=list(request.notes),
        )
        self._profiles[profile.profile_id] = profile
        self._trim_profiles()
        self._record(
            SubstratePolicyAuditEventV1(
                event_type="staged",
                profile_id=profile.profile_id,
                operator_id=request.operator_id,
                rationale=request.rationale,
                rollout_scope=profile.rollout_scope,
                notes=["manual_stage"],
            )
        )

        previous_profile_id = None
        notes = ["profile_staged"]
        if request.activate_now:
            activated, previous_profile_id = self._activate_profile(profile_id=profile.profile_id, operator_id=request.operator_id, rationale=request.rationale)
            profile = activated
            notes.append("profile_activated")
        self._persist()

        return SubstratePolicyAdoptionResultV1(
            request_id=request.request_id,
            profile_id=profile.profile_id,
            action_taken="activated" if request.activate_now else "staged",
            active_scope_summary=self._scope_summary(profile.rollout_scope),
            previous_active_profile_id=previous_profile_id,
            notes=notes,
        )

    def activate(self, *, profile_id: str, operator_id: str | None = None, rationale: str = "") -> SubstratePolicyAdoptionResultV1:
        if profile_id not in self._profiles:
            return SubstratePolicyAdoptionResultV1(
                request_id=f"activate-missing-{profile_id}",
                action_taken="rejected",
                warnings=["profile_not_found"],
            )
        profile, previous = self._activate_profile(profile_id=profile_id, operator_id=operator_id, rationale=rationale)
        self._persist()
        return SubstratePolicyAdoptionResultV1(
            request_id=f"activate-{profile_id}",
            profile_id=profile.profile_id,
            action_taken="activated",
            active_scope_summary=self._scope_summary(profile.rollout_scope),
            previous_active_profile_id=previous,
            notes=["explicit_activation"],
        )

    def rollback(self, request: SubstratePolicyRollbackRequestV1) -> SubstratePolicyRollbackResultV1:
        scope_key = _scope_key(request.rollout_scope)
        current_id = self._active_by_scope.get(scope_key)
        if current_id is None and request.rollback_target != "baseline":
            return SubstratePolicyRollbackResultV1(request_id=request.request_id, action_taken="rejected", warnings=["no_active_profile_for_scope"])

        if request.rollback_target == "baseline":
            previous = current_id
            if previous:
                self._deactivate(previous, state="rolled_back")
            self._active_by_scope.pop(scope_key, None)
            self._record(
                SubstratePolicyAuditEventV1(
                    event_type="rolled_back",
                    profile_id=None,
                    previous_profile_id=previous,
                    operator_id=request.operator_id,
                    rationale=request.rationale,
                    rollout_scope=request.rollout_scope,
                    notes=["rollback_to_baseline"],
                )
            )
            self._persist()
            return SubstratePolicyRollbackResultV1(
                request_id=request.request_id,
                action_taken="rolled_back",
                active_profile_id=None,
                previous_profile_id=previous,
                notes=["baseline_restored"],
            )

        target_id = request.explicit_profile_id if request.rollback_target == "profile" else self._profiles.get(current_id).previous_profile_id if current_id else None
        if not target_id:
            return SubstratePolicyRollbackResultV1(request_id=request.request_id, action_taken="rejected", warnings=["no_previous_profile_available"])
        if target_id not in self._profiles:
            return SubstratePolicyRollbackResultV1(request_id=request.request_id, action_taken="rejected", warnings=["target_profile_not_found"])

        previous = current_id
        if previous:
            self._deactivate(previous, state="rolled_back")
        activated, _ = self._activate_profile(profile_id=target_id, operator_id=request.operator_id, rationale=request.rationale)
        self._record(
            SubstratePolicyAuditEventV1(
                event_type="rolled_back",
                profile_id=activated.profile_id,
                previous_profile_id=previous,
                operator_id=request.operator_id,
                rationale=request.rationale,
                rollout_scope=activated.rollout_scope,
                notes=["rollback_activated_previous"],
            )
        )
        self._persist()
        return SubstratePolicyRollbackResultV1(
            request_id=request.request_id,
            action_taken="rolled_back",
            active_profile_id=activated.profile_id,
            previous_profile_id=previous,
            notes=["rollback_complete"],
        )

    def resolve(
        self,
        *,
        invocation_surface: GraphReviewRuntimeSurfaceV1,
        target_zone: FrontierTargetZoneV1,
        operator_mode: bool,
    ) -> SubstratePolicyResolutionV1:
        active_profiles = [p for p in self._profiles.values() if p.activation_state == "active"]
        matched = [p for p in active_profiles if self._matches_scope(p.rollout_scope, invocation_surface, target_zone, operator_mode)]
        if not matched:
            return SubstratePolicyResolutionV1(mode="baseline", reason="no_active_scope_match")
        matched.sort(key=lambda p: p.activated_at or p.created_at, reverse=True)
        winner = matched[0]
        return SubstratePolicyResolutionV1(
            mode="adopted",
            profile_id=winner.profile_id,
            rollout_scope=winner.rollout_scope,
            overrides=winner.policy_overrides.model_dump(exclude_none=True),
            reason="active_scope_match",
        )

    def inspect(self, *, audit_limit: int = 25) -> SubstratePolicyInspectionV1:
        profiles = list(self._profiles.values())
        return SubstratePolicyInspectionV1(
            active_profiles=[p for p in profiles if p.activation_state == "active"],
            staged_profiles=[p for p in profiles if p.activation_state == "staged"],
            rolled_back_profiles=[p for p in profiles if p.activation_state == "rolled_back"],
            recent_audit_events=list(self._audit_events[-max(1, audit_limit) :]),
        )

    def get_profile(self, profile_id: str) -> SubstratePolicyProfileV1 | None:
        return self._profiles.get(profile_id)

    def list_profiles(self, *, limit: int = 200) -> list[SubstratePolicyProfileV1]:
        bounded_limit = max(1, min(limit, self.max_profiles))
        return sorted(self._profiles.values(), key=lambda p: p.created_at, reverse=True)[:bounded_limit]

    def compare_against_baseline(self, *, resolution: SubstratePolicyResolutionV1, baseline_summary: dict[str, Any]) -> SubstratePolicyComparisonV1:
        if resolution.mode == "baseline":
            return SubstratePolicyComparisonV1(baseline_summary=baseline_summary, comparison_notes=["baseline_only"])
        notes = [f"override:{key}" for key in sorted(resolution.overrides)]
        return SubstratePolicyComparisonV1(
            baseline_summary=baseline_summary,
            active_profile_id=resolution.profile_id,
            active_overrides=resolution.overrides,
            comparison_notes=notes,
        )

    def _activate_profile(self, *, profile_id: str, operator_id: str | None, rationale: str) -> tuple[SubstratePolicyProfileV1, str | None]:
        profile = self._profiles[profile_id]
        scope_key = _scope_key(profile.rollout_scope)
        previous_id = self._active_by_scope.get(scope_key)
        if previous_id and previous_id in self._profiles:
            self._deactivate(previous_id, state="inactive")

        activated = profile.model_copy(
            update={
                "activation_state": "active",
                "activated_at": datetime.now(timezone.utc),
                "deactivated_at": None,
                "previous_profile_id": previous_id,
            }
        )
        self._profiles[profile_id] = activated
        self._active_by_scope[scope_key] = profile_id
        self._record(
            SubstratePolicyAuditEventV1(
                event_type="activated",
                profile_id=profile_id,
                previous_profile_id=previous_id,
                operator_id=operator_id,
                rationale=rationale,
                rollout_scope=activated.rollout_scope,
                notes=["manual_activation"],
            )
        )
        self._persist()
        return activated, previous_id

    def _deactivate(self, profile_id: str, *, state: str) -> None:
        profile = self._profiles.get(profile_id)
        if profile is None:
            return
        self._profiles[profile_id] = profile.model_copy(
            update={"activation_state": state, "deactivated_at": datetime.now(timezone.utc)}
        )
        self._record(
            SubstratePolicyAuditEventV1(
                event_type="deactivated",
                profile_id=profile_id,
                operator_id=profile.operator_id,
                rationale=profile.rationale,
                rollout_scope=profile.rollout_scope,
                notes=[f"deactivated_to:{state}"],
            )
        )
        self._persist()

    @staticmethod
    def _matches_scope(
        scope: SubstratePolicyRolloutScopeV1,
        invocation_surface: GraphReviewRuntimeSurfaceV1,
        target_zone: FrontierTargetZoneV1,
        operator_mode: bool,
    ) -> bool:
        if scope.invocation_surfaces and invocation_surface not in scope.invocation_surfaces:
            return False
        if scope.target_zones and target_zone not in scope.target_zones:
            return False
        if scope.operator_only and not operator_mode:
            return False
        return True

    def _record(self, event: SubstratePolicyAuditEventV1) -> None:
        self._audit_events.append(event)
        if len(self._audit_events) > self.max_audit_events:
            self._audit_events = self._audit_events[-self.max_audit_events :]

    def _trim_profiles(self) -> None:
        if len(self._profiles) <= self.max_profiles:
            return
        by_created = sorted(self._profiles.values(), key=lambda item: item.created_at)
        for profile in by_created[:-self.max_profiles]:
            self._profiles.pop(profile.profile_id, None)

    def _persist(self) -> None:
        if self.sql_db_path:
            self._persist_to_sql()
            return
        if not self.persistence_path:
            return
        path = Path(self.persistence_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "profiles": [profile.model_dump(mode="json") for profile in self._profiles.values()],
            "active_by_scope": [
                {
                    "scope": {"invocation_surfaces": list(key[0]), "target_zones": list(key[1]), "operator_only": key[2]},
                    "profile_id": profile_id,
                }
                for key, profile_id in self._active_by_scope.items()
            ],
            "audit_events": [event.model_dump(mode="json") for event in self._audit_events[-self.max_audit_events :]],
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)

    def _load_if_available(self) -> None:
        if not self.persistence_path:
            return
        path = Path(self.persistence_path)
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._profiles = {}
        for item in payload.get("profiles", []):
            profile = SubstratePolicyProfileV1.model_validate(item)
            self._profiles[profile.profile_id] = profile
        self._active_by_scope = {}
        for item in payload.get("active_by_scope", []):
            scope = SubstratePolicyRolloutScopeV1.model_validate(item.get("scope", {}))
            profile_id = str(item.get("profile_id", ""))
            if profile_id:
                self._active_by_scope[_scope_key(scope)] = profile_id
        self._audit_events = [
            SubstratePolicyAuditEventV1.model_validate(item)
            for item in payload.get("audit_events", [])
        ][-self.max_audit_events :]

    def _ensure_sql_schema(self) -> None:
        if not self.sql_db_path:
            return
        with sqlite3.connect(self.sql_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS substrate_policy_profile (
                    profile_id TEXT PRIMARY KEY,
                    activation_state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    activated_at TEXT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS substrate_policy_active_scope (
                    scope_key TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS substrate_policy_audit (
                    event_id TEXT PRIMARY KEY,
                    profile_id TEXT NULL,
                    event_type TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _persist_to_sql(self) -> None:
        if not self.sql_db_path:
            return
        with sqlite3.connect(self.sql_db_path) as conn:
            conn.execute("DELETE FROM substrate_policy_profile")
            conn.execute("DELETE FROM substrate_policy_active_scope")
            conn.execute("DELETE FROM substrate_policy_audit")
            for profile in self._profiles.values():
                conn.execute(
                    """
                    INSERT INTO substrate_policy_profile(profile_id, activation_state, created_at, activated_at, payload_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        profile.profile_id,
                        profile.activation_state,
                        profile.created_at.isoformat(),
                        profile.activated_at.isoformat() if profile.activated_at else None,
                        json.dumps(profile.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                    ),
                )
            for key, profile_id in self._active_by_scope.items():
                conn.execute(
                    "INSERT INTO substrate_policy_active_scope(scope_key, profile_id) VALUES (?, ?)",
                    (json.dumps([list(key[0]), list(key[1]), key[2]], ensure_ascii=False), profile_id),
                )
            for event in self._audit_events[-self.max_audit_events :]:
                conn.execute(
                    """
                    INSERT INTO substrate_policy_audit(event_id, profile_id, event_type, recorded_at, payload_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.profile_id,
                        event.event_type,
                        event.recorded_at.isoformat(),
                        json.dumps(event.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                    ),
                )
            conn.commit()

    def _load_from_sql(self) -> None:
        if not self.sql_db_path:
            return
        with sqlite3.connect(self.sql_db_path) as conn:
            profiles = conn.execute("SELECT payload_json FROM substrate_policy_profile").fetchall()
            active = conn.execute("SELECT scope_key, profile_id FROM substrate_policy_active_scope").fetchall()
            audits = conn.execute("SELECT payload_json FROM substrate_policy_audit ORDER BY recorded_at ASC").fetchall()
        self._profiles = {}
        for (payload_json,) in profiles:
            profile = SubstratePolicyProfileV1.model_validate(json.loads(payload_json))
            self._profiles[profile.profile_id] = profile
        self._active_by_scope = {}
        for scope_key, profile_id in active:
            invocation_surfaces, target_zones, operator_only = json.loads(scope_key)
            scope = SubstratePolicyRolloutScopeV1(
                invocation_surfaces=list(invocation_surfaces or []),
                target_zones=list(target_zones or []),
                operator_only=bool(operator_only),
            )
            self._active_by_scope[_scope_key(scope)] = profile_id
        self._audit_events = [
            SubstratePolicyAuditEventV1.model_validate(json.loads(payload_json))
            for (payload_json,) in audits
        ][-self.max_audit_events :]

    def _migrate_from_legacy_json_if_needed(self) -> None:
        if not self.sql_db_path or not self.persistence_path:
            return
        if self._profiles:
            return
        path = Path(self.persistence_path)
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._profiles = {}
        for item in payload.get("profiles", []):
            profile = SubstratePolicyProfileV1.model_validate(item)
            self._profiles[profile.profile_id] = profile
        self._active_by_scope = {}
        for item in payload.get("active_by_scope", []):
            scope = SubstratePolicyRolloutScopeV1.model_validate(item.get("scope", {}))
            profile_id = str(item.get("profile_id", ""))
            if profile_id:
                self._active_by_scope[_scope_key(scope)] = profile_id
        self._audit_events = [
            SubstratePolicyAuditEventV1.model_validate(item)
            for item in payload.get("audit_events", [])
        ][-self.max_audit_events :]
        self._persist_to_sql()

    @staticmethod
    def _scope_summary(scope: SubstratePolicyRolloutScopeV1) -> dict[str, Any]:
        return {
            "invocation_surfaces": list(scope.invocation_surfaces),
            "target_zones": list(scope.target_zones),
            "operator_only": scope.operator_only,
        }


def build_substrate_policy_store_from_env() -> SubstratePolicyProfileStore:
    import os

    sql_db_path = str(os.getenv("SUBSTRATE_POLICY_SQL_DB_PATH", "")).strip() or None
    legacy_json_path = str(os.getenv("SUBSTRATE_POLICY_LEGACY_JSON_PATH", "")).strip() or None
    if sql_db_path is None:
        sql_db_path = "/tmp/orion_substrate_policy.sqlite3"
    return SubstratePolicyProfileStore(sql_db_path=sql_db_path, persistence_path=legacy_json_path)
