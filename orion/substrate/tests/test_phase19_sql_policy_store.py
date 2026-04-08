from __future__ import annotations

import json
from datetime import datetime

from orion.core.schemas.substrate_policy_adoption import (
    SubstratePolicyAdoptionRequestV1,
    SubstratePolicyOverridesV1,
    SubstratePolicyRollbackRequestV1,
    SubstratePolicyRolloutScopeV1,
)
from orion.substrate.policy_profiles import SubstratePolicyProfileStore


def test_sql_store_persists_stage_activate_rollback_and_restart(tmp_path) -> None:
    db_path = tmp_path / "policy.sqlite3"
    store = SubstratePolicyProfileStore(sql_db_path=str(db_path))

    staged = store.adopt(
        SubstratePolicyAdoptionRequestV1(
            rollout_scope=SubstratePolicyRolloutScopeV1(invocation_surfaces=["operator_review"], target_zones=["concept_graph"]),
            policy_overrides=SubstratePolicyOverridesV1(normal_revisit_seconds=900, query_cache_enabled=False),
            activate_now=False,
            operator_id="operator-1",
            rationale="stage before activate",
        )
    )
    assert staged.action_taken == "staged"

    activated = store.activate(profile_id=staged.profile_id or "", operator_id="operator-1", rationale="activate")
    assert activated.action_taken == "activated"

    reloaded = SubstratePolicyProfileStore(sql_db_path=str(db_path))
    resolution = reloaded.resolve(invocation_surface="operator_review", target_zone="concept_graph", operator_mode=True)
    assert resolution.mode == "adopted"
    assert resolution.overrides.get("query_cache_enabled") is False

    rollback = reloaded.rollback(
        SubstratePolicyRollbackRequestV1(
            rollback_target="baseline",
            rollout_scope=SubstratePolicyRolloutScopeV1(invocation_surfaces=["operator_review"], target_zones=["concept_graph"]),
            operator_id="operator-1",
            rationale="rollback",
        )
    )
    assert rollback.action_taken == "rolled_back"

    reloaded_again = SubstratePolicyProfileStore(sql_db_path=str(db_path))
    baseline = reloaded_again.resolve(invocation_surface="operator_review", target_zone="concept_graph", operator_mode=True)
    assert baseline.mode == "baseline"


def test_sql_store_can_migrate_legacy_json_state(tmp_path) -> None:
    json_path = tmp_path / "legacy_policy.json"
    db_path = tmp_path / "policy.sqlite3"

    legacy_profile_id = "substrate-policy-profile-legacy"
    payload = {
        "profiles": [
            {
                "profile_id": legacy_profile_id,
                "profile_version": 1,
                "source_recommendation_id": None,
                "source_summary_window": None,
                "rollout_scope": {"invocation_surfaces": ["operator_review"], "target_zones": ["concept_graph"], "operator_only": False},
                "policy_overrides": {"normal_revisit_seconds": 1200},
                "activation_state": "active",
                "previous_profile_id": None,
                "created_at": datetime.utcnow().isoformat(),
                "activated_at": datetime.utcnow().isoformat(),
                "deactivated_at": None,
                "operator_id": "legacy-op",
                "rationale": "legacy",
                "notes": ["legacy_state"],
            }
        ],
        "active_by_scope": [
            {
                "scope": {"invocation_surfaces": ["operator_review"], "target_zones": ["concept_graph"], "operator_only": False},
                "profile_id": legacy_profile_id,
            }
        ],
        "audit_events": [],
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    migrated = SubstratePolicyProfileStore(sql_db_path=str(db_path), persistence_path=str(json_path))
    resolution = migrated.resolve(invocation_surface="operator_review", target_zone="concept_graph", operator_mode=True)
    assert resolution.mode == "adopted"
    assert resolution.profile_id == legacy_profile_id

    reloaded = SubstratePolicyProfileStore(sql_db_path=str(db_path))
    resolution2 = reloaded.resolve(invocation_surface="operator_review", target_zone="concept_graph", operator_mode=True)
    assert resolution2.mode == "adopted"
    assert resolution2.profile_id == legacy_profile_id
