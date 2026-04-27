#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.schemas.substrate_mutation import RecallStrategyProfileV1
from orion.substrate.mutation_queue import SubstrateMutationStore


def _resolve_control_plane_postgres_url() -> str | None:
    control_plane_url = str(os.getenv("SUBSTRATE_CONTROL_PLANE_POSTGRES_URL", "")).strip()
    policy_url = str(os.getenv("SUBSTRATE_POLICY_POSTGRES_URL", "")).strip()
    database_url = str(os.getenv("DATABASE_URL", "")).strip()
    return control_plane_url or policy_url or database_url or None


def build_mutation_store_from_env() -> SubstrateMutationStore:
    return SubstrateMutationStore(
        sql_db_path=str(os.getenv("SUBSTRATE_MUTATION_SQL_DB_PATH", "")).strip() or None,
        postgres_url=_resolve_control_plane_postgres_url(),
    )


def _default_profile_payload(profile_id: str) -> RecallStrategyProfileV1:
    return RecallStrategyProfileV1(
        profile_id=profile_id,
        source_proposal_id=profile_id,
        source_pressure_ids=[],
        source_evidence_refs=["seed:recall_canary_profile"],
        readiness_snapshot={
            "recommendation": "review_candidate",
            "gates_blocked": [],
            "seeded_for_canary": True,
            "manual_canary_ready": True,
        },
        strategy_kind="strategy_profile",
        recall_v2_config_snapshot={
            "retrieval_mode": "recall_v2_shadow",
            "anchor_weighting": "conservative",
            "semantic_expansion": "bounded",
            "temporal_filtering": "enabled",
            "exact_anchor_preference": "enabled",
            "max_results": 12,
            "notes": "canary-only review-only default profile",
        },
        anchor_policy_snapshot={
            "status": "shadow_canary_review_only",
            "production_default": False,
            "live_apply_enabled": False,
            "autonomous_apply_allowed": False,
            "production_write_allowed": False,
        },
        page_index_policy_snapshot={
            "status": "shadow_canary_review_only",
            "production_default": False,
            "live_apply_enabled": False,
        },
        graph_expansion_policy_snapshot={
            "status": "shadow_canary_review_only",
            "production_default": False,
            "live_apply_enabled": False,
        },
        created_by="operator_seed_script",
        status="staged",
    )


def seed_recall_canary_profile(*, store: SubstrateMutationStore, profile_id: str = "recall_v2_shadow_default") -> tuple[RecallStrategyProfileV1, bool]:
    existing = store.get_recall_strategy_profile(profile_id)
    created = existing is None
    baseline = existing if existing is not None else _default_profile_payload(profile_id)
    staged = store.stage_recall_profile(
        profile=baseline.model_copy(
            update={
                "readiness_snapshot": {
                    "recommendation": "review_candidate",
                    "gates_blocked": [],
                    "seeded_for_canary": True,
                    "manual_canary_ready": True,
                },
                "strategy_kind": "strategy_profile",
                "recall_v2_config_snapshot": {
                    "retrieval_mode": "recall_v2_shadow",
                    "anchor_weighting": "conservative",
                    "semantic_expansion": "bounded",
                    "temporal_filtering": "enabled",
                    "exact_anchor_preference": "enabled",
                    "max_results": 12,
                    "notes": "canary-only review-only default profile",
                },
                "anchor_policy_snapshot": {
                    "status": "shadow_canary_review_only",
                    "production_default": False,
                    "live_apply_enabled": False,
                    "autonomous_apply_allowed": False,
                    "production_write_allowed": False,
                },
                "page_index_policy_snapshot": {
                    "status": "shadow_canary_review_only",
                    "production_default": False,
                    "live_apply_enabled": False,
                },
                "graph_expansion_policy_snapshot": {
                    "status": "shadow_canary_review_only",
                    "production_default": False,
                    "live_apply_enabled": False,
                },
                "created_by": "operator_seed_script",
                "status": "staged",
            }
        )
    )
    return staged, created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed a bounded default Recall canary profile into substrate mutation store.")
    parser.add_argument("--profile-id", default="recall_v2_shadow_default", help="Profile ID to seed or upsert.")
    parser.add_argument("--print-json", action="store_true", help="Print seeded profile as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = build_mutation_store_from_env()
    profile, created = seed_recall_canary_profile(store=store, profile_id=str(args.profile_id))
    result: dict[str, Any] = {
        "created": created,
        "profile_id": profile.profile_id,
        "status": profile.status,
        "source_proposal_id": profile.source_proposal_id,
        "production_default": False,
        "live_apply_enabled": False,
        "production_recall_mode": "v1",
        "notes": "canary-only review-only seed",
    }
    print(json.dumps(result, ensure_ascii=True))
    if args.print_json:
        print(json.dumps(profile.model_dump(mode="json"), ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
