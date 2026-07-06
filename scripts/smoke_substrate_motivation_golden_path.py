#!/usr/bin/env python3
"""Golden-path integration smoke for substrate-fed motivation (flags on, no live stack)."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

# Running as `python scripts/...` puts `scripts/` on sys.path[0], which shadows
# stdlib `platform` via `scripts/platform/` and breaks pydantic imports.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, ".."))
if sys.path and os.path.abspath(sys.path[0]) == _script_dir:
    sys.path.pop(0)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from orion.autonomy.capability_policy import CapabilityEvaluationContext, evaluate_capability
from orion.autonomy.episode_fetch import EpisodeFetchRequest, execute_readonly_fetch
from orion.autonomy.substrate_metabolism import metabolize_substrate_signals
from orion.core.schemas.drives import GoalProposalV1
from orion.journaler.schemas import JournalEntryWriteV1
from orion.journaler.worker import build_autonomy_episode_trigger, build_compose_request
from orion.memory.crystallization.intake_autonomy_episode import build_crystallization_from_episode
from orion.schemas.world_pulse import (
    DailyWorldPulseSectionsV1,
    DailyWorldPulseV1,
    SectionRollupV1,
    WorldPulseRunResultV1,
    WorldPulseRunV1,
)
from orion.substrate.endogenous_curiosity import EndogenousCuriosityConfig, endogenous_curiosity_candidates


def _gpu_gap_result() -> WorldPulseRunResultV1:
    run_id = "wp-run-gap-gpu"
    now = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)
    digest = DailyWorldPulseV1(
        run_id=run_id,
        date="2026-07-06",
        generated_at=now,
        title="Daily World Pulse",
        executive_summary="Sparse GPU coverage.",
        sections=DailyWorldPulseSectionsV1(),
        items=[],
        orion_analysis_layer="deterministic",
        coverage_status="sparse",
        section_rollups=[
            SectionRollupV1(
                section="hardware_compute_gpu",
                status="missing",
                article_count=0,
                digest_item_count=0,
                confidence=0.35,
            ),
            SectionRollupV1(
                section="ai_technology",
                status="covered",
                article_count=2,
                digest_item_count=1,
                confidence=1.0,
            ),
        ],
        created_at=now,
    )
    return WorldPulseRunResultV1(
        run=WorldPulseRunV1(
            run_id=run_id,
            date="2026-07-06",
            started_at=now,
            completed_at=now,
            status="completed",
            dry_run=False,
        ),
        digest=digest,
    )


def _goal(**kwargs) -> GoalProposalV1:
    base = dict(
        artifact_id="goal-gap-gpu",
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        kind="memory.goals.proposed.v1",
        goal_statement="Reduce predictive uncertainty for hardware_compute_gpu.",
        proposal_signature="sig",
        drive_origin="predictive",
        proposal_status="proposed",
        provenance={"intake_channel": "orion:world_pulse:run:result"},
    )
    base.update(kwargs)
    return GoalProposalV1.model_validate(base)


async def _run_fetch(store_path: str) -> None:
    os.environ["ORION_ACTION_OUTCOME_STORE_PATH"] = store_path
    backend = AsyncMock(return_value={"urls": ["https://example.com/a"], "success": True})
    req = EpisodeFetchRequest(
        subject="orion",
        goal_artifact_id="goal-gap-gpu",
        spawned_correlation_id="wp-run-gap-gpu",
        query="hardware GPU supply chain news",
        max_articles=2,
    )
    outcome = await execute_readonly_fetch(req, fetch_backend=backend)
    assert outcome.success is True
    assert outcome.kind == "web.fetch.readonly"
    backend.assert_awaited_once_with(req.query, max_articles=req.max_articles)


def main() -> int:
    run_id = "wp-run-gap-gpu"
    goal_id = "goal-gap-gpu"

    os.environ["ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED"] = "true"
    # Single metabolism tick yields predictive delta 0.15; threshold must allow that.
    os.environ["ORION_METABOLISM_MIN_PREDICTIVE_PRESSURE"] = "0.1"
    os.environ["ORION_METABOLISM_MIN_CURIOSITY_STRENGTH"] = "0.5"

    wp_result = _gpu_gap_result()

    metabolism = metabolize_substrate_signals(world_pulse_result=wp_result)
    assert metabolism.drive_deltas.get("predictive", 0.0) > 0.0
    assert any(s.signal_type == "world_coverage_gap" for s in metabolism.curiosity_signals)

    candidates = endogenous_curiosity_candidates(
        coverage_gap_signals=metabolism.curiosity_signals,
        config=EndogenousCuriosityConfig(enabled=True),
    )
    assert len(candidates) >= 1
    assert candidates[0].signal_type == "curiosity_candidate"
    assert "hardware_compute_gpu" in candidates[0].focal_node_refs[0]

    top_strength = max(c.signal_strength for c in candidates)
    ctx = CapabilityEvaluationContext(
        predictive_pressure=metabolism.drive_deltas.get("predictive", 0.0),
        curiosity_strength=top_strength,
        signal_kinds=["world_coverage_gap"],
        goal=_goal(),
        budget_used={},
    )
    decision = evaluate_capability("web.fetch.readonly", ctx)
    assert decision.outcome == "allowed"
    assert decision.auto_execute is True

    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(_run_fetch(os.path.join(tmp, "outcomes.json")))

    trigger = build_autonomy_episode_trigger(
        goal_artifact_id=goal_id,
        spawned_correlation_id=run_id,
        narrative_seed="gap → curiosity → fetch → learnings",
    )
    assert trigger.trigger_kind == "autonomy_episode"
    assert trigger.source_ref == goal_id

    compose_req = build_compose_request(
        trigger,
        session_id="orion",
        user_id="juniper",
        trace_id=run_id,
    )
    assert compose_req.context.metadata["spawned_correlation_id"] == run_id

    entry = JournalEntryWriteV1(
        author="orion",
        mode="digest",
        title="Episode: GPU coverage gap",
        body="## Gap\nhardware_compute_gpu empty\n## Learnings\nFetched two articles.",
        source_kind="autonomy_episode",
        source_ref=goal_id,
        correlation_id=run_id,
    )
    proposal = build_crystallization_from_episode(
        journal_entry=entry,
        spawned_correlation_id=run_id,
        grammar_event_ids=["gram-1", "gram-2"],
    )
    assert proposal.kind == "episode"
    assert proposal.status == "proposed"
    assert proposal.provenance.get("spawned_correlation_id") == run_id

    print("GOLDEN_PATH_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
