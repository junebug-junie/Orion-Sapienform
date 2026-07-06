from __future__ import annotations

from orion.journaler.worker import build_autonomy_episode_trigger, build_compose_request


def test_episode_journal_carries_spawned_correlation_id() -> None:
    trigger = build_autonomy_episode_trigger(
        goal_artifact_id="goal-gap-gpu",
        spawned_correlation_id="wp-run-gap-gpu",
        narrative_seed="gap → curiosity → fetch → learnings",
    )
    assert trigger.trigger_kind == "autonomy_episode"
    assert trigger.source_ref == "goal-gap-gpu"
    req = build_compose_request(trigger, session_id="orion", user_id="juniper", trace_id="wp-run-gap-gpu")
    assert req.context.metadata["spawned_correlation_id"] == "wp-run-gap-gpu"
    assert "gap" in (trigger.prompt_seed or "").lower() or "curiosity" in (trigger.prompt_seed or "").lower()
