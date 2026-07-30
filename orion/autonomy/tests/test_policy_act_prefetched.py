import asyncio
from datetime import datetime, timezone

from orion.autonomy.models import ActionOutcomeRefV1, FetchedArticleRefV1, SubstrateEpisodeIntentV1
from orion.autonomy.policy_act import maybe_execute_substrate_act_after_metabolism


def _prefetched() -> ActionOutcomeRefV1:
    return ActionOutcomeRefV1(
        action_id="fetch-prefetched",
        kind="web.fetch.readonly",
        summary="reused",
        success=True,
        observed_at=datetime.now(timezone.utc),
        query="q",
        articles=[FetchedArticleRefV1(url="https://ex/1", title="t", salience=0.5)],
        salience=0.5,
    )


def test_prefetched_outcome_skips_live_fetch():
    calls = {"count": 0}

    async def _backend(query, *, max_articles):
        calls["count"] += 1
        return {"success": True, "urls": ["https://x"], "articles": []}

    intent = SubstrateEpisodeIntentV1(
        goal_artifact_id="goal-1",
        drive_origin="predictive",
        spawned_correlation_id="run-1",
        subject="orion",
    )
    result = asyncio.run(
        maybe_execute_substrate_act_after_metabolism(
            episode_intent=intent,
            curiosity_signals=[],
            spawned_correlation_id="run-1",
            fetch_backend=_backend,
            journal_dispatch=None,
            budget_used={},
            episode_journal_enabled=False,
            prefetched_outcome=_prefetched(),
        )
    )
    assert calls["count"] == 0
    assert result.fetch_attempted is True
    assert result.fetch_outcome is not None
    assert result.fetch_outcome.action_id == "fetch-prefetched"
