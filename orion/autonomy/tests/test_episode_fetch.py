from unittest.mock import AsyncMock

import pytest

from orion.autonomy.action_outcomes import load_action_outcomes
from orion.autonomy.episode_fetch import EpisodeFetchRequest, execute_readonly_fetch
from orion.autonomy.models import ActionOutcomeRefV1


@pytest.mark.asyncio
async def test_execute_readonly_fetch_writes_action_outcome(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    backend = AsyncMock(return_value={"urls": ["https://example.com/a"], "success": True})
    req = EpisodeFetchRequest(
        subject="orion",
        goal_artifact_id="goal-gap-gpu",
        spawned_correlation_id="wp-run-gap-gpu",
        query="hardware GPU supply chain news",
        max_articles=2,
    )
    outcome = await execute_readonly_fetch(req, fetch_backend=backend)
    assert isinstance(outcome, ActionOutcomeRefV1)
    assert outcome.success is True
    assert outcome.kind == "web.fetch.readonly"
    backend.assert_awaited_once_with(req.query, max_articles=req.max_articles)

    loaded = load_action_outcomes(subject="orion")
    assert len(loaded) == 1
    assert loaded[0].action_id == outcome.action_id


@pytest.mark.asyncio
async def test_execute_readonly_fetch_failure_sets_surprise(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    backend = AsyncMock(return_value={"urls": [], "success": False, "error": "timeout"})
    req = EpisodeFetchRequest(
        subject="orion",
        goal_artifact_id="goal-gap-gpu",
        spawned_correlation_id="wp-run-gap-gpu",
        query="hardware GPU supply chain news",
    )
    outcome = await execute_readonly_fetch(req, fetch_backend=backend)
    assert outcome.success is False
    assert outcome.surprise == 1.0


@pytest.mark.asyncio
async def test_default_fetch_backend_raises() -> None:
    from orion.autonomy.episode_fetch import default_fetch_backend

    with pytest.raises(RuntimeError, match="episode_fetch_backend_not_configured"):
        await default_fetch_backend("query", max_articles=2)
