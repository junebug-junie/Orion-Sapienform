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


@pytest.mark.asyncio
async def test_execute_readonly_fetch_carries_articles_and_salience(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    backend = AsyncMock(
        return_value={
            "success": True,
            "urls": ["https://example.com/a", "https://example.com/b"],
            "articles": [
                {"url": "https://example.com/a", "title": "GPU compute news", "description": "hardware compute"},
                {"url": "https://example.com/b", "title": "Cooking", "description": "recipes"},
            ],
        }
    )
    req = EpisodeFetchRequest(
        subject="orion",
        goal_artifact_id="goal-gap-gpu",
        spawned_correlation_id="wp-run-gap-gpu",
        query="hardware compute gpu recent news coverage",
        gap_terms=("hardware", "compute", "gpu"),
    )
    outcome = await execute_readonly_fetch(req, fetch_backend=backend)

    assert outcome.query == "hardware compute gpu recent news coverage"
    assert len(outcome.articles) == 2
    assert outcome.articles[0].title == "GPU compute news"
    # article 0 covers all 3 gap terms -> salience 1.0; article 1 covers none -> 0.0
    assert outcome.articles[0].salience == 1.0
    assert outcome.articles[1].salience == 0.0
    # aggregate outcome salience is the max over articles
    assert outcome.salience == 1.0


@pytest.mark.asyncio
async def test_execute_readonly_fetch_backcompat_urls_only(tmp_path, monkeypatch) -> None:
    """Older backend returning only urls still yields article refs (salience 0)."""
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    backend = AsyncMock(return_value={"success": True, "urls": ["https://example.com/a"]})
    req = EpisodeFetchRequest(
        subject="orion",
        goal_artifact_id="goal-gap-gpu",
        spawned_correlation_id="wp-run-gap-gpu",
        query="gpu news",
        gap_terms=("gpu",),
    )
    outcome = await execute_readonly_fetch(req, fetch_backend=backend)
    assert [a.url for a in outcome.articles] == ["https://example.com/a"]
    assert outcome.articles[0].salience == 0.0
    assert outcome.salience == 0.0
    assert outcome.query == "gpu news"
