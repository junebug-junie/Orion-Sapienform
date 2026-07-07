from orion.autonomy.models import SubstrateActResultV1, SubstrateEpisodeIntentV1


def test_substrate_episode_intent_v1_fields() -> None:
    intent = SubstrateEpisodeIntentV1(
        goal_artifact_id="episode-wp-run-1",
        drive_origin="predictive",
        spawned_correlation_id="wp-run-1",
        subject="orion",
    )
    assert intent.drive_origin == "predictive"
    assert intent.spawned_correlation_id == "wp-run-1"


def test_substrate_act_result_v1_optional_outcomes() -> None:
    result = SubstrateActResultV1(fetch_attempted=True, journal_attempted=False)
    assert result.fetch_attempted is True


def test_fetched_article_ref_defaults() -> None:
    from orion.autonomy.models import FetchedArticleRefV1

    art = FetchedArticleRefV1(url="https://example.com/a")
    assert art.title == ""
    assert art.description == ""
    assert art.salience == 0.0


def test_action_outcome_ref_carries_articles_and_query() -> None:
    from orion.autonomy.models import ActionOutcomeRefV1, FetchedArticleRefV1

    outcome = ActionOutcomeRefV1(
        action_id="fetch-1",
        kind="web.fetch.readonly",
        summary="fetched 1 article(s)",
        success=True,
        query="gpu recent news coverage",
        articles=[FetchedArticleRefV1(url="https://example.com/a", title="A", salience=0.5)],
        salience=0.5,
    )
    assert outcome.query == "gpu recent news coverage"
    assert outcome.articles[0].title == "A"
    assert outcome.salience == 0.5


def test_action_outcome_ref_backward_compatible_without_new_fields() -> None:
    from orion.autonomy.models import ActionOutcomeRefV1

    # Persisted JSON from before this patch (no query/articles/salience).
    outcome = ActionOutcomeRefV1.model_validate(
        {
            "action_id": "fetch-old",
            "kind": "web.fetch.readonly",
            "summary": "fetched 2 article(s)",
            "success": True,
            "surprise": 0.0,
        }
    )
    assert outcome.query is None
    assert outcome.articles == []
    assert outcome.salience == 0.0
