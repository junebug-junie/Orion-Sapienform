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
