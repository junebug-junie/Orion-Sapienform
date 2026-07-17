from datetime import datetime, timezone

from orion.autonomy.action_outcomes import append_action_outcome, load_action_outcomes
from orion.autonomy.models import ActionOutcomeRefV1


def test_action_outcome_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    fixed = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)
    ref = ActionOutcomeRefV1(
        action_id="fetch-1",
        kind="web.fetch.readonly",
        summary="fetched 2 articles",
        success=True,
        surprise=0.1,
        observed_at=fixed,
    )
    append_action_outcome(subject="orion", outcome=ref)
    loaded = load_action_outcomes(subject="orion")
    assert len(loaded) == 1
