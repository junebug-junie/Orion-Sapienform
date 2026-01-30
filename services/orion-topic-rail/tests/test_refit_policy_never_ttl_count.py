from datetime import datetime, timedelta, timezone

from app.topic_rail.lifecycle import decide_refit


def test_refit_policy_never():
    decision = decide_refit(
        policy="never",
        force_refit=False,
        manifest_created_at=None,
        refit_ttl_hours=168,
        refit_doc_threshold=200,
        new_doc_count=0,
    )
    assert decision.should_refit is False


def test_refit_policy_ttl_expired():
    created_at = (datetime.now(timezone.utc) - timedelta(hours=200)).isoformat()
    decision = decide_refit(
        policy="ttl",
        force_refit=False,
        manifest_created_at=created_at,
        refit_ttl_hours=168,
        refit_doc_threshold=200,
        new_doc_count=0,
    )
    assert decision.should_refit is True


def test_refit_policy_count_threshold():
    decision = decide_refit(
        policy="count",
        force_refit=False,
        manifest_created_at="2025-01-01T00:00:00+00:00",
        refit_ttl_hours=168,
        refit_doc_threshold=200,
        new_doc_count=250,
    )
    assert decision.should_refit is True
