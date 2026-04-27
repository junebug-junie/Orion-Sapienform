from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.world_context import filter_world_context_capsule


def test_filter_world_context_capsule_fail_open_missing():
    capsule, diag = filter_world_context_capsule(
        None,
        min_confidence=0.65,
        max_topics=5,
        max_age_hours=36,
        politics_default="only_when_requested",
    )
    assert capsule is None
    assert diag["capsule_filtered_reason"] == "missing_capsule"


def test_filter_world_context_capsule_filters_expired_and_low_confidence():
    now = datetime.now(timezone.utc)
    capsule, diag = filter_world_context_capsule(
        {
            "generated_at": now.isoformat(),
            "salient_topics": [
                {"topic": "expired", "confidence": 0.9, "expires_at": (now - timedelta(hours=1)).isoformat()},
                {"topic": "low", "confidence": 0.1},
                {"topic": "good", "confidence": 0.9},
            ],
        },
        min_confidence=0.65,
        max_topics=5,
        max_age_hours=36,
        politics_default="only_when_requested",
    )
    assert capsule is not None
    assert len(capsule["salient_topics"]) == 1
    assert capsule["salient_topics"][0]["topic"] == "good"
    assert diag["stance_world_context_items_used"] == 1
