from datetime import datetime, timezone

from orion.schemas.attention_frame import OpenLoopV1
from scripts.attention_loops_store import build_pending_card


def _loop() -> OpenLoopV1:
    return OpenLoopV1(
        id="open-loop-2eb998452183",
        target_type="anomaly",
        description="reactor telemetry mismatch",
        why_it_matters="unresolved anomaly with substrate pressure",
        salience=0.71,
        salience_features={
            "evidence_strength": 0.8, "recurrence": 0.6, "habituation": 0.7,
            "novelty_vs_known": 0.5, "recency": 0.9, "evidence_breadth": 0.5, "dwell": 0.4,
        },
        provenance={"signal_source": "current_turn"},
    )


def test_card_never_id_only():
    card = build_pending_card(
        _loop(), first_seen=datetime.now(timezone.utc), recurrence_count=3,
        narrative="", now=datetime.now(timezone.utc),
    )
    assert card.title and not card.title.startswith("open-loop-")
    assert card.why_it_matters.strip()
    assert "open-loop-2eb998452183" not in card.title
    assert card.top_contributing_features  # rendered in words
    assert all(isinstance(f, str) for f in card.top_contributing_features)


def test_card_features_rendered_in_words():
    card = build_pending_card(
        _loop(), first_seen=datetime.now(timezone.utc), recurrence_count=3,
        narrative="", now=datetime.now(timezone.utc),
    )
    joined = " ".join(card.top_contributing_features).lower()
    assert "evidence" in joined or "recurr" in joined or "recency" in joined
