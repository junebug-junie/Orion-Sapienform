# orion/autonomy/tests/test_evidence_ref_schema.py
from __future__ import annotations

from datetime import datetime

from orion.autonomy.models import AutonomyEvidenceRefV1


def test_evidence_ref_accepts_optional_signal_fields() -> None:
    fixed = datetime(2026, 7, 10, 12, 0, 0)
    ev = AutonomyEvidenceRefV1(
        evidence_id="social_bridge:abc",
        source="social_bridge",
        kind="relational_signal",
        summary="cooldown_active",
        confidence=0.6,
        observed_at=fixed,
        signal_kind="chat_social_hazard",
        dimension="cooldown_active",
        value=1.0,
    )
    assert ev.signal_kind == "chat_social_hazard"
    assert ev.dimension == "cooldown_active"
    assert ev.value == 1.0


def test_evidence_ref_defaults_signal_fields_to_none() -> None:
    ev = AutonomyEvidenceRefV1(
        evidence_id="user_turn:x",
        source="user_message",
        kind="user_turn",
        summary="hi",
    )
    assert ev.signal_kind is None
    assert ev.dimension is None
    assert ev.value is None
    assert ev.observed_at is None
