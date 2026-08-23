from __future__ import annotations

import pytest
from pydantic import ValidationError

from orion.schemas.affectgpt import (
    AffectGptAssessRequestPayload,
    AffectGptAssessResultPayload,
    JuniperMultimodalAffectV1,
)


def test_request_requires_video_and_audio_path():
    with pytest.raises(ValidationError):
        AffectGptAssessRequestPayload()

    req = AffectGptAssessRequestPayload(video_path="/a.mp4", audio_path="/a.wav")
    assert req.subtitle == ""
    assert req.user_message is None


def test_request_forbids_extra_fields():
    with pytest.raises(ValidationError):
        AffectGptAssessRequestPayload(video_path="/a.mp4", audio_path="/a.wav", bogus=1)


def test_result_requires_ok():
    with pytest.raises(ValidationError):
        AffectGptAssessResultPayload()

    res = AffectGptAssessResultPayload(ok=True, raw_response="In the text, ...")
    assert res.error is None
    assert res.face_detection is None
    assert res.subtitle_source is None
    assert res.transcript is None


def test_result_subtitle_source_only_accepts_the_three_real_states():
    for value in ("caller", "transcribed", "none"):
        assert AffectGptAssessResultPayload(ok=True, subtitle_source=value).subtitle_source == value
    with pytest.raises(ValidationError):
        AffectGptAssessResultPayload(ok=True, subtitle_source="bogus")


def test_juniper_multimodal_affect_defaults():
    from datetime import datetime, timezone

    evt = JuniperMultimodalAffectV1(observed_at=datetime.now(timezone.utc), ok=True)
    assert evt.schema_version == "juniper_multimodal_affect.v1"
    assert evt.source == "affectgpt"
    assert evt.subtitle_source is None
    assert evt.transcript is None


def test_schema_id_not_shared_with_text_only_signal():
    # Regression guard for the real naming-collision risk this schema was
    # designed to avoid -- see orion/schemas/affectgpt.py module docstring.
    from orion.schemas.affective_state import JuniperAffectiveStateV1

    assert JuniperAffectiveStateV1.__name__ != JuniperMultimodalAffectV1.__name__
    assert (
        JuniperAffectiveStateV1.model_fields["schema_version"].default
        != JuniperMultimodalAffectV1.model_fields["schema_version"].default
    )
