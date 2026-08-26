import time
import uuid

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionArtifactOutputs, VisionArtifactPayload, VisionObject

from app.projection import (
    artifact_uris_from_artifact,
    build_window_payload,
    envelope_to_http_dict,
    identity_confidence_from_artifact,
    identity_hint_from_artifact,
    stream_key_from_artifact,
)


def _artifact(**kwargs) -> VisionArtifactPayload:
    base = dict(
        artifact_id="a1",
        correlation_id="c1",
        task_type="detect",
        device="cam-1",
        inputs={},
        outputs=VisionArtifactOutputs(objects=[]),
        timing={},
        model_fingerprints={},
    )
    base.update(kwargs)
    return VisionArtifactPayload(**base)


def test_stream_key_from_inputs():
    a = _artifact(inputs={"stream_id": "s42"})
    assert stream_key_from_artifact(a) == "s42"


def test_stream_key_fallback_device():
    a = _artifact(inputs={}, device="edge-9")
    assert stream_key_from_artifact(a) == "edge-9"


def _identity_artifact(candidates: list[dict]) -> VisionArtifactPayload:
    return _artifact(
        task_type="identity_face",
        outputs=VisionArtifactOutputs(
            objects=[],
            identities={
                "candidates": candidates,
                "enrolled_subject": "juniper",
                "gallery_enrolled": True,
            },
        ),
    )


def test_identity_hint_picks_highest_similarity_probable_candidate():
    art = _identity_artifact(
        [
            {"subject": "juniper", "similarity": 0.4, "state": "possible", "detect_confidence": 0.9},
            {"subject": "juniper", "similarity": 0.61, "state": "probable", "detect_confidence": 0.95},
        ]
    )
    hint = identity_hint_from_artifact(art)
    assert hint == {"subject": "juniper", "state": "probable", "similarity": 0.61}


def test_identity_hint_none_when_only_unsure():
    art = _identity_artifact(
        [{"subject": "unknown", "similarity": 0.1, "state": "unsure", "detect_confidence": 0.9}]
    )
    assert identity_hint_from_artifact(art) is None


def test_identity_hint_none_when_no_candidates():
    art = _identity_artifact([])
    assert identity_hint_from_artifact(art) is None


def test_identity_hint_none_when_not_enrolled():
    art = _artifact(
        task_type="identity_face",
        outputs=VisionArtifactOutputs(
            objects=[],
            identities={"candidates": [], "enrolled_subject": "juniper", "gallery_enrolled": False},
        ),
    )
    assert identity_hint_from_artifact(art) is None


def test_identity_hint_none_for_non_identity_artifact():
    """A plain retina_fast artifact has no `identities` key at all -- must
    not raise, must not fabricate a hint."""
    art = _artifact(task_type="retina_fast")
    assert identity_hint_from_artifact(art) is None


# -- identity_confidence_from_artifact ----------------------------------------
# Unlike identity_hint_from_artifact above, this one is deliberately NOT
# silent about "unsure" -- see its own docstring for why (the unified-turn
# clarifying-question feature needs exactly that distinction).


def test_identity_confidence_confirmed_for_probable_match():
    art = _identity_artifact(
        [{"subject": "juniper", "similarity": 0.61, "state": "probable", "detect_confidence": 0.95}]
    )
    assert identity_confidence_from_artifact(art) == "confirmed"


def test_identity_confidence_confirmed_for_possible_match():
    art = _identity_artifact(
        [{"subject": "juniper", "similarity": 0.4, "state": "possible", "detect_confidence": 0.9}]
    )
    assert identity_confidence_from_artifact(art) == "confirmed"


def test_identity_confidence_uncertain_for_a_real_unmatched_face():
    """A face WAS detected and genuinely did not match -- the case
    identity_hint_from_artifact discards, this one exists to preserve."""
    art = _identity_artifact(
        [{"subject": "unknown", "similarity": 0.1, "state": "unsure", "detect_confidence": 0.9}]
    )
    assert identity_confidence_from_artifact(art) == "uncertain"


def test_identity_confidence_confirmed_even_when_an_unsure_face_has_higher_raw_similarity():
    """Regression, 2026-08-26 review finding: an earlier version picked
    "best" by raw similarity across ALL candidates (including unsure ones),
    which was not provably consistent with identity_hint_from_artifact's
    own probable/possible-only selection. This artifact is constructed so a
    naive max-by-similarity would pick the unsure candidate (0.61 > 0.4) --
    the fix delegates the "confirmed" check to identity_hint_from_artifact
    directly, so a hint's mere existence always wins regardless of a lower-
    similarity confirmed candidate elsewhere in the same frame."""
    art = _identity_artifact(
        [
            {"subject": "unknown", "similarity": 0.61, "state": "unsure", "detect_confidence": 0.9},
            {"subject": "juniper", "similarity": 0.4, "state": "possible", "detect_confidence": 0.95},
        ]
    )
    assert identity_confidence_from_artifact(art) == "confirmed"


def test_identity_confidence_picks_the_best_candidate_across_multiple_faces():
    art = _identity_artifact(
        [
            {"subject": "unknown", "similarity": 0.1, "state": "unsure", "detect_confidence": 0.9},
            {"subject": "juniper", "similarity": 0.61, "state": "probable", "detect_confidence": 0.95},
        ]
    )
    assert identity_confidence_from_artifact(art) == "confirmed"


def test_identity_confidence_none_when_no_face_detected():
    """No candidates at all -- genuinely no signal, not "uncertain"."""
    art = _identity_artifact([])
    assert identity_confidence_from_artifact(art) is None


def test_identity_confidence_none_for_not_enrolled_reason():
    """A candidate whose `reason` is `not_enrolled` reflects an empty
    gallery (a config problem), not a stranger at the camera -- must read
    as no-signal, never as 'uncertain', or an operator error would
    masquerade as Orion failing to recognize a real person."""
    art = _identity_artifact(
        [{"subject": "unknown", "similarity": None, "state": "unsure", "reason": "not_enrolled"}]
    )
    assert identity_confidence_from_artifact(art) is None


def test_identity_confidence_none_for_non_identity_artifact():
    art = _artifact(task_type="retina_fast")
    assert identity_confidence_from_artifact(art) is None


def test_artifact_uris_caps():
    a = _artifact(
        inputs={
            "image_uri": "https://example.com/x.jpg",
            "thumb": "https://example.com/t.jpg",
        }
    )
    uris = artifact_uris_from_artifact(a)
    assert "https://example.com/x.jpg" in uris


def test_build_window_payload_cursor_and_schema():
    art = _artifact(
        inputs={"camera_id": "cam-a"},
        outputs=VisionArtifactOutputs(
            objects=[VisionObject(label="cup", score=0.9, box_xyxy=[0, 0, 1, 1])]
        ),
    )
    corr = uuid.uuid4()
    env = BaseEnvelope(
        kind="vision.artifact",
        source=ServiceRef(name="src"),
        correlation_id=corr,
    )
    now = time.time()
    p = build_window_payload(
        stream_id="cam-a",
        items=[(art, now)],
        envs=[env],
        window_start=now - 1,
        window_end=now,
        cursor="vw:000000000001:abc",
        stale_after_ms=5000,
    )
    assert p.schema_version == "vision_window_snapshot.v1"
    assert p.cursor == "vw:000000000001:abc"
    assert str(corr) in p.upstream_event_ids
    assert p.camera_id == "cam-a"


def test_envelope_to_http_dict_stale():
    art = _artifact()
    env = BaseEnvelope(
        kind="vision.artifact",
        source=ServiceRef(name="src"),
        correlation_id=uuid.uuid4(),
    )
    old = time.time() - 999.0
    p = build_window_payload(
        stream_id="default",
        items=[(art, old)],
        envs=[env],
        window_start=old,
        window_end=old,
        cursor="vw:000000000002:def",
        stale_after_ms=1000,
    )
    body = envelope_to_http_dict(p, source="live_state")
    assert body["status"] == "stale"
    assert body["source"] == "live_state"
