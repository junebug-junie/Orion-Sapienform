"""Captions reach the council only for COUNCIL_CAPTION_STREAMS (default cam0).

Spec: docs/superpowers/specs/2026-10-20-walkway-followups-context-spec.md,
item 3 patch A. Live case: walkway caption "A person is pushing a lawn mower
along a sidewalk next to a street." was copied into an unlabelled council row.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionSceneInterpretationV1, VisionWindowPayload

from app import main as council_main
from app.caption_policy import (
    DEFAULT_CAPTION_STREAMS,
    parse_caption_streams,
    window_for_council,
)
from app.evidence_grounding import enforce_evidence_grounding
from app.interpretation import (
    InterpretationParseOutcome,
    _derive_scene_summary,
    build_interpretation_prompt,
    parse_llm_content,
)
from app.settings import Settings

CAPTION = "A person is pushing a lawn mower along a sidewalk next to a street."


def _window(stream_id: str | None, *, camera_id: str | None = None) -> VisionWindowPayload:
    return VisionWindowPayload(
        window_id="w-cap",
        start_ts=1.0,
        end_ts=2.0,
        stream_id=stream_id,
        camera_id=camera_id,
        artifact_ids=["art-1"],
        summary={
            "top_labels": ["person"],
            "object_counts": {"person": 1},
            "captions": [CAPTION],
            "detection_count": 1,
            "evidence": {
                "hard_labels": ["person"],
                "soft_labels": ["mower"],
                "caption_count": 1,
                "host_person_hits": 1,
            },
        },
    )


def test_settings_default_is_room_camera_only() -> None:
    assert Settings().COUNCIL_CAPTION_STREAMS == "cam0"
    assert parse_caption_streams(Settings().COUNCIL_CAPTION_STREAMS) == DEFAULT_CAPTION_STREAMS
    assert council_main.CAPTION_STREAMS == frozenset({"cam0"})


def test_parse_caption_streams() -> None:
    assert parse_caption_streams(" cam0 , walkway,,") == frozenset({"cam0", "walkway"})
    assert parse_caption_streams("") == frozenset()
    assert parse_caption_streams(None) == frozenset()


def test_walkway_prompt_has_no_caption_text() -> None:
    window = window_for_council(_window("walkway"), DEFAULT_CAPTION_STREAMS)
    prompt = build_interpretation_prompt(window)
    assert CAPTION not in prompt
    assert "lawn mower" not in prompt
    assert '"captions": []' in prompt
    # Caption-derived evidence is gone too; detector evidence stays.
    assert "soft_labels" not in prompt
    assert "caption_count" not in prompt
    assert '"hard_labels"' in prompt
    assert '"host_person_hits": 1' in prompt


def test_walkway_scene_summary_never_falls_back_to_caption() -> None:
    window = window_for_council(_window("walkway"), DEFAULT_CAPTION_STREAMS)
    assert _derive_scene_summary({}, window) == "Visual scene interpreted from window summary"
    # Also through the parse path that calls it (model omitted scene_summary).
    outcome = parse_llm_content('{"event_candidates": [], "salient_observations": []}', window)
    assert outcome.interpretation is not None
    assert CAPTION not in outcome.interpretation.scene_summary


def test_cam0_window_keeps_captions() -> None:
    raw = _window("cam0")
    window = window_for_council(raw, DEFAULT_CAPTION_STREAMS)
    assert window is raw
    prompt = build_interpretation_prompt(window)
    assert CAPTION in prompt
    assert '"soft_labels"' in prompt
    assert _derive_scene_summary({}, window) == CAPTION


def test_stream_key_falls_back_to_camera_id_like_the_transition_gate() -> None:
    kept = window_for_council(_window(None, camera_id="cam0"), DEFAULT_CAPTION_STREAMS)
    assert kept.summary["captions"] == [CAPTION]
    dropped = window_for_council(_window(None), DEFAULT_CAPTION_STREAMS)
    assert dropped.summary["captions"] == []


def test_strip_does_not_mutate_the_input_window() -> None:
    raw = _window("walkway")
    window_for_council(raw, DEFAULT_CAPTION_STREAMS)
    assert raw.summary["captions"] == [CAPTION]
    assert raw.summary["evidence"]["soft_labels"] == ["mower"]


def _activity_interpretation() -> VisionSceneInterpretationV1:
    return VisionSceneInterpretationV1(
        window_id="w-cap",
        scene_summary="Person on the sidewalk",
        event_candidates=[
            {
                "event_type": "human_activity",
                "narrative": "A person is using a lawn mower.",
                "entities": ["person"],
                "tags": [],
                "confidence": 0.9,
                "salience": 0.7,
                "evidence_refs": ["art-1"],
            }
        ],
    )


def test_grounding_does_not_tag_caption_inferred_when_captions_were_withheld() -> None:
    walkway = window_for_council(_window("walkway"), DEFAULT_CAPTION_STREAMS)
    grounded, _ = enforce_evidence_grounding(_activity_interpretation(), walkway)
    assert "caption_inferred" not in grounded.event_candidates[0].tags

    cam0 = window_for_council(_window("cam0"), DEFAULT_CAPTION_STREAMS)
    grounded, _ = enforce_evidence_grounding(_activity_interpretation(), cam0)
    assert "caption_inferred" in grounded.event_candidates[0].tags


def test_process_window_sends_caption_free_window_to_council_but_full_one_to_unresolved(
    monkeypatch,
) -> None:
    svc = council_main.CouncilService()
    seen: dict[str, VisionWindowPayload] = {}

    async def fake_generate(window, env, **_kw):
        seen["council"] = window
        return None, InterpretationParseOutcome(interpretation=None, parse_mode="parse_failed")

    async def fake_unresolved(window, interpretation, env, **_kw):
        seen["unresolved"] = window
        return False

    monkeypatch.setattr(svc, "_generate_interpretation", fake_generate)
    monkeypatch.setattr(svc, "_maybe_publish_unresolved", fake_unresolved)
    published: list = []

    async def fake_publish(channel, env):
        published.append(env)

    monkeypatch.setattr(svc.bus, "publish", fake_publish)

    env = BaseEnvelope(
        kind="vision.window",
        source=ServiceRef(name="test"),
        payload=_window("walkway").model_dump(mode="json"),
    )
    asyncio.run(svc._process_window(env))

    assert seen["council"].summary["captions"] == []
    assert "soft_labels" not in seen["council"].summary["evidence"]
    assert seen["unresolved"].summary["captions"] == [CAPTION]
    # host person fallback still publishes, with no caption text in it.
    assert published
    assert all("lawn mower" not in env.model_dump_json() for env in published)
