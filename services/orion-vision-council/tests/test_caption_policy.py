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
            "item_count": 1,
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
    assert parse_caption_streams(" Cam0 , walkway,,") == frozenset({"cam0", "walkway"})
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


def test_grounding_caps_withheld_activity_without_caption_tag() -> None:
    walkway = window_for_council(_window("walkway"), DEFAULT_CAPTION_STREAMS)
    grounded, notes = enforce_evidence_grounding(_activity_interpretation(), walkway)
    cand = grounded.event_candidates[0]
    # No caption reached the council, so no caption tag; the claim is still a
    # still-frame guess, so it keeps the 0.4 cap.
    assert "caption_inferred" not in cand.tags
    assert cand.confidence == 0.4
    assert "capped_confidence:human_activity:captions_withheld" in notes

    cam0 = window_for_council(_window("cam0"), DEFAULT_CAPTION_STREAMS)
    grounded, _ = enforce_evidence_grounding(_activity_interpretation(), cam0)
    assert "caption_inferred" in grounded.event_candidates[0].tags
    assert grounded.event_candidates[0].confidence == 0.4


def test_stream_match_ignores_case() -> None:
    streams = parse_caption_streams("Cam0")
    assert window_for_council(_window("cam0"), streams).summary["captions"] == [CAPTION]
    assert window_for_council(_window("CAM0"), streams).summary["captions"] == [CAPTION]


# The model omits scene_summary (so the parser falls back) and makes an
# activity claim with high confidence.
_LLM_REPLY = (
    '{"event_candidates": [{"event_type": "human_activity", '
    '"narrative": "A person is using the sidewalk.", "entities": ["person"], '
    '"tags": [], "confidence": 0.9, "salience": 0.6, "evidence_refs": ["art-1"]}], '
    '"salient_observations": []}'
)


def _service(monkeypatch, *, prompts: list, published: list, unresolved: list):
    svc = council_main.CouncilService()

    async def fake_llm(prompt, env):
        prompts.append(prompt)
        return _LLM_REPLY

    async def fake_publish(channel, env):
        published.append(env)

    async def fake_unresolved(window, interpretation, env, **_kw):
        unresolved.append(window)
        return False

    monkeypatch.setattr(svc, "_call_llm_raw", fake_llm)
    monkeypatch.setattr(svc.bus, "publish", fake_publish)
    monkeypatch.setattr(svc, "_maybe_publish_unresolved", fake_unresolved)
    return svc


def _env(payload: dict) -> BaseEnvelope:
    return BaseEnvelope(kind="vision.window", source=ServiceRef(name="test"), payload=payload)


def test_process_window_walkway_end_to_end(monkeypatch) -> None:
    prompts, published, unresolved = [], [], []
    svc = _service(monkeypatch, prompts=prompts, published=published, unresolved=unresolved)

    asyncio.run(svc._process_window(_env(_window("walkway").model_dump(mode="json"))))

    assert len(prompts) == 1
    assert "lawn mower" not in prompts[0]
    assert "soft_labels" not in prompts[0]
    # Scene-summary fallback did not use the caption.
    record = svc._recent_interpretations[-1]
    assert CAPTION not in record["scene_summary"]
    # Grounding saw the withheld window: capped, no caption tag.
    assert published
    payload = published[-1].payload
    events = payload["events"] if isinstance(payload, dict) else payload.events
    events = [e if isinstance(e, dict) else e.model_dump() for e in events]
    activity = [e for e in events if "sidewalk" in (e.get("narrative") or "")]
    assert activity and activity[0]["confidence"] == 0.4
    assert "caption_inferred" not in (activity[0].get("tags") or [])
    assert all("lawn mower" not in env.model_dump_json() for env in published)
    # Unresolved keeps the full window (caption attributed there).
    assert unresolved[-1].summary["captions"] == [CAPTION]


def test_process_window_cam0_end_to_end_keeps_captions(monkeypatch) -> None:
    prompts, published, unresolved = [], [], []
    svc = _service(monkeypatch, prompts=prompts, published=published, unresolved=unresolved)

    asyncio.run(svc._process_window(_env(_window("cam0").model_dump(mode="json"))))

    assert CAPTION in prompts[0]
    assert svc._recent_interpretations[-1]["scene_summary"] == CAPTION


def test_handle_rpc_uses_caption_free_window(monkeypatch) -> None:
    prompts, published, unresolved = [], [], []
    svc = _service(monkeypatch, prompts=prompts, published=published, unresolved=unresolved)
    env = BaseEnvelope(
        kind="vision.council.request",
        source=ServiceRef(name="test"),
        payload={"window": _window("walkway").model_dump(mode="json")},
    )

    asyncio.run(svc._handle_rpc(env))

    assert len(prompts) == 1
    assert "lawn mower" not in prompts[0]
    assert CAPTION not in svc._recent_interpretations[-1]["scene_summary"]
    assert published
    assert all("lawn mower" not in e.model_dump_json() for e in published)


def test_caption_only_walkway_window_skips_the_llm(monkeypatch) -> None:
    prompts, published, unresolved = [], [], []
    svc = _service(monkeypatch, prompts=prompts, published=published, unresolved=unresolved)
    window = _window("walkway")
    summary = dict(window.summary)
    summary.update(
        item_count=0,
        top_labels=[],
        object_counts={},
        detection_count=0,
        evidence={"hard_labels": [], "soft_labels": ["mower"], "caption_count": 1},
    )
    window = window.model_copy(update={"summary": summary})

    asyncio.run(svc._process_window(_env(window.model_dump(mode="json"))))

    assert prompts == []
    assert published == []
    assert unresolved and unresolved[-1].summary["captions"] == [CAPTION]


def test_caption_only_cam0_window_still_interprets(monkeypatch) -> None:
    prompts, published, unresolved = [], [], []
    svc = _service(monkeypatch, prompts=prompts, published=published, unresolved=unresolved)
    window = _window("cam0")
    window = window.model_copy(update={"summary": {**window.summary, "item_count": 0}})

    asyncio.run(svc._process_window(_env(window.model_dump(mode="json"))))

    assert len(prompts) == 1
