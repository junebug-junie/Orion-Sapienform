"""Detector-agreement gate for walkway captions (2026-10-15).

Tonight's real failure: the walkway's BLIP captioner described a dark frame
the detector found nobody in as "A camera is placed on a wooden bench in a
garden", and that reached Orion's situation brief as the street line."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.schemas.vision import VisionArtifactOutputs, VisionArtifactPayload, VisionCaption, VisionObject

from app.caption_gate import CaptionGate
from app.main import WindowService

TONIGHT = "A camera is placed on a wooden bench in a garden"
SKELETON = [2060.0, 877.0, 2274.0, 1227.0]
WALKER = [400.0, 900.0, 520.0, 1400.0]


def _gate(**kw) -> CaptionGate:
    base = dict(
        streams="walkway",
        labels="person,dog,bicycle,stroller,vehicle,package,mail truck",
        min_score=0.25,
        max_age_sec=20.0,
        stationary_min_sec=1800.0,
        stationary_iou=0.5,
        stationary_max_gap_sec=300.0,
    )
    base.update(kw)
    return CaptionGate(**base)


def _detect(objects, *, stream="walkway", aid="det") -> VisionArtifactPayload:
    return VisionArtifactPayload(
        artifact_id=aid,
        correlation_id="c",
        task_type="retina_fast",
        device="cuda:0",
        inputs={"stream_id": stream},
        outputs=VisionArtifactOutputs(objects=[VisionObject(label=l, score=s, box_xyxy=b) for l, s, b in objects]),
        timing={},
        model_fingerprints={},
    )


def _caption(text, *, source=None, stream="walkway", frame_ts=None, aid="cap", task_type="caption_frame",
             rejected_reason=None, objects=None) -> VisionArtifactPayload:
    inputs = {"stream_id": stream}
    if source is not None:
        inputs["source_objects"] = source
    if frame_ts is not None:
        inputs["frame_ts"] = frame_ts
    return VisionArtifactPayload(
        artifact_id=aid,
        correlation_id="c",
        task_type=task_type,
        device="cuda:0",
        inputs=inputs,
        outputs=VisionArtifactOutputs(
            objects=objects,
            caption=VisionCaption(text=text, confidence=0.0 if rejected_reason else 1.0, rejected_reason=rejected_reason),
        ),
        timing={},
        model_fingerprints={},
    )


def _src(label, box, score=0.6):
    return {"label": label, "score": score, "box_xyxy": box}


def test_tonights_blip_caption_on_an_empty_dark_frame_is_dropped():
    """No detections in the frame + BLIP's invented bench -> rejected."""
    gate = _gate()
    now = time.time()
    blip = _caption(TONIGHT, task_type="retina_fast", objects=[])  # BLIP rides on the detect artifact
    items, stats, rejections = gate.apply("walkway", [(blip, now)])
    cap = items[0][0].outputs.caption
    assert cap.text == "" and cap.confidence == 0.0
    assert cap.rejected_reason == "gate:no_source_detections"
    assert stats == {"accepted": 0, "rejected": {"no_source_detections": 1}}
    assert rejections[0]["text"] == TONIGHT


def test_vlm_caption_with_no_gate_detection_is_dropped():
    gate = _gate()
    now = time.time()
    art = _caption(TONIGHT, source=[_src("chair", [0, 0, 10, 10], 0.9), _src("person", WALKER, 0.1)], frame_ts=now)
    items, stats, _ = gate.apply("walkway", [(art, now)])
    assert items[0][0].outputs.caption.rejected_reason == "gate:no_gate_detection"
    assert stats["rejected"] == {"no_gate_detection": 1}


def test_moving_person_caption_is_kept():
    gate = _gate()
    now = time.time()
    text = "A man walks a dog along the sidewalk."
    art = _caption(text, source=[_src("person", WALKER), _src("dog", [600, 1200, 700, 1400])], frame_ts=now - 3)
    items, stats, rejections = gate.apply("walkway", [(_detect([("person", 0.6, WALKER)]), now - 3), (art, now)])
    assert items[1][0].outputs.caption.text == text
    assert stats == {"accepted": 1, "rejected": {}}
    assert rejections == []


def test_caption_of_a_stationary_prop_only_is_dropped():
    """The patio skeleton, detected as person in the same box for 35 min."""
    gate = _gate()
    t0 = time.time() - 2400
    for k in range(int(35 * 3)):
        gate.apply("walkway", [(_detect([("person", 0.37, SKELETON)]), t0 + 20.0 * k)])
    now = t0 + 20.0 * 105
    art = _caption("Two skeletons sit at a table on the patio.", source=[_src("person", SKELETON, 0.37)], frame_ts=now - 1)
    items, stats, _ = gate.apply("walkway", [(_detect([("person", 0.37, SKELETON)]), now - 1), (art, now)])
    assert items[1][0].outputs.caption.rejected_reason == "gate:stationary_only"

    # A real walker next to the prop: the caption stands.
    art2 = _caption("A woman walks past the patio.", source=[_src("person", SKELETON, 0.37), _src("person", WALKER)],
                    frame_ts=now)
    items, stats, _ = gate.apply("walkway", [(_detect([("person", 0.37, SKELETON), ("person", 0.7, WALKER)]), now), (art2, now + 1)])
    assert items[1][0].outputs.caption.text == "A woman walks past the patio."


def test_stale_caption_is_dropped():
    gate = _gate()
    now = time.time()
    art = _caption("A man walks by.", source=[_src("person", WALKER)], frame_ts=now - 45)
    items, stats, _ = gate.apply("walkway", [(art, now)])
    assert stats["rejected"] == {"stale": 1}


def test_host_null_answer_is_counted_not_narrated():
    gate = _gate()
    now = time.time()
    art = _caption("", source=[_src("person", WALKER)], frame_ts=now, rejected_reason="null_answer")
    items, stats, _ = gate.apply("walkway", [(art, now)])
    assert stats == {"accepted": 0, "rejected": {"host:null_answer": 1}}
    assert items[0][0].outputs.caption.text == ""


def test_ungated_stream_is_untouched():
    gate = _gate()
    art = _caption(TONIGHT, task_type="retina_fast", objects=[], stream="cam0")
    items, stats, rejections = gate.apply("cam0", [(art, time.time())])
    assert items[0][0] is art and stats is None and rejections == []


def test_empty_gate_streams_disable_it():
    gate = _gate(streams="")
    art = _caption(TONIGHT, task_type="retina_fast", objects=[])
    assert gate.apply("walkway", [(art, time.time())])[1] is None


@pytest.mark.asyncio
async def test_flush_keeps_rejected_caption_out_of_summary_and_counts_it():
    """End to end through the window flush: tonight's caption never reaches
    summary.captions (what the council narrates), and the rejection is counted."""
    svc = WindowService()
    now = time.time()
    buffered = [{"artifact": _caption(TONIGHT, task_type="retina_fast", objects=[]), "ts": now, "env": None}]
    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        await svc._flush_and_publish(stream_id="walkway", buffered=buffered, correlation_id=None, causality_chain=[])
    summary = svc._live_by_stream["walkway"].summary
    assert summary["captions"] == []
    assert summary["evidence"]["caption_count"] == 0
    assert summary["caption_gate"] == {"accepted": 0, "rejected": {"no_source_detections": 1}}
    assert TONIGHT not in str(summary)
    assert svc._m_caption_gate_rejected == {"no_source_detections": 1}


@pytest.mark.asyncio
async def test_flush_keeps_agreeing_vlm_caption():
    svc = WindowService()
    now = time.time()
    text = "A man walks a dog along the sidewalk."
    buffered = [
        {"artifact": _detect([("person", 0.6, WALKER)]), "ts": now - 2, "env": None},
        {"artifact": _caption(text, source=[_src("person", WALKER)], frame_ts=now - 2), "ts": now, "env": None},
    ]
    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        await svc._flush_and_publish(stream_id="walkway", buffered=buffered, correlation_id=None, causality_chain=[])
    summary = svc._live_by_stream["walkway"].summary
    assert summary["captions"] == [text]
    assert summary["caption_gate"] == {"accepted": 1, "rejected": {}}
    assert svc._m_caption_gate_accepted == 1
