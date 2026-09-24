import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.schemas.vision import VisionArtifactOutputs, VisionArtifactPayload, VisionCaption, VisionObject
from orion.vision.caption_echo import CAPTION_PROMPT

from app.projection import summarize_items


def _art(task_type: str, label: str, score: float, caption: str | None = None) -> VisionArtifactPayload:
    cap = VisionCaption(text=caption, confidence=0.5) if caption else None
    return VisionArtifactPayload(
        artifact_id=f"{task_type}-{label}",
        correlation_id="c1",
        task_type=task_type,
        device="cuda:0",
        inputs={"stream_id": "cam0"},
        outputs=VisionArtifactOutputs(
            objects=[VisionObject(label=label, score=score, box_xyxy=[0, 0, 1, 1])],
            caption=cap,
        ),
        timing={},
        model_fingerprints={},
    )


def test_evidence_ignores_edge_detection_artifacts() -> None:
    items = [
        (_art("edge_detection", "person", 0.9), time.time()),
        (_art("retina_fast", "door", 0.8), time.time()),
        (_art("retina_fast", "person", 0.7), time.time()),
    ]
    summary = summarize_items(items)
    ev = summary["evidence"]
    assert ev["edge_person_hits"] == 0
    assert ev["host_person_hits"] == 1
    assert "person" in ev["hard_labels"]
    assert "door" in ev["hard_labels"]


def test_summarize_items_excludes_edge_detection_from_object_counts() -> None:
    items = [(_art("edge_detection", "person", 0.9), time.time())]
    summary = summarize_items(items)
    assert summary["object_counts"] == {}
    assert summary["item_count"] == 1


def test_summarize_items_skips_caption_prompt_echo() -> None:
    items = [
        (_art("retina_fast", "door", 0.8, caption=CAPTION_PROMPT), time.time()),
        (_art("retina_fast", "person", 0.7, caption="A desk with two monitors."), time.time()),
    ]
    summary = summarize_items(items)
    assert summary["captions"] == ["A desk with two monitors."]


def test_unnamed_boxes_carry_geometry_for_the_council_zone_check() -> None:
    art = VisionArtifactPayload(
        artifact_id="a1", correlation_id="c1", task_type="retina_fast", device="cuda:0",
        inputs={"stream_id": "walkway"},
        outputs=VisionArtifactOutputs(
            objects=[VisionObject(label="", score=0.4, box_xyxy=[10, 20, 30, 40]),
                     VisionObject(label="object", score=0.4, box_xyxy=[1, 2, 3, 4]),
                     VisionObject(label="person", score=0.9, box_xyxy=[5, 6, 7, 8])],
            frame_width=640, frame_height=360,
        ),
        timing={}, model_fingerprints={},
    )
    boxes = summarize_items([(art, time.time())])["unnamed_boxes"]
    assert boxes == [
        {"frame": "a1", "box_xyxy": [10.0, 20.0, 30.0, 40.0], "frame_width": 640, "frame_height": 360},
        {"frame": "a1", "box_xyxy": [1.0, 2.0, 3.0, 4.0], "frame_width": 640, "frame_height": 360},
    ]  # named boxes are not forwarded
