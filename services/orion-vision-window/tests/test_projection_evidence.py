import time

from orion.schemas.vision import VisionArtifactOutputs, VisionArtifactPayload, VisionCaption, VisionObject

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


def test_evidence_counts_edge_and_host_person() -> None:
    items = [
        (_art("edge_detection", "person", 0.9), time.time()),
        (_art("retina_fast", "door", 0.8), time.time()),
        (_art("retina_fast", "person", 0.7), time.time()),
        (_art("retina_fast", "screen", 0.6, caption="describe this image. youtube"), time.time()),
    ]
    summary = summarize_items(items)
    ev = summary["evidence"]
    assert ev["edge_person_hits"] == 1
    assert ev["host_person_hits"] == 1
    assert "person" in ev["hard_labels"]
    assert "door" in ev["hard_labels"]
    assert "youtube" in ev["soft_labels"]
    assert ev["caption_count"] == 1
