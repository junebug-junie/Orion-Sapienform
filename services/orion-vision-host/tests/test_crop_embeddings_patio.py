"""The patio rule: a box in a no-embed zone never reaches the embedder.

docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md idea 1
privacy boundary + acceptance check 6. The embedder is mocked; what is under
test is which pixels are handed to it and which objects come back with a
vector.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from PIL import Image

from app.artifacts import build_artifact_payload
from app.crop_embeddings import attach_crop_embeddings, load_zones_fail_closed
from app.models import VisionResult, VisionTask
from app.runner import VisionRunner
from orion.vision.zones import Zone

W, H = 1000, 1000
PATIO = Zone(name="patio", polygon=((0.0, 0.7), (0.35, 0.7), (0.35, 1.0), (0.0, 1.0)), embed=False)
WALKWAY = Zone(name="walkway", polygon=((0.0, 0.35), (1.0, 0.35), (1.0, 1.0), (0.0, 1.0)), embed=True)
ZONES = {"walkway": [PATIO, WALKWAY]}

# bottom-center (150, 950) -> patio; bottom-center (650, 900) -> walkway
PATIO_BOX = [100.0, 700.0, 200.0, 950.0]
WALKWAY_BOX = [600.0, 600.0, 700.0, 900.0]


class _SpyEmbedder:
    def __init__(self, dim: int = 4) -> None:
        self.calls: list[list[tuple[int, int]]] = []
        self.dim = dim

    def __call__(self, crops):
        self.calls.append([c.size for c in crops])
        return np.ones((len(crops), self.dim), dtype=np.float32) * 3.0


def _objects():
    return [
        {"label": "person", "score": 0.9, "box_xyxy": list(PATIO_BOX)},
        {"label": "person", "score": 0.8, "box_xyxy": list(WALKWAY_BOX)},
        {"label": "chair", "score": 0.9, "box_xyxy": [600.0, 600.0, 700.0, 900.0]},
    ]


def _run(objects, *, zones=ZONES, stream_id="walkway", embedder=None):
    embedder = embedder or _SpyEmbedder()
    stats = attach_crop_embeddings(
        objects,
        Image.new("RGB", (W, H)),
        request={"want_crop_embeddings": True, "stream_id": stream_id},
        params={},
        embed_fn=embedder,
        zones_by_stream=zones,
        model_id="siglip-test",
        embed_profile="embed_image",
        min_score=0.35,
    )
    return stats, embedder


def test_patio_box_never_embedded_even_when_requested() -> None:
    objs = _objects()
    stats, spy = _run(objs)
    patio, walk, chair = objs
    assert patio["zone"] == "patio"
    assert patio["embedding"] is None and patio["embedding_ref"] is None
    assert walk["zone"] == "walkway"
    assert walk["embedding"] is not None and walk["embedding_ref"].startswith("crop:embed_image:")
    # L2-normalized
    assert abs(float(np.linalg.norm(walk["embedding"])) - 1.0) < 1e-5
    # untracked label untouched
    assert "zone" not in chair and "embedding" not in chair
    # exactly one crop, the walkway one (100x300), ever reached the embedder
    assert spy.calls == [[(100, 300)]]
    assert stats["embedded"] == 1 and stats["withheld_no_embed_zone"] == 1


def test_patio_only_frame_never_calls_embedder() -> None:
    objs = [{"label": "person", "score": 0.95, "box_xyxy": list(PATIO_BOX)}]
    stats, spy = _run(objs)
    assert spy.calls == []
    assert objs[0]["zone"] == "patio" and objs[0]["embedding"] is None


def test_missing_zones_file_fails_closed(tmp_path: Path) -> None:
    assert load_zones_fail_closed(tmp_path / "nope.yaml") is None
    objs = _objects()
    stats, spy = _run(objs, zones=None)
    assert spy.calls == []
    assert all(o.get("embedding") is None for o in objs)


def test_stream_without_zones_embeds_tracked_boxes() -> None:
    objs = _objects()
    stats, spy = _run(objs, stream_id="cam0")
    assert objs[0]["zone"] is None and objs[0]["embedding"] is not None
    assert len(spy.calls) == 1 and len(spy.calls[0]) == 2  # one batched pass


def test_repo_zones_config_marks_patio_no_embed() -> None:
    zones = load_zones_fail_closed(Path(__file__).resolve().parents[3] / "config" / "vision_zones.yaml")
    assert zones is not None
    patio = [z for z in zones["walkway"] if z.name == "patio"]
    assert patio and patio[0].embed is False


def test_artifact_payload_carries_crop_fields_and_patio_stays_vectorless() -> None:
    objs = _objects()
    _run(objs)
    res = VisionResult(corr_id="c", task_type="retina_fast", device="cpu", artifacts={"objects": objs, "frame_width": W})
    art = build_artifact_payload(res)
    by_zone = {o.zone: o for o in art.outputs.objects if o.zone}
    assert by_zone["patio"].embedding is None
    assert by_zone["walkway"].embedding is not None
    assert art.outputs.frame_width == W


def test_runner_copies_stream_id_from_meta_into_request() -> None:
    task = VisionTask(corr_id="c", reply_channel="r", task_type="retina_fast",
                      request={"image_path": "/x.jpg"}, meta={"stream_id": "walkway"})
    req = VisionRunner._request_with_stream_id(task)
    assert req["stream_id"] == "walkway"
    assert "stream_id" not in task.request
