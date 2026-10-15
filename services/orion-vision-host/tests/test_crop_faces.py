"""Walkway faces: a face vector rides on embedded person crops, nowhere else.

2026-10-15 (Juniper): face recognition for anyone on the walkway. The face
models are mocked; what is under test is which crops reach them and which
objects come back with a face.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from PIL import Image

from app.artifacts import build_artifact_payload
from app.crop_embeddings import ThumbRateLimiter, attach_crop_embeddings, crop_faces
from app.models import VisionResult
from orion.vision.zones import Zone

W, H = 1000, 1000
WALKWAY = Zone(name="walkway", polygon=((0.0, 0.35), (1.0, 0.35), (1.0, 1.0), (0.0, 1.0)), embed=True)
PATIO = Zone(name="patio", polygon=((0.0, 0.7), (0.35, 0.7), (0.35, 1.0), (0.0, 1.0)), embed=True, presence=True)
ZONES = {"walkway": [PATIO, WALKWAY]}

PERSON_BOX = [600.0, 600.0, 700.0, 900.0]
PATIO_PERSON_BOX = [100.0, 700.0, 200.0, 950.0]
DOG_BOX = [800.0, 700.0, 900.0, 800.0]


def _embed(crops):
    return np.ones((len(crops), 4), dtype=np.float32)


class _SpyFaces:
    def __init__(self, hit: bool = True) -> None:
        self.calls: list[list[tuple[int, int]]] = []
        self.hit = hit

    def __call__(self, crops):
        self.calls.append([c.size for c in crops])
        if not self.hit:
            return [None for _ in crops]
        return [([2.0] * 512, 0.99, Image.new("RGB", (20, 24))) for _ in crops]


def _run(objects, *, stream_id="walkway", want_faces=True, faces=None, thumbs=None, tier="triggered"):
    faces = faces if faces is not None else _SpyFaces()
    request = {"want_crop_embeddings": True, "stream_id": stream_id, "dispatch_tier": tier}
    if want_faces:
        request["want_crop_faces"] = True
    stats = attach_crop_embeddings(
        objects, Image.new("RGB", (W, H)), request=request, params={}, embed_fn=_embed,
        zones_by_stream=ZONES, model_id="m", embed_profile="embed_image", min_score=0.35,
        thumb_fn=thumbs, face_fn=faces, face_thumb_limiter=ThumbRateLimiter(0.0),
    )
    return stats, faces


def test_walkway_person_crop_gets_a_unit_face_vector_and_face_thumb() -> None:
    written = []

    def thumbs(img):
        written.append(img.size)
        return f"thumb:{len(written):064d}"

    objs = [{"label": "person", "score": 0.9, "box_xyxy": list(PERSON_BOX)},
            {"label": "dog", "score": 0.9, "box_xyxy": list(DOG_BOX)}]
    stats, spy = _run(objs, thumbs=thumbs)
    person, dog = objs
    assert spy.calls == [[(100, 300)]]  # only the person crop reached the face models
    assert len(person["face_embedding"]) == 512
    assert abs(float(np.linalg.norm(person["face_embedding"])) - 1.0) < 1e-5
    assert person["face_score"] == 0.99
    assert person["face_thumb_ref"].startswith("thumb:")
    assert (20, 24) in written  # the face crop itself, not the person crop
    assert dog["face_embedding"] is None and dog["face_thumb_ref"] is None
    assert stats["faces"] == 1 and stats["face_thumbs"] == 1


def test_patio_person_gets_a_face_too() -> None:
    objs = [{"label": "person", "score": 0.9, "box_xyxy": list(PATIO_PERSON_BOX)}]
    _run(objs)
    assert objs[0]["zone"] == "patio" and objs[0]["face_embedding"] is not None


def test_no_face_without_the_request_flag() -> None:
    objs = [{"label": "person", "score": 0.9, "box_xyxy": list(PERSON_BOX)}]
    stats, spy = _run(objs, want_faces=False)
    assert spy.calls == [] and "face_embedding" not in objs[0] and stats["faces"] == 0
    assert objs[0]["embedding"] is not None


def test_room_cameras_never_get_a_face_via_this_path() -> None:
    # cam0/carbon have no zones on file, so nothing is embedded and no crop
    # reaches the face models -- even if a request wrongly asked for faces.
    for sid in ("cam0", "carbon", ""):
        objs = [{"label": "person", "score": 0.9, "box_xyxy": list(PERSON_BOX)}]
        stats, spy = _run(objs, stream_id=sid)
        assert spy.calls == [], sid
        assert objs[0].get("face_embedding") is None and objs[0].get("embedding") is None


def test_no_confident_face_leaves_crop_vector_alone() -> None:
    objs = [{"label": "person", "score": 0.9, "box_xyxy": list(PERSON_BOX)}]
    stats, _ = _run(objs, faces=_SpyFaces(hit=False))
    assert objs[0]["embedding"] is not None and objs[0]["face_embedding"] is None and stats["faces"] == 0


def test_face_failure_costs_only_the_face() -> None:
    def boom(_crops):
        raise RuntimeError("cuda oom")

    objs = [{"label": "person", "score": 0.9, "box_xyxy": list(PERSON_BOX)}]
    stats, _ = _run(objs, faces=boom)
    assert objs[0]["embedding"] is not None and objs[0]["face_embedding"] is None
    assert stats["face_error"] == 1


def test_face_thumb_only_on_triggered_tier() -> None:
    objs = [{"label": "person", "score": 0.9, "box_xyxy": list(PERSON_BOX)}]
    _run(objs, thumbs=lambda img: "thumb:" + "a" * 64, tier="baseline")
    assert objs[0]["face_embedding"] is not None and objs[0]["face_thumb_ref"] is None


def test_artifact_payload_carries_face_fields() -> None:
    objs = [{"label": "person", "score": 0.9, "box_xyxy": list(PERSON_BOX)}]
    _run(objs, thumbs=lambda img: "thumb:" + "b" * 64)
    res = VisionResult(corr_id="c", task_type="retina_fast", device="cpu",
                       artifacts={"objects": objs, "frame_width": W})
    obj = build_artifact_payload(res).outputs.objects[0]
    assert obj.face_embedding is not None and obj.face_thumb_ref == "thumb:" + "b" * 64
    assert obj.face_score == 0.99


class _FakeMTCNN:
    def __init__(self, detections):
        self.detections = list(detections)
        self.extracted = []

    def detect(self, img):
        boxes, probs = self.detections.pop(0)
        return (None, [None]) if boxes is None else (np.asarray(boxes, dtype=np.float32), list(probs))

    def extract(self, img, boxes, _save):
        self.extracted.append([list(map(float, b)) for b in boxes])
        return torch.ones((len(boxes), 3, 160, 160))


class _FakeResnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.batches = []

    def forward(self, x):
        self.batches.append(int(x.shape[0]))
        return torch.arange(x.shape[0] * 512, dtype=torch.float32).reshape(x.shape[0], 512)


def test_crop_faces_picks_most_confident_face_and_batches() -> None:
    crops = [Image.new("RGB", (100, 300)), Image.new("RGB", (100, 300)), Image.new("RGB", (100, 300))]
    mtcnn = _FakeMTCNN([
        ([[0, 0, 30, 30], [10, 10, 50, 60]], [0.96, 0.999]),   # second face is the confident one
        (None, None),                                         # no face
        ([[0, 0, 40, 40]], [0.80]),                            # below min_prob
    ])
    model = _FakeResnet()
    out = crop_faces(crops, mtcnn=mtcnn, model=model, device="cpu", min_prob=0.95)
    assert out[1] is None and out[2] is None
    vec, prob, face = out[0]
    assert prob == 0.999 and len(vec) == 512
    assert mtcnn.extracted == [[[10.0, 10.0, 50.0, 60.0]]]
    assert model.batches == [1]
    assert face.size[0] > 40  # padded face crop, clipped to the person crop


def test_crop_faces_skips_tiny_faces() -> None:
    mtcnn = _FakeMTCNN([([[0, 0, 10, 12]], [0.999])])
    out = crop_faces([Image.new("RGB", (100, 300))], mtcnn=mtcnn, model=_FakeResnet(), device="cpu",
                     min_prob=0.9, min_side_px=20)
    assert out == [None]
