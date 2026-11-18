"""OSNet shadow body vector (2026-10-16): stored next to SigLIP, never used to match.

What is under test: which crops get an OSNet vector, that an OSNet failure never
costs the SigLIP vector or the detection, that every embedded crop names the
body model that ACTUALLY made its vector, and that /health and /ready say when
the body embedder fell back or OSNet cannot load. The real-weights test runs
inside the built image (the weights are baked in at build) and skips elsewhere.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pytest
from PIL import Image

from app import osnet_shadow as osn
from app.artifacts import build_artifact_payload
from app.crop_embeddings import attach_crop_embeddings
from app.models import VisionResult
from orion.vision.zones import Zone

SERVICE = Path(__file__).resolve().parents[1]
W, H = 1000, 1000
WALKWAY = Zone(name="walkway", polygon=((0.0, 0.35), (1.0, 0.35), (1.0, 1.0), (0.0, 1.0)), embed=True)
ZONES = {"walkway": [WALKWAY]}
PERSON_BOX = [600.0, 600.0, 700.0, 900.0]
PERSON2_BOX = [200.0, 500.0, 300.0, 800.0]
DOG_BOX = [800.0, 700.0, 900.0, 800.0]
BODY = "google/siglip-so400m-patch14-384"


def _siglip(crops):
    return np.ones((len(crops), 4), dtype=np.float32)


class _SpyOsnet:
    def __init__(self, *, fail=False, rows=None):
        self.calls: list[int] = []
        self.fail = fail
        self.rows = rows

    def __call__(self, crops):
        self.calls.append(len(crops))
        if self.fail:
            raise RuntimeError("cuda oom")
        n = len(crops) if self.rows is None else self.rows
        return np.full((n, 512), 3.0, dtype=np.float32)


def _run(objects, *, osnet=None, stream_id="walkway"):
    request = {"want_crop_embeddings": True, "stream_id": stream_id, "dispatch_tier": "triggered"}
    return attach_crop_embeddings(
        objects, Image.new("RGB", (W, H)), request=request, params={}, embed_fn=_siglip,
        zones_by_stream=ZONES, model_id="google/siglip2-so400m-patch14-384", embed_profile="embed_image",
        min_score=0.35, body_model_id=BODY, osnet_fn=osnet, osnet_model_id=osn.MODEL_ID,
    )


def _objs():
    return [
        {"label": "person", "score": 0.9, "box_xyxy": list(PERSON_BOX)},
        {"label": "dog", "score": 0.9, "box_xyxy": list(DOG_BOX)},
        {"label": "person", "score": 0.8, "box_xyxy": list(PERSON2_BOX)},
    ]


# --- which crops ------------------------------------------------------------------------------


def test_osnet_vector_on_every_embedded_person_crop_only() -> None:
    spy = _SpyOsnet()
    objs = _objs()
    stats = _run(objs, osnet=spy)
    person, dog, person2 = objs
    assert spy.calls == [2]                             # one batched pass, persons only
    for o in (person, person2):
        v = np.asarray(o["osnet_embedding"])
        assert v.shape == (512,) and abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
        assert o["osnet_model_id"] == osn.MODEL_ID
        assert o["embedding"] is not None               # the SigLIP vector is still there
    assert dog["embedding"] is not None and dog["osnet_embedding"] is None and dog["osnet_model_id"] is None
    assert stats["osnet"] == 2 and "osnet_error" not in stats


def test_every_embedded_crop_names_the_loaded_body_model_not_the_requested_one() -> None:
    objs = _objs()
    _run(objs, osnet=None)
    assert all(o["embedding_model_id"] == BODY for o in objs)


def test_no_zones_means_no_crop_and_no_osnet() -> None:
    spy = _SpyOsnet()
    objs = _objs()
    _run(objs, osnet=spy, stream_id="cam0")             # no zones on file: fail closed
    assert spy.calls == []
    assert all(o.get("osnet_embedding") is None and o["embedding"] is None for o in objs)


def test_kill_switch_off_means_no_osnet_keys_at_all() -> None:
    objs = _objs()
    stats = _run(objs, osnet=None)
    assert all("osnet_embedding" not in o for o in objs)
    assert stats["osnet"] == 0


# --- failure never costs the body vector ------------------------------------------------------


def test_osnet_failure_keeps_siglip_vectors() -> None:
    objs = _objs()
    stats = _run(objs, osnet=_SpyOsnet(fail=True))
    assert all(o["embedding"] is not None for o in objs)
    assert all(o["osnet_embedding"] is None for o in objs)
    assert stats["osnet_error"] == 1 and stats["embedded"] == 3


def test_osnet_wrong_row_count_drops_osnet_only() -> None:
    objs = _objs()
    stats = _run(objs, osnet=_SpyOsnet(rows=1))
    assert all(o["osnet_embedding"] is None for o in objs) and stats["osnet_error"] == 1
    assert all(o["embedding"] is not None for o in objs)


def test_artifact_payload_carries_osnet_and_body_model() -> None:
    objs = _objs()
    _run(objs, osnet=_SpyOsnet())
    res = VisionResult(corr_id="c", task_type="retina_fast", device="cpu",
                       artifacts={"objects": objs, "frame_width": W})
    person = build_artifact_payload(res).outputs.objects[0]
    assert len(person.osnet_embedding) == 512 and person.osnet_model_id == osn.MODEL_ID
    assert person.embedding_model_id == BODY


# --- OsnetShadow: load failure is contained ---------------------------------------------------


def test_load_failure_returns_no_fn_and_retries_after_the_window() -> None:
    now = [0.0]
    calls = []

    def bad_loader(path, device):
        calls.append(device)
        raise FileNotFoundError("no weights")

    sh = osn.OsnetShadow(enabled=True, weights_path="/nope.pt", loader=bad_loader, clock=lambda: now[0])
    assert sh.embed_fn("cuda:0") is None
    assert sh.embed_fn("cuda:0") is None                # inside the retry window: no second load
    assert calls == ["cuda:0"]
    st = sh.status()
    assert st["state"] == "failed" and "no weights" in st["last_error"] and st["failures"] == 1
    now[0] = osn.RETRY_AFTER_SEC + 1
    assert sh.embed_fn("cuda:0") is None and calls == ["cuda:0", "cuda:0"]


def test_disabled_never_loads() -> None:
    def loader(path, device):  # pragma: no cover - must not run
        raise AssertionError("loaded while disabled")

    sh = osn.OsnetShadow(enabled=False, weights_path="/x.pt", loader=loader)
    assert sh.embed_fn("cuda:0") is None and sh.warm("cuda:0") is False
    assert sh.status()["state"] == "disabled"


def test_embed_counts_vectors_and_errors() -> None:
    import torch

    class _Net(torch.nn.Module):
        def forward(self, x):
            return torch.ones((x.shape[0], 512))

    sh = osn.OsnetShadow(enabled=True, weights_path="/x.pt", loader=lambda p, d: _Net())
    fn = sh.embed_fn("cpu")
    out = fn([Image.new("RGB", (40, 100)), Image.new("RGB", (50, 90))])
    assert out.shape == (2, 512)
    assert sh.status()["state"] == "loaded" and sh.status()["vectors"] == 2


def test_preprocess_is_torchreid_test_transform() -> None:
    t = osn.preprocess([Image.new("RGB", (37, 91), (255, 0, 0))])
    assert tuple(t.shape) == (1, 3, 256, 128)
    # pure red after ImageNet normalisation
    assert abs(float(t[0, 0, 0, 0]) - (1.0 - 0.485) / 0.229) < 1e-4
    assert abs(float(t[0, 1, 0, 0]) - (0.0 - 0.456) / 0.224) < 1e-4


def test_load_refuses_a_wrong_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        osn.load_osnet(tmp_path / "missing.pt", "cpu")
    p = tmp_path / "bad.pt"
    p.write_bytes(b"not the weights")
    with pytest.raises(ValueError, match="sha256"):
        osn.load_osnet(p, "cpu")


@pytest.mark.skipif(not Path(osn.DEFAULT_WEIGHTS_PATH).is_file(), reason="weights are baked into the image only")
def test_real_weights_load_strictly_and_embed() -> None:
    model = osn.load_osnet(Path(osn.DEFAULT_WEIGHTS_PATH), "cpu")
    sh = osn.OsnetShadow(enabled=True, weights_path=osn.DEFAULT_WEIGHTS_PATH, loader=lambda p, d: model)
    fn = sh.embed_fn("cpu")
    crops = [Image.new("RGB", (60, 150), (200, 30, 30)), Image.new("RGB", (60, 150), (30, 30, 200))]
    a, b = fn(crops), fn(crops)
    assert a.shape == (2, 512) and np.isfinite(a).all()
    assert float(np.abs(a - b).max()) == 0.0            # eval mode: deterministic
    assert float(np.linalg.norm(a[0] - a[1])) > 0.0     # not a constant output


# --- the pinned weights: one hash everywhere --------------------------------------------------


def test_dockerfile_pins_the_same_weights_as_the_loader() -> None:
    df = (SERVICE / "Dockerfile").read_text()
    assert f"ARG OSNET_WEIGHTS_SHA256={osn.WEIGHTS_SHA256}" in df
    url = re.search(r"ARG OSNET_WEIGHTS_URL=(\S+)", df).group(1)
    assert f"/{osn.WEIGHTS_REPO}/resolve/{osn.WEIGHTS_REVISION}/{osn.WEIGHTS_FILE}" in url
    assert osn.DEFAULT_WEIGHTS_PATH in df and "sha256sum -c" in df
    assert osn.MODEL_ID.endswith(osn.WEIGHTS_SHA256[:12])


def test_shadow_ships_on_everywhere() -> None:
    from app.settings import Settings

    assert Settings().VISION_OSNET_SHADOW_ENABLED is True
    assert "VISION_OSNET_SHADOW_ENABLED=true" in (SERVICE / ".env_example").read_text()
    assert "VISION_OSNET_SHADOW_ENABLED=${VISION_OSNET_SHADOW_ENABLED:-true}" in (
        SERVICE / "docker-compose.yml").read_text()


# --- the actually-loaded body model is visible ------------------------------------------------


def test_siglip_fallback_is_recorded_as_the_loaded_model(monkeypatch) -> None:
    import transformers

    from app.model_manager import ModelManager

    class _M:
        def eval(self):
            return self

    def proc(mid, **kw):
        if "siglip2" in mid:
            raise ValueError("GemmaTokenizer mismatch")
        return object()

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(proc))
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", staticmethod(lambda mid, **kw: _M()))
    mm = ModelManager()
    mm.load_siglip_image_embedder(profile_name="embed_image", device="cpu", dtype="auto",
                                  model_id="google/siglip2-so400m-patch14-384", fallback_model_id=BODY)
    assert mm.loaded_embedder_id("embed_image", "cpu") == BODY
    (st,) = mm.embedder_status.values()
    assert st["fallback"] is True and st["requested"] == "google/siglip2-so400m-patch14-384"
    assert "GemmaTokenizer" in st["fallback_error"]


def test_degraded_reasons_name_the_fallback_and_osnet_failure() -> None:
    from app.main import model_degraded_reasons

    models = {
        "body_embedder": {
            "embed_image@cuda:0": {"requested": "a", "loaded": BODY, "fallback": True},
            "embed_image@cuda:1": {"requested": "a", "loaded": BODY, "fallback": True},
        },
        "osnet_shadow": {"state": "failed"},
    }
    assert model_degraded_reasons(models) == [f"body_embedder_fallback:{BODY}", "osnet_shadow_failed"]
    assert model_degraded_reasons({"body_embedder": {}, "osnet_shadow": {"state": "loaded"}}) == []


# --- review fixes (2026-10-16) ----------------------------------------------------------------


class _RowsOsnet:
    def __init__(self, rows):
        self.rows = rows

    def __call__(self, crops):
        return np.asarray(self.rows[: len(crops)], dtype=np.float32)


def test_a_nan_row_loses_only_its_own_osnet_vector() -> None:
    objs = _objs()  # persons at rows 0 and 2 of the embed list
    stats = _run(objs, osnet=_RowsOsnet([[float("nan")] * 512, [1.0] * 512]))
    person, dog, person2 = objs
    assert person["osnet_embedding"] is None and person["embedding"] is not None
    assert person2["osnet_embedding"] is not None and stats["osnet"] == 1


def test_wrong_dimension_is_not_stored_as_osnet() -> None:
    objs = _objs()
    stats = _run(objs, osnet=_RowsOsnet([[1.0] * 256, [1.0] * 256]))
    assert all(o["osnet_embedding"] is None for o in objs) and stats["osnet_error"] == 1
    assert all(o["embedding"] is not None for o in objs)


def test_osnet_dim_constant_matches_the_model() -> None:
    from app.crop_embeddings import OSNET_DIM

    assert OSNET_DIM == osn.FEATURE_DIM


def test_body_model_id_is_read_after_the_embedder_ran() -> None:
    seen = []

    def embed(crops):
        seen.append("embedded")
        return np.ones((len(crops), 4), dtype=np.float32)

    def body_id():
        seen.append("asked")
        return BODY

    objs = [{"label": "person", "score": 0.9, "box_xyxy": list(PERSON_BOX)}]
    attach_crop_embeddings(objs, Image.new("RGB", (W, H)), request={"want_crop_embeddings": True,
                           "stream_id": "walkway"}, params={}, embed_fn=embed, zones_by_stream=ZONES,
                           model_id="m", embed_profile="embed_image", min_score=0.35, body_model_id=body_id)
    assert seen == ["embedded", "asked"] and objs[0]["embedding_model_id"] == BODY
    # Nothing to embed: the embedder is never asked, and neither is the id.
    seen.clear()
    _ = attach_crop_embeddings([{"label": "person", "score": 0.9, "box_xyxy": list(PERSON_BOX)}],
                               Image.new("RGB", (W, H)), request={"want_crop_embeddings": True,
                               "stream_id": "cam0"}, params={}, embed_fn=embed, zones_by_stream=ZONES,
                               model_id="m", embed_profile="embed_image", min_score=0.35, body_model_id=body_id)
    assert seen == []


def test_ready_stays_true_when_osnet_fails_and_body_falls_back(monkeypatch) -> None:
    from app import main as host_main

    class _Sched:
        def can_pick_gpu(self):
            return True

        def queue_depth(self):
            return 0

    class _Runner:
        warm_errors: dict = {}

        def model_status(self):
            return {"body_embedder": {"embed_image@cuda:0": {"requested": "x", "loaded": BODY, "fallback": True}},
                    "osnet_shadow": {"state": "failed", "last_error": "load: no weights"}}

    svc = host_main.VisionHostService.__new__(host_main.VisionHostService)
    svc.profiles, svc.bus, svc.sched, svc.runner = object(), None, _Sched(), _Runner()
    monkeypatch.setattr(host_main.settings, "ORION_BUS_ENABLED", False)
    body = svc.readiness_payload()
    assert body["ready"] is True
    assert f"body_embedder_fallback:{BODY}" in body["degraded_reasons"]
    assert "osnet_shadow_failed" in body["degraded_reasons"]
    assert body["models"]["osnet_shadow"]["state"] == "failed"
