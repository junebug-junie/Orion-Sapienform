"""Router/host/retina side of the content-addressed frame path.

Lives here rather than in the repo-level tests/ because it imports the router's
own `app` package. Every service ships an `app`, so a repo-level test that
inserts a service dir on sys.path picks up whichever one was imported first --
it passes alone and fails in a broad run. Verified: that is exactly what
happened to the first version of this file.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "services" / "orion-vision-frame-router"))

from orion.schemas.vision import VisionFramePointerPayload  # noqa: E402

_SHA = "a" * 64


# -- the router -------------------------------------------------------------


def _router_policy():
    """Real FrameDispatchPolicy against the real config file."""
    import yaml

    sys.path.insert(0, str(_REPO / "services" / "orion-vision-frame-router"))
    from app.policy import FrameDispatchPolicy
    from app.settings import Settings

    raw = yaml.safe_load((_REPO / "config" / "vision_frame_router.yaml").read_text())
    return FrameDispatchPolicy(settings=Settings(), raw=raw)


def _envelope(payload: VisionFramePointerPayload):
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    return BaseEnvelope(
        kind="vision.frame.pointer",
        source=ServiceRef(name="test", version="0"),
        payload=payload.model_dump(mode="json"),
    )


def test_router_dispatches_a_content_addressed_frame() -> None:
    """Behavioural, against the real policy and the real config.

    The path checks must not apply to a frame that has no path -- applying them
    would reject every frame from every node that does not share athena's disk,
    which is the entire reason sha256 exists.
    """
    sys.path.insert(0, str(_REPO / "services" / "orion-vision-frame-router"))
    from app.state import RouterState

    policy = _router_policy()
    frame = VisionFramePointerPayload(
        sha256=_SHA, camera_id="cam0", stream_id="cam0", frame_ts=1.0
    )
    decision = policy.decide(_envelope(frame), RouterState(), now=1.0)
    assert decision.reason not in ("missing_image_path", "missing_frame_address",
                                   "image_path_not_visible"), \
        f"content-addressed frame rejected for lack of a path: {decision.reason}"


def test_router_still_rejects_a_frame_with_an_unreadable_path() -> None:
    """The guard must survive: a local frame whose file is gone is still bad."""
    sys.path.insert(0, str(_REPO / "services" / "orion-vision-frame-router"))
    from app.state import RouterState

    policy = _router_policy()
    frame = VisionFramePointerPayload(
        image_path="/definitely/not/here.jpg", camera_id="cam0", stream_id="cam0", frame_ts=1.0
    )
    decision = policy.decide(_envelope(frame), RouterState(), now=1.0, image_path_exists=False)
    assert not decision.should_dispatch
    assert decision.reason == "image_path_not_visible"


def test_router_builds_a_host_request_carrying_the_content_address() -> None:
    sys.path.insert(0, str(_REPO / "services" / "orion-vision-frame-router"))
    from app.state import RouterState

    policy = _router_policy()
    frame = VisionFramePointerPayload(
        sha256=_SHA, camera_id="cam0", stream_id="cam0", frame_ts=1.0
    )
    env = _envelope(frame)
    decision = policy.decide(env, RouterState(), now=1.0)
    task = policy.build_task_request(frame, env, decision)
    assert task.request.get("percept_sha256") == _SHA
    assert "image_path" not in task.request, "empty path forwarded alongside the address"


def test_router_forwards_the_content_address_to_the_host() -> None:
    src = (_REPO / "services" / "orion-vision-frame-router" / "app" / "policy.py").read_text()
    build = src[src.index("def build_task_request"):]
    assert '"percept_sha256"' in build, "sha256 not forwarded into the host request"
    assert '"image_path"' in build, "local path route dropped"


# -- the host ---------------------------------------------------------------


def test_host_resolves_a_content_address_and_validates_it() -> None:
    src = (_REPO / "services" / "orion-vision-host" / "app" / "runner.py").read_text()
    assert "_load_image_from_percept_store" in src
    assert "percept_sha256" in src
    assert "_SHA256_RE" in src, "digest must be validated before it becomes a URL"
    assert "VISION_PERCEPT_STORE_URL" in src


def test_host_still_requires_an_address() -> None:
    """Removing the guard entirely would let an addressless request through."""
    src = (_REPO / "services" / "orion-vision-host" / "app" / "runner.py").read_text()
    assert "image_path or percept_sha256" in src


# -- retina -----------------------------------------------------------------


def test_retina_upload_mode_writes_nothing_locally() -> None:
    """A capture agent on a personal laptop must not accumulate a spool of
    webcam frames. Verified structurally: the upload path uses imencode (memory)
    and never imwrite (disk)."""
    src = (_REPO / "services" / "orion-vision-retina" / "app" / "frame_store.py").read_text()
    upload = src[src.index("def upload_frame"):]
    assert "cv2.imencode" in upload
    assert "imwrite" not in upload, "upload path writes to disk"
    assert "hashlib.sha256" in upload, "must verify the store agrees on the address"


def test_retina_drops_rather_than_spools_on_upload_failure() -> None:
    src = (_REPO / "services" / "orion-vision-retina" / "app" / "main.py").read_text()
    assert "PerceptUploadError" in src
    assert "RETINA_FRAME_MODE" in src
    assert "queue" not in src.lower().split("percept_store")[1][:2000], "spooling on failure"
