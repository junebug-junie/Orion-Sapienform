"""A frame captured on one machine must be interpretable on another.

Until `sha256` existed on `VisionFramePointerPayload`, capture published a
local file path and the router enforced `require_image_path_exists` -- so a
node with no shared filesystem physically could not feed this pipeline, no
matter how it was configured. That blocked carbon's webcam, detectors on
circe, and anything else off-athena.

athena keeps using `image_path`: same filesystem, no HTTP hop, no copy. The
content-addressed route is additive.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from orion.schemas.vision import VisionFramePointerPayload  # noqa: E402

_SHA = "a" * 64


# -- the schema -------------------------------------------------------------


def test_either_address_alone_is_a_valid_pointer() -> None:
    assert VisionFramePointerPayload(image_path="/frames/a.jpg").image_path
    assert VisionFramePointerPayload(sha256=_SHA).sha256 == _SHA


def test_a_frame_may_carry_both() -> None:
    """Uploaded AND on local disk. Consumers pick what they can reach."""
    p = VisionFramePointerPayload(image_path="/frames/a.jpg", sha256=_SHA)
    assert p.image_path and p.sha256


def test_a_pointer_with_no_address_is_rejected() -> None:
    """Without this the failure surfaces far downstream as "task produced no
    artifact", which looks identical to a detector finding nothing.

    Asserts on the MESSAGE, not just that something raised: a bare
    `pytest.raises(Exception)` here passed against a neutralised validator,
    because pydantic raises for unrelated reasons too. The message is what
    proves this specific guard fired.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="image_path, frame_paths, video_path or sha256"):
        VisionFramePointerPayload()


@pytest.mark.parametrize(
    "bad", ["A" * 64, "z" * 64, "a" * 63, "a" * 65, "../../etc/passwd", "not-a-hash"]
)
def test_sha256_must_be_64_lowercase_hex(bad) -> None:
    """It becomes part of a fetch URL downstream. Keeping it to 64 hex chars is
    what stops it being a path or an authority."""
    with pytest.raises(Exception):
        VisionFramePointerPayload(sha256=bad)


def test_existing_producers_are_unaffected() -> None:
    """Backward compatibility: every current caller sends image_path only."""
    p = VisionFramePointerPayload(
        image_path="/mnt/telemetry/vision/frames/x.jpg",
        camera_id="cam0", stream_id="cam0", frame_ts=1.0,
        width=640, height=480, format="jpg",
    )
    assert p.sha256 is None


# Router/host/retina coverage lives with those services
# (services/orion-vision-frame-router/tests/, .../orion-vision-host/tests/):
# every service ships its own `app` package, so a repo-level test that inserts
# a service dir on sys.path picks up whichever `app` was imported first when the
# suite runs broadly. This file therefore imports `orion.schemas` and nothing
# else.
