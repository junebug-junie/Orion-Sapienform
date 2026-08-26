from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.artifacts import build_artifact_payload
from app.main import VisionHostService, should_broadcast_artifact
from app.models import VisionResult


def test_every_publish_artifact_broadcast_call_site_is_guarded():
    """Structural regression pin: found live, 2026-08-26, while verifying
    the fix in _publish_result -- the /v1/vision/task HTTP endpoint has
    its OWN, separate call to _publish_artifact_broadcast that bypassed
    should_broadcast_artifact() entirely, still broadcasting identity data
    unfiltered even after the bus-first path was fixed. Both real call
    sites (2, confirmed) must have the guard on the same line as the call
    -- a third call site added later without it would silently reopen the
    same leak this exists to prevent."""
    main_py = Path(__file__).resolve().parents[1] / "app" / "main.py"
    lines = main_py.read_text(encoding="utf-8").splitlines()

    call_sites = [i for i, line in enumerate(lines) if "await self._publish_artifact_broadcast(" in line
                  or "await service._publish_artifact_broadcast(" in line]
    definition_sites = [i for i, line in enumerate(lines) if "def _publish_artifact_broadcast(" in line]

    assert len(call_sites) == 2, (
        f"expected exactly 2 call sites to _publish_artifact_broadcast, found {len(call_sites)} at "
        f"lines {[i + 1 for i in call_sites]} -- update this test's expected count only after confirming "
        f"the new call site is guarded by should_broadcast_artifact()"
    )
    assert len(definition_sites) == 1

    for call_line in call_sites:
        # 20-line lookback, not a tight 2-line one: the two real call
        # sites have different shapes -- _publish_result's is a flat
        # `if ...: await ...`, but the HTTP /v1/vision/task endpoint's is
        # a nested `if should_broadcast_artifact(...): ... if art_payload:
        # await ...`, ~14 lines between the real guard and the call. A
        # tight window silently missed that second, nested shape when this
        # test was first written -- confirmed live by mutation below.
        guard_window = "\n".join(lines[max(0, call_line - 20) : call_line + 1])
        assert "should_broadcast_artifact(" in guard_window, (
            f"_publish_artifact_broadcast call at main.py:{call_line + 1} has no "
            f"should_broadcast_artifact() guard within 20 lines above it"
        )


def test_should_broadcast_artifact_suppresses_identity_data_reached_via_a_pipeline():
    """Review finding, 2026-08-26, second pass: a task_type-only check is
    bypassable by a config-only change -- adding `- use: identity_face` as
    a step in ANY pipeline. runner.py's _run_pipeline merges every step's
    dict output with zero content filtering, and artifacts.py's generic
    passthrough attaches the merged `identities` key onto the outer
    artifact regardless of the pipeline's own task_type name. This builds
    a REAL VisionArtifactPayload (not a hand-built stand-in) the same way
    a pipeline result actually would, with an outer task_type that is
    NOT "identity_face" -- confirming the content check, not the task_type
    check, is what catches this."""
    res = VisionResult(
        corr_id="c3",
        ok=True,
        task_type="pipeline_retina_dense",  # NOT "identity_face"
        device="cuda:0",
        artifacts={
            "objects": [],
            "identities": {
                "candidates": [{"subject": "juniper", "similarity": 0.71, "state": "probable"}],
                "enrolled_subject": "juniper",
                "gallery_enrolled": True,
            },
        },
    )
    payload = build_artifact_payload(res)

    assert should_broadcast_artifact(res.task_type, payload) is False


def test_should_broadcast_artifact_allows_non_identity_pipeline_result():
    res = VisionResult(
        corr_id="c4",
        ok=True,
        task_type="pipeline_retina_fast",
        device="cuda:0",
        artifacts={"objects": [{"label": "chair", "score": 0.9, "box_xyxy": [0, 0, 1, 1]}]},
    )
    payload = build_artifact_payload(res)

    assert should_broadcast_artifact(res.task_type, payload) is True


def test_should_broadcast_artifact_excludes_identity_face():
    """The one, real, deliberate exclusion -- see this module's own comment
    for why (identity data must not reach the general, identity-unaware
    broadcast channel)."""
    assert should_broadcast_artifact("identity_face") is False


@pytest.mark.parametrize(
    "task_type",
    ["caption_frame", "vqa", "detect_open_vocab", "embed_image", "retina_fast"],
)
def test_should_broadcast_artifact_allows_everything_else(task_type):
    assert should_broadcast_artifact(task_type) is True


@pytest.mark.asyncio
async def test_publish_result_skips_broadcast_for_identity_face():
    """Integration-level check at the actual bus-call boundary, not just the
    pure predicate -- confirms _publish_result really doesn't invoke
    _publish_artifact_broadcast for this task_type, and confirms the direct
    RPC reply (result_payload with the full artifact) is still sent as
    normal to the requester."""
    service = VisionHostService()
    service.bus = MagicMock()
    service.bus.publish = AsyncMock()

    res = VisionResult(
        corr_id="c1",
        ok=True,
        task_type="identity_face",
        device="cpu",
        artifacts={
            "configured": True,
            "implemented": True,
            "kind": "identity",
            "model_id": "facenet-pytorch/vggface2",
            "device": "cpu",
            "identities": {
                "candidates": [{"subject": "juniper", "similarity": 0.61, "state": "probable"}],
                "enrolled_subject": "juniper",
                "gallery_enrolled": True,
            },
        },
    )

    fake_envelope = MagicMock()
    fake_envelope.reply_to = "orion:vision:reply:c1"
    fake_envelope.derive_child.return_value = MagicMock()

    await service._publish_result(res, fake_envelope)

    # Exactly one publish call -- the direct RPC reply. No second call to
    # CHANNEL_VISIONHOST_PUB (the general broadcast).
    assert service.bus.publish.call_count == 1
    published_channel = service.bus.publish.call_args[0][0]
    assert published_channel == "orion:vision:reply:c1"


@pytest.mark.asyncio
async def test_publish_result_still_broadcasts_for_non_identity_task_types():
    """Regression guard the other direction -- this suppression must stay
    scoped to identity_face, not silently swallow every broadcast."""
    service = VisionHostService()
    service.bus = MagicMock()
    service.bus.publish = AsyncMock()

    res = VisionResult(
        corr_id="c2",
        ok=True,
        task_type="caption_frame",
        device="cpu",
        artifacts={"configured": True, "implemented": True, "kind": "caption_frame", "device": "cpu"},
    )

    fake_envelope = MagicMock()
    fake_envelope.reply_to = "orion:vision:reply:c2"
    fake_envelope.derive_child.return_value = MagicMock()

    await service._publish_result(res, fake_envelope)

    assert service.bus.publish.call_count == 2
    published_channels = {call.args[0] for call in service.bus.publish.call_args_list}
    assert "orion:vision:reply:c2" in published_channels
    assert "orion:vision:artifacts" in published_channels
