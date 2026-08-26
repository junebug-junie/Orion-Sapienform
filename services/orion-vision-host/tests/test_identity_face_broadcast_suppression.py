from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.artifacts import build_artifact_payload
from app.main import VisionHostService, should_broadcast_artifact, should_broadcast_identity
from app.models import VisionResult


def test_broadcast_guard_logic_lives_in_exactly_one_place():
    """Structural regression pin, rewritten 2026-08-26 after consolidating
    the guard logic into _maybe_broadcast (review finding: the guard used
    to be hand-rolled at each call site independently, which is exactly
    the shape that let a real bug ship once -- the /v1/vision/task HTTP
    endpoint's own separate call bypassed should_broadcast_artifact()
    entirely, still broadcasting identity data unfiltered even after the
    bus-first path was fixed).

    New invariant: _publish_artifact_broadcast and _publish_identity_
    broadcast are each called from exactly ONE place now (_maybe_broadcast
    itself), and _maybe_broadcast is called from exactly the 2 real
    request paths (_publish_result, http_task). A future call site that
    forgets to route through _maybe_broadcast broadcasts nothing --
    fails safe, not fails open -- so this pins the shape that makes that
    the failure mode, not just the historical call count."""
    main_py = Path(__file__).resolve().parents[1] / "app" / "main.py"
    lines = main_py.read_text(encoding="utf-8").splitlines()

    def _call_sites(needle: str) -> list[int]:
        return [i for i, line in enumerate(lines) if needle in line]

    artifact_calls = _call_sites("await self._publish_artifact_broadcast(")
    identity_calls = _call_sites("await self._publish_identity_broadcast(")
    maybe_broadcast_calls = _call_sites("await self._maybe_broadcast(") + _call_sites(
        "await service._maybe_broadcast("
    )

    assert len(artifact_calls) == 1, (
        f"_publish_artifact_broadcast must be called from exactly one place "
        f"(_maybe_broadcast) -- found {len(artifact_calls)} at lines {[i + 1 for i in artifact_calls]}"
    )
    assert len(identity_calls) == 1, (
        f"_publish_identity_broadcast must be called from exactly one place "
        f"(_maybe_broadcast) -- found {len(identity_calls)} at lines {[i + 1 for i in identity_calls]}"
    )
    assert len(maybe_broadcast_calls) == 2, (
        f"_maybe_broadcast must be called from exactly the 2 real request paths "
        f"(_publish_result, http_task) -- found {len(maybe_broadcast_calls)} at "
        f"lines {[i + 1 for i in maybe_broadcast_calls]}"
    )

    # Both real predicate checks must still live inside _maybe_broadcast's
    # own body, not scattered back out to the call sites. Next line at the
    # SAME 4-space method-def indentation (async or not) ends the body --
    # a naive "next def" search would stop early on should_broadcast_
    # artifact/should_broadcast_identity's own nested `if` blocks, which
    # don't start with "def " anyway, so this is just being explicit about
    # matching only sibling method definitions.
    maybe_broadcast_start = next(i for i, line in enumerate(lines) if "async def _maybe_broadcast(" in line)
    maybe_broadcast_end = next(
        i
        for i, line in enumerate(lines[maybe_broadcast_start + 1 :], start=maybe_broadcast_start + 1)
        if line.startswith("    def ") or line.startswith("    async def ")
    )
    body = "\n".join(lines[maybe_broadcast_start:maybe_broadcast_end])
    assert "should_broadcast_artifact(" in body
    assert "should_broadcast_identity(" in body


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


# -- should_broadcast_identity: the mirror-image predicate for the dedicated
# CHANNEL_VISIONHOST_IDENTITY_PUB lane -----------------------------------


def test_should_broadcast_identity_true_for_identity_face_task_type():
    assert should_broadcast_identity("identity_face") is True


@pytest.mark.parametrize(
    "task_type",
    ["caption_frame", "vqa", "detect_open_vocab", "embed_image", "retina_fast"],
)
def test_should_broadcast_identity_false_for_everything_else_by_task_type_alone(task_type):
    assert should_broadcast_identity(task_type) is False


def test_should_broadcast_identity_true_for_identity_data_reached_via_a_pipeline():
    """Mirror of test_should_broadcast_artifact_suppresses_identity_data_
    reached_via_a_pipeline above -- the same pipeline-composition path that
    can smuggle identity data past a task_type-only check must also be
    caught by should_broadcast_identity's content check, or a pipeline-
    routed identity result would silently never reach the dedicated
    channel at all (neither broadcast, nor suppressed -- just dropped)."""
    res = VisionResult(
        corr_id="c5",
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
    assert should_broadcast_identity(res.task_type, payload) is True


def test_should_broadcast_identity_false_for_non_identity_pipeline_result():
    res = VisionResult(
        corr_id="c6",
        ok=True,
        task_type="pipeline_retina_fast",
        device="cuda:0",
        artifacts={"objects": [{"label": "chair", "score": 0.9, "box_xyxy": [0, 0, 1, 1]}]},
    )
    payload = build_artifact_payload(res)
    assert should_broadcast_identity(res.task_type, payload) is False


@pytest.mark.asyncio
async def test_publish_result_skips_general_broadcast_for_identity_face():
    """Integration-level check at the actual bus-call boundary, not just the
    pure predicate -- confirms _publish_result really doesn't invoke
    _publish_artifact_broadcast (the general CHANNEL_VISIONHOST_PUB) for
    this task_type. 2026-08-26: now expects TWO calls, not one -- the RPC
    reply, plus the new dedicated CHANNEL_VISIONHOST_IDENTITY_PUB publish
    (should_broadcast_identity, _publish_identity_broadcast). The general
    broadcast is still and only ever skipped."""
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

    assert service.bus.publish.call_count == 2
    published_channels = [call.args[0] for call in service.bus.publish.call_args_list]
    assert "orion:vision:reply:c1" in published_channels
    assert "orion:vision:artifacts" not in published_channels, "general broadcast must stay suppressed"
    assert "orion:vision:artifacts:identity" in published_channels, "dedicated identity channel must fire"


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
