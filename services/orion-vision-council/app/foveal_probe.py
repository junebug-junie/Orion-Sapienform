"""Foveal probe: a manually-triggered, event-driven call to a dedicated VLM
host for a real caption on the current frame -- the "Foveal tier" from
docs/superpowers/specs/2026-08-12-perception-frontier-design.md.

Distinct from the always-on peripheral pipeline (retina_fast, dispatched
continuously by orion-vision-frame-router to the SHARED
orion:exec:request:VisionHostService channel). This module targets a
SEPARATE, isolated vision-host intake channel instead -- deliberately never
the shared one. Two consumers racing on the shared channel is not a
hypothetical: it happened live on 2026-08-25 (a second orion-vision-host
instance on circe subscribed to the shared channel and its fast
"image_not_found" rejections silently beat athena's real, slower replies for
2m13s, killing every host_trigger update in that window -- see PR #1859 and
this session's own record). CHANNEL_FOVEAL_HOST_REQUEST must point at an
isolated channel only the foveal host answers.

Three real hops, each independently testable:
1. `resolve_latest_frame_path` -- find the newest captured frame on local
   disk (same convention orion-vision-host's own runner.py uses for
   request.use_latest_frame).
2. `upload_frame_bytes` -- hand it to orion-percept-store, get back a
   content-addressed sha256 (mirrors orion-vision-retina's own
   frame_store.upload_bytes; not imported cross-service per this repo's
   service-boundary convention -- each service owns its own small client).
3. `build_foveal_task_envelope` + `run_foveal_probe` -- RPC the foveal host
   with that sha256 and return its real answer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionTaskRequestPayload, VisionTaskResultPayload


class NoFrameAvailableError(RuntimeError):
    """No captured frame exists yet under the configured frames directory."""


class PerceptUploadError(RuntimeError):
    """The frame could not be handed to the percept store."""


class FovealReplyDecodeError(RuntimeError):
    """The foveal host's RPC reply arrived but the bus envelope itself
    failed to decode (malformed bytes, schema drift, wrong codec version).

    Deliberately NOT PerceptUploadError -- code review caught that reusing
    that class here meant a bus-level decode failure was reported to callers
    as `error_code="upload_failed"`, sending anyone debugging it toward
    percept-store connectivity when the upload had already succeeded.
    """


class FovealTaskFailedError(RuntimeError):
    """The RPC round-trip itself worked (envelope decoded fine), but the
    foveal host's own VisionTaskResultPayload.ok was False -- e.g. a
    disabled profile, a bad percept_sha256, an OOM on the host's GPU.

    Code review caught that the debug endpoint was returning top-level
    `ok: true` whenever the bus-level decode succeeded, without ever looking
    at this domain-level field -- reproduced live during this same PR's own
    testing (a `percept_sha256` field-name bug made the foveal host reply
    `ok: false`, and the endpoint still reported success at the top level).
    """

    def __init__(self, result: "VisionTaskResultPayload"):
        self.result = result
        super().__init__(f"foveal host task failed: error={result.error!r} error_code={result.error_code!r}")


class FovealHostNotConfiguredError(RuntimeError):
    """CHANNEL_FOVEAL_HOST_REQUEST is unset -- refuse rather than call nothing."""


def resolve_latest_frame_path(frames_dir: str) -> Optional[Path]:
    """Newest .jpg under frames_dir by mtime, or None if the directory is
    missing or empty. Pure filesystem read, no upload, no bus -- mirrors
    orion-vision-host's own runner.py::_resolve_latest_frame_path so both
    services agree on "the current frame" without importing across the
    service boundary."""
    d = Path(frames_dir)
    if not d.is_dir():
        return None
    candidates = sorted(d.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def upload_frame_bytes(
    data: bytes,
    *,
    base_url: str,
    token: str | None = None,
    timeout_sec: float = 10.0,
) -> str:
    """POST raw JPEG bytes to orion-percept-store, return the content hash.
    Hash-verified on response -- never trusts the store's own claim without
    checking it against what was actually sent. stdlib urllib only (AGENTS.md
    section 10: no dependency for a task this small)."""
    local_sha = hashlib.sha256(data).hexdigest()
    url = str(base_url).rstrip("/")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/octet-stream")
    if token:
        req.add_header("X-Orion-Percept-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise PerceptUploadError(f"percept upload to {url} failed: {exc}") from exc

    sha256 = str(body.get("sha256") or "")
    if sha256 != local_sha:
        raise PerceptUploadError(
            f"percept store returned {sha256[:12]!r} for content hashing to {local_sha[:12]!r}"
        )
    return sha256


def build_foveal_task_envelope(
    *,
    sha256: str,
    question: Optional[str],
    reply_to: str,
    service_name: str,
    service_version: str,
    correlation_id: Optional[uuid.UUID] = None,
) -> BaseEnvelope:
    """caption_frame when question is None/empty, vlm (VQA) when it's set --
    same task-type split runner.py already implements for the shared vision
    host; the foveal host answers the identical contract, only reached over
    a different, isolated channel.

    Two bugs live-caught in a row on 2026-08-25, both fixed here:

    1. The request field is `percept_sha256`, not `sha256` -- runner.py's
       `_load_image_from_request` only ever reads
       `request.get("percept_sha256")`. `sha256` is the percept-store's OWN
       response field name (see `upload_frame_bytes` above) -- a different
       contract that happens to share a name; don't conflate the two again.
    2. The VQA task_type is `"vqa"`, not `"vlm"` -- `config/vision_profiles.
       yaml`'s `task_routing` maps `vqa: vlm_vqa`; there is no `vlm` entry,
       so `resolve_target("vlm")` fell through to treating "vlm" itself as
       the profile name, which is never in VISION_ENABLED_PROFILES (that
       lists real profile names: vlm_caption, vlm_vqa) -- came back a clean
       `error_code=profile_disabled`, not a crash, but still wrong. "vlm" is
       the profile's `kind:` in the yaml, not its task_type; conflating a
       profile's kind with its task_type was the mistake.
    """
    request: dict[str, Any] = {"percept_sha256": sha256}
    task_type = "caption_frame"
    if question:
        request["question"] = question
        task_type = "vqa"

    task = VisionTaskRequestPayload(task_type=task_type, request=request)
    kwargs: dict[str, Any] = dict(
        kind="vision.task.request",
        source=ServiceRef(name=service_name, version=service_version),
        # BaseEnvelope.payload is a plain Dict[str, Any], not a typed model --
        # unlike derive_child() (which does this conversion itself), a direct
        # BaseEnvelope(...) construction requires the dict already.
        payload=task.model_dump(mode="json"),
        reply_to=reply_to,
    )
    if correlation_id is not None:
        kwargs["correlation_id"] = correlation_id
    return BaseEnvelope(**kwargs)


async def run_foveal_probe(
    bus: OrionBusAsync,
    settings: Any,
    *,
    question: Optional[str] = None,
) -> dict[str, Any]:
    """Orchestrates the full probe. Raises a specific typed error at whichever
    hop fails, so a caller (the debug endpoint) can report exactly what went
    wrong rather than a generic 500."""
    if not settings.CHANNEL_FOVEAL_HOST_REQUEST:
        raise FovealHostNotConfiguredError("CHANNEL_FOVEAL_HOST_REQUEST is unset")

    # File I/O and urllib.request.urlopen are both blocking. run_foveal_probe
    # is called from a FastAPI request handler on orion-vision-council's own
    # event loop -- the SAME loop CouncilService._consume/_consume_rpc use
    # for the always-on peripheral vision pipeline (app/main.py). Without
    # to_thread, a single probe call stalls that pipeline for the full
    # upload timeout plus network latency every time it's triggered. Code
    # review caught this: the sibling module this deliberately mirrors,
    # orion-vision-retina/app/frame_store.py, states in its own docstring
    # "Callers run it off the event loop via asyncio.to_thread" and every
    # call site there honors it -- this file dropped that wrapper.
    def _resolve_and_upload() -> tuple[Path, str]:
        frame_path = resolve_latest_frame_path(settings.FOVEAL_FRAMES_DIR)
        if frame_path is None:
            raise NoFrameAvailableError(
                f"no .jpg frames found under {settings.FOVEAL_FRAMES_DIR}"
            )
        data = frame_path.read_bytes()
        sha256 = upload_frame_bytes(
            data,
            base_url=settings.FOVEAL_PERCEPT_STORE_URL,
            token=settings.FOVEAL_PERCEPT_STORE_TOKEN or None,
            timeout_sec=settings.FOVEAL_PERCEPT_UPLOAD_TIMEOUT_SEC,
        )
        return frame_path, sha256

    frame_path, sha256 = await asyncio.to_thread(_resolve_and_upload)

    corr_id = uuid.uuid4()
    reply_to = f"{settings.CHANNEL_FOVEAL_HOST_REPLY_PREFIX}:{corr_id}"
    envelope = build_foveal_task_envelope(
        sha256=sha256,
        question=question,
        reply_to=reply_to,
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        correlation_id=corr_id,
    )

    msg = await bus.rpc_request(
        settings.CHANNEL_FOVEAL_HOST_REQUEST,
        envelope,
        reply_channel=reply_to,
        timeout_sec=settings.FOVEAL_HOST_TIMEOUT_SEC,
    )
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise FovealReplyDecodeError(f"foveal host reply decode failed: {decoded.error}")

    result = VisionTaskResultPayload.model_validate(decoded.envelope.payload)
    if not result.ok:
        raise FovealTaskFailedError(result)

    return {
        "frame_path": str(frame_path),
        "sha256": sha256,
        "reply": decoded.envelope.payload,
    }
