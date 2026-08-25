"""Foveal probe: a manually-triggered, event-driven call for a real caption
on the current frame -- the "Foveal tier" from
docs/superpowers/specs/2026-08-12-perception-frontier-design.md.

Distinct from the always-on peripheral pipeline (retina_fast, dispatched
continuously by orion-vision-frame-router to orion-vision-host's
`orion:exec:request:VisionHostService` channel, which only ever runs the
small BLIP-family captioner configured in `config/vision_profiles.yaml`).

**2026-08-25 architecture change: this module no longer RPCs a vision-host.**
It originally targeted a second, isolated orion-vision-host instance
(CHANNEL_FOVEAL_HOST_REQUEST -> a dedicated per-node channel, deliberately
never the shared one -- see PR #1859's race incident). Live-tested that path
after real config landed: it worked end-to-end but returned an EMPTY caption
(`"text": "", "caption_rejected:too_short"`), because `config/
vision_profiles.yaml`'s `vlm_caption` profile still carries the unfilled
placeholder `model_id: "REPLACE_ME/qwen2-vl_or_llava_next"`, which falls back
to `orion-vision-host`'s own `VISION_VLM_MODEL_ID` default -- itself another
BLIP-family model, not a real VLM. So the "richer-than-BLIP" tier was
mechanically wired but had nothing richer behind it.

Per this session's own existing-mechanism check (CLAUDE.md's metric/mechanism
gate, item 5): a real, already-warm, already-proven vision-capable model
already exists -- circe's `chat` lane (`modalities.vision=true`, confirmed
live via `/props`) -- and `orion-llm-gateway/app/vision.py` already
implements attachment-to-model-input plumbing for it, including a
`kind="percept"` attachment path built specifically for camera frames (see
`AttachmentRefV1`/`resolve_attachment_url` in `orion/core/bus/bus_schemas.py`
and `app/vision.py`). `orion-vision-council` is already a registered producer
on `orion:exec:request:LLMGatewayService` (it uses this same channel for its
own metacog interpretation calls, `_call_llm_raw` in `app/main.py`) and the
gateway's live `.env` already allowlists `orion-athena-percept-store` as an
attachment host. So routing the foveal probe through the gateway needed zero
new bus contract, zero new channel, and zero new gateway-side config -- only
a different envelope shape at hop 3 below.

Three real hops, each independently testable:
1. `resolve_latest_frame_path` -- find the newest captured frame on local
   disk (same convention orion-vision-host's own runner.py uses for
   request.use_latest_frame).
2. `upload_frame_bytes` -- hand it to orion-percept-store, get back a
   content-addressed sha256 (mirrors orion-vision-retina's own
   frame_store.upload_bytes; not imported cross-service per this repo's
   service-boundary convention -- each service owns its own small client).
3. `build_foveal_chat_envelope` + `run_foveal_probe` -- RPC the LLM gateway
   with a `ChatRequestPayload` carrying a `kind="percept"` attachment
   referencing that sha256, on the vision-capable route, and return its real
   answer.
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
from orion.core.bus.bus_schemas import (
    AttachmentRefV1,
    BaseEnvelope,
    ChatRequestPayload,
    LLMMessage,
    ServiceRef,
)

from .llm_reply import GATEWAY_ERROR_PREFIX, extract_chat_result_text

DEFAULT_CAPTION_PROMPT = (
    "Describe what you see in this image in one or two plain, concrete "
    "sentences. Name objects and people actually visible; do not guess at "
    "anything outside the frame."
)


class NoFrameAvailableError(RuntimeError):
    """No captured frame exists yet under the configured frames directory."""


class PerceptUploadError(RuntimeError):
    """The frame could not be handed to the percept store."""


class FovealReplyDecodeError(RuntimeError):
    """The LLM gateway's RPC reply arrived but the bus envelope itself
    failed to decode (malformed bytes, schema drift, wrong codec version).

    Deliberately NOT PerceptUploadError -- reusing that class here would mean
    a bus-level decode failure gets reported to callers as
    `error_code="upload_failed"`, sending anyone debugging it toward
    percept-store connectivity when the upload had already succeeded (this
    exact confusion was code-review-caught when the vision-host-RPC version
    of this module was first built).
    """


class FovealTaskFailedError(RuntimeError):
    """The RPC round-trip itself worked (envelope decoded fine), but the
    gateway's answer was not a usable caption -- empty content, or the
    `"[Error: ...]"` embedded-error convention `_call_llm_raw` (app/main.py)
    already uses for this identical reply contract. Covers every configured
    vision route being unreachable, the target worker's own `/props`
    capability probe reporting blind, and a genuinely rejected/empty
    completion.

    `error_code` distinguishes the two cases a caller can actually act on --
    "gateway_error" (the gateway itself reported a failure as embedded
    content) vs "empty_response" (the RPC succeeded but produced nothing).
    Coarser than the retired vision-host path's structured error_code (e.g.
    profile_disabled), since the gateway's chat contract doesn't carry one,
    but a real improvement over collapsing both into one constant string.
    """

    def __init__(self, detail: str, *, error_code: str = "empty_response"):
        self.detail = detail
        self.error_code = error_code
        super().__init__(f"foveal chat call produced no usable caption: {detail!r}")


class FovealNotConfiguredError(RuntimeError):
    """FOVEAL_PERCEPT_STORE_URL or FOVEAL_LLM_ROUTE is unset -- refuse rather
    than call nothing.

    Unlike the retired vision-host path, there is no separate "foveal
    channel" left to be unconfigured: this probe now shares
    CHANNEL_LLM_REQUEST with the council's own always-on metacog calls, which
    is never empty in a running council. The two prerequisites unique to
    this path are the percept store the frame gets uploaded to and the
    gateway route to ask. Checked before the frame read/upload runs so a
    blank route fails fast instead of spending a real upload on every call
    before the gateway eventually replies with an embedded routing error.
    """


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


def build_foveal_chat_envelope(
    *,
    sha256: str,
    percept_store_url: str,
    frame_bytes_len: int,
    question: Optional[str],
    reply_to: str,
    llm_route: str,
    service_name: str,
    service_version: str,
    caption_prompt: str = DEFAULT_CAPTION_PROMPT,
    correlation_id: Optional[uuid.UUID] = None,
) -> BaseEnvelope:
    """A caption prompt when `question` is None/empty, the question itself
    otherwise -- same caption-vs-question split the retired vision-host task
    contract made, expressed here as a chat prompt instead of a task_type.

    The image travels as a `kind="percept"` AttachmentRefV1 (content address
    only, never inline bytes -- see AttachmentRefV1's own docstring on why:
    replicating base64'd frames through Redis/Postgres/trace stores already
    took this host down once via TOAST OOM, 2026-07-23). `source_url` is
    required by the schema but the gateway does NOT use it for percept-kind
    attachments -- it reconstructs `<LLM_GATEWAY_PERCEPT_BASE_URL>/<sha256>`
    from its own trusted config instead (`resolve_attachment_url` in
    `orion-llm-gateway/app/vision.py`), precisely so a client-supplied URL can
    never redirect the gateway somewhere else. Filled in here only so the
    field is a real, dereferenceable URL rather than a placeholder string.
    """
    # .strip() BEFORE the truthiness check, not after -- a whitespace-only
    # question (e.g. ?question=%20) must fall back to the caption prompt
    # rather than send the model zero instruction text. Code review caught
    # `question.strip() if question else caption_prompt` doing the strip too
    # late: a non-empty-but-blank `question` took the truthy branch first.
    stripped_question = question.strip() if question else ""
    prompt = stripped_question or caption_prompt
    attachment = AttachmentRefV1(
        kind="percept",
        sha256=sha256,
        mime="image/jpeg",
        bytes=frame_bytes_len,
        source_url=f"{str(percept_store_url).rstrip('/')}/{sha256}",
    )
    chat_request = ChatRequestPayload(
        route=llm_route,
        messages=[LLMMessage(role="user", content=prompt)],
        attachments=[attachment],
    )
    kwargs: dict[str, Any] = dict(
        kind="llm.chat.request",
        source=ServiceRef(name=service_name, version=service_version),
        # BaseEnvelope.payload is a plain Dict[str, Any], not a typed model --
        # unlike derive_child() (which does this conversion itself), a direct
        # BaseEnvelope(...) construction requires the dict already.
        payload=chat_request.model_dump(mode="json"),
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
    if not str(getattr(settings, "FOVEAL_PERCEPT_STORE_URL", "") or ""):
        raise FovealNotConfiguredError("FOVEAL_PERCEPT_STORE_URL is unset")
    # Checked here, before the real disk read + upload, not left to surface
    # as an embedded gateway error after spending a real upload on every
    # call -- code review caught this was only guarded for the percept
    # store, not for a blank/typo'd route.
    if not str(getattr(settings, "FOVEAL_LLM_ROUTE", "") or ""):
        raise FovealNotConfiguredError("FOVEAL_LLM_ROUTE is unset")

    # File I/O and urllib.request.urlopen are both blocking. run_foveal_probe
    # is called from a FastAPI request handler on orion-vision-council's own
    # event loop -- the SAME loop CouncilService._consume/_consume_rpc use
    # for the always-on peripheral vision pipeline (app/main.py). Without
    # to_thread, a single probe call stalls that pipeline for the full
    # upload timeout plus network latency every time it's triggered.
    def _resolve_and_upload() -> tuple[Path, bytes, str]:
        frame_path = resolve_latest_frame_path(settings.FOVEAL_FRAMES_DIR)
        if frame_path is None:
            raise NoFrameAvailableError(
                f"no .jpg frames found under {settings.FOVEAL_FRAMES_DIR}"
            )
        data = frame_path.read_bytes()
        if not data:
            # sha256(b"") is well-defined and would pass upload_frame_bytes's
            # hash check, silently turning a truncated/partial capture (e.g.
            # racing a mid-write frame) into a "successful" 0-byte upload --
            # the attachment sent downstream would then lie about carrying
            # real image data instead of surfacing the actual problem.
            raise NoFrameAvailableError(f"frame at {frame_path} is empty (0 bytes)")
        sha256 = upload_frame_bytes(
            data,
            base_url=settings.FOVEAL_PERCEPT_STORE_URL,
            token=settings.FOVEAL_PERCEPT_STORE_TOKEN or None,
            timeout_sec=settings.FOVEAL_PERCEPT_UPLOAD_TIMEOUT_SEC,
        )
        return frame_path, data, sha256

    frame_path, data, sha256 = await asyncio.to_thread(_resolve_and_upload)

    corr_id = uuid.uuid4()
    reply_to = f"{settings.CHANNEL_LLM_REPLY_PREFIX}:foveal:{corr_id}"
    envelope = build_foveal_chat_envelope(
        sha256=sha256,
        percept_store_url=settings.FOVEAL_PERCEPT_STORE_URL,
        frame_bytes_len=len(data),
        question=question,
        reply_to=reply_to,
        llm_route=settings.FOVEAL_LLM_ROUTE,
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        correlation_id=corr_id,
    )

    msg = await bus.rpc_request(
        settings.CHANNEL_LLM_REQUEST,
        envelope,
        reply_channel=reply_to,
        timeout_sec=settings.FOVEAL_HOST_TIMEOUT_SEC,
    )
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise FovealReplyDecodeError(f"foveal LLM gateway reply decode failed: {decoded.error}")

    # Same extraction helper _call_llm_raw (app/main.py) uses for this
    # identical reply contract -- not ChatResultPayload.text directly, which
    # only reads top-level content/text and would miss a choices[]/raw[]-
    # shaped reply that helper is built to handle.
    text = extract_chat_result_text(decoded.envelope.payload).strip()
    if not text:
        raise FovealTaskFailedError("empty response", error_code="empty_response")
    if text.startswith(GATEWAY_ERROR_PREFIX):
        raise FovealTaskFailedError(text, error_code="gateway_error")

    return {
        "frame_path": str(frame_path),
        "sha256": sha256,
        "caption": text,
        "reply": decoded.envelope.payload,
    }
