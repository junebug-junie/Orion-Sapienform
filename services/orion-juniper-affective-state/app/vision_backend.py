"""Read Juniper's affect from video alone, via the mesh's own VL lane.

**Why this replaced AffectGPT (2026-08-26).** The previous backend
(`orion-affectgpt-worker`, AffectGPT 7B on circe GPU1) failed in three ways at
once, all confirmed live against `juniper_multimodal_affect_log` and two
instrumented captures:

1. *It answered from the text branch and ignored the face.* Its only obtainable
   checkpoint is `multiface_audio_face_text`; handed an empty subtitle it
   returned "it is not possible to infer the character's emotional state from
   the subtitle content" -- while holding a face crop with a 100% detection
   rate. That refusal was published `ok=True` and mirrored verbatim into
   Juniper's chat prompt for turn `ddddfe40`.
2. *It could not be relied on for a subtitle.* The clip is recorded AFTER
   Juniper finishes speaking, by design, so `subtitle_source="none"` is the
   normal case for every `chat_turn_*` capture, not an edge case.
3. *It confabulated, and it misgendered.* Every read it ever produced that
   actually committed to something described the subject as "the man" (3 of 3
   in the stored log), and one cited "the acoustic characteristics of the
   voice" from an audio track measured at -49.2 dB peak -- silence.

The same frame, sent to `circe:8011` (Qwen3.6-35B-A3B, already running, already
`multimodal`), produced a specific, correctly-hedged read in 5.7s and asserted
no gender at all. This module is that path.

**It goes through orion-llm-gateway, not straight at circe:8011.** This service
runs on circe, so a direct call would be a localhost hop and tempting. The
gateway is used anyway because it owns three things this module must not
reimplement: the live `/props` vision-capability probe (config claiming
`supports_vision` has been wrong before -- see `app/vision.py`'s own docstring),
the route table, and the "bytes enter at the last possible moment" attachment
resolution. Frames travel as `AttachmentRefV1(kind="percept")` URLs and become
base64 exactly once, one hop before the model.

**Audio is gone from this path entirely, not merely optional.** A transcript is
passed as context when one exists and simply omitted when it does not -- there
is no subtitle branch left for an empty string to collapse into. That is the
specific defect that produced the garbage read, and removing the branch is what
fixes it, not a better-worded prompt.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Optional
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import (
    AttachmentRefV1,
    BaseEnvelope,
    ChatRequestPayload,
    LLMMessage,
    ServiceRef,
)
from orion.core.llm_json import parse_json_object
from orion.schemas.affectgpt import AffectReadV1

from .frame_sample import FrameSampleError, FrameSampleResult, sample_frames

logger = logging.getLogger("juniper-affective-state.vision")

# The gateway's own intake channel + reply convention, copied from the pattern
# in orion/memory/crystallization/concept_relation.py rather than invented.
_REPLY_PREFIX = "orion:exec:result:LLMGatewayService"

# What the model is asked to return. Kept here rather than in the schema module
# because it is prompt engineering, not contract -- AffectReadV1 is the
# contract, and it is validated independently of whatever this asks for.
_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "valence": {"type": "number", "minimum": -1, "maximum": 1},
        "arousal": {"type": "number", "minimum": 0, "maximum": 1},
        "primary_affect": {"type": "string"},
        "cues": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "cannot_tell": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["valence", "arousal", "primary_affect", "cues", "confidence"],
}

# The identity-inference ban is the first instruction, not a trailing caveat.
# The replaced backend called Juniper "the man" in every read that committed to
# anything; this is the structural fix for that, and it is also the right
# posture for a recurring read of someone's face regardless of whether any
# given model would have got it right.
_SYSTEM_PROMPT = (
    "You read facial and postural affect from a short sequence of webcam "
    "stills of one person.\n"
    "\n"
    "Absolute rules:\n"
    "- Do NOT infer or state the person's gender, age, ethnicity, identity, "
    "or appearance. Never use gendered words for them. Refer to them only as "
    "'the person' or 'they'.\n"
    "- Base every claim on something you can actually see in these frames. "
    "Name that evidence in 'cues'.\n"
    "- You are given stills only. You have NO audio. Never describe voice, "
    "tone, speech, or sound.\n"
    "- If you cannot judge something, put it in 'cannot_tell' and lower "
    "'confidence'. An honest low-confidence read is correct and useful; a "
    "confident guess is not.\n"
    "\n"
    "The frames are in chronological order across a few seconds, so you may "
    "describe change (settling, tensing, looking away) if you can see it.\n"
    "\n"
    "'valence' is -1 (strongly negative) to 1 (strongly positive). 'arousal' "
    "is 0 (calm, still) to 1 (highly activated). 'primary_affect' is a short "
    "phrase in your own words.\n"
    "\n"
    "Reply with ONLY a JSON object matching the requested schema."
)


class VisionAffectError(RuntimeError):
    """A stage of the vision read failed. Carries a stable error_code."""

    def __init__(self, message: str, *, error_code: str):
        super().__init__(message)
        self.error_code = error_code


class VisionAffectResult:
    """What the caller needs to build a JuniperMultimodalAffectV1."""

    def __init__(
        self,
        *,
        affect: AffectReadV1,
        raw_response: str,
        face_detection: dict[str, Any],
        frames_used: int,
        timings: dict[str, Any],
        model: Optional[str],
    ):
        self.affect = affect
        self.raw_response = raw_response
        self.face_detection = face_detection
        self.frames_used = frames_used
        self.timings = timings
        self.model = model


def _upload_percept(
    data: bytes, *, base_url: str, token: str, timeout_sec: float
) -> str:
    """POST one JPEG, return its sha256. Verifies the store agrees with us.

    urllib, not requests/httpx: this service's requirements.txt carries neither,
    and CLAUDE.md section 10 says not to add a dependency for a stdlib task.
    Same call shape orion-vision-retina's frame_store.upload_frame already uses.
    """
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
        raise VisionAffectError(
            f"percept upload to {url} failed: {exc}", error_code="percept_upload_failed"
        ) from exc
    sha256 = str(body.get("sha256") or "")
    if sha256 != local_sha:
        # Content-addressed store disagreeing about what we just sent means the
        # URL we are about to hand the gateway would resolve to different bytes.
        raise VisionAffectError(
            f"percept store returned {sha256[:12]!r} for content hashing to {local_sha[:12]!r}",
            error_code="percept_hash_mismatch",
        )
    return sha256


def _build_prompt(transcript: Optional[str], frame_count: int) -> str:
    lines = [
        f"{frame_count} webcam stills of one person, in chronological order "
        "across a few seconds. Read their affect."
    ]
    text = (transcript or "").strip()
    if text:
        # Only ever added when real text exists. There is deliberately no
        # "(no subtitle)" placeholder branch -- that placeholder IS the bug
        # this backend replaced.
        lines.append(
            "For context, the person said this around the time these frames "
            f"were captured: {text!r}. Use it only as context; your read must "
            "still rest on what you can see."
        )
    return "\n\n".join(lines)


async def assess_via_vision(
    bus: OrionBusAsync,
    *,
    video_path: str,
    transcript: Optional[str],
    settings: Any,
) -> VisionAffectResult:
    """Sample frames -> percept-store -> gateway RPC -> validated AffectReadV1.

    Raises VisionAffectError with a stable error_code on any stage failure.
    Deliberately does NOT swallow into a neutral-looking result: the caller
    turns this into ok=False, because a failed read must never be
    indistinguishable from a calm one.
    """
    loop = asyncio.get_running_loop()
    timings: dict[str, Any] = {}

    t0 = loop.time()
    try:
        sampled: FrameSampleResult = await asyncio.to_thread(
            sample_frames,
            video_path,
            max_frames=int(settings.AFFECT_VISION_MAX_FRAMES),
            jpeg_quality=int(settings.AFFECT_VISION_JPEG_QUALITY),
        )
    except FrameSampleError as exc:
        raise VisionAffectError(str(exc), error_code="frame_sample_failed") from exc
    timings["sample_s"] = round(loop.time() - t0, 3)

    base_url = str(settings.PERCEPT_STORE_BASE_URL or "").strip()
    if not base_url:
        raise VisionAffectError(
            "PERCEPT_STORE_BASE_URL is not configured", error_code="percept_unconfigured"
        )

    t1 = loop.time()
    # Concurrent: N independent HTTP POSTs of ~15KB each. Sequential would add
    # a round trip per frame to a path that already sits in front of a live
    # chat turn. return_exceptions=True so one failure does not leave sibling
    # threads writing into a request that has already returned -- the same
    # reasoning main.py's capture_and_assess documents for its own gather.
    upload_results = await asyncio.gather(
        *[
            asyncio.to_thread(
                _upload_percept,
                f.jpeg,
                base_url=base_url,
                token=str(settings.PERCEPT_STORE_TOKEN or ""),
                timeout_sec=float(settings.PERCEPT_STORE_TIMEOUT_SEC),
            )
            for f in sampled.frames
        ],
        return_exceptions=True,
    )
    failures = [r for r in upload_results if isinstance(r, BaseException)]
    if failures:
        first = failures[0]
        code = getattr(first, "error_code", "percept_upload_failed")
        raise VisionAffectError(
            f"{len(failures)}/{len(upload_results)} frame uploads failed: {first}",
            error_code=code,
        )
    timings["upload_s"] = round(loop.time() - t1, 3)

    attachments = [
        AttachmentRefV1(
            kind="percept",
            sha256=sha,
            mime="image/jpeg",
            bytes=len(frame.jpeg),
            width=frame.width,
            height=frame.height,
            source_url=f"{base_url.rstrip('/')}/{sha}",
        )
        for sha, frame in zip(upload_results, sampled.frames)
    ]

    if not attachments:
        # Unreachable today (sample_frames raises on an empty clip), asserted
        # anyway: a chat request with ZERO images and a prompt reading
        # "0 webcam stills... read their affect" would invite the model to
        # invent one, and any confident JSON it returned would be published
        # ok=True and mirrored. Making the empty-shell rule structural here
        # rather than a consequence of a guard in a different module.
        raise VisionAffectError(
            "no frames survived sampling", error_code="no_frames"
        )

    route = str(settings.AFFECT_VISION_LLM_ROUTE or "chat")
    rpc_corr = str(uuid4())
    reply_channel = f"{_REPLY_PREFIX}:{rpc_corr}"
    payload = ChatRequestPayload(
        messages=[
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=_build_prompt(transcript, len(attachments))),
        ],
        route=route,
        attachments=attachments,
        options={
            "llm_route": route,
            "purpose": "classify",
            "max_tokens": int(settings.AFFECT_VISION_MAX_TOKENS),
            "temperature": 0.2,
            "structured_output_method": "json_object_schema",
            "structured_output_schema": _RESPONSE_SCHEMA,
            "chat_template_kwargs": {"enable_thinking": False},
            # This read is about Juniper's face, not a conversational turn --
            # it must not be ingested as a Spark candidate the way an ordinary
            # chat completion would be.
            "skip_spark_candidate_publish": True,
        },
    )
    env = BaseEnvelope(
        kind="llm.chat.request",
        source=ServiceRef(
            name=settings.SERVICE_NAME,
            version=settings.SERVICE_VERSION,
            node=settings.NODE_NAME,
        ),
        correlation_id=rpc_corr,
        reply_to=reply_channel,
        payload=payload.model_dump(mode="json"),
    )

    t2 = loop.time()
    try:
        msg = await bus.rpc_request(
            str(settings.CHANNEL_LLM_INTAKE),
            env,
            reply_channel=reply_channel,
            timeout_sec=float(settings.AFFECT_VISION_RPC_TIMEOUT_S),
        )
    except asyncio.TimeoutError as exc:
        raise VisionAffectError(
            f"llm gateway did not reply within {settings.AFFECT_VISION_RPC_TIMEOUT_S}s",
            error_code="timeout",
        ) from exc
    timings["generate_s"] = round(loop.time() - t2, 3)
    timings["total_s"] = round(loop.time() - t0, 3)

    if not isinstance(msg, dict):
        raise VisionAffectError(
            f"gateway reply was not a message dict: {type(msg).__name__}",
            error_code="invalid_reply",
        )
    decoded = bus.codec.decode(msg.get("data"))
    # `decoded.envelope` is checked as well as `decoded.ok`, matching the
    # sibling RPC path in this service (main.py's _call_worker). Without it an
    # ok-but-envelope-less decode raises AttributeError, which the caller's
    # generic handler reports as error_code="vision_unexpected" -- destroying
    # the stable error code exactly on the failure an operator would triage.
    if not decoded.ok or not decoded.envelope:
        raise VisionAffectError(
            f"undecodable gateway reply: {decoded.error}", error_code="invalid_reply"
        )
    reply_payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
    content = str(reply_payload.get("content") or reply_payload.get("text") or "").strip()
    if not content:
        # An empty completion is a real failure, not a neutral affect read.
        # CLAUDE.md: raw_len=0 is never a success state.
        raise VisionAffectError(
            "gateway returned an empty completion", error_code="empty_completion"
        )

    try:
        obj = parse_json_object(content)
        affect = AffectReadV1.model_validate(obj)
    except Exception as exc:  # noqa: BLE001 -- any parse/validation failure is one failure mode
        raise VisionAffectError(
            f"model output did not validate as AffectReadV1: {exc}",
            error_code="invalid_affect_json",
        ) from exc

    return VisionAffectResult(
        affect=affect,
        raw_response=content,
        face_detection=sampled.as_meta(),
        frames_used=len(attachments),
        timings=timings,
        # ChatResultPayload's field is `model_used`, NOT `model` -- caught by
        # the live smoke on 2026-08-26, which returned None here and would
        # have written a permanently-NULL model_ckpt column for every vision
        # row. `raw.model` is llama.cpp's own echo of the served path, kept as
        # a fallback because the gateway does not always populate model_used.
        model=(
            reply_payload.get("model_used")
            or (reply_payload.get("raw") or {}).get("model")
            or reply_payload.get("served_by")
        ),
    )
