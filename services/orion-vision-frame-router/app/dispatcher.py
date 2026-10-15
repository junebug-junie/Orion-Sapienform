from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import TYPE_CHECKING

from loguru import logger
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionFramePointerPayload, VisionTaskResultPayload

from orion.vision.stationary import box_iou
from orion.schemas.vision_organ_projection import FAILURE_HOST_ERROR, FAILURE_INVALID_REPLY, FAILURE_TIMEOUT

from . import grammar_emit
from .envelopes import make_host_task_envelope, make_secondary_task_envelope
from .host_trigger import extract_host_trigger_labels, stream_id_from_host_result
from .metrics import RouterMetrics
from .policy import FrameDispatchPolicy
from .settings import Settings
from .state import RouterState

if TYPE_CHECKING:
    from orion.core.bus.async_service import OrionBusAsync


class FrameDispatcher:
    def __init__(
        self,
        *,
        settings: Settings,
        policy: FrameDispatchPolicy,
        state: RouterState,
        metrics: RouterMetrics,
        bus: OrionBusAsync | None,
    ) -> None:
        self.settings = settings
        self.policy = policy
        self.state = state
        self.metrics = metrics
        self.bus = bus
        self._state_lock = asyncio.Lock()
        self._background: set[asyncio.Task] = set()

    async def handle_frame_envelope(self, env: BaseEnvelope) -> None:
        try:
            await self._handle_frame_envelope_inner(env)
        except Exception as exc:
            self.metrics.last_error = f"frame_handler_error: {exc}"

    async def _handle_frame_envelope_inner(self, env: BaseEnvelope) -> None:
        try:
            frame = VisionFramePointerPayload.model_validate(env.payload)
        except Exception as exc:
            self.metrics.last_error = f"invalid_frame_payload: {exc}"
            return

        self.metrics.record_seen()
        camera_id = frame.camera_id or "unknown"
        organ_stream = _organ_stream(frame.stream_id, camera_id)
        # DRY_RUN never sends tasks, so every dispatch would time out and read as
        # a dead host: the organ counts frames only.
        organ = grammar_emit.get_recorder()
        if organ is not None:
            organ.record_frame(organ_stream)
        organ_tasks = organ if not self.settings.DRY_RUN else None
        image_path = (frame.image_path or "").strip()
        image_path_exists: bool | None = None
        if image_path and self.policy.require_image_path_exists(camera_id):
            image_path_exists = await asyncio.to_thread(os.path.isfile, image_path)

        async with self._state_lock:
            decision = self.policy.decide(
                env,
                self.state,
                now=time.time(),
                image_path_exists=image_path_exists,
            )
            if not decision.should_dispatch:
                self.metrics.record_skip(decision.reason)
                if organ is not None:
                    organ.record_skip(organ_stream, decision.reason)
                return

            task = self.policy.build_task_request(frame, env, decision)
            corr = str(env.correlation_id)
            reply_to = f"{self.settings.CHANNEL_REPLY_PREFIX}:{corr}"
            task_env = make_host_task_envelope(
                frame_env=env,
                frame=frame,
                task=task,
                service_name=self.settings.SERVICE_NAME,
                service_version=self.settings.SERVICE_VERSION,
                reply_to=reply_to,
            )

            if not self.settings.DRY_RUN and self.bus:
                await self.bus.publish(self.settings.CHANNEL_HOST_INTAKE, task_env)

            want_caption = bool((task.request or {}).get("want_caption"))
            logger.info(
                "[ROUTER] dispatch tier={} task_type={} want_caption={} camera_id={} stream_id={}",
                decision.dispatch_tier,
                task.task_type,
                want_caption,
                camera_id,
                frame.stream_id,
            )

            self.state.mark_dispatched(
                correlation_id=corr,
                camera_id=frame.camera_id or "unknown",
                image_path=frame.image_path or "",
                task_type=task.task_type,
                reply_to=reply_to,
                now=time.time(),
                frame_ts=frame.frame_ts,
                stream_id=frame.stream_id,
                want_caption=want_caption,
                dispatch_tier=decision.dispatch_tier,
                percept_sha256=frame.sha256,
            )
            self.metrics.record_dispatch()
            if organ_tasks is not None:
                organ_tasks.record_dispatch(organ_stream)

            # Secondary, independent dispatch -- see policy.decide_identity's
            # docstring for why this is not folded into decision/task above.
            # Own corr_id (make_secondary_task_envelope), own reply_to, own
            # RouterState.mark_dispatched call: the reply is handled by the
            # same _handle_reply_envelope_inner as any other task this router
            # owns (it clears pending / counts metrics identically), but its
            # actual CONTENT reaches presence.py and orion-vision-council
            # over orion-vision-host's dedicated identity broadcast channel,
            # not through anything this dispatcher does with the reply.
            now = time.time()
            if self.policy.decide_identity(decision, camera_id=camera_id, state=self.state, now=now):
                identity_task = self.policy.build_identity_task_request(frame, env, decision)
                cam = self.state.camera(camera_id)
                cam.last_identity_dispatch_ts = now
                still_url = str(decision.identity_dispatch_cfg.get("still_url") or "").strip()
                if still_url and not self.settings.DRY_RUN:
                    # The still takes 2-5 s to grab, so it runs off this handler (which holds the
                    # state lock for every camera); decide_identity skips this camera meanwhile.
                    cam.identity_still_pending = True
                    timeout = float(decision.identity_dispatch_cfg.get("still_timeout_sec", 10.0))
                    job = asyncio.create_task(
                        self._identity_with_still(
                            frame, env, identity_task, camera_id, organ_tasks, organ_stream, still_url, timeout
                        )
                    )
                    self._background.add(job)
                    job.add_done_callback(self._background.discard)
                else:
                    await self._publish_identity(frame, env, identity_task, camera_id, organ_tasks, organ_stream)

    async def _identity_with_still(
        self, frame, env, identity_task, camera_id, organ_tasks, organ_stream, still_url: str, timeout: float
    ) -> None:
        """Swap the substream frame for a full-resolution still from the camera's capture service
        (orion-vision-edge POST /still) before the face check. A 640x480 frame leaves a face at
        the desk about 25 px wide; the face model works from 160 px. Any failure falls back to
        the substream frame, so a broken still never costs the face check itself."""
        request = dict(identity_task.request)
        meta = dict(identity_task.meta or {})
        try:
            still = await asyncio.to_thread(request_still, still_url, timeout)
            request["image_path"] = still["image_path"]
            request.pop("percept_sha256", None)  # the still is a different image than the frame's hash
            request["image_source"] = "hires_still"
            meta["still_grab_ms"] = still.get("grab_ms")
            meta["still_size"] = [still.get("width"), still.get("height")]
            meta["still_frame_ts"] = still.get("frame_ts")   # frame_ts stays the triggering frame's
            self.metrics.identity_still_total += 1
        except Exception as exc:  # noqa: BLE001
            request["image_source"] = "stream_frame"
            meta["still_error"] = str(exc) if isinstance(exc, StillError) else type(exc).__name__
            self.metrics.identity_still_fallback_total += 1
            logger.warning("[ROUTER] still_fallback camera_id={} error={}", camera_id, meta["still_error"])
        task = identity_task.model_copy(update={"request": request, "meta": meta})
        # max_inflight_total was checked when the check was decided, 2-10 s ago; at most one still
        # is pending per camera, so this publish can overshoot the cap by one per camera.
        async with self._state_lock:
            try:
                await self._publish_identity(frame, env, task, camera_id, organ_tasks, organ_stream)
            except Exception as exc:  # noqa: BLE001 -- a background task has no caller to raise to
                self.metrics.last_error = f"identity_still_publish_error: {exc}"
            finally:
                self.state.camera(camera_id).identity_still_pending = False

    async def _publish_identity(self, frame, env, identity_task, camera_id, organ_tasks, organ_stream) -> None:
        """Publish the identity_face task and book it. Caller holds the state lock."""
        now = time.time()
        identity_corr = str(uuid.uuid4())
        identity_reply_to = f"{self.settings.CHANNEL_REPLY_PREFIX}:{identity_corr}"
        identity_env = make_secondary_task_envelope(
            frame_env=env,
            frame=frame,
            task=identity_task,
            service_name=self.settings.SERVICE_NAME,
            service_version=self.settings.SERVICE_VERSION,
            reply_to=identity_reply_to,
            correlation_id=identity_corr,
        )
        if not self.settings.DRY_RUN and self.bus:
            await self.bus.publish(self.settings.CHANNEL_HOST_INTAKE, identity_env)
        self.state.mark_dispatched(
            correlation_id=identity_corr,
            camera_id=frame.camera_id or "unknown",
            image_path=str(identity_task.request.get("image_path") or frame.image_path or ""),
            task_type=identity_task.task_type,
            reply_to=identity_reply_to,
            now=now,
            frame_ts=frame.frame_ts,
            stream_id=frame.stream_id,
            # is_primary=False: does not consume the primary tier's
            # per-camera inflight slot or re-pace its dispatch
            # clock -- see mark_dispatched's own docstring for the
            # real bug this prevents (three review passes found it
            # independently, 2026-08-26).
            is_primary=False,
        )
        self.metrics.record_identity_dispatch()
        if organ_tasks is not None:
            organ_tasks.record_dispatch(organ_stream, identity=True)
        logger.info(
            "[ROUTER] identity_dispatch camera_id={} stream_id={} corr={} image_source={}",
            camera_id,
            frame.stream_id,
            identity_corr,
            identity_task.request.get("image_source", "stream_frame"),
        )

    async def handle_reply_envelope(self, env: BaseEnvelope) -> None:
        try:
            await self._handle_reply_envelope_inner(env)
        except Exception as exc:
            self.metrics.last_error = f"reply_handler_error: {exc}"

    async def _handle_reply_envelope_inner(self, env: BaseEnvelope) -> None:
        # Ownership (clear_pending, needs only env.correlation_id) is checked
        # before the full VisionTaskResultPayload.model_validate -- narrows
        # typed-object construction and trigger-label extraction to corr_ids
        # this router actually dispatched, instead of running that for every
        # reply on the orion:vision:reply:* wildcard pattern. See README.md's
        # "Wildcard reply-channel fan-out" section for what this does and
        # does not close, and test_reply_for_unowned_correlation_id_never_
        # deserializes_payload / test_owned_reply_with_malformed_payload_is_
        # not_counted_as_a_valid_reply below for the two behaviors it locks
        # in. clear_pending itself still needs the lock (mutates self.state,
        # shared with _handle_frame_envelope_inner); model_validate and the
        # pure label/stream_id helpers do not, so they run lock-free.
        corr = str(env.correlation_id)
        async with self._state_lock:
            cleared = self.state.clear_pending(corr, now=time.time())
        if not cleared:
            return

        if cleared.lane == "caption":
            self._handle_caption_reply(env, cleared)
            return

        try:
            result = VisionTaskResultPayload.model_validate(env.payload)
        except Exception as exc:
            # Owned corr_id, malformed payload: count as an error, not a
            # silently-dropped timeout and not a valid reply -- the pending
            # slot is already cleared above (no reason to hold it open for
            # sweep_timeouts when vision-host demonstrably did respond).
            self.metrics.last_error = f"invalid_reply_payload: {exc}"
            self.metrics.host_errors_total += 1
            _organ_failure(cleared, FAILURE_INVALID_REPLY)
            return

        self.metrics.host_replies_total += 1
        if not result.ok:
            self.metrics.host_errors_total += 1
            _organ_failure(cleared, result.error_code or FAILURE_HOST_ERROR)
            return
        _organ_reply_ok(cleared, result)

        stream_id = stream_id_from_host_result(result, fallback_stream_id=cleared.stream_id)
        allowed = set(self.policy.trigger_labels_for(cleared.camera_id, stream_id))
        labels = extract_host_trigger_labels(result, allowed=allowed)
        if labels and stream_id:
            async with self._state_lock:
                self.state.record_activity(stream_id, labels, now=time.time())
            self.metrics.host_trigger_updates_total += 1
            logger.info("[ROUTER] host_trigger stream={} labels={}", stream_id, labels)

        await self._maybe_dispatch_caption(env, cleared, result, stream_id)

    async def _maybe_dispatch_caption(
        self, reply_env: BaseEnvelope, task, result: VisionTaskResultPayload, stream_id: str | None
    ) -> None:
        """Ask the VLM lane for a caption of the frame the detector just read,
        if this stream opted in (``caption_dispatch``), the frame was on an
        allowed tier, and the detector found at least one gate-label object.

        Chained after the detect reply, never in front of it: detection is
        never held up by the ~3 s VLM call. The lane has its own in-flight cap
        and per-camera clock; it never takes an athena host slot. A missed
        caption (busy, timed out, lane down) is just no caption -- nothing
        falls back to the host's BLIP, which this stream does not request.
        """
        if not task.is_primary or task.task_type == "identity_face":
            return
        cfg = self.policy.caption_dispatch_for(task.camera_id, stream_id)
        if not cfg:
            return
        tiers = [str(t) for t in (cfg.get("tiers") or ["triggered"])]
        if task.dispatch_tier not in tiers:
            return
        objects = self.policy.caption_gate_objects(result, cfg)
        if not objects:
            self.metrics.record_caption_skip("no_detection")
            return
        if not (task.image_path or task.percept_sha256):
            self.metrics.record_caption_skip("no_frame_address")
            return
        now = time.time()
        async with self._state_lock:
            cam = self.state.camera(task.camera_id)
            min_s = float(cfg.get("min_seconds_between_dispatch", 8.0))
            if cam.last_caption_dispatch_ts is not None and (now - cam.last_caption_dispatch_ts) < min_s:
                self.metrics.record_caption_skip("rate_limited")
                return
            if _same_scene(objects, cam, cfg, now=now):
                self.metrics.record_caption_skip("same_scene")
                return
            if self.state.inflight_lane("caption") >= int(cfg.get("max_inflight", 1)):
                self.metrics.record_caption_skip("lane_inflight_limit")
                return
            caption_task = self.policy.build_caption_task_request(
                task, stream_id=stream_id, source_objects=objects, cfg=cfg
            )
            corr = str(uuid.uuid4())
            reply_to = f"{self.settings.CHANNEL_REPLY_PREFIX}:{corr}"
            caption_env = reply_env.derive_child(
                kind="vision.task.request",
                source=ServiceRef(name=self.settings.SERVICE_NAME, version=self.settings.SERVICE_VERSION),
                payload=caption_task,
                reply_to=reply_to,
                correlation_id=corr,
            )
            if not self.settings.DRY_RUN and self.bus:
                await self.bus.publish(self.settings.CHANNEL_CAPTION_INTAKE, caption_env)
            cam.last_caption_dispatch_ts = now
            cam.last_caption_objects = [{"label": o["label"], "box_xyxy": o["box_xyxy"]} for o in objects]
            self.state.mark_dispatched(
                correlation_id=corr,
                camera_id=task.camera_id,
                image_path=task.image_path,
                task_type="caption_frame",
                reply_to=reply_to,
                now=now,
                frame_ts=task.frame_ts,
                stream_id=stream_id,
                is_primary=False,
                want_caption=True,
                lane="caption",
                dispatch_tier=task.dispatch_tier,
                timeout_s=float(cfg.get("timeout_sec", 20.0)),
            )
            self.metrics.caption_dispatched_total += 1
            organ = grammar_emit.get_recorder()
            if organ is not None and not self.settings.DRY_RUN:
                # Requested at dispatch: a lane that never answers then reads
                # as "asked, no caption", not as "never asked".
                organ.record_caption_lane(_organ_stream(stream_id, task.camera_id), requested=True)
        logger.info(
            "[ROUTER] caption_dispatch camera_id={} stream_id={} corr={} labels={}",
            task.camera_id,
            stream_id,
            corr,
            sorted({o["label"] for o in objects}),
        )

    def _handle_caption_reply(self, env: BaseEnvelope, task) -> None:
        """A VLM-lane caption reply. Counted here, honestly by outcome; the
        caption itself reaches orion-vision-window over the lane's own
        (walkway-scoped) artifact broadcast, where the detector-agreement gate
        decides whether it becomes narrative."""
        try:
            result = VisionTaskResultPayload.model_validate(env.payload)
        except Exception as exc:  # noqa: BLE001
            self.metrics.last_error = f"invalid_caption_reply_payload: {exc}"
            self.metrics.record_caption_failure(FAILURE_INVALID_REPLY)
            return
        if not result.ok:
            self.metrics.record_caption_failure(result.error_code or FAILURE_HOST_ERROR)
            logger.warning(
                "[ROUTER] caption_failed stream_id={} corr={} error={}",
                task.stream_id, task.correlation_id, result.error_code or result.error,
            )
            return
        outputs = result.artifact.outputs if result.artifact is not None else None
        caption = outputs.caption if outputs is not None else None
        text = str(caption.text or "").strip() if caption is not None else ""
        organ = grammar_emit.get_recorder()
        if organ is not None and text:
            # Only the yield moves; replies_ok/failed stay the athena host's.
            organ.record_caption_lane(_organ_stream(task.stream_id, task.camera_id), captioned=True)
        if text:
            self.metrics.caption_captioned_total += 1
            logger.info("[ROUTER] caption_ok stream_id={} corr={} text={!r}", task.stream_id, task.correlation_id, text[:160])
        else:
            reason = (caption.rejected_reason if caption is not None else None) or "no_caption"
            self.metrics.record_caption_no_caption(reason)
            logger.info("[ROUTER] caption_none stream_id={} corr={} reason={}", task.stream_id, task.correlation_id, reason)

    async def sweep_timeouts(self, *, now: float) -> int:
        async with self._state_lock:
            expired = self.state.expired_correlation_ids(
                now=now, timeout_s=self.settings.TASK_TIMEOUT_SECONDS
            )
            cleared = 0
            for cid in expired:
                task = self.state.clear_pending(cid, now=now)
                if task:
                    if task.lane == "caption":
                        # The VLM lane, not the athena host: its own counter.
                        self.metrics.record_caption_failure(FAILURE_TIMEOUT)
                    else:
                        self.metrics.host_timeouts_total += 1
                        if not self.settings.DRY_RUN:
                            _organ_failure(task, FAILURE_TIMEOUT)
                    cleared += 1
            return cleared


def _same_scene(objects: list[dict], cam, cfg: dict, *, now: float) -> bool:
    """Every gate box now matches (same label, IoU >= same_scene_iou) a box of
    the last captioned frame, captioned within same_scene_ttl_sec."""
    ttl = float(cfg.get("same_scene_ttl_sec", 0) or 0)
    if ttl <= 0 or not cam.last_caption_objects or cam.last_caption_dispatch_ts is None:
        return False
    if (now - cam.last_caption_dispatch_ts) > ttl:
        return False
    thr = float(cfg.get("same_scene_iou", 0.5))
    return all(
        any(p["label"] == o["label"] and box_iou(p["box_xyxy"], o["box_xyxy"]) >= thr for p in cam.last_caption_objects)
        for o in objects
    )


def _organ_stream(stream_id: str | None, camera_id: str | None) -> str:
    return str(stream_id or camera_id or "unknown")


def _organ_failure(task, failure_class: str) -> None:
    organ = grammar_emit.get_recorder()
    if organ is not None:
        organ.record_failure(_organ_stream(task.stream_id, task.camera_id), failure_class)


def _organ_reply_ok(task, result: VisionTaskResultPayload) -> None:
    organ = grammar_emit.get_recorder()
    if organ is None:
        return
    outputs = result.artifact.outputs if result.artifact is not None else None
    objects = outputs.objects if outputs is not None else None
    caption = outputs.caption if outputs is not None else None
    organ.record_reply_ok(
        _organ_stream(task.stream_id, task.camera_id),
        primary=task.is_primary,
        # None (no objects key) is an embed-only/identity reply, not an empty detection
        objects=None if objects is None else len(objects),
        caption_requested=task.want_caption,
        caption_present=bool(caption is not None and str(caption.text or "").strip()),
    )


class StillError(RuntimeError):
    """The capture service answered but gave no still (its own error code)."""


def request_still(url: str, timeout: float) -> dict:
    """POST to orion-vision-edge's /still; returns its JSON (image_path, width, height, grab_ms)."""
    req = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        try:
            code = json.loads(exc.read().decode("utf-8") or "{}").get("error")
        except Exception:  # noqa: BLE001
            code = None
        raise StillError(code or f"http_{exc.code}") from exc
    if not body.get("ok") or not body.get("image_path"):
        raise StillError(str(body.get("error") or "no_image_path"))
    return body
