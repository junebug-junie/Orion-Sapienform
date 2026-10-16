from __future__ import annotations

import os
import re

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "vision-retina"
    SERVICE_VERSION: str = "0.2.0"
    LOG_LEVEL: str = "INFO"

    ORION_BUS_URL: str = "redis://localhost:6379/0"
    ORION_BUS_ENFORCE_CATALOG: bool = False

    CHANNEL_RETINA_PUB: str = "orion:vision:frames"
    CHANNEL_RETINA_ERROR: str = "orion:vision:retina:error"
    CHANNEL_SYSTEM_HEALTH: str = "orion:system:health"

    RETINA_SOURCE_TYPE: str = "folder"
    RETINA_SOURCE: str = "/mnt/telemetry/vision/intake"
    RETINA_SOURCE_PATH: str | None = None
    RETINA_CAMERA_ID: str = "retina-cam-01"
    RETINA_STREAM_ID: str = "retina-stream-01"

    RETINA_FPS: float = 1.0
    RETINA_WIDTH: int | None = None
    RETINA_HEIGHT: int | None = None

    FRAME_STORAGE_DIR: str = "/mnt/telemetry/vision/frames"

    # How this node addresses the frames it publishes.
    #   local         -- write to FRAME_STORAGE_DIR, publish image_path.
    #                    Correct ONLY when the vision host shares this
    #                    filesystem. athena. Default, so nothing changes.
    #   percept_store -- encode in memory, POST to orion-percept-store, publish
    #                    sha256. The only option for a node with no shared
    #                    disk. Nothing is written locally, on purpose: a
    #                    capture agent on a personal laptop must not accumulate
    #                    a spool of webcam frames.
    RETINA_FRAME_MODE: str = "local"
    RETINA_PERCEPT_STORE_URL: str = ""
    RETINA_PERCEPT_STORE_TOKEN: str = ""
    RETINA_PERCEPT_TIMEOUT_SEC: float = 10.0
    FRAME_RETENTION_SECONDS: int = 300
    JPEG_QUALITY: int = 90

    HEALTH_INTERVAL_SECONDS: float = 10.0
    SOURCE_RECONNECT_SECONDS: float = 5.0

    # RTSP cameras only (webcam/folder/mock ignore these). See
    # app/sources.py RtspFrameSource and the 2026-10-15 walkway stall.
    #   TRANSPORT       tcp interleaves RTP on the RTSP connection: no UDP
    #                   loss/reorder (the walkway's smeared lower frame and
    #                   `RTP: bad cseq` lines). udp/http also accepted.
    #   IO_TIMEOUT      FFmpeg socket timeout + OpenCV read timeout: a quiet
    #                   socket returns from cap.read() after this, not never.
    #   STALL_SECONDS   no frame for this long -> WARNING. If the reader thread
    #                   also made no progress that long (wedged in FFmpeg),
    #                   abandon it and start a fresh capture. 0 disables. Must
    #                   exceed the reader's own recovery cycle
    #                   (3 x IO_TIMEOUT + SOURCE_RECONNECT_SECONDS).
    RETINA_RTSP_TRANSPORT: str = "tcp"
    RETINA_RTSP_IO_TIMEOUT_SECONDS: float = 5.0
    RETINA_RTSP_STALL_SECONDS: float = 30.0
    #   RECONNECT_QUIET_SECONDS  no RTSP (re)connect until this long after the
    #                   last camera session (reader reconnect, watchdog
    #                   replacement, pause/resume, heal, process start). A
    #                   Reolink keeps a dropped session alive; reconnecting
    #                   inside that window muxes the stale session in (bad
    #                   cseq + every frame smeared, 2026-10-15 and 2026-10-16).
    #                   Live 2026-10-16: 180 s stop/start was clean. 90 s is
    #                   above RFC 2326's 60 s default session timeout (Reolink
    #                   does not document its own). 0 disables.
    #   SESSION_STATE_DIR  where the last-session time is persisted so a
    #                   container restart/deploy also waits. Unset/unreadable
    #                   at boot -> wait one full quiet period.
    #   HEAL_*          live self-heal: every HEAL_CHECK_EVERY served frames,
    #                   score smear/decoder-gray (evals/frame_smear.py math).
    #                   When >= HEAL_MIN_SHARE of checks over
    #                   HEAL_WINDOW_SECONDS are corrupt, drop the session, wait
    #                   the quiet period (doubling per heal in a row, max 8x),
    #                   reconnect. Never reboots the camera.
    RETINA_RTSP_RECONNECT_QUIET_SECONDS: float = 90.0
    RETINA_RTSP_SESSION_STATE_DIR: str = "/var/lib/orion-vision-retina"
    RETINA_RTSP_HEAL_ENABLED: bool = True
    RETINA_RTSP_HEAL_CHECK_EVERY: int = 10
    RETINA_RTSP_HEAL_WINDOW_SECONDS: float = 120.0
    RETINA_RTSP_HEAL_MIN_SHARE: float = 0.8

    # On-demand video+audio clip capture for AffectGPT (app/clip_capture.py).
    # Separate from RETINA_SOURCE/RETINA_CAMERA_ID above -- those drive the
    # existing single-frame capture_loop, this is a distinct on-demand path
    # triggered via POST /capture/clip or the bus RPC twin. Defaults below
    # are live-verified correct for carbon's actual devices, 2026-08-22 (see
    # clip_capture.py module docstring).
    RETINA_CLIP_ENABLED: bool = False
    # Shared-secret gate for POST /capture/clip, same convention as
    # orion-percept-store's PERCEPT_STORE_TOKEN. Added after review
    # (2026-08-22): docs/operations/carbon-webcam.md's whole documented
    # security posture is "nothing needs to reach carbon inbound" -- true
    # when this app's HTTP surface was health-only, no longer true once a
    # POST here triggers a live webcam+mic recording. Empty disables the
    # check (acceptable only on a closed tailnet, same caveat as percept
    # store) -- strongly set this one given what the endpoint actually does.
    RETINA_CLIP_TOKEN: str = ""
    RETINA_CLIP_FFMPEG_BIN: str = "ffmpeg"
    RETINA_CLIP_VIDEO_DEVICE: str = "/dev/video0"
    # PipeWire's pulse-compatible socket (the modern default on most desktop
    # Linux) accepts "default" as the default source name, same as plain
    # PulseAudio. Override if carbon's audio backend names sources differently.
    RETINA_CLIP_AUDIO_INPUT: str = "default"
    RETINA_CLIP_DURATION_SEC: float = 8.0
    RETINA_CLIP_FRAMERATE: int = 15
    RETINA_CLIP_WIDTH: int | None = None
    RETINA_CLIP_HEIGHT: int | None = None
    RETINA_CLIP_TIMEOUT_SEC: float = 30.0
    # Minimum seconds between the END of one capture and the START of the
    # next, enforced regardless of caller/path (HTTP or bus RPC). Real risk
    # this closes (review finding, 2026-08-22): the bus RPC path has no
    # auth beyond bus reachability, so nothing else stops a bus-connected
    # caller from firing capture requests back-to-back -- this bounds that
    # to a known worst-case rate rather than de facto continuous recording.
    RETINA_CLIP_MIN_INTERVAL_SEC: float = 30.0

    # Bus-reachable twin of POST /capture/clip -- see orion/bus/channels.yaml
    # for why this exists (carbon has no reachable inbound HTTP surface at
    # all). Gated by RETINA_CLIP_ENABLED same as the HTTP route; no separate
    # token check here since the bus itself is the trust boundary (unlike an
    # HTTP port, nothing needs to be additionally exposed to reach it).
    CHANNEL_RETINA_CLIP_INTAKE: str = "orion:exec:request:RetinaClipCaptureService"
    CHANNEL_RETINA_CLIP_REPLY_PREFIX: str = "orion:retina:clip:reply"

    @field_validator("RETINA_CLIP_WIDTH", "RETINA_CLIP_HEIGHT", mode="before")
    @classmethod
    def _blank_clip_dims_to_none(cls, v: object) -> object:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("RETINA_RTSP_TRANSPORT", mode="before")
    @classmethod
    def _check_rtsp_transport(cls, v: object) -> str:
        t = str(v or "tcp").strip().lower()
        if t not in {"tcp", "udp", "http"}:
            raise ValueError(f"RETINA_RTSP_TRANSPORT must be tcp, udp or http, got {v!r}")
        return t

    @field_validator("JPEG_QUALITY", mode="before")
    @classmethod
    def _clamp_jpeg_quality(cls, v: object) -> int:
        try:
            n = int(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 90
        return max(1, min(100, n))

    @field_validator("RETINA_WIDTH", "RETINA_HEIGHT", mode="before")
    @classmethod
    def _blank_to_none(cls, v: object) -> object:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @model_validator(mode="after")
    def _check_rtsp_heal(self) -> "Settings":
        if self.RETINA_RTSP_RECONNECT_QUIET_SECONDS < 0:
            raise ValueError("RETINA_RTSP_RECONNECT_QUIET_SECONDS must be >= 0")
        if self.RETINA_RTSP_HEAL_CHECK_EVERY < 1:
            raise ValueError("RETINA_RTSP_HEAL_CHECK_EVERY must be >= 1")
        if self.RETINA_RTSP_HEAL_WINDOW_SECONDS <= 0:
            raise ValueError("RETINA_RTSP_HEAL_WINDOW_SECONDS must be > 0")
        if not 0 < self.RETINA_RTSP_HEAL_MIN_SHARE <= 1:
            raise ValueError("RETINA_RTSP_HEAL_MIN_SHARE must be in (0, 1]")
        return self

    def rtsp_session_state_path(self) -> str | None:
        """Per-stream file under RETINA_RTSP_SESSION_STATE_DIR (None if unset)."""
        d = (self.RETINA_RTSP_SESSION_STATE_DIR or "").strip()
        if not d:
            return None
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", self.RETINA_STREAM_ID or "stream")
        return os.path.join(d, f"rtsp_session_{safe}.json")

    @model_validator(mode="after")
    def _check_rtsp_stall_window(self) -> "Settings":
        stall = self.RETINA_RTSP_STALL_SECONDS
        floor = 3 * self.RETINA_RTSP_IO_TIMEOUT_SECONDS + self.SOURCE_RECONNECT_SECONDS
        if stall != 0 and stall <= floor:
            raise ValueError(
                f"RETINA_RTSP_STALL_SECONDS={stall} must be 0 or > {floor} "
                "(3 x RETINA_RTSP_IO_TIMEOUT_SECONDS + SOURCE_RECONNECT_SECONDS), "
                "or the watchdog races the reader's own reconnect"
            )
        return self

    @model_validator(mode="after")
    def _apply_source_path_alias(self) -> "Settings":
        if self.RETINA_SOURCE_PATH and "RETINA_SOURCE" not in self.model_fields_set:
            self.RETINA_SOURCE = self.RETINA_SOURCE_PATH
        return self


def get_settings() -> Settings:
    return Settings()
