"""On-demand short video+audio clip capture via ffmpeg subprocess, for
AffectGPT (see services/orion-affectgpt-worker/README.md). Distinct from
sources.py's WebcamFrameSource: OpenCV's VideoCapture has no audio support
at all, so a genuinely new capture path was needed rather than an extension
of the existing single-frame one.

**UNVERIFIED against real hardware.** This was written without access to
carbon (tailnet policy blocks SSH there for this session, same as the
percept-store/webcam wiring before it) -- no webcam, no microphone, no
ffmpeg build to test against. The subprocess construction and error handling
are tested (tests/test_clip_capture.py, mocked ffmpeg), but the actual
capture -- real device names, real audio backend, real timing -- has not
been exercised live. Must be verified on carbon before anything is built on
top of it. See README "Live verification needed".

Nothing is written to disk beyond a TemporaryDirectory that is always
cleaned up -- same "no spool on a personal laptop" principle as
frame_store.upload_frame: this runs on carbon, a personal machine, and must
not accumulate a backlog of clips of someone's face and voice.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass


class ClipCaptureError(RuntimeError):
    """ffmpeg failed to produce a usable clip."""


@dataclass
class ClipCaptureResult:
    video_bytes: bytes
    audio_bytes: bytes
    duration_sec: float


async def capture_clip(
    *,
    ffmpeg_bin: str,
    video_device: str,
    audio_input: str,
    duration_sec: float,
    video_framerate: int,
    width: int | None,
    height: int | None,
    timeout_sec: float,
) -> ClipCaptureResult:
    """Record `duration_sec` of video (v4l2) and audio (pulse) concurrently,
    read the results into memory, and delete the temp files.

    Concurrent, not sequential: video and audio are captured from two
    independent devices, so running them one after another (simpler, but
    considered and rejected) would have the two clips describe two different
    few-second windows instead of the same moment -- a real accuracy cost for
    what is supposed to be a read of "affect right now".
    """
    with tempfile.TemporaryDirectory(prefix="orion-retina-clip-") as tmpdir:
        video_path = os.path.join(tmpdir, "clip.mp4")
        audio_path = os.path.join(tmpdir, "clip.wav")

        video_cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "v4l2",
            "-framerate",
            str(video_framerate),
            *(["-video_size", f"{width}x{height}"] if width and height else []),
            "-i",
            video_device,
            "-t",
            str(duration_sec),
            "-an",
            "-pix_fmt",
            "yuv420p",
            video_path,
        ]
        # 16kHz mono: matches what AffectGPT's own audio pipeline resamples
        # to anyway (my_affectgpt's load_audio expects speech-model-rate
        # input) -- not required for correctness (it would resample
        # regardless) but keeps the clip small and avoids a format surprise.
        audio_cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "pulse",
            "-i",
            audio_input,
            "-t",
            str(duration_sec),
            "-ac",
            "1",
            "-ar",
            "16000",
            audio_path,
        ]

        await asyncio.gather(
            _run_ffmpeg(video_cmd, timeout_sec=timeout_sec),
            _run_ffmpeg(audio_cmd, timeout_sec=timeout_sec),
        )

        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            raise ClipCaptureError(f"ffmpeg produced no video output (device={video_device!r})")
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            raise ClipCaptureError(f"ffmpeg produced no audio output (input={audio_input!r})")

        with open(video_path, "rb") as f:
            video_bytes = f.read()
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

    return ClipCaptureResult(
        video_bytes=video_bytes, audio_bytes=audio_bytes, duration_sec=duration_sec
    )


async def _run_ffmpeg(cmd: list[str], *, timeout_sec: float) -> None:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ClipCaptureError(f"ffmpeg timed out after {timeout_sec}s: {' '.join(cmd)}")
    if proc.returncode != 0:
        tail = stderr.decode(errors="replace")[-2000:]
        raise ClipCaptureError(f"ffmpeg exited {proc.returncode}: {' '.join(cmd)}\n{tail}")
