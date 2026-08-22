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
import signal
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

        # Both processes are started up front and their handles kept, rather
        # than hiding them inside a helper coroutine, so a sibling failure
        # can kill the OTHER real OS process directly. An earlier version
        # relied on asyncio Task.cancel() propagating a CancelledError
        # through proc.communicate() to trigger cleanup -- review finding
        # (2026-08-22): asyncio.gather() does not cancel sibling awaitables
        # when one of them raises. Without this, a fast-failing audio
        # capture left the still-running video ffmpeg process orphaned,
        # recording into a tempdir about to be deleted, and holding
        # /dev/video0 open until it naturally finished -- breaking the
        # *next* capture call too.
        #
        # start_new_session=True + killing the whole process GROUP
        # (os.killpg), not just the tracked PID (proc.kill()), matters for a
        # real reason found live in this module's own test suite: a plain
        # proc.kill() on a process whose own child is still writing to the
        # same stdout/stderr pipes does NOT make communicate()/wait()
        # return promptly -- the pipe only reaches EOF once every process
        # holding its write end has exited, and an orphaned grandchild
        # (reparented, signal never delivered to it) keeps that pipe open
        # for however long IT still has left to run. Confirmed live: this
        # was the actual 5s-vs-0.0004s difference in the regression test
        # below, not a red herring.
        video_proc = await asyncio.create_subprocess_exec(
            *video_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        audio_proc = await asyncio.create_subprocess_exec(
            *audio_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        procs = {"video": (video_proc, video_cmd), "audio": (audio_proc, audio_cmd)}

        try:
            await asyncio.gather(
                *(
                    _wait_ffmpeg(proc, cmd, timeout_sec=timeout_sec)
                    for proc, cmd in procs.values()
                )
            )
        except Exception:
            for proc, _cmd in procs.values():
                _killpg(proc)
            for proc, _cmd in procs.values():
                if proc.returncode is None:
                    await proc.wait()
            raise

        video_bytes, audio_bytes = await asyncio.to_thread(
            _read_clip_files, video_path, audio_path, video_device, audio_input
        )

    return ClipCaptureResult(
        video_bytes=video_bytes, audio_bytes=audio_bytes, duration_sec=duration_sec
    )


def _read_clip_files(
    video_path: str, audio_path: str, video_device: str, audio_input: str
) -> tuple[bytes, bytes]:
    """Blocking file I/O, run via asyncio.to_thread -- not done inline in the
    coroutine (review finding, 2026-08-22): this service's event loop also
    runs the continuous webcam capture_loop and health_loop, and every other
    blocking call in this module (including this file's own upload_bytes
    calls one function away) already goes through to_thread."""
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        raise ClipCaptureError(f"ffmpeg produced no video output (device={video_device!r})")
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        raise ClipCaptureError(f"ffmpeg produced no audio output (input={audio_input!r})")
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    return video_bytes, audio_bytes


async def _wait_ffmpeg(
    proc: asyncio.subprocess.Process, cmd: list[str], *, timeout_sec: float
) -> None:
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        _killpg(proc)
        await proc.wait()
        raise ClipCaptureError(f"ffmpeg timed out after {timeout_sec}s: {' '.join(cmd)}")
    if proc.returncode != 0:
        tail = stderr.decode(errors="replace")[-2000:]
        raise ClipCaptureError(f"ffmpeg exited {proc.returncode}: {' '.join(cmd)}\n{tail}")


def _killpg(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the whole process group (see start_new_session=True above),
    not just proc.pid -- see the comment in capture_clip for why a plain
    proc.kill() is not enough. Best-effort: the process may already have
    exited (returncode set, or the group already reaped) between the caller
    checking and this running."""
    if proc.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
