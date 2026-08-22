"""Tests the subprocess construction and error handling of clip_capture.py
with a fake ffmpeg (a tiny script that writes real bytes or fails on
command). Does NOT exercise real hardware -- no webcam, no microphone here.
See services/orion-vision-retina/app/clip_capture.py module docstring: the
actual capture is UNVERIFIED and must be checked live on carbon before being
trusted.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "orion-vision-retina"))

import pytest

from app.clip_capture import ClipCaptureError, capture_clip


def _write_fake_ffmpeg(path: str, *, exit_code: int = 0, write_output: bool = True) -> None:
    """A fake ffmpeg: finds its own last argument (the output path) and
    writes fixed magic bytes there, honoring -f {v4l2,pulse} to pick which
    magic bytes so both concurrent invocations produce distinguishable,
    sniffable output -- same shape real ffmpeg would produce for -f v4l2
    (video) vs -f pulse (audio)."""
    script = f"""#!/bin/bash
set -e
is_audio=0
for arg in "$@"; do
    if [ "$arg" = "pulse" ]; then is_audio=1; fi
done
out="${{@: -1}}"
if [ "{exit_code}" != "0" ]; then
    echo "fake ffmpeg forced failure" >&2
    exit {exit_code}
fi
if [ "{str(write_output).lower()}" = "true" ]; then
    if [ "$is_audio" = "1" ]; then
        printf 'RIFF\\x00\\x00\\x00\\x00WAVE0000000000000000000000000000000' > "$out"
    else
        printf '\\x00\\x00\\x00\\x18ftypisom0000000000000000000000000000000' > "$out"
    fi
fi
exit 0
"""
    with open(path, "w") as f:
        f.write(script)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


@pytest.fixture
def fake_ffmpeg_ok(tmp_path):
    path = str(tmp_path / "fake_ffmpeg_ok.sh")
    _write_fake_ffmpeg(path, exit_code=0, write_output=True)
    return path


@pytest.fixture
def fake_ffmpeg_fails(tmp_path):
    path = str(tmp_path / "fake_ffmpeg_fail.sh")
    _write_fake_ffmpeg(path, exit_code=1, write_output=False)
    return path


@pytest.fixture
def fake_ffmpeg_empty_output(tmp_path):
    path = str(tmp_path / "fake_ffmpeg_empty.sh")
    _write_fake_ffmpeg(path, exit_code=0, write_output=False)
    return path


@pytest.mark.asyncio
async def test_captures_video_and_audio_concurrently(fake_ffmpeg_ok):
    result = await capture_clip(
        ffmpeg_bin=fake_ffmpeg_ok,
        video_device="/dev/video0",
        audio_input="default",
        duration_sec=1.0,
        video_framerate=15,
        width=None,
        height=None,
        timeout_sec=10.0,
    )
    assert result.video_bytes.startswith(b"\x00\x00\x00\x18ftyp")
    assert result.audio_bytes.startswith(b"RIFF")
    assert result.duration_sec == 1.0


@pytest.mark.asyncio
async def test_ffmpeg_nonzero_exit_raises_clip_capture_error(fake_ffmpeg_fails):
    with pytest.raises(ClipCaptureError, match="ffmpeg exited"):
        await capture_clip(
            ffmpeg_bin=fake_ffmpeg_fails,
            video_device="/dev/video0",
            audio_input="default",
            duration_sec=1.0,
            video_framerate=15,
            width=None,
            height=None,
            timeout_sec=10.0,
        )


@pytest.mark.asyncio
async def test_missing_output_raises_clip_capture_error_not_a_crash(fake_ffmpeg_empty_output):
    with pytest.raises(ClipCaptureError, match="produced no"):
        await capture_clip(
            ffmpeg_bin=fake_ffmpeg_empty_output,
            video_device="/dev/video0",
            audio_input="default",
            duration_sec=1.0,
            video_framerate=15,
            width=None,
            height=None,
            timeout_sec=10.0,
        )


@pytest.mark.asyncio
async def test_nonexistent_ffmpeg_binary_raises_cleanly(tmp_path):
    with pytest.raises((ClipCaptureError, FileNotFoundError, OSError)):
        await capture_clip(
            ffmpeg_bin=str(tmp_path / "does_not_exist"),
            video_device="/dev/video0",
            audio_input="default",
            duration_sec=1.0,
            video_framerate=15,
            width=None,
            height=None,
            timeout_sec=10.0,
        )


@pytest.mark.asyncio
async def test_no_files_left_on_disk_after_capture(fake_ffmpeg_ok, tmp_path, monkeypatch):
    """Privacy discipline: nothing survives beyond the TemporaryDirectory."""
    import tempfile

    seen_dirs: list[str] = []
    real_tempdir = tempfile.TemporaryDirectory

    class _Tracking(real_tempdir):
        def __enter__(self):
            d = super().__enter__()
            seen_dirs.append(d)
            return d

    monkeypatch.setattr(tempfile, "TemporaryDirectory", _Tracking)

    await capture_clip(
        ffmpeg_bin=fake_ffmpeg_ok,
        video_device="/dev/video0",
        audio_input="default",
        duration_sec=1.0,
        video_framerate=15,
        width=None,
        height=None,
        timeout_sec=10.0,
    )
    assert seen_dirs, "TemporaryDirectory was never used"
    assert not os.path.exists(seen_dirs[0]), "capture directory survived past the with-block"
