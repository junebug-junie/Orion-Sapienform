"""Run from repo root or the service dir alike; keep RtspFrameSource's
process-global FFmpeg options from leaking between tests."""

import sys
from pathlib import Path

import pytest

_SERVICE = str(Path(__file__).resolve().parents[1])
if _SERVICE not in sys.path:
    sys.path.insert(0, _SERVICE)


@pytest.fixture(autouse=True)
def _no_leaked_ffmpeg_options(monkeypatch):
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    yield
