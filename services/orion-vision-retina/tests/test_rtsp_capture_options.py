"""RTSP gets TCP + an FFmpeg socket timeout; webcam/folder/mock do not.

The option name `timeout` (microseconds) is the one the FFmpeg bundled with
this image's OpenCV 4.10 (FFmpeg 5.1) honours -- measured in the built image;
`stimeout` and `rw_timeout` were ignored there.
"""

import os

import cv2
import pytest

from app import sources
from app.settings import Settings
from app.sources import RtspFrameSource, create_frame_source, rtsp_capture_options


def test_options_string_defaults_to_tcp_with_microsecond_timeout():
    assert rtsp_capture_options("tcp", 5.0) == "rtsp_transport;tcp|timeout;5000000"
    assert rtsp_capture_options(" UDP ", 2.5) == "rtsp_transport;udp|timeout;2500000"
    with pytest.raises(ValueError):
        rtsp_capture_options("carrier-pigeon", 5)


def test_rtsp_source_sets_ffmpeg_capture_options(monkeypatch):
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    src = create_frame_source("rtsp", "rtsp://cam/x", rtsp_io_timeout_seconds=4)
    assert isinstance(src, RtspFrameSource)
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp|timeout;4000000"
    assert src.stall_seconds == 15.0


def test_rtsp_default_opener_passes_open_and_read_timeouts(monkeypatch):
    seen = {}

    class _Cap:
        def set(self, *_a):
            return True

    def fake_vc(url, api, params):
        seen.update(url=url, api=api, params=params)
        return _Cap()

    monkeypatch.setattr(sources.cv2, "VideoCapture", fake_vc)
    src = RtspFrameSource("rtsp://cam/x", io_timeout_seconds=5)
    src._open_sync()
    assert seen["api"] == cv2.CAP_FFMPEG
    p = dict(zip(seen["params"][::2], seen["params"][1::2]))
    assert p[cv2.CAP_PROP_READ_TIMEOUT_MSEC] == 5000
    assert p[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC] == 10000


@pytest.mark.parametrize("kind,src_arg", [("webcam", "0"), ("folder", "/nope"), ("mock", "/nope")])
def test_non_rtsp_sources_leave_ffmpeg_options_alone(monkeypatch, kind, src_arg):
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    create_frame_source(kind, src_arg)
    assert "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ


def test_settings_ship_tcp_and_watchdog_on(monkeypatch):
    for k in ("RETINA_RTSP_TRANSPORT", "RETINA_RTSP_STALL_SECONDS", "RETINA_RTSP_IO_TIMEOUT_SECONDS"):
        monkeypatch.delenv(k, raising=False)
    s = Settings(_env_file=None)
    assert s.RETINA_RTSP_TRANSPORT == "tcp"
    assert s.RETINA_RTSP_STALL_SECONDS > s.RETINA_RTSP_IO_TIMEOUT_SECONDS > 0
    with pytest.raises(ValueError):
        Settings(_env_file=None, RETINA_RTSP_TRANSPORT="quic")
