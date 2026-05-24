from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.state import RouterState


def test_mark_dispatched_tracks_inflight() -> None:
    rs = RouterState()
    rs.mark_dispatched(
        correlation_id="abc",
        camera_id="cam1",
        image_path="/tmp/x.jpg",
        task_type="retina_fast",
        reply_to="orion:vision:reply:abc",
        now=100.0,
        frame_ts=99.0,
    )
    assert rs.inflight_total() == 1
    assert "abc" in rs.cameras["cam1"].inflight
    assert rs.pending["abc"].task_type == "retina_fast"


def test_clear_pending_drops_inflight() -> None:
    rs = RouterState()
    rs.mark_dispatched(
        correlation_id="abc",
        camera_id="cam1",
        image_path="/tmp/x.jpg",
        task_type="retina_fast",
        reply_to="orion:vision:reply:abc",
        now=100.0,
        frame_ts=None,
    )
    rs.clear_pending("abc", now=110.0)
    assert rs.inflight_total() == 0
    assert "abc" not in rs.pending
