from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.hub.association import build_hub_association_bundle
from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
)
from orion.substrate.felt_state_reader import SubstrateFeltStateReader


def _fresh_broadcast() -> AttentionBroadcastProjectionV1:
    return AttentionBroadcastProjectionV1(
        generated_at=datetime.now(timezone.utc),
        frame=AttentionFrameV1(open_loops=[]),
        attended_node_ids=["node-a", "node-b"],
    )


def _make_reader(*, max_age_sec: int = 120) -> SubstrateFeltStateReader:
    reader = SubstrateFeltStateReader(
        enabled=False, database_url="postgresql://unused", max_age_sec=max_age_sec
    )
    reader._enabled = True
    reader._engine = None
    return reader


def _now() -> datetime:
    return datetime.now(timezone.utc)


def test_association_read_fail_closed_when_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    stale = _fresh_broadcast().model_copy(
        update={"generated_at": datetime.now(timezone.utc) - timedelta(seconds=300)}
    )

    monkeypatch.setenv("ORION_ATTENTION_BROADCAST_ENABLED", "true")

    def _fake_read(*, reader_factory=None, **kwargs):
        return stale, None, "felt_state_reader"

    monkeypatch.setattr("orion.hub.association._read_association_data", _fake_read)

    bundle = build_hub_association_bundle(correlation_id="corr-stale", repair_bundle=None)
    assert bundle.broadcast_stale is True
    assert bundle.broadcast is not None


def test_association_read_fail_closed_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORION_ATTENTION_BROADCAST_ENABLED", "false")

    bundle = build_hub_association_bundle(correlation_id="corr-off", repair_bundle=None)
    assert bundle.broadcast is None
    assert bundle.broadcast_stale is True
    assert bundle.read_source == "felt_state_reader"


def test_association_fresh_broadcast_not_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORION_ATTENTION_BROADCAST_ENABLED", "true")
    broadcast = _fresh_broadcast()

    def fake_fetch(lane):
        if lane.ctx_key == "attention_broadcast":
            return (broadcast.model_dump(mode="json"), _now())
        return None

    reader = _make_reader()
    reader._fetch_lane = fake_fetch  # type: ignore[assignment]

    bundle = build_hub_association_bundle(
        correlation_id="corr-fresh",
        repair_bundle=None,
        reader_factory=lambda: reader,
    )
    assert bundle.broadcast_stale is False
    assert bundle.broadcast is not None
    assert bundle.broadcast.attended_node_ids == ["node-a", "node-b"]


def test_association_missing_projection_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORION_ATTENTION_BROADCAST_ENABLED", "true")

    def fake_fetch(lane):
        return None

    reader = _make_reader()
    reader._fetch_lane = fake_fetch  # type: ignore[assignment]

    bundle = build_hub_association_bundle(
        correlation_id="corr-missing",
        repair_bundle=None,
        reader_factory=lambda: reader,
    )
    assert bundle.broadcast is None
    assert bundle.broadcast_stale is True
    assert bundle.read_source == "felt_state_reader"


def test_association_execution_trajectory_slice_from_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORION_ATTENTION_BROADCAST_ENABLED", "true")
    broadcast = _fresh_broadcast()
    trajectory = {"active_step": "observe", "tick": 3}

    def fake_fetch(lane):
        if lane.ctx_key == "attention_broadcast":
            return (broadcast.model_dump(mode="json"), _now())
        if lane.ctx_key == "execution_trajectory_projection":
            return (trajectory, _now())
        return None

    reader = _make_reader()
    reader._fetch_lane = fake_fetch  # type: ignore[assignment]

    bundle = build_hub_association_bundle(
        correlation_id="corr-trajectory",
        repair_bundle=None,
        reader_factory=lambda: reader,
    )
    assert bundle.broadcast_stale is False
    assert bundle.execution_trajectory_slice == trajectory
