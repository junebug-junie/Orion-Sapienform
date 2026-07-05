from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.hub.association import build_hub_association_bundle
from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
)


def _fresh_broadcast() -> AttentionBroadcastProjectionV1:
    return AttentionBroadcastProjectionV1(
        generated_at=datetime.now(timezone.utc),
        frame=AttentionFrameV1(open_loops=[]),
        attended_node_ids=["node-a", "node-b"],
    )


def test_association_read_fail_closed_when_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    stale = _fresh_broadcast().model_copy(
        update={"generated_at": datetime.now(timezone.utc) - timedelta(seconds=300)}
    )

    monkeypatch.setenv("ORION_ATTENTION_BROADCAST_ENABLED", "true")

    def _fake_read(*, max_age_sec: int = 120, **kwargs):
        return stale, "felt_state_reader"

    monkeypatch.setattr("orion.hub.association._read_broadcast", _fake_read)

    bundle = build_hub_association_bundle(correlation_id="corr-stale", repair_bundle=None)
    assert bundle.broadcast_stale is True
    assert bundle.broadcast is not None


def test_association_read_fail_closed_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORION_ATTENTION_BROADCAST_ENABLED", "false")

    bundle = build_hub_association_bundle(correlation_id="corr-off", repair_bundle=None)
    assert bundle.broadcast is None
    assert bundle.broadcast_stale is True
    assert bundle.read_source == "felt_state_reader"
