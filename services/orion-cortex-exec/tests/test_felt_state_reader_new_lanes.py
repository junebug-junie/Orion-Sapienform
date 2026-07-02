from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.substrate_felt_state_reader import _LANES, SubstrateFeltStateReader


def _lane(ctx_key):
    return next(lane for lane in _LANES if lane.ctx_key == ctx_key)


def _reader() -> SubstrateFeltStateReader:
    return SubstrateFeltStateReader(
        enabled=True,
        database_url="postgresql://unused/unused",
        max_age_sec=120,
    )


def test_attention_broadcast_lane_registered():
    lane = _lane("attention_broadcast")
    assert lane.table == "substrate_attention_broadcast_projection"
    assert lane.payload_col == "projection_json"
    assert lane.ts_col == "generated_at"
    assert lane.projection_id == "substrate.attention.broadcast.v1"
    assert lane.max_age_sec is None  # global 120s gate applies


def test_episode_lane_registered_with_extended_max_age():
    lane = _lane("episode_summary")
    assert lane.table == "substrate_episode_summaries"
    assert lane.payload_col == "episode_json"
    assert lane.ts_col == "created_at"
    assert lane.projection_id is None
    assert lane.max_age_sec == 1800


def test_hydrate_injects_episode_older_than_global_gate(monkeypatch):
    reader = _reader()

    def fake_fetch(lane):
        if lane.ctx_key == "episode_summary":
            # 15 minutes old: stale under the global 120s gate, fresh under
            # the lane's 1800s override.
            return (
                {"episode_id": "ep1", "status": "proposal"},
                datetime.now(timezone.utc) - timedelta(seconds=900),
            )
        return None

    monkeypatch.setattr(reader, "_fetch_lane", fake_fetch)
    ctx: dict = {}
    reader.hydrate(ctx)
    assert ctx["episode_summary"] == {"episode_id": "ep1", "status": "proposal"}


def test_hydrate_rejects_episode_older_than_lane_max_age(monkeypatch):
    reader = _reader()

    def fake_fetch(lane):
        if lane.ctx_key == "episode_summary":
            return (
                {"episode_id": "ep_old", "status": "proposal"},
                datetime.now(timezone.utc) - timedelta(seconds=3600),
            )
        return None

    monkeypatch.setattr(reader, "_fetch_lane", fake_fetch)
    ctx: dict = {}
    reader.hydrate(ctx)
    assert "episode_summary" not in ctx


def test_hydrate_rejects_stale_attention_broadcast(monkeypatch):
    reader = _reader()

    def fake_fetch(lane):
        if lane.ctx_key == "attention_broadcast":
            return (
                {"selected_action_type": "focus"},
                datetime.now(timezone.utc) - timedelta(seconds=600),
            )
        return None

    monkeypatch.setattr(reader, "_fetch_lane", fake_fetch)
    ctx: dict = {}
    reader.hydrate(ctx)
    assert "attention_broadcast" not in ctx
