from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.substrate_felt_state_reader import (
    _LANES,
    SubstrateFeltStateReader,
    hydrate_felt_state_ctx,
    reset_reader_for_tests,
)


def _make_reader(max_age_sec: int = 120) -> SubstrateFeltStateReader:
    # Build disabled (no create_engine / no DB), then flip to enabled with no engine.
    reader = SubstrateFeltStateReader(
        enabled=False, database_url="postgresql://unused", max_age_sec=max_age_sec
    )
    reader._enabled = True
    reader._engine = None
    return reader


def _now() -> datetime:
    return datetime.now(timezone.utc)


def test_fresh_rows_all_lanes_hydrated():
    reader = _make_reader()

    def fake_fetch(lane):
        return ({"lane": lane.ctx_key, "value": 1}, _now())

    reader._fetch_lane = fake_fetch  # type: ignore[assignment]

    ctx: dict = {}
    reader.hydrate(ctx)

    for lane in _LANES:
        assert ctx.get(lane.ctx_key) == {"lane": lane.ctx_key, "value": 1}


def test_stale_row_not_injected():
    reader = _make_reader(max_age_sec=120)
    stale_ts = _now() - timedelta(seconds=10000)

    def fake_fetch(lane):
        return ({"lane": lane.ctx_key}, stale_ts)

    reader._fetch_lane = fake_fetch  # type: ignore[assignment]

    ctx: dict = {}
    reader.hydrate(ctx)

    for lane in _LANES:
        assert lane.ctx_key not in ctx


def test_existing_ctx_key_not_overwritten():
    reader = _make_reader()
    sentinel = {"preexisting": True}

    def fake_fetch(lane):
        return ({"lane": lane.ctx_key}, _now())

    reader._fetch_lane = fake_fetch  # type: ignore[assignment]

    ctx: dict = {"execution_trajectory_projection": sentinel}
    reader.hydrate(ctx)

    assert ctx["execution_trajectory_projection"] is sentinel


def test_disabled_is_noop():
    reader = SubstrateFeltStateReader(
        enabled=False, database_url="postgresql://unused", max_age_sec=120
    )

    def fake_fetch(lane):
        raise AssertionError("should not be called when disabled")

    reader._fetch_lane = fake_fetch  # type: ignore[assignment]

    ctx: dict = {}
    reader.hydrate(ctx)

    assert ctx == {}


def test_one_lane_raises_fail_open_others_hydrate():
    reader = _make_reader()

    def fake_fetch(lane):
        if lane.ctx_key == "self_state":
            raise RuntimeError("boom")
        return ({"lane": lane.ctx_key}, _now())

    reader._fetch_lane = fake_fetch  # type: ignore[assignment]

    ctx: dict = {}
    # Must not raise.
    reader.hydrate(ctx)

    assert "self_state" not in ctx
    for lane in _LANES:
        if lane.ctx_key == "self_state":
            continue
        assert ctx.get(lane.ctx_key) == {"lane": lane.ctx_key}


def test_entrypoint_flag_unset_is_noop(monkeypatch):
    monkeypatch.delenv("ENABLE_SUBSTRATE_FELT_STATE_CTX", raising=False)
    reset_reader_for_tests()

    ctx: dict = {}
    hydrate_felt_state_ctx(ctx)

    assert ctx == {}

    reset_reader_for_tests()
