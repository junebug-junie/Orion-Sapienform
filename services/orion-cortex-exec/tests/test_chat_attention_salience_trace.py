from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app import chat_attention_salience_trace as cast
from orion.schemas.attention_frame import (
    AttentionFrameV1,
    CuriosityCandidateActionV1,
    OpenLoopV1,
)


def _frame(
    *,
    selected_id: str | None = "loop-1",
    loop_salience: float = 0.62,
    turn_id: str | None = "turn-1",
    correlation_id: str | None = "corr-1",
) -> AttentionFrameV1:
    loop = OpenLoopV1(
        id="loop-1",
        target_type="plan",
        description="Zephyr Bridge",
        salience=loop_salience,
        salience_features={"evidence_strength": 0.5, "evidence_breadth": 0.3},
    )
    selected = (
        CuriosityCandidateActionV1(action_type="watch", open_loop_id=selected_id, score=loop_salience)
        if selected_id
        else None
    )
    return AttentionFrameV1(
        generated_at=datetime.now(timezone.utc),
        turn_id=turn_id,
        session_id="sess-1",
        correlation_id=correlation_id,
        open_loops=[loop],
        selected_action=selected,
    )


@pytest.fixture(autouse=True)
def _reset_engine(monkeypatch: pytest.MonkeyPatch):
    cast.reset_chat_attention_salience_trace_engine_for_tests()
    monkeypatch.setenv(
        "SUBSTRATE_FELT_STATE_DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/conjourney",
    )
    monkeypatch.setenv("ENABLE_CHAT_ATTENTION_SALIENCE_TRACE", "true")
    monkeypatch.setenv("CHAT_ATTENTION_SALIENCE_TRACE_TIMEOUT_SEC", "0.8")
    yield
    cast.reset_chat_attention_salience_trace_engine_for_tests()


def test_build_row_uses_scope_chat_not_reverie() -> None:
    row = cast.build_chat_salience_trace_row(_frame())
    assert row is not None
    assert row["scope"] == "chat"
    assert row["loop_id"] == "loop-1"
    assert row["theme_key"] == "loop-1"
    assert row["description"] == "Zephyr Bridge"
    assert row["correlation_id"] == "corr-1"
    assert row["salience"] == pytest.approx(0.62)
    assert row["weights_version"] == "gwt-coalition-v1"
    assert row["features"] == {"evidence_strength": 0.5, "evidence_breadth": 0.3}


def test_build_row_none_when_nothing_selected() -> None:
    assert cast.build_chat_salience_trace_row(_frame(selected_id=None)) is None


def test_build_row_none_when_selected_loop_missing() -> None:
    frame = _frame(selected_id="loop-does-not-exist")
    assert cast.build_chat_salience_trace_row(frame) is None


def test_trace_id_is_stable_and_idempotent_by_correlation_and_loop() -> None:
    row_a = cast.build_chat_salience_trace_row(_frame())
    row_b = cast.build_chat_salience_trace_row(_frame())
    assert row_a["trace_id"] == row_b["trace_id"]


def test_trace_id_does_not_collapse_across_distinct_turns_when_correlation_id_is_absent() -> None:
    # Regression for the 2026-08-19 review finding: with correlation_id=None
    # (a real, allowed AttentionFrameV1 state -- see
    # orion/substrate/attention_frame.py's own fallback-to-None), the trace_id
    # must still disambiguate on turn_id so two distinct real turns that both
    # select "loop-1" don't collapse into one ON CONFLICT DO NOTHING no-op.
    row_a = cast.build_chat_salience_trace_row(_frame(correlation_id=None, turn_id="turn-a"))
    row_b = cast.build_chat_salience_trace_row(_frame(correlation_id=None, turn_id="turn-b"))
    assert row_a["trace_id"] != row_b["trace_id"]


@pytest.mark.asyncio
async def test_persist_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def _fake_persist_sync(row: dict) -> bool:
        calls.append(row)
        return True

    monkeypatch.setattr(cast, "_persist_sync", _fake_persist_sync)
    ok = await cast.persist_chat_attention_salience_trace(_frame())
    assert ok is True
    assert calls and calls[0]["scope"] == "chat"


@pytest.mark.asyncio
async def test_persist_disabled_returns_false_without_touching_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_CHAT_ATTENTION_SALIENCE_TRACE", "false")

    def _boom(row: dict) -> bool:
        raise AssertionError("must not be called when disabled")

    monkeypatch.setattr(cast, "_persist_sync", _boom)
    ok = await cast.persist_chat_attention_salience_trace(_frame())
    assert ok is False


@pytest.mark.asyncio
async def test_persist_no_selection_returns_false_without_touching_db(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(row: dict) -> bool:
        raise AssertionError("must not be called when nothing was selected")

    monkeypatch.setattr(cast, "_persist_sync", _boom)
    ok = await cast.persist_chat_attention_salience_trace(_frame(selected_id=None))
    assert ok is False


@pytest.mark.asyncio
async def test_persist_dsn_unset_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUBSTRATE_FELT_STATE_DATABASE_URL", raising=False)
    monkeypatch.delenv("ENDOGENOUS_RUNTIME_SQL_DATABASE_URL", raising=False)
    ok = await cast.persist_chat_attention_salience_trace(_frame())
    assert ok is False


@pytest.mark.asyncio
async def test_persist_timeout_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_ATTENTION_SALIENCE_TRACE_TIMEOUT_SEC", "0.05")

    def _slow(row: dict) -> bool:
        import time

        time.sleep(0.3)
        return True

    monkeypatch.setattr(cast, "_persist_sync", _slow)
    ok = await cast.persist_chat_attention_salience_trace(_frame())
    assert ok is False


@pytest.mark.asyncio
async def test_persist_exception_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(row: dict) -> bool:
        raise RuntimeError("db down")

    monkeypatch.setattr(cast, "_persist_sync", _boom)
    ok = await cast.persist_chat_attention_salience_trace(_frame())
    assert ok is False


def test_dsn_falls_back_to_endogenous_runtime_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUBSTRATE_FELT_STATE_DATABASE_URL", raising=False)
    monkeypatch.setenv(
        "ENDOGENOUS_RUNTIME_SQL_DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/conjourney",
    )
    assert cast._dsn() == "postgresql://postgres:postgres@localhost:5432/conjourney"
