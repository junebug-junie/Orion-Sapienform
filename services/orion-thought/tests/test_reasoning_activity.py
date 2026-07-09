"""Unit tests for the reasoning-activity rolling-window projection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.reasoning_activity import ReasoningActivityStore, _decode_reasoning_call
from orion.schemas.telemetry.reasoning import ReasoningActivityV1, ReasoningCallV1

NOW = datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc)


def _call(
    *,
    offset_sec: float = 0.0,
    reasoning_present: bool = False,
    thinking_enabled: bool = False,
    completion_tokens: int | None = None,
    thinking_tokens: int | None = None,
    mode: str = "brain",
    corr: str = "c1",
) -> ReasoningCallV1:
    return ReasoningCallV1(
        correlation_id=corr,
        verb="stance_react",
        mode=mode,
        reasoning_present=reasoning_present,
        thinking_enabled=thinking_enabled,
        completion_tokens=completion_tokens,
        thinking_tokens=thinking_tokens,
        emitted_at=NOW - timedelta(seconds=offset_sec),
    )


def test_aggregates_rate_sums_p50_and_by_mode() -> None:
    store = ReasoningActivityStore(window_sec=120.0, max_calls=100)
    # 4 calls: 3 reasoning_present, 2 thinking_enabled.
    store.record(_call(reasoning_present=True, thinking_enabled=True, completion_tokens=10, mode="brain"))
    store.record(_call(reasoning_present=True, completion_tokens=20, mode="brain"))
    store.record(_call(reasoning_present=True, thinking_enabled=True, completion_tokens=30, mode="reflex"))
    store.record(_call(reasoning_present=False, completion_tokens=40, mode="reflex"))

    snap = store.snapshot(NOW)
    assert isinstance(snap, ReasoningActivityV1)
    assert snap.call_count == 4
    assert snap.reasoning_call_count == 3
    assert snap.thinking_call_count == 2
    assert snap.reasoning_present_rate == 3 / 4
    assert snap.completion_tokens_sum == 100
    # median of [10,20,30,40] = 25.0
    assert snap.completion_tokens_p50 == 25.0
    assert snap.by_mode == {"brain": 2, "reflex": 2}


def test_window_filtering_excludes_old_call() -> None:
    store = ReasoningActivityStore(window_sec=60.0, max_calls=100)
    store.record(_call(offset_sec=10, completion_tokens=5))   # in window
    store.record(_call(offset_sec=120, completion_tokens=99))  # older than 60s

    snap = store.snapshot(NOW)
    assert snap.call_count == 1
    assert snap.completion_tokens_sum == 5


def test_buffer_cap_keeps_most_recent() -> None:
    store = ReasoningActivityStore(window_sec=10_000.0, max_calls=3)
    for i in range(10):
        store.record(_call(completion_tokens=i, corr=f"c{i}"))

    snap = store.snapshot(NOW)
    assert snap.call_count == 3
    # Only the last 3 (7,8,9) survive the deque cap.
    assert snap.completion_tokens_sum == 7 + 8 + 9


def test_empty_store_zeroed_projection() -> None:
    store = ReasoningActivityStore(window_sec=120.0, max_calls=100)
    snap = store.snapshot(NOW)
    assert snap.call_count == 0
    assert snap.reasoning_call_count == 0
    assert snap.thinking_call_count == 0
    assert snap.reasoning_present_rate == 0.0
    assert snap.completion_tokens_sum == 0
    assert snap.completion_tokens_p50 == 0.0
    assert snap.thinking_tokens_sum is None
    assert snap.by_mode == {}
    assert snap.generated_at == NOW
    assert snap.window_sec == 120.0


def test_thinking_tokens_sum_none_then_sum() -> None:
    store = ReasoningActivityStore(window_sec=120.0, max_calls=100)
    store.record(_call(completion_tokens=10))
    store.record(_call(completion_tokens=20))
    # No call carried thinking_tokens -> None.
    assert store.snapshot(NOW).thinking_tokens_sum is None

    store.record(_call(completion_tokens=30, thinking_tokens=7))
    store.record(_call(completion_tokens=40, thinking_tokens=3))
    # Two calls carry thinking_tokens -> sum of those present.
    assert store.snapshot(NOW).thinking_tokens_sum == 10


def test_p50_zero_when_no_completion_tokens() -> None:
    store = ReasoningActivityStore(window_sec=120.0, max_calls=100)
    store.record(_call(reasoning_present=True))  # no completion_tokens
    snap = store.snapshot(NOW)
    assert snap.call_count == 1
    assert snap.completion_tokens_sum == 0
    assert snap.completion_tokens_p50 == 0.0


def test_by_mode_capped_at_16_entries() -> None:
    store = ReasoningActivityStore(window_sec=10_000.0, max_calls=1000)
    for i in range(40):
        store.record(_call(mode=f"mode-{i}", corr=f"c{i}"))
    snap = store.snapshot(NOW)
    assert len(snap.by_mode) == 16


def test_record_never_raises_on_bad_input() -> None:
    store = ReasoningActivityStore(window_sec=120.0, max_calls=10)
    # A non-ReasoningCallV1 object has no emitted_at; record must swallow nothing
    # here (append succeeds) but snapshot must not crash on the junk element.
    store.record(object())  # type: ignore[arg-type]
    store.record(_call(completion_tokens=5))
    # snapshot never raises even with a junk element in the buffer.
    snap = store.snapshot(NOW)
    assert isinstance(snap, ReasoningActivityV1)


class _FakeCodec:
    class _Decoded:
        def __init__(self, ok: bool, payload) -> None:
            self.ok = ok

            class _Env:
                pass

            self.envelope = _Env()
            self.envelope.payload = payload

    def __init__(self, ok: bool, payload) -> None:
        self._ok = ok
        self._payload = payload

    def decode(self, data):
        return self._Decoded(self._ok, self._payload)


class _FakeBus:
    def __init__(self, ok: bool, payload) -> None:
        self.codec = _FakeCodec(ok, payload)


def test_decode_valid_message() -> None:
    payload = {
        "correlation_id": "c1",
        "verb": "stance_react",
        "mode": "brain",
        "reasoning_present": True,
        "completion_tokens": 12,
        "emitted_at": NOW.isoformat(),
    }
    bus = _FakeBus(ok=True, payload=payload)
    call = _decode_reasoning_call(bus, {"data": b"{}"})
    assert call is not None
    assert call.correlation_id == "c1"
    assert call.completion_tokens == 12


def test_decode_malformed_message_returns_none() -> None:
    # Missing required emitted_at -> validation fails -> None, no exception.
    bus = _FakeBus(ok=True, payload={"correlation_id": "c1"})
    assert _decode_reasoning_call(bus, {"data": b"{}"}) is None

    # extra="forbid": unknown field -> None.
    bad = {
        "correlation_id": "c1",
        "emitted_at": NOW.isoformat(),
        "not_a_field": True,
    }
    assert _decode_reasoning_call(_FakeBus(ok=True, payload=bad), {"data": b"{}"}) is None

    # decode not ok -> None.
    assert _decode_reasoning_call(_FakeBus(ok=False, payload=None), {"data": b"x"}) is None

    # non-dict payload -> None.
    assert _decode_reasoning_call(_FakeBus(ok=True, payload="nope"), {"data": b"x"}) is None


def test_record_path_survives_malformed_then_records_valid() -> None:
    """The worker's record path (decode -> record) must not crash on a bad message."""
    store = ReasoningActivityStore(window_sec=120.0, max_calls=10)

    bad = _decode_reasoning_call(_FakeBus(ok=True, payload={"bogus": 1}), {"data": b"{}"})
    assert bad is None  # decode swallowed it; nothing to record

    good = _decode_reasoning_call(
        _FakeBus(
            ok=True,
            payload={"correlation_id": "c1", "completion_tokens": 3, "emitted_at": NOW.isoformat()},
        ),
        {"data": b"{}"},
    )
    assert good is not None
    store.record(good)
    assert store.snapshot(NOW).call_count == 1
