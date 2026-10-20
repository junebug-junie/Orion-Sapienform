"""Shed backoff: a heat-shed refusal makes callers wait (growing, capped, jittered), one probe per
window, and the very next call goes through once the pool stops shedding. Non-shed refusals are
untouched. orion/gpu_pool/shed_backoff.py; client wiring in orion/gpu_pool/client.py."""
from __future__ import annotations

import pytest

from orion.gpu_pool import client as pool_client
from orion.gpu_pool.client import LeaseUnavailable, gpu_lease
from orion.gpu_pool.shed_backoff import BASE_SEC, CAP_SEC, ShedBackoff, gateway_shed_refusal
from orion.gpu_pool.tests.test_client_attach import ScriptedBus, _grant
from orion.schemas.gpu_pool import GpuLeaseReplyV1

HOT = "shed:cabinet_hot"


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _backoff(clock: Clock, *, rng=lambda: 0.5, logs: list[str] | None = None) -> ShedBackoff:
    # rng 0.5 -> zero jitter, so delays are exact.
    return ShedBackoff(name="t", clock=clock, rng=rng, log=(logs.append if logs is not None else lambda m: None))


def _refuse_probe(b: ShedBackoff, clock: Clock, key: str = "background") -> float:
    """Advance to the end of the current window, take the probe, refuse it; return the new delay."""
    rem = b.remaining(key)
    clock.t += rem
    assert b.check(key) is None, "the window is over: this call is the probe"
    b.resolve(key, HOT)
    return b.remaining(key)


def test_refusal_backoff_doubles_and_caps():
    clock = Clock()
    b = _backoff(clock)
    assert b.check("background") is None
    b.resolve("background", HOT)
    delays = [b.remaining("background")]
    for _ in range(6):
        delays.append(_refuse_probe(b, clock))
    assert delays == [BASE_SEC, 2 * BASE_SEC, 4 * BASE_SEC, CAP_SEC, CAP_SEC, CAP_SEC, CAP_SEC]


def test_calls_inside_the_window_are_refused_locally_and_counted():
    clock = Clock()
    logs: list[str] = []
    b = _backoff(clock, logs=logs)
    b.check("background")
    b.resolve("background", HOT)
    for _ in range(10):
        clock.t += 1.0
        wait = b.check("background")
        assert wait is not None and 0 < wait <= BASE_SEC
    assert len(logs) == 1 and "gpu_shed_backoff_start" in logs[0]   # once per episode, not per call
    clock.t += BASE_SEC
    assert b.check("background") is None
    b.resolve("background", None)
    assert len(logs) == 2 and "gpu_shed_backoff_end" in logs[1] and "skipped=10" in logs[1]


def test_only_one_probe_per_window():
    clock = Clock()
    b = _backoff(clock)
    b.check("background")
    b.resolve("background", HOT)
    clock.t += BASE_SEC
    assert b.check("background") is None          # the probe
    assert b.check("background") is not None      # a concurrent call waits for the probe
    b.abandon("background")                       # probe ended without a verdict
    assert b.check("background") is None          # another call may probe now


def test_stale_probe_stops_blocking():
    clock = Clock()
    b = _backoff(clock)
    b.check("background")
    b.resolve("background", HOT)
    clock.t += BASE_SEC
    assert b.check("background") is None
    clock.t += b.probe_stale_sec + 1
    assert b.check("background") is None


def test_heat_clears_next_call_is_prompt():
    clock = Clock()
    b = _backoff(clock)
    b.check("background")
    b.resolve("background", HOT)
    for _ in range(5):
        _refuse_probe(b, clock)
    clock.t += b.remaining("background")
    assert b.check("background") is None
    b.resolve("background", None)                 # granted: heat cleared
    assert b.check("background") is None and b.remaining("background") is None
    # A later episode starts from the base delay again, not from the cap.
    b.resolve("background", HOT)
    assert b.remaining("background") == BASE_SEC


def test_non_shed_refusal_is_not_backed_off_and_ends_an_episode():
    clock = Clock()
    b = _backoff(clock)
    b.resolve("background", "min_ctx_exceeds_class:65536")
    assert b.check("background") is None and b.remaining("background") is None
    b.resolve("background", HOT)
    clock.t += BASE_SEC
    b.check("background")
    b.resolve("background", "deadline")
    assert b.check("background") is None


def test_keys_are_independent():
    clock = Clock()
    b = _backoff(clock)
    b.resolve("background", HOT)
    assert b.check("background") is not None
    assert b.check("system") is None


def test_in_flight_refusals_do_not_grow_the_window():
    clock = Clock()
    b = _backoff(clock)
    b.resolve("background", HOT)
    b.resolve("background", HOT)   # a second call already in flight when the first came back
    assert b.remaining("background") == BASE_SEC


def test_jitter_stays_within_bounds():
    clock = Clock()
    lo = _backoff(clock, rng=lambda: 0.0)
    hi = _backoff(clock, rng=lambda: 0.999999)
    lo.resolve("k", HOT)
    hi.resolve("k", HOT)
    assert lo.remaining("k") == pytest.approx(BASE_SEC * 0.8)
    assert hi.remaining("k") == pytest.approx(BASE_SEC * 1.2, rel=1e-4)


def test_hint_extends_but_is_capped():
    clock = Clock()
    b = _backoff(clock)
    b.resolve("k", HOT, hint_sec=100.0)
    assert b.remaining("k") == 100.0
    b2 = _backoff(clock)
    b2.resolve("k", HOT, hint_sec=10_000.0)
    assert b2.remaining("k") == pytest.approx(CAP_SEC * 1.2)


def test_gateway_shed_refusal_reads_the_gateway_error_shape():
    live = {"content": "", "raw": {"error": "gpu_pool_unavailable",
                                   "details": {"reason": HOT, "route": "metacog_background", "work_class": "metacog"}}}
    assert gateway_shed_refusal(live) == (HOT, None)
    live["raw"]["details"]["retry_after_sec"] = 42.0
    assert gateway_shed_refusal(live) == (HOT, 42.0)
    assert gateway_shed_refusal({"raw": {"error": "gpu_pool_unavailable", "details": {"reason": "deadline"}}}) is None
    assert gateway_shed_refusal({"content": "ok", "raw": {}}) is None
    assert gateway_shed_refusal(None) is None


# --- the pool client --------------------------------------------------------------------------

def _install(monkeypatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr(pool_client, "SHED_BACKOFF", _backoff(clock))
    return clock


class Pool:
    def __init__(self, reason: str | None):
        self.reason = reason

    def __call__(self, req):
        if req.verb in ("acquire", "attach"):
            if self.reason is None:
                return GpuLeaseReplyV1(status="granted", lease_id="l-1", grant=_grant("l-1"))
            return GpuLeaseReplyV1(status="unavailable", lease_id="l-1", reason=self.reason)
        return GpuLeaseReplyV1(status="ok", lease_id=req.lease_id)


def _acquires(bus) -> int:
    return sum(1 for r in bus.requests if r.verb == "acquire")


async def _try(bus, **kw):
    async with gpu_lease(bus, work_class="metacog", holder="vision-council", priority="background",
                         deadline_sec=5, **kw):
        pass


@pytest.mark.asyncio
async def test_client_shed_refusal_is_answered_locally_until_the_window_ends(monkeypatch):
    clock = _install(monkeypatch)
    pool = Pool(HOT)
    bus = ScriptedBus(pool)
    with pytest.raises(LeaseUnavailable) as first:
        await _try(bus)
    assert first.value.reason == HOT and not first.value.local
    assert first.value.retry_after_sec == BASE_SEC
    for _ in range(20):
        clock.t += 1.0
        with pytest.raises(LeaseUnavailable) as again:
            await _try(bus)
        assert again.value.reason == HOT and again.value.local and again.value.retry_after_sec > 0
    assert _acquires(bus) == 1                      # 21 calls, 1 reached the pool
    pool.reason = None                              # heat clears
    clock.t += BASE_SEC
    await _try(bus)                                 # the probe is granted
    await _try(bus)                                 # and the next call goes straight through
    assert _acquires(bus) == 3


@pytest.mark.asyncio
async def test_client_non_shed_refusal_unchanged(monkeypatch):
    _install(monkeypatch)
    bus = ScriptedBus(Pool("min_ctx_exceeds_class:65536"))
    for _ in range(3):
        with pytest.raises(LeaseUnavailable) as exc:
            await _try(bus)
        assert not exc.value.local and exc.value.retry_after_sec is None
    assert _acquires(bus) == 3


@pytest.mark.asyncio
async def test_client_never_gates_retryable_or_holds(monkeypatch):
    _install(monkeypatch)
    pool_client.SHED_BACKOFF.resolve("background", HOT)
    bus = ScriptedBus(Pool(None))
    await _try(bus, retryable=True)
    async with gpu_lease(bus, work_class="agent", holder="durable-runs:r", priority="background", kind="hold",
                         deadline_sec=5):
        pass
    assert _acquires(bus) == 2
    with pytest.raises(LeaseUnavailable) as exc:
        await _try(bus)                              # a plain one-shot is still gated
    assert exc.value.local
