"""Shed backoff: a heat-shed refusal makes callers wait (growing, capped, jittered), one probe per
window, and the very next call goes through once the pool stops shedding. Non-shed refusals are
untouched. orion/gpu_pool/shed_backoff.py; client wiring in orion/gpu_pool/client.py."""
from __future__ import annotations

import asyncio

import pytest

from orion.gpu_pool import client as pool_client
from orion.gpu_pool.client import LeaseUnavailable, gpu_lease
from orion.gpu_pool.shed_backoff import BASE_SEC, CAP_SEC, ShedBackoff, gateway_error_reason, gateway_shed_refusal
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


def _probe(b: ShedBackoff, clock: Clock, key: str = "background") -> int:
    """Advance to the end of the current window and take the probe; return its token."""
    clock.t += b.remaining(key)
    wait, token = b.admit(key)
    assert wait is None and token is not None, "the window is over: this call is the probe"
    return token


def _refuse_probe(b: ShedBackoff, clock: Clock, key: str = "background") -> float:
    b.resolve(key, HOT, token=_probe(b, clock, key))
    return b.remaining(key)


def test_refusal_backoff_doubles_and_caps():
    clock = Clock()
    b = _backoff(clock)
    assert b.admit("background") == (None, None)
    b.resolve("background", HOT)
    delays = [b.remaining("background")]
    for _ in range(6):
        delays.append(_refuse_probe(b, clock))
    assert delays == [BASE_SEC, 2 * BASE_SEC, 4 * BASE_SEC, CAP_SEC, CAP_SEC, CAP_SEC, CAP_SEC]


def test_calls_inside_the_window_are_refused_locally_and_counted():
    clock = Clock()
    logs: list[str] = []
    b = _backoff(clock, logs=logs)
    b.resolve("background", HOT)
    for _ in range(10):
        clock.t += 1.0
        wait, token = b.admit("background")
        assert wait is not None and 0 < wait <= BASE_SEC and token is None
    assert len(logs) == 1 and "gpu_shed_backoff_start" in logs[0]   # once per episode, not per call
    b.resolve("background", None, token=_probe(b, clock))
    assert len(logs) == 2 and "gpu_shed_backoff_end" in logs[1] and "skipped=10" in logs[1]


def test_only_one_probe_per_window():
    clock = Clock()
    b = _backoff(clock)
    b.resolve("background", HOT)
    token = _probe(b, clock)
    assert b.check("background") is not None      # a concurrent call waits for the probe
    b.abandon("background", token=token)          # probe ended without a verdict
    assert b.check("background") is None          # another call may probe now


def test_neutral_outcome_releases_the_probe_without_ending_or_growing():
    clock = Clock()
    b = _backoff(clock)
    b.resolve("background", HOT)
    for reason in ("deadline", "cancelled", "pool_unreachable", "gpu_pool_unavailable"):
        token = _probe(b, clock)
        b.resolve("background", reason, token=token)
        assert b.reason("background") == HOT and b.remaining("background") == 0.0
    assert _refuse_probe(b, clock) == 2 * BASE_SEC   # still the second step: neutral did not grow it


def test_stale_probe_stops_blocking():
    clock = Clock()
    b = _backoff(clock)
    b.resolve("background", HOT)
    _probe(b, clock)
    assert b.check("background") is not None
    clock.t += b.probe_stale_sec + 1
    assert b.admit("background")[1] is not None


def test_heat_clears_next_call_is_prompt():
    clock = Clock()
    b = _backoff(clock)
    b.resolve("background", HOT)
    for _ in range(5):
        _refuse_probe(b, clock)
    b.resolve("background", None, token=_probe(b, clock))   # granted: heat cleared
    assert b.admit("background") == (None, None) and b.remaining("background") is None
    # A later episode starts from the base delay again, not from the cap.
    b.resolve("background", HOT)
    assert b.remaining("background") == BASE_SEC


def test_non_shed_refusal_is_not_backed_off_and_a_judged_one_ends_an_episode():
    clock = Clock()
    b = _backoff(clock)
    b.resolve("background", "min_ctx_exceeds_class:65536")
    assert b.check("background") is None and b.remaining("background") is None
    b.resolve("background", HOT)
    b.resolve("background", "no_serviceable_role", token=_probe(b, clock))
    assert b.admit("background") == (None, None)


def test_only_the_probe_moves_the_episode():
    clock = Clock()
    b = _backoff(clock)
    b.resolve("background", HOT)
    b.resolve("background", HOT)            # a second call already in flight: no growth
    assert b.remaining("background") == BASE_SEC
    b.resolve("background", None)           # a pre-episode call's late grant: ignored
    assert b.reason("background") == HOT
    token = _probe(b, clock)
    b.resolve("background", HOT)            # an in-flight non-probe refusal while the probe runs
    assert b.check("background") is not None, "the probe is still outstanding"
    b.resolve("background", HOT, token=token)
    assert b.remaining("background") == 2 * BASE_SEC
    b.resolve("background", None, token=token)   # a late duplicate from the spent probe: ignored
    assert b.reason("background") == HOT


def test_keys_are_independent():
    clock = Clock()
    b = _backoff(clock)
    b.resolve("background", HOT)
    assert b.check("background") is not None
    assert b.check("system") is None


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
    assert gateway_error_reason(live) == HOT
    assert gateway_error_reason({"raw": {"error": "gpu_pool_unavailable", "details": {}}}) == "gpu_pool_unavailable"
    assert gateway_error_reason({"content": "ok", "raw": {}}) is None


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


def _queued(req):
    if req.verb == "acquire":
        return GpuLeaseReplyV1(status="queued", lease_id="l-q", position=1)
    return GpuLeaseReplyV1(status="ok", lease_id=req.lease_id)


@pytest.mark.asyncio
async def test_client_probe_the_pool_queues_ends_the_episode(monkeypatch):
    """Heat cleared but the probe waits in line behind real work: the pool would have refused a shed
    lease on its first tick, so after QUEUED_PROBE_CLEARS_SEC the episode ends -- other calls are not
    refused locally while the probe waits."""
    clock = _install(monkeypatch)
    monkeypatch.setattr(pool_client, "QUEUED_PROBE_CLEARS_SEC", 0.05)
    pool_client.SHED_BACKOFF.resolve("background", HOT)
    clock.t += BASE_SEC
    bus = ScriptedBus(_queued)
    task = asyncio.ensure_future(_try(bus))
    await asyncio.sleep(0.15)
    assert pool_client.SHED_BACKOFF.reason("background") is None
    assert pool_client.SHED_BACKOFF.admit("background") == (None, None)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_client_cancelled_probe_is_released(monkeypatch):
    clock = _install(monkeypatch)
    pool_client.SHED_BACKOFF.resolve("background", HOT)
    clock.t += BASE_SEC
    bus = ScriptedBus(_queued)
    task = asyncio.ensure_future(_try(bus))
    await asyncio.sleep(0.01)
    assert pool_client.SHED_BACKOFF.check("background") is not None   # the probe is outstanding
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pool_client.SHED_BACKOFF.reason("background") == HOT          # episode neither ended
    assert pool_client.SHED_BACKOFF.remaining("background") == 0.0       # nor grew
    assert pool_client.SHED_BACKOFF.admit("background")[1] is not None   # next call probes


@pytest.mark.asyncio
async def test_client_probe_deadline_is_neutral(monkeypatch):
    clock = _install(monkeypatch)
    monkeypatch.setattr(pool_client, "QUEUED_PROBE_CLEARS_SEC", 60.0)
    pool_client.SHED_BACKOFF.resolve("background", HOT)
    clock.t += BASE_SEC
    bus = ScriptedBus(_queued)
    with pytest.raises(LeaseUnavailable) as exc:
        async with gpu_lease(bus, work_class="metacog", holder="world-model", priority="background",
                             deadline_sec=0.05):
            pass
    assert exc.value.reason == "deadline"
    assert pool_client.SHED_BACKOFF.reason("background") == HOT
    assert pool_client.SHED_BACKOFF.remaining("background") == 0.0
