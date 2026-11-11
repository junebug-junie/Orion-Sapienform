"""Attention patch 3b: each durable run's market net rides on its pool hold request, so the pool can
serve its background queue by value (app/run_bids.py ``value_for``, app/pool_hold.py, register)."""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]

from app.run_bids import READING_FAMILY, VALUE_PRODUCER, RunBidder, RunValue, reading_want  # noqa: E402
from orion.gpu_pool.client import acquire_hold, request_payload  # noqa: E402
from orion.physiology.families import load_families  # noqa: E402
from orion.schemas.gpu_pool import GpuLeaseReplyV1, GpuLeaseRequestV1  # noqa: E402

FAMS, _ = load_families(ROOT / "config/physiology/families.yaml")
T0 = datetime(2026, 11, 11, 6, 0, tzinfo=timezone.utc)


class _Decision:
    def __init__(self, want, net):
        self.decision_id, self.want_id, self.family = f"{want.want_id}@x", want.want_id, want.family
        self.outcome, self.reason, self.binding_constraint, self.net = "deferred", "capacity_reserved", "lane:agent", net


class _Market:
    def __init__(self, net=0.74, delay=0.0):
        self.families, self.net, self.delay = FAMS, net, delay
        self.bids, self.released, self.opened = [], [], []

    async def bid(self, want, now=None, *, reserve_at_open=False):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.bids.append(want)
        return _Decision(want, self.net)

    def release(self, want_id):
        self.released.append(want_id)
        return 1

    def mark_opened(self, want_id, at=None):
        self.opened.append(want_id)
        return 1


def _reading(stage, run_id, seed="seed-1"):
    return {"run_id": run_id, "workflow": "reading.turn", "correlation_id": f"corr-{run_id}",
            "requested_at": T0.isoformat(), "brief": {"seed_id": seed, "stage": stage}, "admission": {}}


def _self_inquiry(run_id="run-si"):
    return {"run_id": run_id, "workflow": "curiosity.investigate", "correlation_id": "c", "requested_at": T0.isoformat(),
            "brief": {"prompt": "p", "session_id": "orion_curiosity", "line": "self_inquiry"},
            "admission": {"deadline_at": None}}


def test_reading_stage_1_gets_a_value_only_bid_and_stage_2_reuses_its_net():
    async def go():
        m = _Market(net=0.74)
        b = RunBidder(enabled=True, market_getter=lambda: m)
        await b.on_submit(_reading(1, "reading-s1"))
        assert [w.family for w in m.bids] == [READING_FAMILY] and m.bids[0].verb == "look"
        assert m.bids[0].producer == VALUE_PRODUCER     # told apart from Hub's looks in want_ledger
        assert m.released == ["reading-s1"]           # value only: it reserves nothing
        assert b.on_submit(_reading(2, "reading-s2")) is None   # stage 2 never bids
        v1 = await b.value_for(_reading(1, "reading-s1"))
        v2 = await b.value_for(_reading(2, "reading-s2"))
        assert v1 == RunValue("reading-s1", 0.74, READING_FAMILY) and v2 == v1
        assert await b.value_for(_reading(2, "reading-other", seed="seed-2")) is None
        assert b.counts["value_bid"] == 1
    asyncio.run(go())


def test_reading_want_is_stage_1_only():
    assert reading_want(_reading(2, "r"), 2) is None and reading_want(_self_inquiry(), 2) is None
    w = reading_want(_reading(1, "r"), 2)
    assert (w.want_id, w.family, w.tier, w.correlation_id) == ("r", READING_FAMILY, 2, "corr-r")


def test_a_shadow_bid_still_in_flight_is_awaited_so_the_first_hold_carries_its_net():
    async def go():
        m = _Market(net=0.6, delay=0.05)
        b = RunBidder(enabled=True, market_getter=lambda: m)
        b.on_submit(_self_inquiry())
        v = await b.value_for(_self_inquiry(), wait_sec=1.0)
        assert v == RunValue("run-si", 0.6, "durable.self_inquiry")
        assert m.released == []                        # an ordinary shadow bid keeps its reservation path
    asyncio.run(go())


def test_no_bid_means_no_value():
    async def go():
        b = RunBidder(enabled=True, market_getter=lambda: _Market())
        assert await b.value_for({"run_id": "upkeep-1", "workflow": "dream.carry"}) is None
    asyncio.run(go())


# --- the wire ------------------------------------------------------------------------------------

def test_unset_value_fields_never_reach_the_wire():
    req = GpuLeaseRequestV1(verb="acquire", request_id="r", holder="h", work_class="agent", kind="hold")
    assert not {"want_id", "want_net", "want_family"} & set(request_payload(req))
    req = req.model_copy(update={"want_id": "w", "want_net": 0.5, "want_family": "f"})
    assert request_payload(req)["want_net"] == 0.5


def _old_pool_schema():
    """GpuLeaseRequestV1 as a pool built before patch 3b validates it: no want_* fields, extra forbidden."""
    from pydantic import ConfigDict, create_model

    fields = {k: (f.annotation, f) for k, f in GpuLeaseRequestV1.model_fields.items() if not k.startswith("want_")}
    return create_model("GpuLeaseRequestV1", __config__=ConfigDict(extra="forbid"), **fields)


def test_an_old_pool_refusing_the_fields_gets_the_hold_asked_again_without_them(monkeypatch):
    """Review finding 7: against the REAL refusal text an old pool sends (services/orion-gpu-pool/app/
    main.py: ``f"invalid:{exc}"[:300]`` of the pydantic error), not a hand-written one."""
    old = _old_pool_schema()
    sent = []

    async def fake_rpc(bus, req, *, source, timeout_sec):
        payload = request_payload(req)
        sent.append(payload)
        try:
            old.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 -- exactly as the old pool's handler
            return GpuLeaseReplyV1(status="unavailable", reason=f"invalid:{exc}"[:300])
        return GpuLeaseReplyV1(status="queued", lease_id="L1", position=3)

    import orion.gpu_pool.client as client
    monkeypatch.setattr(client, "lease_rpc", fake_rpc)
    reply = asyncio.run(acquire_hold(None, holder="durable-runs:r", work_class="agent", request_id="r:1",
                                     want_id="r", want_net=0.7, want_family="durable.self_inquiry"))
    assert reply.status == "queued" and len(sent) == 2 and "want_net" not in sent[1]
    sent.clear()
    reply = asyncio.run(acquire_hold(None, holder="durable-runs:r", work_class="agent", request_id="r:2"))
    assert reply.status == "queued" and len(sent) == 1               # nothing to drop: asked once
    # Any other refusal is passed through, never retried without the fields.
    async def refuse(bus, req, *, source, timeout_sec):
        sent.append(request_payload(req))
        return GpuLeaseReplyV1(status="unavailable", reason="unknown_class:agent")
    monkeypatch.setattr(client, "lease_rpc", refuse)
    sent.clear()
    reply = asyncio.run(acquire_hold(None, holder="durable-runs:r", work_class="agent", request_id="r:3",
                                     want_id="r", want_net=0.7, want_family="durable.self_inquiry"))
    assert reply.reason == "unknown_class:agent" and len(sent) == 1


# --- register: the first hold request carries the value --------------------------------------------

class _Store:
    def __init__(self, request):
        self.request, self.events = request, []

    async def get_run(self, run_id):
        return {"request": self.request}

    async def record_event(self, run_id, kind, payload, event_id=None):
        self.events.append(kind)


class _Holds:
    def __init__(self):
        self.calls = []

    def placement(self, admission):
        return "agent", "background", 0

    async def acquire(self, run_id, request_id, admission, *, correlation_id, **kw):
        self.calls.append(kw)
        return GpuLeaseReplyV1(status="queued", lease_id="L1", position=1)


class _Bids:
    def __init__(self, value):
        self.value, self.asked = value, 0

    async def bind(self, request, now):
        return None

    async def value_for(self, request):
        self.asked += 1
        return self.value


def _runtime(value):
    from app.admission_runtime import AdmissionRuntime

    rt = object.__new__(AdmissionRuntime)
    rt.store, rt.holds, rt.bids = _Store(_self_inquiry()), _Holds(), _Bids(value)
    rt.now = lambda: T0
    return rt


def test_register_sends_the_runs_value_on_a_new_hold_request_only():
    v = RunValue("run-si", 0.6, "durable.self_inquiry")
    rt = _runtime(v)
    out = asyncio.run(rt.register({"run_id": "run-si", "correlation_id": "c"}))
    assert out["status"] == "waiting_resource" and rt.holds.calls == [{"value": v}]
    # Re-asking with the hold's own request id is idempotent at the pool: no value lookup, none sent.
    rt2 = _runtime(v)
    asyncio.run(rt2.register({"run_id": "run-si", "correlation_id": "c", "hold_seq": 1,
                              "hold": {"request_id": "run-si:1", "lease_id": "L1"}}))
    assert rt2.holds.calls == [{}] and rt2.bids.asked == 0
    # No bid: the request goes out exactly as before.
    rt3 = _runtime(None)
    asyncio.run(rt3.register({"run_id": "run-si", "correlation_id": "c"}))
    assert rt3.holds.calls == [{}]
