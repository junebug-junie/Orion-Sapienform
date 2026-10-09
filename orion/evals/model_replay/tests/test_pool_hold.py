from __future__ import annotations

import asyncio

import pytest

from orion.evals.model_replay.pool_hold import (
    BONSAI_PROFILE, Q4_PROFILE, HoldLedger, HoldRefused, PoolHolds, release_leftovers,
)

GRANTS = {
    "memory_distill": {"role": "agent", "profile_name": Q4_PROFILE, "url": "http://c:8015", "cards": ["gpu1"]},
    "agent": {"role": "agent-gpu2", "profile_name": BONSAI_PROFILE, "url": "http://c:8016", "cards": ["gpu2"]},
}


class FakePool:
    """Grants by work class; records every verb. ``plan`` overrides grants per hold call (in order)."""

    def __init__(self, plan=None, queued=False):
        self.n = 0
        self.plan = list(plan or [])
        self.queued = queued
        self.verbs = []
        self.live = set()

    async def hold(self, work_class, actor):
        self.n += 1
        lid = f"L{self.n}"
        self.verbs.append(("hold", work_class, lid))
        self.live.add(lid)
        grant = dict(self.plan.pop(0) if self.plan else GRANTS[work_class])
        if grant.get("refuse"):
            return {"ok": False, "reason": "actuation_paused", "detail": {"status": "unavailable", "lease_id": lid}}
        grant.update(lease_id=lid, generation=1, served_by="x")
        if self.queued:
            self._pending = grant
            return {"ok": True, "detail": {"status": "queued", "lease_id": lid}}
        return {"ok": True, "detail": {"status": "granted", "lease_id": lid, "grant": grant}}

    async def wait_granted(self, lease_id, timeout_sec):
        return self._pending

    async def release(self, lease_id, actor):
        self.verbs.append(("release", lease_id))
        self.live.discard(lease_id)
        return {"ok": True}

    async def cancel(self, lease_id, actor):
        self.verbs.append(("cancel", lease_id))
        self.live.discard(lease_id)
        return {"ok": True}


def _holds(pool, tmp_path):
    return PoolHolds(transport=pool, ledger=HoldLedger(tmp_path / "holds.jsonl"))


def test_both_seats_verified_and_released(tmp_path):
    pool = FakePool()

    async def go():
        async with _holds(pool, tmp_path) as seats:
            assert seats["q4"].role == "agent" and seats["bonsai"].role == "agent-gpu2"
            assert len(pool.live) == 2      # gpu1 + gpu2 (one hold per role)

    asyncio.run(go())
    assert pool.live == set()
    assert [v[1] for v in pool.verbs if v[0] == "hold"] == ["memory_distill", "agent"]
    assert HoldLedger(tmp_path / "holds.jsonl").outstanding() == []


def test_released_when_task_body_raises(tmp_path):
    pool = FakePool()

    async def go():
        async with _holds(pool, tmp_path):
            raise RuntimeError("model crashed")

    with pytest.raises(RuntimeError):
        asyncio.run(go())
    assert pool.live == set()


def test_released_when_cancelled(tmp_path):
    pool = FakePool()

    async def go():
        async with _holds(pool, tmp_path):
            await asyncio.sleep(3600)

    async def main():
        t = asyncio.create_task(go())
        await asyncio.sleep(0.01)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t

    asyncio.run(main())
    assert pool.live == set()


def test_wrong_seat_refused_and_everything_released(tmp_path):
    # class `agent` lands on hecate's agent-deep instead of gpu2: refuse, give gpu1 back too.
    pool = FakePool(plan=[GRANTS["memory_distill"], {"role": "agent-deep", "profile_name": Q4_PROFILE,
                                                      "url": "http://h:8021", "cards": ["hecate-gpu0"]}])

    async def go():
        async with _holds(pool, tmp_path):
            pytest.fail("must not enter")

    with pytest.raises(HoldRefused, match="wrong seat"):
        asyncio.run(go())
    assert pool.live == set()


def test_wrong_profile_on_gpu2_refused(tmp_path):
    pool = FakePool(plan=[GRANTS["memory_distill"], {**GRANTS["agent"], "profile_name": Q4_PROFILE}])
    with pytest.raises(HoldRefused):
        asyncio.run(_holds(pool, tmp_path).__aenter__())
    assert pool.live == set()


def test_pool_refusal_releases_earlier_holds(tmp_path):
    pool = FakePool(plan=[GRANTS["memory_distill"], {"refuse": True}])
    with pytest.raises(HoldRefused, match="refused"):
        asyncio.run(_holds(pool, tmp_path).__aenter__())
    assert pool.live == set()


def test_queued_hold_waits_for_grant(tmp_path):
    pool = FakePool(queued=True)
    holds = PoolHolds(transport=pool, ledger=HoldLedger(tmp_path / "h.jsonl"), models=("q4",))

    async def go():
        async with holds as seats:
            assert seats["q4"].url == "http://c:8015"

    asyncio.run(go())
    assert pool.live == set()


def test_leftovers_after_sigkill_are_released(tmp_path):
    ledger = HoldLedger(tmp_path / "holds.jsonl")
    ledger.append(action="acquired", lease_id="A")
    ledger.append(action="acquired", lease_id="B")
    ledger.append(action="released", lease_id="A")
    assert ledger.outstanding() == ["B"]
    pool = FakePool()
    pool.live = {"B"}
    released = asyncio.run(release_leftovers(pool, ledger, "t"))
    assert released == ["B"] and pool.live == set() and ledger.outstanding() == []


def test_lease_ended_reads_pool_replies():
    from orion.evals.model_replay.pool_hold import lease_ended

    assert lease_ended({"ok": True})
    assert lease_ended({"ok": False, "detail": {"status": "unavailable", "reason": "cancelled"}})   # release of a queued lease
    assert lease_ended({"ok": False, "reason": "not_cancelable_from_released"})
    assert lease_ended({"ok": False, "reason": "unknown_lease"})
    assert not lease_ended({"ok": False, "reason": "TimeoutError: "})
    assert not lease_ended({"ok": False, "detail": {"status": "granted"}})


def test_bus_transport_sends_only_control_envelopes():
    import json as _json

    from orion.evals.model_replay.pool_hold import BusControlTransport
    from orion.schemas.gpu_pool import GPU_POOL_CONTROL_REQUEST_CHANNEL

    sent = []

    class Bus:
        async def rpc_request(self, channel, env, *, reply_channel, timeout_sec):
            sent.append((channel, env.payload, env.reply_to == reply_channel))
            reply = {"payload": {"ok": True, "reason": None,
                                 "detail": {"status": "queued", "lease_id": "L9"}}}
            return {"data": _json.dumps(reply)}

    t = BusControlTransport("redis://unused")
    t._bus = Bus()

    async def go():
        res = await t.hold("memory_distill", "bonsai-replay-eval")
        await t.release("L9", "bonsai-replay-eval")
        await t.cancel("L9", "bonsai-replay-eval")
        return res

    res = asyncio.run(go())
    assert res["detail"]["lease_id"] == "L9" and "L9" in t._grants
    assert [s[0] for s in sent] == [GPU_POOL_CONTROL_REQUEST_CHANNEL] * 3
    assert [s[1]["verb"] for s in sent] == ["hold", "release", "cancel"]
    assert sent[0][1]["work_class"] == "memory_distill" and all(s[2] for s in sent)


def test_hold_rpc_timeout_sweeps_the_pool_for_our_lease(tmp_path):
    """The pool created the lease but the reply never came: the sweep finds and releases it."""
    pool = FakePool()
    calls = {"n": 0}

    async def hold(work_class, actor):
        calls["n"] += 1
        if work_class == "agent":
            pool.live.add("GHOST")
            raise TimeoutError("no reply")
        return await FakePool.hold(pool, work_class, actor)

    pool.hold = hold

    async def sweep():
        return sorted(pool.live)

    holds = PoolHolds(transport=pool, ledger=HoldLedger(tmp_path / "h.jsonl"), sweep=sweep)
    with pytest.raises(TimeoutError):
        asyncio.run(holds.__aenter__())
    assert pool.live == set()
    rows = (tmp_path / "h.jsonl").read_text()
    assert '"requested"' in rows and '"GHOST"' in rows


def test_second_cancel_during_release_still_releases_everything(tmp_path):
    pool = FakePool()
    real_release = pool.release

    async def slow_release(lease_id, actor):
        await asyncio.sleep(0.05)
        return await real_release(lease_id, actor)

    pool.release = slow_release

    async def main():
        holds = _holds(pool, tmp_path)
        await holds.__aenter__()
        task = asyncio.create_task(holds.release_all(reason="x"))
        await asyncio.sleep(0.01)
        task.cancel()                      # cancel while the first release is in flight
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.2)           # shielded releases finish

    asyncio.run(main())
    assert pool.live == set()


def test_seat_problem_reads_pool_state():
    from orion.evals.model_replay.pool_hold import HOLDER, seat_problem

    roles = [{"role": "agent", "status": "confirmed", "profile_name": Q4_PROFILE},
             {"role": "agent-gpu2", "status": "confirmed", "profile_name": BONSAI_PROFILE}]
    assert seat_problem({"roles": roles, "leases": []}) is None
    busy = [{"role": "agent-gpu2", "kind": "hold", "status": "granted", "holder": "durable-runs:x"}]
    assert "held by durable-runs:x" in seat_problem({"roles": roles, "leases": busy})
    ours = [{"role": "agent-gpu2", "kind": "hold", "status": "granted", "holder": HOLDER}]
    assert seat_problem({"roles": roles, "leases": ours}) is None
    swapped = [roles[0], {**roles[1], "profile_name": Q4_PROFILE}]
    assert "want ternary" in seat_problem({"roles": swapped, "leases": []})


def test_leftovers_sweep_finds_unledgered_lease(tmp_path):
    pool = FakePool()
    pool.live = {"UNSEEN"}

    async def sweep():
        return sorted(pool.live)

    released = asyncio.run(release_leftovers(pool, HoldLedger(tmp_path / "h.jsonl"), "t", sweep=sweep))
    assert released == ["UNSEEN"] and pool.live == set()
