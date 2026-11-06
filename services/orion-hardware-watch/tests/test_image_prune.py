"""The disk rule (ladder retirement PR 4): pure verdict, the background ask, the clock order, /health."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from orion.hardware_watch.rules import (
    IMAGE_PRUNE_MIN_DISK_PCT, DiskReading, ImagePruneRuleConfig, ImagePruneState, image_prune_verdict,
)

from app.image_prune import ImagePruneRule, read_docker_volume, skill_answer

T0 = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
CFG = ImagePruneRuleConfig()          # 70 %, 12 h cooldown, 30 min retry, 10 min start delay
COOL = timedelta(seconds=CFG.cooldown_sec)
LATE = T0 + timedelta(hours=1)        # well past the start delay


def _state(**kw) -> ImagePruneState:
    return ImagePruneState(started_at=T0, **kw)


def _skill_reply(decision: str, **extra) -> dict:
    body = {"decision": decision, "acted": decision.startswith("pruned"),
            "disk_before": {"used_pct": 81.3}, "dangling_before": {"available": True, "dangling_count": 172},
            **extra}
    return {"ok": True, "result": {"status": "success", "final_text": json.dumps(body)}}


# --- the pure verdict -----------------------------------------------------------------------

def test_the_floor_is_the_skills_own():
    assert IMAGE_PRUNE_MIN_DISK_PCT == 70.0 and CFG.min_disk_pct == 70.0


def test_asks_at_or_above_the_floor_and_never_below():
    assert image_prune_verdict(DiskReading(70.0), LATE, _state()).ask
    assert image_prune_verdict(DiskReading(81.3), LATE, _state()).ask
    v = image_prune_verdict(DiskReading(69.99), LATE, _state())
    assert (v.ask, v.reason) == (False, "below_threshold")


def test_an_unreadable_disk_never_asks():
    for reading in (DiskReading(None, "not_mounted"), DiskReading(float("nan"))):
        v = image_prune_verdict(reading, LATE, _state())
        assert (v.ask, v.reason) == (False, "disk_unreadable")


def test_the_start_delay_holds_the_first_ticks_after_a_restart():
    v = image_prune_verdict(DiskReading(81.0), T0 + timedelta(seconds=30), _state())
    assert (v.ask, v.reason) == (False, "starting") and v.wait_sec == pytest.approx(570.0)
    assert image_prune_verdict(DiskReading(81.0), T0 + timedelta(seconds=600), _state()).ask


def test_one_ask_at_a_time():
    v = image_prune_verdict(DiskReading(81.0), LATE, _state(in_flight=True))
    assert (v.ask, v.reason) == (False, "in_flight")


@pytest.mark.parametrize("outcome", ["pruned", "declined_nothing_to_reclaim", "pruned_nothing",
                                     "declined_no_pressure"])
def test_an_answered_ask_holds_the_full_cooldown(outcome):
    st = _state(last_finished_at=LATE, last_outcome=outcome)
    v = image_prune_verdict(DiskReading(81.0), LATE + COOL - timedelta(seconds=1), st)
    assert (v.ask, v.reason) == (False, "cooldown")
    assert image_prune_verdict(DiskReading(81.0), LATE + COOL, st).ask


@pytest.mark.parametrize("outcome", ["rpc_error", "unreadable_reply", "exec_error", "declined_unmeasurable",
                                     "failed"])
def test_an_unanswered_ask_waits_only_the_retry(outcome):
    st = _state(last_finished_at=LATE, last_outcome=outcome)
    v = image_prune_verdict(DiskReading(81.0), LATE + timedelta(minutes=29), st)
    assert (v.ask, v.reason) == (False, "retry_wait")
    assert image_prune_verdict(DiskReading(81.0), LATE + timedelta(minutes=30), st).ask


def test_an_ask_finished_after_the_ticks_clock_is_age_zero_not_negative():
    """The #3134 order: the tick's clock, then the ask's answer lands 2.4 s later. That is age 0 (held
    for the whole cooldown), never a negative age that a careless check would read as long ago."""
    now = LATE
    st = _state(last_finished_at=now + timedelta(seconds=2.4), last_outcome="pruned")
    v = image_prune_verdict(DiskReading(81.0), now, st)
    assert (v.ask, v.reason) == (False, "cooldown") and v.wait_sec == pytest.approx(CFG.cooldown_sec)


# --- reading the disk ------------------------------------------------------------------------

def test_a_plain_directory_is_not_read_as_the_docker_volume(tmp_path):
    r = read_docker_volume(str(tmp_path))
    assert r.used_pct is None and r.error == "not_a_mount_point"
    assert read_docker_volume(str(tmp_path / "missing")).error == "not_mounted"


def test_a_real_mount_point_reads_a_percentage():
    r = read_docker_volume("/")
    assert r.error is None and 0.0 < r.used_pct <= 100.0


# --- reading the skill's answer --------------------------------------------------------------

def test_skill_answer_reads_the_skills_own_decision():
    a = skill_answer(_skill_reply("pruned", images_removed=171, bytes_reclaimed=3366912))
    assert a["outcome"] == "pruned" and a["dangling_before"] == 172 and a["bytes_reclaimed"] == 3366912
    assert a["skill_disk_used_pct"] == 81.3 and a["plan_status"] == "success"


def test_skill_answer_never_invents_a_decision():
    assert skill_answer({"ok": True, "result": {"final_text": "not json"}})["outcome"] == "unreadable_reply"
    assert skill_answer({"ok": True, "result": {"final_text": "{}"}})["outcome"] == "unreadable_reply"
    assert skill_answer({"ok": True})["outcome"] == "unreadable_reply"
    assert skill_answer({"ok": False, "error": "boom"}) == {"outcome": "exec_error", "error": "boom"}


# --- the rule object: background ask, clock after the read -----------------------------------

class _Clock:
    def __init__(self, t: datetime):
        self.t = t

    def __call__(self) -> datetime:
        return self.t


def _rule(clock, ask, disk=81.3, enabled=True, cfg=CFG, reads=None):
    def read(_path):
        if reads is not None:
            reads.append(clock())
        return DiskReading(disk)
    return ImagePruneRule(enabled=enabled, mount_path="/hostfs/docker", cfg=cfg, ask=ask, read_disk=read,
                          clock=clock)


def test_a_switched_off_rule_never_reads_or_asks():
    calls = []

    async def ask(attempt_id):
        calls.append(attempt_id)
        return _skill_reply("pruned")

    async def go():
        clock = _Clock(LATE)
        rule = _rule(clock, ask, enabled=False)
        assert await rule.tick() is None
        return rule
    rule = asyncio.run(go())
    assert calls == [] and rule.snapshot()["enabled"] is False and rule.last_tick is None


def test_a_tick_asks_once_in_the_background_then_holds_for_the_cooldown():
    calls = []
    release = None

    async def ask(attempt_id):
        calls.append(attempt_id)
        if len(calls) == 1:
            await release.wait()         # the first ask is still running at the second tick
        return _skill_reply("declined_nothing_to_reclaim")

    async def go():
        nonlocal release
        release = asyncio.Event()
        clock = _Clock(T0)
        rule = _rule(clock, ask)
        clock.t = LATE
        v1 = await rule.tick()
        v2 = await rule.tick()           # same instant, ask still running
        release.set()
        await asyncio.gather(rule._in_flight)
        clock.t = LATE + timedelta(hours=1)
        v3 = await rule.tick()
        clock.t = LATE + COOL + timedelta(seconds=1)
        v4 = await rule.tick()
        await asyncio.gather(rule._in_flight)
        return rule, [v1.reason, v2.reason, v3.reason, v4.reason]
    rule, reasons = asyncio.run(go())
    assert reasons == ["ask", "in_flight", "cooldown", "ask"]
    assert len(calls) == 2 and all(c.startswith("image_prune_disk_rule:") for c in calls)
    snap = rule.snapshot()
    assert snap["asks"] == 2 and snap["outcomes"] == {"declined_nothing_to_reclaim": 2}
    assert snap["last_ask"]["outcome"] == "declined_nothing_to_reclaim" and snap["last_ask"]["dangling_before"] == 172
    assert snap["verdicts"] == {"ask": 2, "in_flight": 1, "cooldown": 1}


def test_a_failed_first_ask_after_a_restart_is_harmless_and_retried():
    """cortex-exec mid-restart: the RPC raises. The tick does not crash, the outcome is rpc_error, and
    the next ask comes after the 30 min retry, not the 12 h cooldown."""
    calls = []

    async def ask(attempt_id):
        calls.append(attempt_id)
        raise RuntimeError("RpcNoListener: orion:cortex:exec:request:background")

    async def go():
        clock = _Clock(T0)
        rule = _rule(clock, ask)
        clock.t = T0 + timedelta(seconds=600)
        await rule.tick()
        await asyncio.gather(rule._in_flight)
        clock.t = T0 + timedelta(seconds=600 + 1799)
        held = await rule.tick()
        clock.t = T0 + timedelta(seconds=600 + 1800)
        again = await rule.tick()
        await asyncio.gather(rule._in_flight)
        return rule, held, again
    rule, held, again = asyncio.run(go())
    assert held.reason == "retry_wait" and again.reason == "ask" and len(calls) == 2
    assert rule.snapshot()["outcomes"] == {"rpc_error": 2}
    assert "RpcNoListener" in rule.snapshot()["last_ask"]["error"]


def test_the_clock_is_taken_after_the_disk_read_with_a_real_read_lag():
    """Live order: the disk read takes 2.4 s. The verdict's clock is the time after that read, so an
    ask that finished during the read is age 0, never negative."""
    async def ask(attempt_id):
        return _skill_reply("pruned")

    async def go():
        clock = _Clock(T0)
        rule = _rule(clock, ask)

        def slow_read(_path):
            clock.t = clock.t + timedelta(seconds=2.4)    # the read takes 2.4 s
            return DiskReading(81.3)
        rule._read_disk = slow_read
        clock.t = LATE
        await rule.tick()
        await asyncio.gather(rule._in_flight)
        finished = rule._last_finished_at
        assert finished == LATE + timedelta(seconds=2.4)
        clock.t = LATE + COOL - timedelta(seconds=3)
        before = await rule.tick()   # read ends 0.6 s before the cooldown ends: still in cooldown
        clock.t = LATE + COOL
        after = await rule.tick()    # read ends 2.4 s later: one cooldown after the finish
        await asyncio.gather(rule._in_flight)
        return before, after, rule
    before, after, rule = asyncio.run(go())
    assert before.reason == "cooldown" and after.reason == "ask"
    assert rule.last_tick["at"] == (LATE + COOL + timedelta(seconds=2.4)).isoformat()


def test_health_exposes_the_rule():
    from app import main

    async def ask(attempt_id):
        return _skill_reply("pruned", bytes_reclaimed=10)

    async def go():
        clock = _Clock(T0)
        rule = _rule(clock, ask)
        clock.t = LATE
        await rule.tick()
        await asyncio.gather(rule._in_flight)
        main.image_prune = rule
        try:
            return await main.health()
        finally:
            main.image_prune = None
    body = asyncio.run(go())
    ip = body["image_prune"]
    assert ip["enabled"] is True and ip["verb"] == "skills.runtime.image_prune.v1"
    assert ip["last_tick"]["verdict"] == "ask" and ip["last_tick"]["disk_used_pct"] == 81.3
    assert ip["last_ask"]["outcome"] == "pruned" and ip["last_ask"]["bytes_reclaimed"] == 10


def test_the_bus_ask_sends_dispatchs_request_for_the_verb_in_execute_mode():
    from orion.core.bus.bus_schemas import ServiceRef

    from app.image_prune import build_bus_ask

    sent = {}

    class _Decoded:
        ok = True
        error = None

        def __init__(self, payload):
            self.envelope = type("E", (), {"payload": payload})()

    class _Codec:
        def decode(self, data):
            return _Decoded(data)

    class _Bus:
        codec = _Codec()

        async def rpc_request(self, channel, env, *, reply_channel, timeout_sec, require_listener):
            sent.update(channel=channel, env=env, reply=reply_channel, timeout=timeout_sec,
                        require_listener=require_listener)
            return {"data": _skill_reply("pruned")}

    ask = build_bus_ask(_Bus(), request_channel="orion:cortex:exec:request:background",
                        result_prefix="orion:exec:result", source=ServiceRef(name="orion-hardware-watch"),
                        timeout_sec=720.0)
    payload = asyncio.run(ask("image_prune_disk_rule:abc"))
    assert skill_answer(payload)["outcome"] == "pruned"
    assert sent["channel"] == "orion:cortex:exec:request:background" and sent["timeout"] == 720.0
    assert sent["require_listener"] is True and sent["reply"].startswith("orion:exec:result:")
    body = sent["env"].payload
    assert body["plan"]["verb_name"] == "skills.runtime.image_prune.v1"
    assert body["context"]["skill_args"] == {"mode": "execute"}
    assert body["args"]["trigger_source"] == "orion-hardware-watch"
    assert body["args"]["request_id"] == "image_prune_disk_rule:abc"
