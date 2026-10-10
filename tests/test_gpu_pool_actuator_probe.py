"""orion/gpu_pool/actuator_probe.py (shared probe core) and the CLI over it."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from orion.gpu_pool import actuator_probe as ap
from orion.gpu_pool.config import launch_digest, load_pool_config
from orion.schemas.gpu_pool import GPU_POOL_ACTUATE_REQUEST_CHANNEL, GPU_POOL_ACTUATE_RESULT_CHANNEL

REPO = Path(__file__).resolve().parents[1]
CFG = load_pool_config(REPO / "config" / "gpu_pool.yaml")
ROLE = "agent-gpu2"


def _script():
    spec = importlib.util.spec_from_file_location("gpu_pool_actuator_probe_cli", REPO / "scripts" / "gpu_pool_actuator_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _refused(reason):
    return [{"status": "accepted"}, {"status": "refused", "reason": reason}]


class TestClassify:
    def test_incident_status_is_config_unloadable(self) -> None:
        v = ap.classify("status", _refused("config_unloadable:ValidationError"), role=ROLE)
        assert (v.kind, v.reason, v.ok) == ("config_unloadable", "config_unloadable:ValidationError", False)
        assert v.line == "NOT OK: refused config_unloadable:ValidationError"

    def test_digest_config_unloadable_is_also_unloadable(self) -> None:
        v = ap.classify("digest", _refused("config_unloadable:ValidationError"))
        assert v.kind == "config_unloadable"
        assert v.line == "UNEXPECTED: refused config_unloadable:ValidationError"

    def test_digest_agree_mismatch_and_other(self) -> None:
        assert ap.classify("digest", _refused("profile_not_allowed")).kind == "ok"
        assert ap.classify("digest", _refused("launch_digest_mismatch")).kind == "digest_mismatch"
        assert ap.classify("digest", _refused("busy")).kind == "other_refusal"
        assert ap.classify("digest", _refused("deadline_passed")).kind == "other_refusal"

    def test_status_succeeded_is_ok(self) -> None:
        v = ap.classify("status", [{"status": "succeeded", "observed": {"diffusion": "running"},
                                    "in_flight": False, "last_action_id": "a1"}])
        assert v.ok and v.line == "OK: observed={'diffusion': 'running'} in_flight=False last_action_id=a1"

    def test_no_terminal_result_is_no_answer(self) -> None:
        assert ap.classify("status", []).kind == "no_answer"
        assert ap.classify("status", [{"status": "accepted"}, {"status": "progress"}]).kind == "no_answer"

    def test_last_terminal_wins(self) -> None:
        got = [{"status": "refused", "reason": "busy"}, {"status": "refused", "reason": "config_unloadable:X"}]
        assert ap.classify("status", got).kind == "config_unloadable"


def test_build_request_status_and_digest() -> None:
    s = ap.build_request(CFG, ROLE, "status")
    d = ap.build_request(CFG, ROLE, "digest")
    assert (s.action, s.profile) == ("status", None)
    assert (d.action, d.profile) == ("load", ap.PROBE_PROFILE)
    assert d.launch_digest == launch_digest(CFG, ROLE) and d.actuator == "circe"
    assert s.action_id.startswith("probe-status:agent-gpu2:") and s.generation == 1


def test_build_request_refuses_role_without_launch() -> None:
    with pytest.raises(ValueError, match="no launch block"):
        ap.build_request(CFG, "chat", "status")
    with pytest.raises(SystemExit, match="no launch block"):
        _script().build(CFG, "chat", "status")


class FakePubSub:
    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()

    async def get_message(self, ignore_subscribe_messages=True, timeout=1.0):
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=min(timeout, 0.01))
        except asyncio.TimeoutError:
            return None


class FakeBus:
    """Answers each actuate request with responder(request_payload) -> list of result payloads,
    plus a foreign result first (another action_id) to prove filtering."""

    def __init__(self, responder) -> None:
        self.responder = responder
        self.published: list[tuple[str, dict]] = []
        self.subscribed: list[str] = []
        self._subs: list[FakePubSub] = []

    @asynccontextmanager
    async def subscribe(self, *channels):
        self.subscribed.extend(channels)
        ps = FakePubSub()
        self._subs.append(ps)
        yield ps

    async def publish(self, channel, env) -> None:
        payload = env.payload
        self.published.append((channel, payload))
        foreign = {"action_id": "pool-action-1", "status": "refused", "reason": "busy"}
        for res in [foreign, *self.responder(payload)]:
            for ps in self._subs:
                ps.queue.put_nowait({"data": json.dumps({"payload": res}).encode()})


def _answer(status, reason=None):
    return lambda req: [{"action_id": req["action_id"], "status": "accepted"},
                        {"action_id": req["action_id"], "status": status, "reason": reason}]


@pytest.mark.asyncio
async def test_probe_publishes_and_classifies_only_its_own_answer() -> None:
    bus = FakeBus(_answer("refused", "config_unloadable:ValidationError"))
    v = await ap.probe(bus, CFG, ROLE, "status", 2.0, source="test")
    assert v.kind == "config_unloadable"
    assert bus.subscribed == [GPU_POOL_ACTUATE_RESULT_CHANNEL]
    assert [c for c, _ in bus.published] == [GPU_POOL_ACTUATE_REQUEST_CHANNEL]
    assert all(r["action_id"] != "pool-action-1" for r in v.results)


@pytest.mark.asyncio
async def test_probe_no_answer_after_wait() -> None:
    bus = FakeBus(lambda req: [])
    v = await ap.probe(bus, CFG, ROLE, "digest", 0.05)
    assert v.kind == "no_answer"


def test_cli_output_and_exit_code(monkeypatch, capsys) -> None:
    """The CLI keeps its line format and exit code (1 when any check is not OK)."""
    import orion.core.bus.async_service as bus_mod

    answers = {"status": _answer("refused", "config_unloadable:ValidationError"),
               "load": _answer("refused", "profile_not_allowed")}

    class CliBus(FakeBus):
        def __init__(self, url) -> None:
            super().__init__(lambda req: answers[req["action"]](req))

        async def connect(self) -> None:
            pass

        async def close(self) -> None:
            pass

    monkeypatch.setattr(bus_mod, "OrionBusAsync", CliBus)
    monkeypatch.setenv("ORION_BUS_URL", "redis://fake:6379/0")
    cli = _script()
    rc = asyncio.run(cli.probe(ROLE, ["status", "digest"], 1.0, str(REPO / "config" / "gpu_pool.yaml")))
    out = capsys.readouterr().out.splitlines()
    dig = launch_digest(CFG, ROLE)[:16]
    assert out == [f"status  {ROLE} digest={dig} -> NOT OK: refused config_unloadable:ValidationError",
                   f"digest  {ROLE} digest={dig} -> OK: launch digests agree"]
    assert rc == 1
    answers["status"] = lambda req: [{"action_id": req["action_id"], "status": "succeeded",
                                      "observed": {}, "in_flight": False, "last_action_id": None}]
    assert asyncio.run(cli.probe(ROLE, ["status", "digest"], 1.0, str(REPO / "config" / "gpu_pool.yaml"))) == 0
