"""The GPU pool discovery bridge, worker side: a worker announces its role, profile and host port
on orion:llm:worker:announce, and stays silent when not configured for the pool."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app import main as host_main
from orion.schemas.gpu_pool import LLM_WORKER_ANNOUNCE_CHANNEL, LlmWorkerAnnounceV1


class Bus:
    def __init__(self):
        self.sent = []

    async def publish(self, channel, env):
        self.sent.append((channel, env))


def _settings(**kw):
    base = dict(llm_role="metacog", llm_announce_port=8012, llm_announce_host="circe",
                llm_profile_name="qwen3-8b-q5km-v100-16gb-atlas-metacog-16k",
                service_name="llamacpp-host", service_version="0.1.0")
    base.update(kw)
    return SimpleNamespace(**base)


def test_announces_role_profile_and_host_port():
    bus = Bus()
    asyncio.run(host_main._announce_worker(bus, _settings()))
    [(channel, env)] = bus.sent
    assert channel == LLM_WORKER_ANNOUNCE_CHANNEL and env.kind == "llm.worker.announce.v1"
    ann = LlmWorkerAnnounceV1.model_validate(env.payload)
    assert (ann.role, ann.port, ann.profile_name) == ("metacog", 8012, "qwen3-8b-q5km-v100-16gb-atlas-metacog-16k")


def test_silent_without_role_or_port():
    bus = Bus()
    asyncio.run(host_main._announce_worker(bus, _settings(llm_role=None)))
    asyncio.run(host_main._announce_worker(bus, _settings(llm_announce_port=None)))
    assert bus.sent == []


def test_every_compose_worker_declares_a_role_and_port():
    from pathlib import Path

    for name in ("docker-compose.atlas-workers.yml", "docker-compose.dsv41.yml"):
        text = (Path(__file__).resolve().parents[1] / name).read_text()
        assert text.count("LLM_PROFILE_NAME=") == text.count("LLM_ROLE=") == text.count("LLM_ANNOUNCE_PORT="), name


def test_heartbeat_loop_survives_bus_down_then_hung_then_recovers(monkeypatch):
    """A bus down at boot, then a hung publish, must not kill or wedge the loop: it reconnects."""
    calls = {"n": 0}
    published = []

    class FlakyBus:
        def __init__(self, *a, **k):
            pass

        async def connect(self):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("bus down at boot")

        async def publish(self, channel, env):
            if calls["n"] == 2:
                await asyncio.sleep(3600)  # hung socket
            published.append(channel)

        async def close(self):
            pass

    monkeypatch.setattr(host_main, "OrionBusAsync", FlakyBus)
    monkeypatch.setattr(host_main, "HEARTBEAT_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(host_main, "HEARTBEAT_STEP_TIMEOUT_SEC", 0.05)

    async def scenario():
        task = asyncio.create_task(host_main.heartbeat_loop(_settings(orion_bus_url="redis://x")))
        for _ in range(300):
            if LLM_WORKER_ANNOUNCE_CHANNEL in published:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        await task
    asyncio.run(scenario())
    assert "orion:system:health" in published and LLM_WORKER_ANNOUNCE_CHANNEL in published
    assert calls["n"] >= 3


def test_strata_profile_announces_only_and_launches_no_server(monkeypatch):
    """The chat lane's engine is Strata in its own container: this host must heartbeat, never spawn llama-server."""
    beats = []

    async def heartbeat(settings):
        beats.append(settings.llm_profile_name)

    async def no_spawn(*a, **k):
        raise AssertionError("a strata profile must not launch llama-server")

    profile = SimpleNamespace(name="qwen38-flash-next-iq3s-strata-1xv100-circe-chat", backend="strata")
    monkeypatch.setattr(type(host_main.settings), "resolve_profile", lambda self: profile)
    monkeypatch.setattr(host_main.settings, "cuda_visible_devices_override", None)
    monkeypatch.setattr(host_main, "heartbeat_loop", heartbeat)
    monkeypatch.setattr(host_main, "build_llama_server_cmd_and_env", no_spawn)
    monkeypatch.setattr(host_main.asyncio, "create_subprocess_exec", no_spawn)
    asyncio.run(host_main._main_async())
    assert beats == [host_main.settings.llm_profile_name]


def test_the_strata_chat_profile_loads_in_this_host():
    from pathlib import Path

    import yaml

    from app.profiles import LLMProfile

    name = "qwen38-flash-next-iq3s-strata-1xv100-circe-chat"
    raw = yaml.safe_load((Path(__file__).resolve().parents[3] / "config" / "llm_profiles.yaml").read_text())
    assert LLMProfile(name=name, **raw["profiles"][name]).backend == "strata"
