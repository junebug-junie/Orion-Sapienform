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
