import asyncio
import os
import sys
from uuid import uuid4

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.logic import (  # noqa: E402
    SKILL_BIOMETRICS_SNAPSHOT_V1,
    SKILL_GPU_NVIDIA_SMI_SNAPSHOT_V1,
    build_skill_cortex_orch_envelope,
    dispatch_cortex_request,
)
from app.main import should_run_interval  # noqa: E402
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402


class _FakeBus:
    def __init__(self) -> None:
        self.published = []

    async def publish(self, channel: str, envelope: BaseEnvelope) -> None:
        self.published.append((channel, envelope))


def test_should_run_interval_respects_startup_and_spacing():
    assert should_run_interval(now_monotonic=10.0, last_run_monotonic=None, interval_seconds=600, run_on_startup=True) is True
    assert should_run_interval(now_monotonic=10.0, last_run_monotonic=None, interval_seconds=600, run_on_startup=False) is False
    assert should_run_interval(now_monotonic=700.0, last_run_monotonic=50.0, interval_seconds=600, run_on_startup=False) is True
    assert should_run_interval(now_monotonic=500.0, last_run_monotonic=50.0, interval_seconds=600, run_on_startup=False) is False


def test_scheduler_skill_dispatch_builds_cortex_requests():
    bus = _FakeBus()
    parent = BaseEnvelope(kind="actions.scheduler.tick.v1", source=ServiceRef(name="orion-actions"), correlation_id=str(uuid4()), payload={})

    biometrics_env = build_skill_cortex_orch_envelope(
        parent,
        source=ServiceRef(name="orion-actions"),
        verb=SKILL_BIOMETRICS_SNAPSHOT_V1,
        session_id="skills_scheduler",
        user_id="operators",
        metadata={"schedule": "periodic_skills"},
        options={"source": "test"},
        recall_enabled=False,
    )
    gpu_env = build_skill_cortex_orch_envelope(
        parent,
        source=ServiceRef(name="orion-actions"),
        verb=SKILL_GPU_NVIDIA_SMI_SNAPSHOT_V1,
        session_id="skills_scheduler",
        user_id="operators",
        metadata={"schedule": "periodic_skills"},
        options={"source": "test"},
        recall_enabled=False,
    )

    asyncio.run(dispatch_cortex_request(bus=bus, channel="orion:cortex:request", envelope=biometrics_env))
    asyncio.run(dispatch_cortex_request(bus=bus, channel="orion:cortex:request", envelope=gpu_env))

    assert [env.payload["verb"] for _, env in bus.published] == [
        SKILL_BIOMETRICS_SNAPSHOT_V1,
        SKILL_GPU_NVIDIA_SMI_SNAPSHOT_V1,
    ]
