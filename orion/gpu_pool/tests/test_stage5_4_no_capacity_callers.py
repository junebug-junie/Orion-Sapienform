"""Stage 5.4 gate: nothing calls the durable-runs /capacity permit broker any more.

World-model and the visual chain (the only two callers left after stage 4) now take GPU pool
leases/holds. The broker itself (orion/durable_admission/capacity*.py, durable-runs /capacity) is
deleted in 5.6; until then this proves it has no client, so zero new durable_gateway_permits rows
is the expected live state after deploy -- not a quiet failure. Kill means kill: no service may
grow a new caller or a fallback to it.

Also pins the other half of 5.4: the pool's visual_baseline swap guard and its thought
/visual-chain/activity read are gone from the pool.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# The broker, its schemas, and their own tests/evals: deleted together in 5.6.
ALLOWED = (
    "orion/durable_admission/",
    "orion/schemas/resource_admission.py",
    "orion/schema_skew_discovery.py",
    "orion/tests/test_capacity_client.py",
    "orion/gpu_pool/tests/",
    "services/orion-durable-runs/",
)
# A client: importing the permit client, constructing a permit, or POSTing to the routes.
CALLER = re.compile(r"capacity_client|GpuCapacityPermit|/capacity/(acquire|renew|release)|:8121/capacity|8124/capacity")


def _sources():
    for base in ("orion", "services", "scripts"):
        for path in (ROOT / base).rglob("*"):
            if path.suffix not in (".py", ".js", ".sh", ".yml", ".yaml") or not path.is_file():
                continue
            rel = path.relative_to(ROOT).as_posix()
            if "/node_modules/" in rel or rel.startswith(ALLOWED):
                continue
            yield rel, path


def _is_test(rel: str) -> bool:
    return "/tests/" in rel or "/evals/" in rel


def test_no_service_calls_the_capacity_permit_broker():
    callers = []
    for rel, path in _sources():
        if _is_test(rel):
            continue   # tests may name it to assert it is gone
        text = path.read_text(errors="ignore")
        if CALLER.search(text):
            callers.append(rel)
    assert callers == [], f"/capacity permit callers remain (stage 5.4 removed them): {callers}"


def test_world_and_diffusion_env_templates_carry_no_permit_keys():
    for rel in ("services/orion-world-model/.env_example", "services/orion-world-model/docker-compose.yml",
                "services/orion-thought/.env_example", "services/orion-thought/docker-compose.yml"):
        text = (ROOT / rel).read_text()
        for key in ("WM_GPU2_CAPACITY_", "ORION_VISUAL_CHAIN_GPU2_CAPACITY_", "ORION_VISUAL_ELASTIC_"):
            assert key not in text, f"{rel} still carries {key}*"


def test_pool_no_longer_reads_the_visual_chain():
    for rel in ("services/orion-gpu-pool/app/guards.py", "services/orion-gpu-pool/app/settings.py",
                "services/orion-gpu-pool/app/main.py", "services/orion-gpu-pool/docker-compose.yml",
                "services/orion-gpu-pool/.env_example", "services/orion-gpu-pool/Dockerfile",
                "config/gpu_pool.yaml"):
        text = (ROOT / rel).read_text()
        assert "GPU_POOL_VISUAL_ACTIVITY_URL" not in text and "visual_activity_url" not in text, rel
        assert "/visual-chain/activity" not in text or rel.endswith("guards.py"), rel   # guards.py docstring names it
