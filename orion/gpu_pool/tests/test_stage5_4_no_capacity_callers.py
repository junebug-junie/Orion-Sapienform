"""Stage 5.4 gate: nothing calls the durable-runs /capacity permit broker any more.

World-model and the visual chain (the only two callers left after stage 4) now take GPU pool
leases/holds. Stage 5.6 deleted the broker itself (the capacity store and client, the durable-runs
/capacity routes, the Capacity*V1 schemas), so the allow-list below is empty: no non-test file may
name a client, a route, the old package or its env keys. Kill means kill: no service may grow a new
caller, a fallback, or a second broker.

Also pins the other half of 5.4: the pool's visual_baseline swap guard and its thought
/visual-chain/activity read are gone from the pool.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# Stage 5.6: the broker, its schemas, tests and evals are deleted. The only exemption is the
# stage 6.6 dead-key reporter, which names retired keys (DURABLE_RUNS_CAPACITY_*) precisely so it
# can delete them from live .env files -- a removal list, not a caller.
ALLOWED: tuple[str, ...] = ("scripts/report_dead_env_keys.py",)
# A client: importing the permit client, constructing a permit, or POSTing to the routes.
CALLER = re.compile(r"capacity_client|GpuCapacityPermit|/capacity/(acquire|renew|release)|:8121/capacity|8124/capacity"
                    r"|orion\.durable_admission|orion/durable_admission/|PostgresCapacityStore|Capacity(Acquire|Token|Permit)V1"
                    r"|DURABLE_RUNS_CAPACITY_|[\"']/capacity[\"']")


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
    assert callers == [], f"/capacity permit broker or callers remain (5.4 removed callers, 5.6 the broker): {callers}"


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


def test_the_broker_package_and_its_tables_left_the_code():
    """5.6: the package is gone (the run registry moved to orion/durable_runs), and no live code,
    SQL model or metric definition anywhere reads or writes the four dropped tables. Exempt: tests,
    evals, docs, the migrations that created/drop them, and the snapshot-and-drop script."""
    assert not (ROOT / "orion" / "durable_admission").exists()
    assert (ROOT / "orion" / "durable_runs" / "registry_store.py").is_file()
    dropped = re.compile(r"durable_gateway_permits|durable_resource_demands|durable_resource_leases|durable_elastic_slot")
    hits = []
    exempt = ("services/orion-sql-db/", "scripts/gpu_pool_stage5_snapshot_and_drop.sh")
    for base in ("orion", "services", "scripts", "config"):
        for path in (ROOT / base).rglob("*"):
            rel = path.relative_to(ROOT).as_posix()
            if (path.suffix not in (".py", ".sql", ".sh", ".yaml", ".yml", ".json") or not path.is_file()
                    or _is_test(rel) or "/node_modules/" in rel or rel.startswith(exempt)):
                continue
            # Module docstrings may name them as deleted (registry_store's does); code and SQL must not.
            code = path.read_text(errors="ignore")
            if code.lstrip().startswith('"""'):
                code = code.split('"""', 2)[-1]
            code = "\n".join(line for line in code.splitlines() if not line.lstrip().startswith("#"))
            if dropped.search(code):
                hits.append(rel)
    assert hits == [], hits
