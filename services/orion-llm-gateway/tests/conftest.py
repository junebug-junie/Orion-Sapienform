import contextlib
import os
import sys
from typing import Any, Callable, Dict, List, Optional

import pytest

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# The real routes -> class map (the container reads /app/config/gpu_pool.yaml).
os.environ.setdefault("GPU_POOL_CONFIG_PATH", os.path.join(REPO_ROOT, "config", "gpu_pool.yaml"))

# Role -> fake llama.cpp URL / ctx per slot. Class -> role the fake pool grants (first choice).
FAKE_ROLE_URLS = {
    "chat": "http://pool-chat:8011",
    "agent": "http://pool-agent:8015",
    "agent-gpu2": "http://pool-agent-gpu2:8016",
    "metacog": "http://pool-metacog:8012",
    "fast": "http://pool-fast:8013",
}
FAKE_CTX = {"chat": 131072, "agent": 32768, "agent-gpu2": 32768, "metacog": 16384, "fast": 4096}
FAKE_CLASS_ROLE = {"chat": "chat", "agent": "agent", "metacog": "metacog", "fast": "fast"}


class FakePool:
    """Stands in for orion.gpu_pool.client.gpu_lease: records every acquire and release."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.releases: List[str] = []
        self.active = 0
        self.max_active = 0
        self.unavailable: Optional[str] = None
        # Optional override: kwargs -> role name (or raise LeaseUnavailable).
        self.choose: Optional[Callable[[Dict[str, Any]], str]] = None

    def grant(self, role: str, n: int):
        from orion.schemas.gpu_pool import GpuLeaseGrantV1

        return GpuLeaseGrantV1(
            lease_id=f"lease-{n}", generation=1, role=role, cards=["gpuX"], url=FAKE_ROLE_URLS[role],
            profile_name=f"profile-{role}", model_file=f"{role}.gguf", ctx_per_slot=FAKE_CTX[role],
            served_by=f"circe-worker-{role}",
        )

    @contextlib.asynccontextmanager
    async def gpu_lease(self, bus: Any, **kw: Any):
        from orion.gpu_pool.client import Lease, LeaseUnavailable

        self.calls.append(dict(kw))
        if self.unavailable:
            raise LeaseUnavailable(self.unavailable)
        role = self.choose(kw) if self.choose else FAKE_CLASS_ROLE[kw["work_class"]]
        grant = self.grant(role, len(self.calls))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            yield Lease(grant.lease_id, grant)
        except BaseException:
            self.releases.append("upstream_error")
            raise
        else:
            self.releases.append("ok")
        finally:
            self.active -= 1


@pytest.fixture
def fake_pool(monkeypatch: pytest.MonkeyPatch) -> FakePool:
    from app import pool_placement

    pool = FakePool()
    monkeypatch.setattr(pool_placement, "gpu_lease", pool.gpu_lease)
    monkeypatch.setattr(pool_placement, "_bus", object())
    return pool
