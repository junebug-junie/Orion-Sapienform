import asyncio
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

# Role -> fake llama.cpp URL / ctx per slot. ctx values are the LIVE per-slot contexts (pool state,
# 2026-09-25): a test that only passes because a fake role is bigger than the real one is lying.
FAKE_ROLE_URLS = {
    "chat": "http://pool-chat:8011",
    "agent": "http://pool-agent:8015",
    "agent-gpu2": "http://pool-agent-gpu2:8016",
    "metacog": "http://pool-metacog:8012",
    "fast": "http://pool-fast:8013",
}
FAKE_CTX = {"chat": 65536, "agent": 131072, "agent-gpu2": 131072, "metacog": 4096, "fast": 4096}


class FakePool:
    """Stands in for orion.gpu_pool.client.gpu_lease, with the scheduler rules that matter here:

    * a class's roles are tried in config/gpu_pool.yaml preference order; a borrower only reaches a
      lendable card (gpu0/chat) while ``lent`` is set;
    * a role is only granted when its ctx_per_slot >= min_ctx_tokens; when no role of the class is
      that big the answer is immediate: ``min_ctx_exceeds_class:<largest ctx in the class>``;
    * each role has ``slots`` concurrent leases; with none free the acquire waits (tests shrink the
      wait with ``max_wait_sec``) and then fails ``deadline``, like a queued lease past deadline_at;
    * the release outcome honours ``lease.release_outcome``, like the real client;
    * stage 4 holds: ``holds[lease_id] = GpuLeaseRefV1`` is a live hold that occupies one slot of its
      role (``busy``). A call with ``hold=`` attaches: it runs on the hold's role in the slot the hold
      reserves (never waits for another), or is refused ``attach_refused`` when the hold is unknown
      or its generation stale -- the pool-side rule the gateway codes against.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.releases: List[str] = []
        self.leases: List[Any] = []
        self.active = 0
        self.max_active = 0
        self.unavailable: Optional[str] = None
        # Optional preference: kwargs -> role name tried first (still subject to ctx and slots).
        self.choose: Optional[Callable[[Dict[str, Any]], str]] = None
        self.slots: Dict[str, int] = {role: 4 for role in FAKE_CTX}
        self.busy: Dict[str, int] = {role: 0 for role in FAKE_CTX}
        self.lent = False
        self.max_wait_sec = 0.05
        self.withdrawn = 0
        self.on_grant: Optional[Callable[[Any], None]] = None
        self.urls: Dict[str, str] = dict(FAKE_ROLE_URLS)  # tests may point a role at a local server
        self.holds: Dict[str, Any] = {}
        self.attach_refused = "unknown_lease"

    def add_hold(self, ref: Any) -> None:
        """A durable run's hold, granted: it takes one slot of its role until the test ends."""
        self.holds[ref.lease_id] = ref
        self.busy[ref.role] += 1

    def grant(self, role: str, n: int):
        from orion.schemas.gpu_pool import GpuLeaseGrantV1

        return GpuLeaseGrantV1(
            lease_id=f"lease-{n}", generation=1, role=role, cards=["gpuX"], url=self.urls[role],
            profile_name=f"profile-{role}", model_file=f"{role}.gguf", ctx_per_slot=FAKE_CTX[role],
            served_by=f"circe-worker-{role}",
        )

    def _candidates(self, kw: Dict[str, Any]) -> List[str]:
        from app import pool_placement
        from orion.gpu_pool.client import LeaseUnavailable

        cfg = pool_placement.pool_config()
        roles = [r for r in cfg.classes[kw["work_class"]].roles if r in FAKE_CTX]
        min_ctx = int(kw.get("min_ctx_tokens") or 0)
        largest = max(FAKE_CTX[r] for r in roles)
        if min_ctx > largest:
            raise LeaseUnavailable(f"min_ctx_exceeds_class:{largest}")
        usable = [r for r in roles if FAKE_CTX[r] >= min_ctx
                  and (cfg.owns(kw["work_class"], r) or self.lent or not cfg.lendable_cards(r))]
        if self.choose:
            preferred = self.choose(kw)
            if preferred in usable:
                usable = [preferred] + [r for r in usable if r != preferred]
        return usable

    async def _place(self, kw: Dict[str, Any]) -> str:
        from orion.gpu_pool.client import LeaseUnavailable

        candidates = self._candidates(kw)
        loop = asyncio.get_running_loop()
        give_up = loop.time() + min(float(kw.get("deadline_sec") or 0), self.max_wait_sec)
        while True:
            for role in candidates:
                if self.busy[role] < self.slots[role]:
                    return role
            if loop.time() >= give_up:
                raise LeaseUnavailable("deadline")
            await asyncio.sleep(0.005)

    @contextlib.asynccontextmanager
    async def gpu_lease(self, bus: Any, **kw: Any):
        from orion.gpu_pool.client import Lease, LeaseUnavailable

        self.calls.append(dict(kw))
        if self.unavailable:
            raise LeaseUnavailable(self.unavailable)
        hold = kw.get("hold")
        if hold is not None:
            live = self.holds.get(hold.lease_id)
            if live is None or live.generation != hold.generation:
                raise LeaseUnavailable(self.attach_refused)
            role = live.role  # the child runs in the slot its hold reserves: no queue, no second slot
        else:
            try:
                role = await self._place(kw)
            except asyncio.CancelledError:
                self.withdrawn += 1
                raise
        grant = self.grant(role, len(self.calls))
        lease = Lease(grant.lease_id, grant)
        self.leases.append(lease)
        if hold is None:
            self.busy[role] += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.on_grant is not None:
            self.on_grant(lease)
        outcome = "ok"
        try:
            yield lease
        except BaseException:
            outcome = "upstream_error"
            raise
        finally:
            if hold is None:
                self.busy[role] -= 1
            self.active -= 1
            self.releases.append(lease.release_outcome or outcome)


@pytest.fixture
def fake_pool(monkeypatch: pytest.MonkeyPatch) -> FakePool:
    from app import pool_placement

    pool = FakePool()
    monkeypatch.setattr(pool_placement, "gpu_lease", pool.gpu_lease)
    monkeypatch.setattr(pool_placement, "_bus", object())
    pool_placement.reset_pool_unreachable()
    return pool
