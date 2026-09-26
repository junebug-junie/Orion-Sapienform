"""The REAL GPU pool runtime, in process, for durable-runs tests (stage 4.5).

``services/orion-gpu-pool/app`` is loaded by file path under its own package name (durable-runs
owns ``app``): the real scheduler, lease graph (MemorySaver), in-memory store, hold placement,
``attach``, ``status``, recall and expiry. Only the llama.cpp servers are fixtures (a prober that
answers the way circe's do today) and the clock is fake.

Two ways to reach it:

* ``InProcessPool.install(bus)`` answers ``orion:gpu_pool:lease:request`` on a TypedBus through the
  real client (orion.gpu_pool.client) and the real codec, and publishes its lifecycle events on
  that bus (``orion:gpu_pool:event``) -- the acceptance path.
* ``PoolBus(pool)`` is a minimal bus with just ``rpc_request`` + ``codec`` for AdmissionRuntime
  unit tests that do not need a whole TypedBus.
"""
from __future__ import annotations

import contextlib
import importlib
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

from langgraph.checkpoint.memory import MemorySaver

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.gpu_pool.client import Lease, LeaseUnavailable
from orion.gpu_pool.config import load_pool_config
from orion.gpu_pool.discovery import Probe, load_profiles
from orion.gpu_pool.lease_graph import build_lease_graph
from orion.schemas.gpu_pool import (
    GPU_LEASE_REPLY_KIND, GPU_POOL_LEASE_REQUEST_CHANNEL, GpuLeaseReplyV1, GpuLeaseRequestV1, LlmWorkerAnnounceV1,
)

ROOT = Path(__file__).resolve().parents[3]
_PKG = "durable_tests_gpu_pool_app"
if _PKG not in sys.modules:
    package = ModuleType(_PKG)
    package.__path__ = [str(ROOT / "services/orion-gpu-pool/app")]
    sys.modules[_PKG] = package
PoolRuntime = importlib.import_module(f"{_PKG}.runtime").PoolRuntime
MemoryStore = importlib.import_module(f"{_PKG}.store").MemoryStore

CFG = load_pool_config()
PROFILES = load_profiles()
SOURCE = ServiceRef(name="durable-tests-gpu-pool")
# What circe's llama.cpp servers report today (same table as services/orion-gpu-pool/tests).
LIVE = {
    "chat": ("qwen36-35b-a3b-udq5km-2xv100-32gb-deep-cognition", "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf", 1, 65536),
    "agent": ("qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex", "Qwen3.8-27B-UD-Q4_K_XL.gguf", 1, 131072),
    "metacog": ("qwen3-8b-q5km-v100-16gb-atlas-metacog-16k", "Qwen_Qwen3-8B-Q5_K_M.gguf", 4, 4096),
    "fast": ("qwen3-8b-q4km-v100-16gb-balanced", "Qwen_Qwen3-8B-Q4_K_M.gguf", 4, 4096),
}


class Clock:
    def __init__(self):
        self.t = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t

    def advance(self, sec):
        self.t += timedelta(seconds=sec)


class _PoolSideBus:
    """What the pool runtime publishes on. Forwards to the test bus when one is installed."""

    def __init__(self, pool):
        self.pool = pool

    async def publish(self, channel, env):
        self.pool.published.append((channel, env))
        if self.pool.bus is not None:
            await self.pool.bus.publish(channel, env)

    def record_hop_success(self, hop, ms):
        pass

    def record_hop_timeout(self, hop, ms=None):
        pass


class InProcessPool:
    def __init__(self, *, clock: Clock | None = None, down=(), live=None, actuate=()):
        self.clock = clock or Clock()
        self.down = set(down)
        self.live = dict(live or LIVE)
        self.bus = None
        self.published: list = []
        self.requests: list[GpuLeaseRequestV1] = []

        async def prober(role, url, kind, health):
            if role in self.down or (kind != "service" and role not in self.live):
                return Probe(False, error="refused", checked_at=self.clock())
            if kind == "service":
                return Probe(True, checked_at=self.clock())
            _, file, slots, ctx = self.live[role]
            return Probe(True, {"model_path": f"/models/gguf/{file}", "total_slots": slots,
                                "default_generation_settings": {"n_ctx": ctx}, "modalities": {"vision": False}},
                         checked_at=self.clock())

        self.rt = PoolRuntime(cfg=CFG, profiles=PROFILES, store=MemoryStore(),
                              graph=build_lease_graph(lambda: CFG, MemorySaver()), bus=_PoolSideBus(self),
                              prober=prober, now=self.clock, probe_interval_sec=0, actuate_roles=actuate)
        if actuate:
            # Guards read clear (cabinet cool, visual baseline not overdue): the pool may load.
            self.rt.guard_states = {name: None for name in self.rt.guard_states}

    async def announce(self):
        for role, (profile, _, _, _) in list(self.live.items()):
            if role in self.down:
                continue
            await self.rt.on_announce(LlmWorkerAnnounceV1(host="circe", role=role, profile_name=profile,
                                                          port=CFG.roles[role].port, announced_at=self.rt.now()))

    async def boot(self):
        await self.rt.start()
        await self.announce()
        await self.rt.tick()
        return self

    async def later(self, sec: float, *, beat=(), every: float = 30.0):
        """Advance time like production: workers re-announce, holders in ``beat`` heartbeat inside
        their TTL, the pool ticks."""
        left = sec
        while left > 0:
            dt = min(every, left)
            self.clock.advance(dt)
            left -= dt
            for lease_id in beat:
                await self.rt.heartbeat(lease_id)
            await self.announce()
            await self.rt.tick()

    async def dispatch(self, req: GpuLeaseRequestV1) -> GpuLeaseReplyV1:
        # Mirrors services/orion-gpu-pool/app/main.py dispatch_lease (its main imports the bus chassis).
        if not self.rt._started:
            await self.boot()
        self.requests.append(req)
        rt = self.rt
        if req.verb == "acquire":
            return await rt.acquire(req)
        if req.verb == "attach":
            return await rt.attach(req)
        if not req.lease_id:
            return GpuLeaseReplyV1(status="unknown_lease", reason="lease_id required")
        verbs = {"status": rt.status, "heartbeat": rt.heartbeat, "cancel": rt.cancel}
        if req.verb in verbs:
            return await verbs[req.verb](req.lease_id)
        if req.verb == "release":
            return await rt.release(req.lease_id, req.outcome or "ok", req.detail)
        return GpuLeaseReplyV1(status="unavailable", lease_id=req.lease_id, reason=f"unknown_verb:{req.verb}")

    def install(self, bus) -> None:
        """Serve lease RPCs on ``bus`` (a TypedBus) and publish pool events on it."""
        self.bus = bus

        async def handle(env):
            reply = await self.dispatch(GpuLeaseRequestV1.model_validate(env.payload))
            if env.reply_to:
                await bus.publish(env.reply_to, BaseEnvelope(kind=GPU_LEASE_REPLY_KIND, source=SOURCE,
                    correlation_id=env.correlation_id, payload=reply.model_dump(mode="json")))

        bus.handlers[GPU_POOL_LEASE_REQUEST_CHANNEL] = handle

    async def lease(self, lease_id: str) -> dict | None:
        return await self.rt.store.lease(lease_id)

    def leases(self, **match) -> list[dict]:
        rows = list(self.rt.store.leases.values())
        return [r for r in rows if all(r.get(k) == v for k, v in match.items())]

    def events(self, name: str | None = None) -> list[dict]:
        out = [env.payload for channel, env in self.published if channel == "orion:gpu_pool:event"]
        return [e for e in out if name is None or e.get("event") == name]

    def verbs(self) -> list[str]:
        return [r.verb for r in self.requests]

    def gateway_gpu_lease(self, grants: list, url_for_role=None):
        """A stand-in for ``orion.gpu_pool.client.gpu_lease`` inside the gateway, backed by THIS pool:
        a call carrying a hold ref is an ``attach``; any other call is a plain ``acquire``. A call that
        the pool does not grant at once raises -- in these tests that can only mean it queued behind
        the run's own hold, which is exactly the self-deadlock stage 4 must never produce."""
        pool = self

        @contextlib.asynccontextmanager
        async def fixture_gpu_lease(_bus, **kw):
            hold = kw.get("hold")
            common = dict(request_id=kw.get("request_id") or uuid.uuid4().hex, holder=kw["holder"],
                          work_class=kw["work_class"], priority=kw.get("priority", "system"),
                          turn_correlation_id=kw.get("turn_correlation_id"),
                          deadline_at=pool.clock() + timedelta(seconds=float(kw.get("deadline_sec", 60))))
            if hold is not None:
                req = GpuLeaseRequestV1(verb="attach", kind="request", hold_lease_id=hold.lease_id,
                                        hold_generation=hold.generation, **common)
            else:
                req = GpuLeaseRequestV1(verb="acquire", kind=kw.get("kind", "request"), **common)
            reply = await pool.dispatch(req)
            grants.append({**kw, "pool_verb": req.verb, "pool_status": reply.status, "lease_id": reply.lease_id})
            if reply.status != "granted" or reply.grant is None:
                if reply.lease_id:
                    await pool.dispatch(GpuLeaseRequestV1(verb="cancel", lease_id=reply.lease_id))
                raise LeaseUnavailable(f"fixture_pool_{reply.status}:{reply.reason}", reply.lease_id)
            grant = reply.grant
            if url_for_role is not None:
                grant = grant.model_copy(update={"url": url_for_role(grant.role)})
            try:
                yield Lease(reply.lease_id, grant)
            finally:
                await pool.dispatch(GpuLeaseRequestV1(verb="release", lease_id=reply.lease_id, outcome="ok"))

        return fixture_gpu_lease


class PoolBus:
    """Just enough bus for PoolHolds (rpc_request + codec), straight to an InProcessPool."""

    def __init__(self, pool: InProcessPool):
        self.pool, self.codec = pool, OrionCodec()
        self.fail_next: int = 0   # simulate an unreachable pool for the next N RPCs

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec, health_label=None):
        assert channel == GPU_POOL_LEASE_REQUEST_CHANNEL and envelope.reply_to == reply_channel
        if self.fail_next:
            self.fail_next -= 1
            raise TimeoutError("fixture: pool unreachable")
        wire = self.codec.decode(self.codec.encode(envelope)).envelope
        reply = await self.pool.dispatch(GpuLeaseRequestV1.model_validate(wire.payload))
        env = BaseEnvelope(kind=GPU_LEASE_REPLY_KIND, source=SOURCE, correlation_id=envelope.correlation_id,
                           payload=reply.model_dump(mode="json"))
        return {"type": "message", "channel": reply_channel, "data": self.codec.encode(env)}
