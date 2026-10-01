"""Stage 7.1 bake-off (scripts/bench/stage7_1_bakeoff.sh): the exact control sequence it sends, through
the real runtime. Envelopes come from scripts/bench/stage7_1_pool_ctl.py's own build(), validated the way
app.main._on_control does.

Pinned:
- the order is hold -> pause: the pool refuses an operator hold while paused (so pause -> hold can never work);
- while the diffusion hold is granted, other diffusion work and world work queue (nothing loads ~24 GB next
  to the bake-off worker);
- the pause is recorded as by "stage7-1-bakeoff" (cleanup resumes only a pause with that name);
- release + resume hand gpu2 back: the queued diffusion lease is granted.
Spec: docs/superpowers/specs/2026-09-30-gpu-pool-stage7-concurrency.md (PR row 7.1)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from orion.schemas.gpu_pool import GpuPoolControlV1
from tests.test_runtime import acq, run
from tests.test_stage5_7_enforce import boot, enforce

_PATH = Path(__file__).resolve().parents[3] / "scripts" / "bench" / "stage7_1_pool_ctl.py"
_spec = importlib.util.spec_from_file_location("stage7_1_pool_ctl", _PATH)
ctl = importlib.util.module_from_spec(_spec)
sys.modules["stage7_1_pool_ctl"] = ctl
_spec.loader.exec_module(ctl)


async def send(rt, action, **kw):
    _, env = ctl.build(action, **kw)
    return await rt.control(GpuPoolControlV1.model_validate(env.payload))


def test_bakeoff_sequence_hold_then_pause_then_release_then_resume():
    async def go():
        rt, _ = enforce()
        await boot(rt)
        rt._reconciling.clear()   # the boot reconcile is not this test's subject

        held = await send(rt, "hold")
        assert held.ok and held.detail["status"] == "granted", held
        lease = held.detail["lease_id"]

        other = await rt.acquire(acq("diffusion"))
        world = await rt.acquire(acq("world"))
        assert other.status == "queued"                      # diffusion cannot load next to the worker
        assert world.status == "queued"                      # serialize_with: world waits too

        paused = await send(rt, "pause")
        assert paused.ok and paused.reason == "paused"
        assert rt.paused["by"] == ctl.ACTOR                  # what cleanup checks before resuming

        # The order matters: an operator hold is refused while paused.
        late = await send(rt, "hold")
        assert not late.ok and late.reason == "actuation_paused"

        rel = await send(rt, "release", lease_id=lease)
        assert rel.ok
        res = await send(rt, "resume")
        assert res.ok and rt.paused is None
        await rt.tick()
        assert (await rt.store.lease(other.lease_id))["status"] == "granted"
    run(go())
