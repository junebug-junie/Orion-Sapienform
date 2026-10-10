#!/usr/bin/env python3
"""Ask a GPU pool host actuator two read-only questions over the bus, and print its answers.

Used by docs/runbooks/2026-09-29-gpu-pool-stage5-3-cutover.md before and after each deploy step:

- ``status``: is the controller up, can it parse its own checkout's config/gpu_pool.yaml
  (``config_unloadable:*`` = image not rebuilt after a pull), what does it observe on the card,
  and is anything in flight. A pure read.
- ``digest``: does the controller's checkout agree with THIS checkout's YAML? Sends a ``load`` naming
  a profile that is never in any allow-list. The controller checks the launch digest before the
  profile, and both before the generation fence or any docker call, so the answer is always a
  refusal: ``profile_not_allowed`` = digests agree; ``launch_digest_mismatch`` = the two hosts are on
  different commits (or the controller image predates the YAML). No generation is spent and no
  container is touched (controller tests: test_launch_exec.py
  ``test_profile_outside_allow_list_refused_before_any_docker_call``).

The pool sees these results too and logs them as stale (not its action_ids); its card state does
not change. Run from a checkout at the commit the POOL runs (or pass --config with the pool's YAML):

    ORION_BUS_URL=redis://100.92.216.81:6379/0 PYTHONPATH=. .venv/bin/python \
        scripts/gpu_pool_actuator_probe.py [--role agent-gpu2] [--check status digest]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.gpu_pool import actuator_probe  # noqa: E402
from orion.gpu_pool.actuator_probe import PROBE_PROFILE  # noqa: E402,F401  (re-exported for callers)
from orion.gpu_pool.config import PoolConfig, load_pool_config  # noqa: E402
from orion.schemas.gpu_pool import GpuActuateV1  # noqa: E402

# The probe core lives in orion/gpu_pool/actuator_probe.py (shared with orion-mesh-guardian's GPU
# watch); this file is the operator CLI over it.
SOURCE = actuator_probe.DEFAULT_SOURCE


def build(cfg: PoolConfig, role: str, check: str, *, now=None) -> GpuActuateV1:
    """The request for one check (see actuator_probe.build_request)."""
    try:
        return actuator_probe.build_request(cfg, role, check, now=now)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


def verdict(check: str, results: list[dict]) -> str:
    """One line a human can act on, from the results carrying this probe's action_id."""
    return actuator_probe.classify(check, results).line


async def probe(role: str, checks: list[str], wait_sec: float, config: str | None = None) -> int:
    from orion.core.bus.async_service import OrionBusAsync

    cfg = load_pool_config(config) if config else load_pool_config()
    build(cfg, role, checks[0])  # a role with no launch block exits before touching the bus
    bus = OrionBusAsync(url=os.environ["ORION_BUS_URL"])
    await bus.connect()
    bad = 0
    try:
        for check in checks:
            got = await actuator_probe.probe(bus, cfg, role, check, wait_sec, SOURCE)
            bad += not got.ok
            print(f"{check:7s} {role} digest={got.launch_digest[:16]} -> {got.line}")
    finally:
        await bus.close()
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--role", default="agent-gpu2")
    ap.add_argument("--check", nargs="+", choices=("status", "digest"), default=["status", "digest"])
    ap.add_argument("--wait-sec", type=float, default=90.0,
                    help="per check; status waits on `docker compose ps` (up to ~60 s)")
    ap.add_argument("--config", default=None,
                    help="the pool's config/gpu_pool.yaml when it differs from this checkout's "
                         "(e.g. `git show <pool commit>:config/gpu_pool.yaml > /tmp/pool.yaml`)")
    args = ap.parse_args()
    return asyncio.run(probe(args.role, args.check, args.wait_sec, args.config))


if __name__ == "__main__":
    sys.exit(main())
