"""The lane door, the eval helper and the gateway agree (leased-evals Phase 2):

* what the door lets through without a lease id (no-model reads) is exactly what an eval may read from the
  granted URL (orion/evals/leased_llm.py WORKER_READ_*), so a leased probe never trips the door;
* the lease header name matches on both sides, and no shared-secret header is left on either."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _evict_app() -> dict:
    gone = {k: sys.modules.pop(k) for k in [k for k in sys.modules if k == "app" or k.startswith("app.")]}
    return gone


def _door():
    """Import the door's ``app`` package without colliding with another service's ``app`` loaded in the same
    pytest session (the gateway's): set that one aside, import, then put it back exactly as it was."""
    saved = _evict_app()
    sys.path.insert(0, str(REPO / "services" / "orion-lane-door"))
    try:
        import app.door as door  # noqa: PLC0415
        import app.settings as door_settings  # noqa: PLC0415
        return door, door_settings
    finally:
        sys.path.pop(0)
        _evict_app()
        sys.modules.update(saved)


def test_the_door_exempts_exactly_the_reads_an_eval_may_make():
    from orion.evals import leased_llm

    door, _ = _door()
    assert door.EXEMPT_GET_PATHS == leased_llm.WORKER_READ_GET_PATHS
    assert door.EXEMPT_POST_PATHS == leased_llm.WORKER_READ_POST_PATHS
    assert not (door.EXEMPT_GET_PATHS | door.EXEMPT_POST_PATHS) & leased_llm.INFERENCE_PATHS


def test_door_and_gateway_name_the_same_lease_header_and_no_key():
    door, door_settings = _door()
    src = (REPO / "services" / "orion-llm-gateway" / "app" / "lane_door.py").read_text()
    assert 'LEASE_HEADER = "X-Orion-Lease-Id"' in src and door.LEASE_HEADER == b"x-orion-lease-id"
    # Kill means kill: the shared secret is gone from both sides.
    assert "Lane-Key" not in src and not hasattr(door, "KEY_HEADER")
    assert not hasattr(door_settings, "key_hash")


def test_every_path_the_pool_and_the_lane_controller_read_is_exempt():
    """Review finding: the pool's vLLM probe reads /orion/server-info; unexempt, agent-deep could never read
    READY in shadow and would go down in enforce. Pin every probe path to the door's exempt list."""
    import re

    import yaml

    door, _ = _door()
    src = (REPO / "services" / "orion-gpu-pool" / "app" / "main.py").read_text()
    probe = src[src.index("async def _probe"):]
    probe = probe[:probe.index("\nasync def ", 10) if "\nasync def " in probe[10:] else len(probe)]
    paths = set(re.findall(r'client\.get\(f"\{url\}(/[A-Za-z0-9_./-]+)"', probe))
    assert {"/health", "/v1/models", "/orion/server-info", "/props"} <= paths, paths
    cfg = yaml.safe_load((REPO / "config" / "gpu_pool.yaml").read_text())
    for role in (cfg.get("roles") or {}).values():
        if isinstance(role, dict) and role.get("kind") == "llm":
            ready = (role.get("launch") or {}).get("ready")
            if ready:
                paths.add(ready)
    assert paths <= door.EXEMPT_GET_PATHS, sorted(paths - door.EXEMPT_GET_PATHS)
