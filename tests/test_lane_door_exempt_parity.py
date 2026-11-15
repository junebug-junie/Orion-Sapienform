"""The lane door, the eval helper and the gateway agree (leased-evals Phase 2):

* what the door lets through without a key (no-model reads) is exactly what an eval may read from the
  granted URL (orion/evals/leased_llm.py WORKER_READ_*), so a leased probe never trips the door;
* the door and the gateway hash the key the same way, so the gateway's key check means something;
* the header names match on both sides."""
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


def test_door_and_gateway_hash_the_key_the_same_way_and_name_the_same_headers():
    door, door_settings = _door()
    src = (REPO / "services" / "orion-llm-gateway" / "app" / "lane_door.py").read_text()
    # The gateway module imports its pydantic settings; compare its hash function by executing just it.
    ns: dict = {}
    exec("import hashlib\n" + src[src.index("def key_hash"):src.index("def _key")], ns)  # noqa: S102
    for key in ("", "a", "k" * 64):
        assert ns["key_hash"](key) == door_settings.key_hash(key)
    assert 'KEY_HEADER = "X-Orion-Lane-Key"' in src and door.KEY_HEADER == b"x-orion-lane-key"
    assert 'LEASE_HEADER = "X-Orion-Lease-Id"' in src and door.LEASE_HEADER == b"x-orion-lease-id"
