"""Ensure orion-thought ``app`` resolves to this service during tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_THOUGHT_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _THOUGHT_ROOT.parents[1]


def _ensure_thought_paths() -> None:
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    for p in (str(_REPO_ROOT), str(_THOUGHT_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_THOUGHT_ROOT))


def pytest_configure() -> None:
    # Runs before collection, so module-top ``import orion...``/``import app...``
    # in test files resolve. The autouse fixture below only fires at test
    # execution, which is too late for collection-time imports.
    _ensure_thought_paths()


@pytest.fixture(autouse=True)
def _thought_service_isolation() -> None:
    _ensure_thought_paths()
    yield


@pytest.fixture(autouse=True)
def _gpu2_capacity_off_by_default(monkeypatch, request):
    """visual_chain_gpu2_capacity_enabled defaults to True in production
    (PR that added the GPU2 capacity mutex), but most tests in this suite
    call run_visual_chain_once/call_diffusion_generate without mocking
    GpuCapacityPermit. Left at its production default, those tests would
    each attempt a real acquire() against durable-runs -- unreachable in a
    bare pytest process -- and the client's own retry loop treats that as
    transient and polls for the full budget_sec (180s default) before
    giving up, hanging the whole suite for minutes per test (confirmed
    live: exactly this hang, reproduced running the full suite).
    test_visual_chain_gpu2_capacity.py explicitly re-enables this per test.

    Confirmed live this needs THREE separate patch points, not one:
    `_thought_service_isolation` deletes every `app.*` module from
    sys.modules every test, so a test file that does `from app import
    visual_chain` once at its own MODULE level (e.g.
    test_visual_chain_thermal_gate.py, imported at collection time) keeps a
    reference to that ORIGINAL object forever -- a later fresh `from app
    import visual_chain` inside this fixture creates a genuinely different
    object with its own Settings() instance, and patching only that one
    silently does nothing for the module-level-import file. Patching only
    the env var doesn't help either: that file's `settings` singleton was
    already constructed once at collection time, before any per-test
    fixture ever ran. So: env var (covers any future fresh construction),
    the currently-cached sys.modules entry (covers function-local re-import
    files), AND the specific object already bound on the requesting test's
    own module, if it imported one at module level (covers the stale-
    reference case) -- whichever of these actually apply for a given test,
    all three are cheap and harmless to set regardless."""
    monkeypatch.setenv("ORION_VISUAL_CHAIN_GPU2_CAPACITY_ENABLED", "false")
    seen: set[int] = set()

    def _disable(visual_chain_module) -> None:
        settings_obj = getattr(visual_chain_module, "settings", None)
        if settings_obj is not None and id(settings_obj) not in seen:
            seen.add(id(settings_obj))
            monkeypatch.setattr(settings_obj, "visual_chain_gpu2_capacity_enabled", False)

    cached = sys.modules.get("app.visual_chain")
    if cached is not None:
        _disable(cached)
    module_level = getattr(request.module, "visual_chain", None)
    if module_level is not None:
        _disable(module_level)
