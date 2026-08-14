"""The connect-time FCC sandbox sync must never fire from a test process.

Regression for a live incident on 2026-08-14. The hub suite drives the real
``websocket_endpoint()`` (test_workflow_schedule_runtime_paths.py), which reaches
``_sync_fcc_sandbox_background`` with the real ``HUB_AGENT_CLAUDE_WORKSPACE``
loaded from ``.env`` -- ``/mnt/orion-fcc/repo``, Orion's actual checkout.

That was survivable while the sync only read ``git status`` and declined on a
dirty tree. Once it gained the rescue-and-reset path, a plain ``pytest`` run
stashed Orion's working tree and moved its HEAD: confirmed by
``git log -g refs/stash`` showing an entry authored by the test runner
(``Athena Orchestration``) at 09:37:03Z, mid-suite.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, HUB_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# Same channel defaults the other websocket_handler-importing test declares --
# settings validation runs at import time.
os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

from scripts import websocket_handler  # noqa: E402


def test_sync_is_skipped_under_pytest(monkeypatch):
    """The guard keys off PYTEST_CURRENT_TEST, which pytest sets for us."""
    called: list[str] = []
    monkeypatch.setattr(
        websocket_handler,
        "sync_fcc_sandbox",
        lambda workspace: called.append(workspace) or "synced",
    )

    from orion.fcc.sandbox_sync import last_sync_state

    asyncio.run(websocket_handler._sync_fcc_sandbox_background("conn-under-test"))

    assert called == [], "sync_fcc_sandbox must not run against a live workspace in tests"
    assert last_sync_state()["last_attempt"]["result"] == "skipped_test_process"


def test_sync_runs_when_not_a_test_process(monkeypatch):
    """The guard must gate on the test markers only -- not disable the sync itself,
    which would turn the whole feature into a no-op in production.

    Note this test must neutralise BOTH markers, and monkeypatch sync_fcc_sandbox, to
    simulate production. That is deliberate friction: the earlier single-marker guard
    could be switched off by deleting one env var, and a future test copying that
    pattern without also stubbing the sync would reproduce the original incident.
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setitem(websocket_handler.sys.modules, "pytest", None)
    monkeypatch.delitem(websocket_handler.sys.modules, "pytest")
    monkeypatch.setattr(websocket_handler, "active_turns", lambda: [])
    called: list[str] = []
    monkeypatch.setattr(
        websocket_handler,
        "sync_fcc_sandbox",
        lambda workspace: called.append(workspace) or "synced",
    )

    asyncio.run(websocket_handler._sync_fcc_sandbox_background("conn-real"))

    assert len(called) == 1


def test_guard_holds_when_only_the_env_marker_is_cleared(monkeypatch):
    """The env marker is set per test *item*, so a task outliving the item sees it
    unset -- which is exactly what this fire-and-forget task is. `sys.modules` is the
    marker that survives, and it alone must be enough to hold the guard."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(websocket_handler, "active_turns", lambda: [])
    called: list[str] = []
    monkeypatch.setattr(
        websocket_handler,
        "sync_fcc_sandbox",
        lambda workspace: called.append(workspace) or "synced",
    )

    asyncio.run(websocket_handler._sync_fcc_sandbox_background("conn-late-thread"))

    assert called == [], "sys.modules must hold the guard when the env marker is gone"


def test_status_endpoint_reports_the_configured_workspace_on_a_cold_hub():
    """Regression: the route spread sync state over a leading "workspace" key, so the
    state's own None overrode the configured path and a freshly booted hub reported
    workspace=null -- exactly when an operator checks it and exactly when the sandbox
    is most likely stale."""
    import orion.fcc.sandbox_sync as sandbox_sync
    from scripts.api_routes import api_fcc_sandbox_sync

    # Cold-boot state: nothing has synced yet.
    sandbox_sync._LAST_ATTEMPT = {"result": None, "at": None, "workspace": None}
    sandbox_sync._LAST_SUCCESS = None

    payload = api_fcc_sandbox_sync()

    assert payload["configured_workspace"], "operator must still learn which path is watched"
    assert payload["last_attempt"]["result"] is None
    assert payload["last_success"] == {}


def test_status_endpoint_reports_a_recorded_verdict():
    from orion.fcc.sandbox_sync import record_sync_skip
    from scripts.api_routes import api_fcc_sandbox_sync

    record_sync_skip("/mnt/orion-fcc/repo", "skipped_turn_in_flight_external")

    payload = api_fcc_sandbox_sync()
    assert payload["last_attempt"]["result"] == "skipped_turn_in_flight_external"
    assert payload["last_attempt"]["workspace"] == "/mnt/orion-fcc/repo"


@pytest.mark.parametrize(
    "test_file",
    ["tests/test_workflow_schedule_runtime_paths.py"],
)
def test_known_websocket_endpoint_driver_still_exists(test_file):
    """If this file is renamed or stops driving websocket_endpoint, the guard is
    still correct -- but the comment pointing at the incident goes stale, so fail
    loudly rather than let the reasoning rot."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / test_file
    assert path.is_file(), f"{test_file} moved; update the guard's incident note"
    assert "websocket_endpoint" in path.read_text(encoding="utf-8")
