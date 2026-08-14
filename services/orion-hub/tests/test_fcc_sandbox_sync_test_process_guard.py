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

    asyncio.run(websocket_handler._sync_fcc_sandbox_background("conn-under-test"))

    assert called == [], "sync_fcc_sandbox must not run against a live workspace in tests"
    assert websocket_handler.last_sync_state()["result"] == "skipped_test_process"


def test_sync_runs_when_not_a_test_process(monkeypatch):
    """The guard must gate on the test marker only -- not disable the sync itself,
    which would turn the whole feature into a no-op in production."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(websocket_handler, "active_turns", lambda: [])
    called: list[str] = []
    monkeypatch.setattr(
        websocket_handler,
        "sync_fcc_sandbox",
        lambda workspace: called.append(workspace) or "synced",
    )

    asyncio.run(websocket_handler._sync_fcc_sandbox_background("conn-real"))

    assert len(called) == 1


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
