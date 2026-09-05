"""Tests for chat_stance.py's _publish_chat_stance_belief() (self-model
rebuild arc, 2026-09-05) -- the real wiring point a code review flagged as
untested: it's the one thing that calls publish_chat_stance_belief_log_sync
from inside the actual chat turn, off asyncio.to_thread rather than inline,
with a real UnifiedRelationalBeliefSetV1-shaped object."""
import asyncio
import importlib.util
import sys
import types
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = SERVICE_DIR / "app"
PACKAGE_NAME = "orion_cortex_exec"
APP_PACKAGE_NAME = f"{PACKAGE_NAME}.app"
if PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(SERVICE_DIR)]
    sys.modules[PACKAGE_NAME] = pkg
if APP_PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(APP_PACKAGE_NAME)
    pkg.__path__ = [str(APP_DIR)]
    sys.modules[APP_PACKAGE_NAME] = pkg

REPO_ROOT = SERVICE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_chat_stance_key = f"{APP_PACKAGE_NAME}.chat_stance"
if _chat_stance_key in sys.modules:
    chat_stance = sys.modules[_chat_stance_key]
else:
    spec = importlib.util.spec_from_file_location(_chat_stance_key, APP_DIR / "chat_stance.py")
    chat_stance = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = chat_stance
    spec.loader.exec_module(chat_stance)


class _FakeBeliefs:
    def __init__(self, *, anchors=None, degraded_producers=None, lineage=None):
        self.anchors = anchors or {}
        self.degraded_producers = degraded_producers or []
        self.lineage = lineage or {}


def test_publish_chat_stance_belief_noop_when_beliefs_none():
    calls = []
    orig = chat_stance.publish_chat_stance_belief_log_sync

    def spy(**kwargs):
        calls.append(kwargs)

    chat_stance.publish_chat_stance_belief_log_sync = spy
    try:
        asyncio.run(chat_stance._publish_chat_stance_belief({}, None))
    finally:
        chat_stance.publish_chat_stance_belief_log_sync = orig

    assert calls == []


def test_publish_chat_stance_belief_calls_publish_with_real_beliefs(monkeypatch):
    calls = []

    def spy(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(chat_stance, "publish_chat_stance_belief_log_sync", spy)

    beliefs = _FakeBeliefs(
        anchors={"orion": object()},
        degraded_producers=["producer_x"],
        lineage={"orion": "producer_x"},
    )
    ctx = {"correlation_id": "corr-1", "session_id": "sess-1", "turn_change_appraisal": {"shift_kind": "repair"}}

    asyncio.run(chat_stance._publish_chat_stance_belief(ctx, beliefs))

    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["anchors"] == beliefs.anchors
    assert kwargs["degraded_producers"] == beliefs.degraded_producers
    assert kwargs["lineage"] == beliefs.lineage
    # Passed through as-read (lowercase); normalization is
    # publish_chat_stance_belief_log_sync's own responsibility -- this test
    # just confirms the real appraisal value reaches the call at all.
    assert kwargs["shift_kind"] == "repair"
    assert kwargs["ctx"] is ctx


def test_publish_chat_stance_belief_never_raises_when_publish_fails(monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("simulated redis failure")

    monkeypatch.setattr(chat_stance, "publish_chat_stance_belief_log_sync", broken)

    beliefs = _FakeBeliefs()
    # Must not raise.
    asyncio.run(chat_stance._publish_chat_stance_belief({}, beliefs))


def test_publish_chat_stance_belief_offloads_to_a_thread(monkeypatch):
    """The whole point of the review fix: this must not call
    publish_chat_stance_belief_log_sync directly on the event loop."""
    seen_thread_ids = []
    main_thread_id = None

    def spy(**kwargs):
        import threading

        seen_thread_ids.append(threading.get_ident())

    monkeypatch.setattr(chat_stance, "publish_chat_stance_belief_log_sync", spy)

    async def run():
        import threading

        nonlocal main_thread_id
        main_thread_id = threading.get_ident()
        await chat_stance._publish_chat_stance_belief({}, _FakeBeliefs())

    asyncio.run(run())

    assert len(seen_thread_ids) == 1
    assert seen_thread_ids[0] != main_thread_id
