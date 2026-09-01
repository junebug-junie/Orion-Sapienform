"""The unattended graph-review tick: seed when empty, drain one due item, stay out of the self-relationship zone.

Regression cover for the 2026-09-01 finding that both halves of the review loop
were operator-endpoint-only, so ``substrate_review_queue_item`` had never held a
row and the downstream mutation scheduler starved for 4620 consecutive cycles.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
hub_scripts_pkg = HUB_ROOT / "scripts" / "__init__.py"
if (
    "scripts" not in sys.modules
    or not str(getattr(sys.modules.get("scripts"), "__file__", "")).startswith(str(HUB_ROOT))
):
    spec = importlib.util.spec_from_file_location(
        "scripts",
        str(hub_scripts_pkg),
        submodule_search_locations=[str(HUB_ROOT / "scripts")],
    )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["scripts"] = module
        spec.loader.exec_module(module)

from scripts import api_routes  # noqa: E402


class _FakeQueueStore:
    """Only the three surfaces the scheduled tick touches."""

    def __init__(self, *, total: int = 0, due: int = 0) -> None:
        self.total = total
        self.due = due
        self.refreshed = 0

    def refresh_from_storage(self) -> None:
        self.refreshed += 1

    def snapshot(self, *, limit: int = 200):
        class _Snap:
            queue_items = [object()] * self.total

        return _Snap()

    def list_eligible(self, *, now, limit: int = 200):
        return [object()] * self.due


@pytest.fixture
def wired(monkeypatch):
    """Install a fake queue store plus recording stubs for bootstrap/cycle."""
    calls: dict[str, list] = {"bootstrap": [], "cycle": []}
    store = _FakeQueueStore()

    def _fake_bootstrap(*, limit: int = 12):
        calls["bootstrap"].append({"limit": limit})
        store.total = 3
        store.due = 3
        return {"items_enqueued": 3, "notes": ["bootstrap_seeded"]}

    def _fake_cycle(*, allow_followup: bool, explicit_queue_item_id=None):
        calls["cycle"].append(
            {"allow_followup": allow_followup, "explicit_queue_item_id": explicit_queue_item_id}
        )
        return {
            "result": {"outcome": "executed", "selected_queue_item_id": "q-1"},
            "queue_after": {"count": store.total, "due_count": store.due},
        }

    monkeypatch.setattr(api_routes, "SUBSTRATE_REVIEW_QUEUE_STORE", store)
    monkeypatch.setattr(api_routes, "_bootstrap_substrate_review_frontier", _fake_bootstrap)
    monkeypatch.setattr(api_routes, "_execute_substrate_review_cycle", _fake_cycle)
    return store, calls


def test_empty_queue_bootstraps_then_executes(wired) -> None:
    store, calls = wired
    store.total, store.due = 0, 0

    payload = api_routes.execute_substrate_review_scheduled_cycle()

    assert len(calls["bootstrap"]) == 1
    assert payload["bootstrapped"] is True
    assert payload["items_enqueued"] == 3
    assert payload["status"] == "executed"
    assert payload["execution_outcome"] == "executed"
    assert payload["selected_queue_item_id"] == "q-1"


def test_non_empty_queue_never_reseeds(wired) -> None:
    """Reseeding around pending future-dated items would grow the queue every tick."""
    store, calls = wired
    store.total, store.due = 5, 0

    payload = api_routes.execute_substrate_review_scheduled_cycle()

    assert calls["bootstrap"] == []
    assert payload["bootstrapped"] is False
    assert payload["status"] == "idle_none_due"
    assert calls["cycle"] == []


def test_non_empty_queue_with_due_items_executes_without_seeding(wired) -> None:
    store, calls = wired
    store.total, store.due = 5, 2

    payload = api_routes.execute_substrate_review_scheduled_cycle()

    assert calls["bootstrap"] == []
    assert len(calls["cycle"]) == 1
    assert payload["status"] == "executed"
    assert payload["queue_before"] == 5
    assert payload["due_now"] == 2


def test_scheduled_cycle_never_allows_frontier_followup(wired) -> None:
    """The containment claim: follow-up is the only path to a self_relationship_graph item.

    ``review_runtime._select_item`` gates that zone on ``invocation_surface``
    alone and honours no override on the non-explicit path, so an unattended
    cycle running under ``operator_review`` would be free to consolidate Orion's
    self-relationship model if follow-up were ever enabled here.
    """
    store, calls = wired
    store.total, store.due = 1, 1

    api_routes.execute_substrate_review_scheduled_cycle()

    assert len(calls["cycle"]) == 1
    assert calls["cycle"][0]["allow_followup"] is False
    assert calls["cycle"][0]["explicit_queue_item_id"] is None


def test_seeded_but_nothing_due_reports_distinctly_and_skips_cycle(wired, monkeypatch) -> None:
    """A bootstrap that schedules everything into the future must not be reported as executed."""
    store, calls = wired
    store.total, store.due = 0, 0

    def _seed_future_only(*, limit: int = 12):
        calls["bootstrap"].append({"limit": limit})
        store.total = 3
        store.due = 0  # every seeded item dated forward
        return {"items_enqueued": 3, "notes": ["bootstrap_seeded"]}

    monkeypatch.setattr(api_routes, "_bootstrap_substrate_review_frontier", _seed_future_only)

    payload = api_routes.execute_substrate_review_scheduled_cycle()

    assert payload["status"] == "seeded_none_due"
    assert payload["bootstrapped"] is True
    assert calls["cycle"] == []
    assert payload["execution_outcome"] is None


def test_bootstrap_limit_is_forwarded(wired) -> None:
    store, calls = wired
    store.total, store.due = 0, 0

    api_routes.execute_substrate_review_scheduled_cycle(bootstrap_limit=4)

    assert calls["bootstrap"][0]["limit"] == 4


def test_tick_refreshes_queue_from_storage_before_deciding(wired) -> None:
    """Hub holds an in-process queue store; a stale view would reseed a queue that has rows."""
    store, _calls = wired
    store.total, store.due = 0, 0

    api_routes.execute_substrate_review_scheduled_cycle()

    assert store.refreshed >= 1


def test_payload_shape_is_loggable_and_carries_tick_id(wired) -> None:
    import json

    store, _calls = wired
    store.total, store.due = 0, 0
    now = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)

    payload = api_routes.execute_substrate_review_scheduled_cycle(now=now)

    assert payload["event"] == "review_scheduler_tick"
    assert payload["tick_id"].startswith("review-scheduler-")
    assert payload["at"] == now.isoformat()
    json.dumps(payload, sort_keys=True, default=str)
