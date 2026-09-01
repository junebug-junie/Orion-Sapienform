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
    """Only the surfaces the scheduled tick touches.

    ``total`` and ``usable`` are modelled separately on purpose: their divergence
    IS the failure mode (a queue full of suppressed items is non-empty and
    permanently un-drainable). A fake that conflated them would hide it. The
    real coupling is covered against the real GraphReviewQueue in
    test_substrate_review_queue_pruning.py.
    """

    def __init__(self, *, total: int = 0, usable: int = 0, due: int = 0) -> None:
        self.total = total
        self.usable = usable
        self.due = due
        self.refreshed = 0
        self.pruned_with: list[float] = []
        self.calls: list[str] = []
        self._kind = "postgres"
        self._degraded = False

    def refresh_from_storage(self) -> None:
        self.refreshed += 1
        self.calls.append("refresh")

    def source_kind(self) -> str:
        return self._kind

    def degraded(self) -> bool:
        return self._degraded

    def last_error(self) -> str | None:
        return None

    def prune_finished(self, *, older_than_sec: float, now=None) -> int:
        self.pruned_with.append(older_than_sec)
        self.calls.append("prune")
        return 0

    def usable_items(self, *, limit: int = 200):
        self.calls.append("usable")
        return [object()] * self.usable

    def snapshot(self, *, limit: int = 200):
        self.calls.append("snapshot")

        class _Snap:
            queue_items = [object()] * self.total

        return _Snap()

    def list_eligible(self, *, now, limit: int = 200):
        self.calls.append("eligible")
        return [object()] * self.due


@pytest.fixture
def wired(monkeypatch):
    """Install a fake queue store plus recording stubs for bootstrap/cycle."""
    calls: dict[str, list] = {"bootstrap": [], "cycle": []}
    store = _FakeQueueStore()

    def _fake_bootstrap(*, limit: int = 12):
        calls["bootstrap"].append({"limit": limit})
        store.total = 3
        store.usable = 3
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
    store.total, store.usable, store.due = 0, 0, 0

    payload = api_routes.execute_substrate_review_scheduled_cycle()

    assert len(calls["bootstrap"]) == 1
    assert payload["bootstrapped"] is True
    assert payload["items_enqueued"] == 3
    assert payload["status"] == "executed"
    assert payload["execution_outcome"] == "executed"
    assert payload["selected_queue_item_id"] == "q-1"


def test_pending_future_dated_items_are_not_reseeded(wired) -> None:
    """Items that can still become due must not trigger a reseed every tick."""
    store, calls = wired
    store.total, store.usable, store.due = 5, 5, 0

    payload = api_routes.execute_substrate_review_scheduled_cycle()

    assert calls["bootstrap"] == []
    assert payload["bootstrapped"] is False
    assert payload["status"] == "idle_none_due"
    assert calls["cycle"] == []


def test_queue_full_of_unusable_items_still_reseeds(wired) -> None:
    """The absorbing state: a non-empty queue whose every item is spent.

    An earlier version of this file asserted that total=5/due=0 must report
    idle_none_due unconditionally -- which enshrined exactly the bug that killed
    the loop, because nothing distinguishes "not due yet" from "never due
    again". Gating the reseed on emptiness meant the first generation of items
    suppressing (6 executed cycles at the schema's
    suppress_after_low_value_cycles=2) left the queue permanently un-drainable.
    """
    store, calls = wired
    store.total, store.usable, store.due = 5, 0, 0

    payload = api_routes.execute_substrate_review_scheduled_cycle()

    assert len(calls["bootstrap"]) == 1
    assert payload["bootstrapped"] is True
    assert payload["queue_total"] == 5
    assert payload["usable_before"] == 0


def test_non_empty_queue_with_due_items_executes_without_seeding(wired) -> None:
    store, calls = wired
    store.total, store.usable, store.due = 5, 5, 2

    payload = api_routes.execute_substrate_review_scheduled_cycle()

    assert calls["bootstrap"] == []
    assert len(calls["cycle"]) == 1
    assert payload["status"] == "executed"
    assert payload["queue_total"] == 5
    assert payload["usable_before"] == 5
    assert payload["due_now"] == 2


def test_scheduled_cycle_never_allows_frontier_followup(wired) -> None:
    """Frontier follow-up is operator work with its own endpoint.

    This is NOT what keeps the loop out of the self_relationship_graph zone --
    an earlier version of this docstring claimed it was, and review disproved it
    three ways (the follow-up executor is unwired on this deployment,
    frontier_curiosity.py:239 refuses that zone unconditionally, and
    consolidation.py:279 echoes the request's own zone). The real guard is
    review_schedule.py:84, covered by
    test_substrate_review_queue_pruning.py::test_self_relationship_zone_never_enters_the_queue.
    """
    store, calls = wired
    store.total, store.usable, store.due = 1, 1, 1

    api_routes.execute_substrate_review_scheduled_cycle()

    assert len(calls["cycle"]) == 1
    assert calls["cycle"][0]["allow_followup"] is False
    assert calls["cycle"][0]["explicit_queue_item_id"] is None


def test_seeded_but_nothing_due_reports_distinctly_and_skips_cycle(wired, monkeypatch) -> None:
    """A bootstrap that schedules everything into the future must not be reported as executed."""
    store, calls = wired
    store.total, store.usable, store.due = 0, 0, 0

    def _seed_future_only(*, limit: int = 12):
        calls["bootstrap"].append({"limit": limit})
        store.total = 3
        store.usable = 3
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
    store.total, store.usable, store.due = 0, 0, 0

    api_routes.execute_substrate_review_scheduled_cycle(bootstrap_limit=4)

    assert calls["bootstrap"][0]["limit"] == 4


def test_tick_refreshes_and_prunes_before_reading_the_queue(wired) -> None:
    """Order matters, not just occurrence.

    ``assert store.refreshed >= 1`` passed even with the refresh moved *after*
    the read it exists to make correct, so it proved nothing the method name
    claimed. Assert the actual sequence instead: refresh, then prune, then read.
    """
    store, _calls = wired
    store.total, store.usable, store.due = 0, 0, 0

    api_routes.execute_substrate_review_scheduled_cycle()

    assert store.calls.index("refresh") < store.calls.index("prune")
    assert store.calls.index("prune") < store.calls.index("usable")


def test_payload_shape_is_loggable_and_carries_tick_id(wired) -> None:
    import json

    store, _calls = wired
    store.total, store.usable, store.due = 0, 0, 0
    now = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)

    payload = api_routes.execute_substrate_review_scheduled_cycle(now=now)

    assert payload["event"] == "review_scheduler_tick"
    assert payload["tick_id"].startswith("review-scheduler-")
    assert payload["at"] == now.isoformat()
    # default=str would make this total and therefore unfalsifiable -- the log
    # emitter uses default=str, but the payload should be plain JSON on its own.
    json.dumps(payload, sort_keys=True)
