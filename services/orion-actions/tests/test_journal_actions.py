from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from uuid import uuid4

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.logic import ActionDedupe, build_journal_cortex_orch_envelope  # noqa: E402
from app.main import _dispatch_journal_notifications, should_journal_from_collapse  # noqa: E402
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.journaler.schemas import JournalEntryWriteV1  # noqa: E402
from orion.journaler.worker import (  # noqa: E402
    build_collapse_stored_trigger,
    build_manual_trigger,
    build_scheduler_trigger,
    cooldown_key_for_trigger,
)
from orion.schemas.collapse_mirror import CollapseMirrorStoredV1  # noqa: E402


class _FakeNotify:
    """Minimal fake mirroring test_async_notify_producers.py's `_FakeNotify` --
    records every `send` (email) / `chat_message` (in-app) call so tests can assert
    exactly how many dispatch attempts actually happened."""

    def __init__(self) -> None:
        self.send_calls: list = []
        self.chat_calls: list = []

    def send(self, request):
        self.send_calls.append(request)
        return SimpleNamespace(ok=True, status="queued", notification_id="notif-1", detail=None)

    def chat_message(self, **kwargs):
        self.chat_calls.append(kwargs)
        return SimpleNamespace(ok=True, notification_id="chat-1", detail=None)


class TestJournalActions(unittest.TestCase):
    def test_actions_journal_cooldown_blocks_repeat_dispatch_for_same_dense_mirror(self):
        deduper = ActionDedupe(ttl_seconds=3600)
        trigger = build_collapse_stored_trigger(
            CollapseMirrorStoredV1(
                mirror_id="mirror-1",
                stored_at="2026-03-21T12:00:00Z",
                correlation_id="corr-1",
                is_causally_dense=True,
                summary="A dense shift.",
                trigger="alignment",
            )
        )
        key = cooldown_key_for_trigger(trigger)
        self.assertTrue(deduper.try_acquire(key))
        deduper.mark_done(key)
        self.assertFalse(deduper.try_acquire(key))

    def test_actions_builds_correct_journal_orch_request_from_stored_event(self):
        parent = BaseEnvelope(
            kind="collapse.mirror.stored.v1",
            source=ServiceRef(name="sql-writer"),
            correlation_id=str(uuid4()),
            payload={},
        )
        trigger = build_collapse_stored_trigger(
            CollapseMirrorStoredV1(
                mirror_id="mirror-42",
                stored_at="2026-03-21T12:00:00Z",
                correlation_id="corr-42",
                is_causally_dense=True,
                summary="Manual journal summary",
                trigger="prompt",
                what_changed_summary="something clicked",
                mantra="stay steady",
            )
        )
        env = build_journal_cortex_orch_envelope(
            parent,
            source=ServiceRef(name="orion-actions"),
            trigger=trigger,
            session_id="orion_journal",
            user_id="juniper_primary",
        )
        self.assertEqual(env.kind, "cortex.orch.request")
        self.assertEqual(env.payload["verb"], "journal.compose")
        self.assertEqual(env.payload["context"]["metadata"]["journal_trigger"]["source_ref"], "mirror-42")
        self.assertEqual(env.payload["context"]["metadata"]["journal_mode"], "response")

    def test_stored_event_dense_gate_controls_journaling(self):
        self.assertFalse(should_journal_from_collapse(False, dense_only=True))
        self.assertTrue(should_journal_from_collapse(False, dense_only=False))
        self.assertTrue(should_journal_from_collapse(True, dense_only=True))

    def test_cooldown_keys_separate_trigger_classes(self):
        daily = build_scheduler_trigger(summary="Daily cadence", source_ref="2026-03-21")
        manual = build_manual_trigger(summary="Daily cadence", source_ref="2026-03-21")
        metacog = build_manual_trigger(summary="Daily cadence", source_ref="2026-03-21").model_copy(
            update={"trigger_kind": "metacog_digest", "source_kind": "metacog"}
        )
        self.assertNotEqual(cooldown_key_for_trigger(daily), cooldown_key_for_trigger(manual))
        self.assertNotEqual(cooldown_key_for_trigger(daily), cooldown_key_for_trigger(metacog))
        self.assertNotEqual(cooldown_key_for_trigger(manual), cooldown_key_for_trigger(metacog))


def _daily_summary_entry(entry_id: str = "entry-daily-1") -> JournalEntryWriteV1:
    return JournalEntryWriteV1(
        entry_id=entry_id,
        author="orion",
        mode="daily",
        title="Morning Reflection",
        body="Summarize daily priorities and anchors.",
        source_kind="scheduler",
        source_ref="2026-07-13",
        correlation_id="corr-daily-1",
        trigger_kind="daily_summary",
    )


def test_scheduler_daily_journal_produces_exactly_one_email_dispatch() -> None:
    """Regression for the double-email bug: two independent codepaths used to email
    the same trigger_kind=daily_summary/source_kind=scheduler entry_id under two
    different dedupe-key namespaces (actions:journal:daily:scheduler:{entry_id} and
    actions:journal:persisted:{entry_id}), so neither path's dedupe ever saw the
    other's send. There is now exactly one call site
    (_dispatch_journal_notifications, invoked once from the post-persist consumer)
    and one dedupe-key namespace (actions:journal:notify:{entry_id}) -- even a
    redelivered journal.entry.created.v1 event for the same entry_id must not
    double-send."""
    notify = _FakeNotify()
    deduper = ActionDedupe(ttl_seconds=3600)
    entry = _daily_summary_entry()

    first = _dispatch_journal_notifications(entry, correlation_id="corr-daily-1", notify=notify, deduper=deduper)
    assert first["deduped"] is False
    assert first["email_ok"] is True

    # Simulate a redelivery of the same journal.entry.created.v1 event (or, in the
    # old world, the second of the two now-removed codepaths) hitting dispatch again.
    second = _dispatch_journal_notifications(entry, correlation_id="corr-daily-1", notify=notify, deduper=deduper)
    assert second["deduped"] is True

    assert len(notify.send_calls) == 1
    email_req = notify.send_calls[0]
    assert email_req.dedupe_key == f"actions:journal:notify:{entry.entry_id}"
    # in_app is off for daily_summary per JOURNAL_DISPATCH_REGISTRY (0 measured
    # engagement -- see orion/journaler/dispatch_registry.py docstring).
    assert len(notify.chat_calls) == 0


def test_resolve_policy_is_fail_closed_for_an_unregistered_trigger_kind() -> None:
    """`orion.journaler.schemas.JournalTriggerKind` is a closed Literal of 8 values,
    all of which are registered today -- but `resolve_policy` itself must not assume
    that stays true forever (e.g. a producer publishing a `journal.entry.created.v1`
    payload from a newer/older schema version with a trigger_kind string
    JOURNAL_DISPATCH_REGISTRY hasn't been taught about yet). Any such string must
    resolve to an all-disabled policy, not "email everything"."""
    from orion.journaler import resolve_policy

    policy = resolve_policy("totally_unregistered_kind")
    assert policy.email_enabled is False
    assert policy.in_app_enabled is False


def test_missing_trigger_kind_sends_nothing() -> None:
    """Fail-closed via the real dispatch path: a journal entry with no trigger_kind
    at all (e.g. a pre-Part-A producer payload, or any future gap) resolves against
    resolve_policy("") -- which has no registry row -- so it must send neither email
    nor in-app, not default to "email everything"."""
    notify = _FakeNotify()
    deduper = ActionDedupe(ttl_seconds=3600)
    entry = JournalEntryWriteV1(
        entry_id="entry-unregistered-1",
        author="orion",
        mode="manual",
        title="Missing trigger kind",
        body="This entry has no trigger_kind at all.",
        source_kind="manual",
        correlation_id="corr-unregistered-1",
        trigger_kind=None,
    )

    result = _dispatch_journal_notifications(entry, correlation_id="corr-unregistered-1", notify=notify, deduper=deduper)

    assert result["deduped"] is False
    assert result["email_ok"] is True  # not required -> True, but nothing was sent
    assert result["in_app_ok"] is True
    assert len(notify.send_calls) == 0
    assert len(notify.chat_calls) == 0


if __name__ == "__main__":
    unittest.main()
