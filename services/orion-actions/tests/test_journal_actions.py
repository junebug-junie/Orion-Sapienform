from __future__ import annotations

import os
import sys
import threading
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.logic import ActionDedupe, build_journal_cortex_orch_envelope  # noqa: E402
from app.main import (  # noqa: E402
    _dispatch_journal_notifications,
    _journal_email_daily_cap_key,
    should_journal_from_collapse,
)
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


def _metacog_digest_entry(entry_id: str = "entry-metacog-1") -> JournalEntryWriteV1:
    return JournalEntryWriteV1(
        entry_id=entry_id,
        author="orion",
        mode="daily",
        title="Metacog Digest",
        body="Reflect on today's metacognitive triggers.",
        source_kind="metacog",
        source_ref="2026-07-13",
        correlation_id="corr-2",
        trigger_kind="metacog_digest",
    )


def test_second_freeform_journal_of_the_day_is_not_emailed_even_with_a_different_trigger_kind() -> None:
    """The operator was getting up to 4 separate freeform-journal emails a day
    (daily_summary, metacog_digest, world_pulse_digest, notify_summary can each fire
    independently). Only the first journal email of the local calendar day should
    actually send; later ones -- even from a different trigger_kind and a different
    entry_id -- must persist (mode not blocked) but skip the email step.

    `now_utc` is pinned to the same instant for both calls -- relying on real
    wall-clock time here would make the test flake if it happened to straddle local
    midnight (settings.actions_daily_timezone, default America/Denver)."""
    notify = _FakeNotify()
    deduper = ActionDedupe(ttl_seconds=3600)
    same_moment = datetime(2026, 7, 13, 15, 0, tzinfo=timezone.utc)

    first = _dispatch_journal_notifications(
        _daily_summary_entry(entry_id="entry-daily-1"), correlation_id="corr-1", notify=notify, deduper=deduper, now_utc=same_moment
    )
    assert first["deduped"] is False
    assert first["email_ok"] is True
    assert len(notify.send_calls) == 1

    second = _dispatch_journal_notifications(
        _metacog_digest_entry(), correlation_id="corr-2", notify=notify, deduper=deduper, now_utc=same_moment
    )

    # A different entry_id/trigger_kind is not the same dedupe -- this is not the
    # entry-level double-send guard, it's the daily cap.
    assert second["deduped"] is False
    assert second["email_ok"] is True  # not required -> True, but nothing was actually sent
    assert len(notify.send_calls) == 1  # still just the first email


def test_daily_email_cap_resets_on_the_next_local_calendar_day() -> None:
    """The cap is per-day, not permanent -- a journal email on day 2 must still
    send even though day 1 already used its slot."""
    notify = _FakeNotify()
    deduper = ActionDedupe(ttl_seconds=3600)
    day_one = datetime(2026, 7, 13, 15, 0, tzinfo=timezone.utc)
    day_two = datetime(2026, 7, 14, 15, 0, tzinfo=timezone.utc)

    first = _dispatch_journal_notifications(
        _daily_summary_entry(entry_id="entry-daily-1"), correlation_id="corr-1", notify=notify, deduper=deduper, now_utc=day_one
    )
    assert first["email_ok"] is True
    assert len(notify.send_calls) == 1

    second = _dispatch_journal_notifications(
        _metacog_digest_entry(), correlation_id="corr-2", notify=notify, deduper=deduper, now_utc=day_two
    )
    assert second["email_ok"] is True
    assert len(notify.send_calls) == 2  # new day -> cap reset -> second email actually sent


def test_daily_cap_slot_is_released_when_notify_raises_instead_of_leaking_forever() -> None:
    """Regression: if NotificationRequest construction or notify.send() raises, the
    daily-cap key must not stay stuck 'in-flight' -- ActionDedupe has no TTL on
    in-flight entries (only on completed ones), so a leaked key would silently block
    every journal email for the rest of the process's life, not just the rest of the
    day."""

    class _RaisingNotify:
        def send(self, request):
            raise RuntimeError("smtp exploded")

    deduper = ActionDedupe(ttl_seconds=3600)
    moment = datetime(2026, 7, 13, 15, 0, tzinfo=timezone.utc)

    with pytest.raises(RuntimeError):
        _dispatch_journal_notifications(
            _daily_summary_entry(entry_id="entry-daily-1"),
            correlation_id="corr-1",
            notify=_RaisingNotify(),
            deduper=deduper,
            now_utc=moment,
        )

    # The failed send must have released the cap key, not left it in-flight forever.
    notify = _FakeNotify()
    second = _dispatch_journal_notifications(
        _metacog_digest_entry(), correlation_id="corr-2", notify=notify, deduper=deduper, now_utc=moment
    )
    assert second["email_ok"] is True
    assert len(notify.send_calls) == 1


def test_action_dedupe_try_acquire_is_thread_safe_under_concurrent_callers() -> None:
    """Regression for a confirmed-live race: `_dispatch_journal_notifications` runs
    via `asyncio.to_thread` from a bus consumer with `concurrent_handlers` enabled by
    default, so two journal entries can call `try_acquire` on the same shared
    daily-cap key from two real OS threads at once. Without a lock in `ActionDedupe`,
    the check-then-add was a TOCTOU race letting both threads win the same key."""
    import concurrent.futures

    deduper = ActionDedupe(ttl_seconds=3600)
    key = "shared-daily-cap-key"
    barrier = threading.Barrier(32)

    def _attempt() -> bool:
        barrier.wait()  # maximize actual thread overlap on try_acquire
        return deduper.try_acquire(key)

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(lambda _: _attempt(), range(32)))

    winners = [r for r in results if r]
    assert len(winners) == 1


def test_daily_cap_key_rejects_a_naive_now_utc_instead_of_silently_misreading_it() -> None:
    """A naive datetime passed as `now_utc` would otherwise be silently
    reinterpreted by `.astimezone()` as the server's local system time rather than
    UTC, producing a wrong calendar-day cap key with no error. Fail loud instead."""
    naive = datetime(2026, 7, 13, 15, 0)  # no tzinfo
    with pytest.raises(ValueError):
        _journal_email_daily_cap_key(now_utc=naive)


def test_daily_cap_duration_is_independent_of_the_dedupers_generic_ttl() -> None:
    """Regression: the cap's `mark_done` used to inherit whatever `ttl_seconds` the
    passed-in `ActionDedupe` instance happened to be constructed with -- which in
    production is `ACTIONS_NOTIFY_DEDUPE_WINDOW_SECONDS`, a setting that also governs
    unrelated per-notification redelivery-dedupe windows elsewhere in the file. If an
    operator ever tunes that setting down for its original purpose, the cap would
    silently stop holding for the full day. Prove the cap now survives well past a
    deliberately tiny generic ttl, because `mark_done` is called with an explicit
    `ttl_seconds` override computed from time-until-local-midnight, not the
    deduper's own `self.ttl_seconds`."""
    notify = _FakeNotify()
    deduper = ActionDedupe(ttl_seconds=1)  # absurdly short generic ttl
    moment = datetime(2026, 7, 13, 15, 0, tzinfo=timezone.utc)

    first = _dispatch_journal_notifications(
        _daily_summary_entry(entry_id="entry-daily-1"), correlation_id="corr-1", notify=notify, deduper=deduper, now_utc=moment
    )
    assert first["email_ok"] is True
    assert len(notify.send_calls) == 1

    # Well past the deduper's 1-second generic ttl, but still the same local day --
    # the cap must still hold because mark_done overrode the ttl to "until midnight."
    later_same_day = datetime(2026, 7, 13, 20, 0, tzinfo=timezone.utc)
    second = _dispatch_journal_notifications(
        _metacog_digest_entry(), correlation_id="corr-2", notify=notify, deduper=deduper, now_utc=later_same_day
    )
    assert second["email_ok"] is True  # not required -> True, but nothing was actually sent
    assert len(notify.send_calls) == 1  # still just the first email


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
