from __future__ import annotations

import os
import sys
import unittest
from uuid import uuid4

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.logic import ActionDedupe, build_journal_cortex_orch_envelope  # noqa: E402
from app.main import should_journal_from_collapse  # noqa: E402
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.journaler.worker import (  # noqa: E402
    build_collapse_stored_trigger,
    build_manual_trigger,
    build_scheduler_trigger,
    cooldown_key_for_trigger,
)
from orion.schemas.collapse_mirror import CollapseMirrorStoredV1  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
