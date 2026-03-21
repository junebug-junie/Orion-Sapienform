from __future__ import annotations

import json
import unittest

from orion.journaler.worker import (
    build_manual_trigger,
    build_compose_request,
    build_write_payload,
    draft_from_cortex_result,
)
from orion.journaler.schemas import JournalEntryDraftV1


class TestJournalEndToEnd(unittest.TestCase):
    def test_happy_path_trigger_to_write_payload(self):
        trigger = build_manual_trigger(summary="Capture today's arc", prompt_seed="seed", source_ref="manual-1")
        req = build_compose_request(trigger, session_id="journal-session", user_id="juniper", trace_id="corr-1")
        self.assertEqual(req.verb, "journal.compose")

        orch_payload = {
            "ok": True,
            "status": "success",
            "final_text": json.dumps({"mode": "manual", "title": "Arc", "body": "I felt a clean convergence today."}),
        }
        draft = draft_from_cortex_result(orch_payload)
        self.assertIsInstance(draft, JournalEntryDraftV1)

        write = build_write_payload(draft, trigger=trigger, correlation_id="corr-1", author="orion")
        self.assertEqual(write.mode, "manual")
        self.assertEqual(write.source_ref, "manual-1")
        self.assertEqual(write.correlation_id, "corr-1")


if __name__ == "__main__":
    unittest.main()
