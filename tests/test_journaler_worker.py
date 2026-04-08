from __future__ import annotations

from datetime import datetime, timezone
import unittest

from orion.journaler.schemas import JournalEntryDraftV1, JournalEntryWriteV1, JournalTriggerV1
from orion.journaler.worker import (
    build_collapse_stored_trigger,
    build_compose_request,
    build_manual_trigger,
    build_scheduler_trigger,
    build_write_payload,
    cooldown_key_for_trigger,
    draft_from_cortex_result,
    journal_mode_for_trigger,
)
from orion.schemas.collapse_mirror import CollapseMirrorStoredV1


class TestJournalerSchemasAndWorker(unittest.TestCase):
    def test_schema_validation_and_extra_forbid(self):
        trigger = JournalTriggerV1.model_validate(
            {
                "trigger_kind": "manual",
                "source_kind": "manual",
                "summary": "Write about today's state.",
                "prompt_seed": "seed",
            }
        )
        self.assertEqual(trigger.trigger_kind, "manual")

        draft = JournalEntryDraftV1.model_validate({"mode": "manual", "title": None, "body": "Body"})
        self.assertEqual(draft.body, "Body")

        write = JournalEntryWriteV1.model_validate(
            {
                "entry_id": "entry-1",
                "created_at": datetime(2026, 3, 21, tzinfo=timezone.utc).isoformat(),
                "author": "orion",
                "mode": "manual",
                "title": None,
                "body": "Body",
                "source_kind": "manual",
                "source_ref": None,
                "correlation_id": "corr-1",
            }
        )
        self.assertEqual(write.entry_id, "entry-1")

        with self.assertRaises(Exception):
            JournalTriggerV1.model_validate(
                {
                    "trigger_kind": "manual",
                    "source_kind": "manual",
                    "summary": "x",
                    "unexpected": True,
                }
            )

    def test_trigger_maps_to_compose_request(self):
        trigger = build_manual_trigger(summary="Daily reflection", prompt_seed="short seed", source_ref="manual-1")
        self.assertEqual(journal_mode_for_trigger(trigger), "manual")
        req = build_compose_request(
            trigger,
            session_id="journal-session",
            user_id="juniper_primary",
            trace_id="corr-1",
            recall_profile="reflect.v1",
        )
        self.assertEqual(req.verb, "journal.compose")
        self.assertEqual(req.context.session_id, "journal-session")
        self.assertEqual(req.context.metadata["journal_trigger"]["source_ref"], "manual-1")
        self.assertEqual(req.context.metadata["journal_mode"], "manual")
        self.assertTrue(req.recall.enabled)

    def test_stored_collapse_trigger_maps_to_response_mode(self):
        trigger = build_collapse_stored_trigger(
            CollapseMirrorStoredV1(
                mirror_id="mirror-1",
                stored_at="2026-03-21T12:00:00Z",
                correlation_id="corr-1",
                is_causally_dense=True,
                summary="A dense mirror was stored.",
                trigger="drift",
                what_changed_summary="attention sharpened",
                mantra="stay steady",
            )
        )
        self.assertEqual(trigger.source_ref, "mirror-1")
        self.assertEqual(journal_mode_for_trigger(trigger), "response")
        self.assertIn("What changed: attention sharpened", trigger.prompt_seed)

    def test_draft_maps_to_write_payload(self):
        draft = JournalEntryDraftV1(mode="digest", title="Signals", body="Today I noticed a shift.")
        trigger = JournalTriggerV1(
            trigger_kind="metacog_digest",
            source_kind="metacog",
            source_ref="2026-03-21T00:00:00Z",
            summary="Pressure eased.",
        )
        created_at = datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)
        write = build_write_payload(
            draft,
            trigger=trigger,
            correlation_id="corr-2",
            author="orion",
            entry_id="entry-2",
            created_at=created_at,
        )
        self.assertEqual(write.mode, "digest")
        self.assertEqual(write.source_kind, "metacog")
        self.assertEqual(write.correlation_id, "corr-2")
        self.assertEqual(write.created_at, created_at)

    def test_cooldown_keys_are_separated_and_collapse_is_per_source(self):
        daily = build_scheduler_trigger(summary="same", source_ref="2026-03-21")
        manual = build_manual_trigger(summary="same", source_ref="2026-03-21")
        metacog = JournalTriggerV1(trigger_kind="metacog_digest", source_kind="metacog", source_ref="2026-03-21", summary="same")
        collapse_a = build_collapse_stored_trigger(
            CollapseMirrorStoredV1(
                mirror_id="mirror-a",
                stored_at="2026-03-21T12:00:00Z",
                correlation_id="corr-a",
                is_causally_dense=True,
                summary="dense",
                trigger="t",
            )
        )
        collapse_b = build_collapse_stored_trigger(
            CollapseMirrorStoredV1(
                mirror_id="mirror-b",
                stored_at="2026-03-21T12:05:00Z",
                correlation_id="corr-b",
                is_causally_dense=True,
                summary="dense",
                trigger="t",
            )
        )
        self.assertNotEqual(cooldown_key_for_trigger(daily), cooldown_key_for_trigger(manual))
        self.assertNotEqual(cooldown_key_for_trigger(daily), cooldown_key_for_trigger(metacog))
        self.assertNotEqual(cooldown_key_for_trigger(manual), cooldown_key_for_trigger(metacog))
        self.assertNotEqual(cooldown_key_for_trigger(collapse_a), cooldown_key_for_trigger(collapse_b))
        self.assertEqual(cooldown_key_for_trigger(collapse_a), cooldown_key_for_trigger(collapse_a))


    def test_draft_from_cortex_result_accepts_clean_json(self):
        payload = {
            "ok": True,
            "status": "success",
            "final_text": '{"mode":"manual","title":"Arc","body":"Kept going."}',
        }
        draft = draft_from_cortex_result(payload)
        self.assertEqual(draft.mode, "manual")
        self.assertEqual(draft.title, "Arc")

    def test_draft_from_cortex_result_accepts_fenced_json(self):
        payload = {
            "ok": True,
            "status": "success",
            "final_text": """Here is the draft:
```json
{"mode":"manual","title":"Arc","body":"Kept going."}
```""",
        }
        draft = draft_from_cortex_result(payload)
        self.assertEqual(draft.title, "Arc")
        self.assertEqual(draft.body, "Kept going.")

    def test_draft_from_cortex_result_accepts_preface_and_trailing_text(self):
        payload = {
            "ok": True,
            "status": "success",
            "final_text": 'Draft: {"mode":"manual","title":"Arc","body":"Kept going."}\n(complete)',
        }
        draft = draft_from_cortex_result(payload)
        self.assertEqual(draft.mode, "manual")
        self.assertEqual(draft.body, "Kept going.")

    def test_draft_from_cortex_result_accepts_think_tag_preface(self):
        payload = {
            "ok": True,
            "status": "success",
            "final_text": '<think>reasoning not for output</think>{"mode":"manual","title":"Arc","body":"Kept going."}',
        }
        draft = draft_from_cortex_result(payload)
        self.assertEqual(draft.mode, "manual")
        self.assertEqual(draft.title, "Arc")

    def test_draft_from_cortex_result_pure_reasoning_without_json_fails_clearly(self):
        payload = {
            "ok": True,
            "status": "success",
            "correlation_id": "corr-think-only",
            "final_text": "<think>no final answer emitted</think>",
        }
        with self.assertRaises(ValueError) as ctx:
            draft_from_cortex_result(payload)
        self.assertIn("journal_draft_parse_failed", str(ctx.exception))
        self.assertIn('"think_tags_detected": true', str(ctx.exception))
        self.assertIn("corr-think-only", str(ctx.exception))

    def test_draft_from_cortex_result_observed_think_preview_pattern_recovers(self):
        payload = {
            "ok": True,
            "status": "success",
            "final_text": (
                "<think>I should check grounding and format.</think>\n"
                "I will now return the requested JSON.\n"
                '{"mode":"manual","title":"Grounded Arc","body":"Kept going."}'
            ),
        }
        draft = draft_from_cortex_result(payload)
        self.assertEqual(draft.title, "Grounded Arc")
        self.assertEqual(draft.body, "Kept going.")

    def test_draft_from_cortex_result_uses_final_answer_not_reasoning_field(self):
        payload = {
            "ok": True,
            "status": "success",
            "final_text": '{"mode":"manual","title":"Final","body":"Visible answer."}',
            "reasoning_content": "<think>hidden chain of thought</think>",
        }
        draft = draft_from_cortex_result(payload)
        self.assertEqual(draft.title, "Final")
        self.assertEqual(draft.body, "Visible answer.")

    def test_draft_from_cortex_result_reads_nested_result_final_text(self):
        payload = {
            "ok": True,
            "status": "success",
            "result": {"final_text": '{"mode":"manual","title":"Nested","body":"From result."}'},
        }
        draft = draft_from_cortex_result(payload)
        self.assertEqual(draft.title, "Nested")
        self.assertEqual(draft.body, "From result.")

    def test_draft_from_cortex_result_reads_nested_cortex_result_final_text(self):
        payload = {
            "ok": True,
            "status": "success",
            "cortex_result": {"final_text": '{"mode":"manual","title":"Wrapped","body":"From cortex_result."}'},
        }
        draft = draft_from_cortex_result(payload)
        self.assertEqual(draft.title, "Wrapped")
        self.assertEqual(draft.body, "From cortex_result.")


    def test_draft_from_cortex_result_missing_required_key_fails_clearly(self):
        payload = {
            "ok": True,
            "status": "success",
            "correlation_id": "corr-missing",
            "final_text": '{"mode":"manual","title":"Arc"}',
        }
        with self.assertRaises(ValueError) as ctx:
            draft_from_cortex_result(payload)
        self.assertIn("journal_draft_missing_required_key:body", str(ctx.exception))
        self.assertIn('"missing_required_key": "body"', str(ctx.exception))
        self.assertIn("corr-missing", str(ctx.exception))

    def test_draft_from_cortex_result_regression_starts_with_open_brace_but_incomplete(self):
        payload = {
            "ok": True,
            "status": "success",
            "correlation_id": "corr-open-brace",
            "final_text": "{",
        }
        with self.assertRaises(ValueError) as ctx:
            draft_from_cortex_result(payload)
        self.assertIn("journal_draft_parse_failed", str(ctx.exception))
        self.assertIn("corr-open-brace", str(ctx.exception))

    def test_draft_from_cortex_result_raises_structured_parse_error(self):
        payload = {
            "ok": True,
            "status": "success",
            "verb": "journal.compose",
            "correlation_id": "corr-parse",
            "final_text": '{"mode":"manual","title":"Arc","body":"unterminated}',
        }
        with self.assertRaises(ValueError) as ctx:
            draft_from_cortex_result(payload)
        self.assertIn("journal_draft_parse_failed", str(ctx.exception))
        self.assertIn("corr-parse", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
