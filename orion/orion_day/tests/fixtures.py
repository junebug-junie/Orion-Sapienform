"""A fixture day for Orion's Day tests: canned rows per source, served by SQL identity.

Rows mirror the live shapes checked on 2026-09-30 (asyncpg returns json/jsonb as text,
which gather decodes; a few are given as dicts to cover both forms).
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

from orion.dream import introspect_sql
from orion.orion_day import gather
from orion.world_pulse_read import introspect as wp_introspect

LETTER_DATE = date(2026, 9, 29)
T0 = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)  # inside the Denver day


def _t(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


CURIOSITY_FINISH = [
    {"run_id": "ab61e4ccd47b", "workflow": "curiosity.investigate", "generated_at": _t(10),
     "detail": json.dumps({"line": "investigate", "continue_line": True, "reach_out": False,
                           "journal_entry_id": "curiosity-investigation:ab61e4ccd47b",
                           "finding_text": "capped finding"})},
    {"run_id": "f218860792b4", "workflow": "curiosity.investigate", "generated_at": _t(20),
     "detail": {"line": "self_inquiry", "self_question_family": "anatomy", "continue_line": True,
                "reach_out": False, "self_definition": {"text": "I am one inference process on one substrate."},
                "journal_entry_id": "curiosity-self-inquiry:f218860792b4"}},
    {"run_id": "nojournal0001", "workflow": "curiosity.investigate", "generated_at": _t(30),
     "detail": {"line": "investigate", "finding_text": "Only the capped finding survived."}},
    {"run_id": "selfsense0001", "workflow": "self_sense_eval", "generated_at": _t(40),
     "detail": {"line": "self_sense_eval", "published": 4}},
]
CURIOSITY_FAILED = [
    {"run_id": "failed000001", "workflow": "curiosity.investigate", "generated_at": _t(50), "error": "HoldLost: x"},
]
CURIOSITY_JOURNALS = [
    {"entry_id": "curiosity-investigation:ab61e4ccd47b", "created_at": _t(10), "title": "Curiosity",
     "body": "I tested the hop written_at prior. " * 40, "source_ref": "curiosity:ab61e4ccd47b"},
    {"entry_id": "curiosity-self-inquiry:f218860792b4", "created_at": _t(20), "title": "Self-inquiry",
     "body": "# What I read\nThe service inventory held at 99.", "source_ref": "curiosity:f218860792b4"},
]
CURIOSITY_OUTCOMES = [
    {"run_id": "ab61e4ccd47b", "completed_at": _t(10), "turn_ok": True, "realized_nats": 0.02,
     "unknown_reason": None, "n_tested": 1, "n_moved": 1, "n_formed": 0,
     "per_prior": json.dumps([{"kind": "tested", "prior_id": "self:hop_written_at_missing", "before": 0.6, "after": 0.7}])},
]
SELF_SENSE = [
    {"run_id": "selfsense0001", "question_key": "what_are_you", "question": "What are you?",
     "answer_text": "A persistent mind made of services and memory.", "answer_source": "turn",
     "self_label_score": 2, "grounded_record_score": 1, "created_at": _t(40)},
]
READING_ROWS = [
    {"seed_id": "reading:07d283a9", "request_id": None, "url": "https://example.org/gguf",
     "title": "Transformers now runs llama.cpp quants", "status": "done", "stage2_status": "done",
     "created_at": _t(60), "handoff_at": _t(61), "stage2_completed_at": _t(62), "landing_at": _t(63),
     "handoff_json": json.dumps({"read_evidence": [{"url": "https://example.org/gguf", "tool_name": "WebFetch",
                                                     "content_chars": 5000}]}),
     "stage2_result_json": json.dumps({"summary": "GGUF quants load through transformers now. " * 60}),
     "trace_id": None, "stage2_trace_id": None, "why_now": "Relevant to my own serving stack."},
]
READING_JOURNALS = [
    {"entry_id": "8c62d21d", "created_at": _t(64), "title": "Transformers now runs llama.cpp quants",
     "body": "Stage 2 reflection on the GGUF reading.", "source_ref": "world_pulse_read_stage2:4a7e1760"},
]
DREAM_NARRATIVES = [
    {"id": 20, "dream_date": LETTER_DATE, "tldr": "A library of recent work", "themes": json.dumps(["archive"]),
     "narrative": "I walked between stacks of PRs.", "occurred_at": _t(5)},
]
DREAM_HYPOTHESES = [
    {"hypothesis_id": "dh-1a69e4d982ac", "cycle_id": "cyc-1", "claim": "The README update preceded the message.",
     "why": "Temporal adjacency in two fragments.", "occurred_at": _t(6), "expires_at": _t(600)},
]
REVERIE_THOUGHTS = [
    {"thought_id": f"th-{i:04d}-aaaaaaaa", "created_at": _t(100 + i), "salience": round(0.3 + (i % 7) / 10, 2),
     "interpretation": (f"Thought {i} about loop {i % 5}: " + "the coalition keeps circling. " * 8),
     "expectation": "the loop closes" if i % 3 == 0 else None,
     "expectation_verdict": "unmet" if i % 6 == 0 else None,
     "chain_id": f"chain-{i // 3}", "hollow": i % 11 == 0}
    for i in range(300)
] + [
    # an exact duplicate opening from another chain: deduped
    {"thought_id": "th-dup-bbbbbbbb", "created_at": _t(99), "salience": 0.99,
     "interpretation": "Thought 6 about loop 1: " + "the coalition keeps circling. " * 8,
     "expectation": None, "expectation_verdict": None, "chain_id": "chain-other", "hollow": False},
]
REVERIE_CHAINS = [
    {"chain_id": f"chain-{j}", "created_at": _t(100 + 3 * j), "theme_key": f"open-loop-{j % 4}",
     "terminal_reason": "max_steps" if j % 2 else "no_coalition", "ema_salience": 0.5 + (j % 3) / 10,
     "thought_count": 3}
    for j in range(100)
]
VISUAL_REVERIES = [
    {"sha256": "a550f6957904e7ec" + "0" * 48, "chain_id": "vchain-1", "step_index": 0, "created_at": _t(7),
     "mime": "image/png", "width": 1024, "height": 1024, "bytes": 123456,
     "path": "/mnt/storage-lukewarm/orion/reverie-visual/a550.png",
     "description": "A glowing light bulb with a blue and yellow light trail.", "theme_key": "light"},
]
CHAT_COMPACTOR = {"entry_id": "d5f4c123", "created_at": datetime(2026, 9, 30, 12, 1, tzinfo=timezone.utc), "title": "Reading Queue Timeout",
                  "body": "Juniper and I talked about the reading queue timeout.",
                  "source_ref": "chat_history_compactor_pass:chat_compactor:day:2026-09-29"}
WORLD_PULSE = {"run_id": "1765808d", "date": "2026-09-29", "title": "Daily World Pulse",
               "executive_summary": "12 tracked items.",
               "payload_json": json.dumps({"items": [
                   {"title": "Fuel standards rolled back", "category": "us_politics",
                    "summary": "Fuel standards rolled back (2 articles)", "why_it_matters": "Emissions."},
                   {"category": "no_title_dropped"}]})}


class FakeRecord(dict):
    """asyncpg Record stand-in: mapping access by column name."""


class FakeConn:
    """Serves canned rows keyed by the exact SQL constant gather sends. Records every call."""

    def __init__(self, *, fail: set[str] | None = None, github: dict | None = None, chat: dict | None = CHAT_COMPACTOR):
        self.calls: list[tuple[str, tuple]] = []
        self.fail = fail or set()
        self.fetch_map = {
            gather.CURIOSITY_FINISH_SQL: ("curiosity_runs", CURIOSITY_FINISH),
            gather.CURIOSITY_FAILED_SQL: ("curiosity_failed", CURIOSITY_FAILED),
            gather.CURIOSITY_JOURNALS_SQL: ("curiosity_runs", CURIOSITY_JOURNALS),
            gather.CURIOSITY_OUTCOMES_SQL: ("curiosity_runs", CURIOSITY_OUTCOMES),
            gather.SELF_SENSE_SQL: ("self_sense", SELF_SENSE),
            wp_introspect._WINDOW_SQL: ("readings", READING_ROWS),
            gather.JOURNALS_BY_PREFIX_SQL: ("reading_journals", READING_JOURNALS),
            introspect_sql.NARRATIVE_WINDOW_SQL: ("dream_narratives", DREAM_NARRATIVES),
            gather.DREAM_HYPOTHESES_SEEN_SQL: ("dream_hypotheses", DREAM_HYPOTHESES),
            gather.REVERIE_THOUGHTS_SQL: ("reverie_thoughts", REVERIE_THOUGHTS),
            gather.REVERIE_CHAINS_SQL: ("reverie_chains", REVERIE_CHAINS),
            gather.VISUAL_REVERIES_SQL: ("visual_reveries", VISUAL_REVERIES),
        }
        self.github = github
        self.chat = chat

    async def fetch(self, sql: str, *args: Any):
        self.calls.append((sql, args))
        source, rows = self.fetch_map[sql]
        if source in self.fail:
            raise RuntimeError(f"relation for {source} does not exist")
        return [FakeRecord(r) for r in rows]

    async def fetchrow(self, sql: str, *args: Any):
        self.calls.append((sql, args))
        if sql == gather.WORLD_PULSE_DIGEST_SQL:
            if "world_pulse_digest" in self.fail:
                raise RuntimeError("world pulse down")
            return FakeRecord(WORLD_PULSE)
        if sql == gather.JOURNAL_BY_ID_OR_REF_SQL:
            ref, written_after = args[1], args[2]
            row = self.chat if ref.startswith("chat_history_compactor_pass:") else self.github
            # The SQL's own `created_at >= $3` guard, applied to the canned row.
            return FakeRecord(row) if row and row["created_at"] >= written_after else None
        raise AssertionError(f"unexpected fetchrow: {sql[:60]}")
