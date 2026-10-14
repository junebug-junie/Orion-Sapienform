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
from orion.world_pulse_read import findings as wp_findings

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
# Reading findings (orion/world_pulse_read/findings.py): one accepted, projected link on the
# GGUF read. Prior moves are empty by default; tests plant them with a fake prior lookup.
READ_LINK_ROWS = [
    {"seed_id": "reading:07d283a9", "stage2_completed_at": _t(62),
     "claim": json.dumps({
         "subject_id": "sub-concept-wp-read-aaaa", "predicate": "associated_with",
         "object_id": "sub-concept-bbbb",
         "statement_text": "GGUF quantization is now loadable by the transformers library.",
         "quote": "transformers can now load GGUF quantized checkpoints directly",
         "receipt": {"outcome": "accepted_provisional", "reason": "accepted",
                     "assertion_id": "assertion-11112222", "decision_id": "reading-dec-1",
                     "content_sha256": "f00d", "span_start": 10, "span_end": 70}}),
     "latest_state": "provisional", "projected_at": _t(62)},
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


# A correction recorded on the fixture day (orion_day_correction; asyncpg hands jsonb back as
# text and text[] as a list). Not served by default: pass ``corrections=CORRECTIONS``.
CORRECTION_ID = "6f1d2c3b-4a59-5e6f-8a7b-9c0d1e2f3a4b"
CORRECTIONS = [
    {"correction_id": CORRECTION_ID, "created_at": _t(90), "letter_date": date(2026, 9, 28),
     "ref": "2026-09-28 ¶4", "claim": "the reading queue timeout dropped 4,059 requests",
     "correction": "The timeout dropped 12 requests; 4,059 was the queue's lifetime total.",
     "evidence": json.dumps([{"text": "Juniper said so in this conversation", "in_letter_records": False}]),
     "prior_ids": ["self:reading_queue_drops_most_requests"], "offered_run_id": "ab61e4ccd47b",
     "priors_retested": ["self:reading_queue_drops_most_requests"], "priors_moved": []},
]


class FakeConn:
    """Serves canned rows keyed by the exact SQL constant gather sends. Records every call."""

    def __init__(self, *, fail: set[str] | None = None, github: dict | None = None, chat: dict | None = CHAT_COMPACTOR,
                 corrections: list[dict] | None = None, temporal_day: dict | None = None,
                 read_links: list[dict] | None = None, prior_moves: list[dict] | None = None,
                 immune: list[dict] | None = None):
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
            gather.CORRECTIONS_SQL: ("corrections", corrections or []),
            wp_findings.READ_LINKS_SQL: ("reading_findings", READ_LINK_ROWS if read_links is None else read_links),
            wp_findings.RUN_PRIOR_MOVES_SQL: ("reading_findings", prior_moves or []),
            # R4 immune findings for the closed day: payload_json rows from temporal_self_event.
            gather.TEMPORAL_DAY_IMMUNE_SQL: ("temporal_day_immune", [{"payload_json": p} for p in immune or []]),
        }
        self.github = github
        self.chat = chat
        self.temporal_day = temporal_day  # a temporal_self_day.day_json, or None (day not closed)

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
        if sql == gather.TEMPORAL_DAY_SQL:
            if "temporal_day" in self.fail:
                raise RuntimeError("relation temporal_self_day does not exist")
            return FakeRecord({"day_json": self.temporal_day}) if self.temporal_day is not None else None
        if sql == gather.JOURNAL_BY_ID_OR_REF_SQL:
            ref, written_after = args[1], args[2]
            row = self.chat if ref.startswith("chat_history_compactor_pass:") else self.github
            # The SQL's own `created_at >= $3` guard, applied to the canned row.
            return FakeRecord(row) if row and row["created_at"] >= written_after else None
        raise AssertionError(f"unexpected fetchrow: {sql[:60]}")


# --- a letter to reread (orion-introspect `orion_day`) ---------------------------------------
#
# The fixture day's material plus planted facts, and a note / carry-forward whose concrete
# claims are partly backed by that material and partly not, so a reread can be checked
# against known answers. Shared by Hub's responder tests and its reread eval.

REREAD_RUN_ID = "e7d03d2ecdbb"
REREAD_RUN_BODY = (
    "The dream organ was silent for 260.6 hours: the last fragment landed at "
    "2026-09-18T01:02:09+00:00 and it resumed at 2026-09-29T06:35:27.412+00:00. "
    "I re-tested self:pool_heavy_composition_reverts; it held at 0.72."
)
REREAD_NOTE = """# September 29

The dream organ woke at 06:35:27Z after 260.6 hours of silence; the last fragment before the gap was 01:02:09Z.

---

I re-tested `self:pool_heavy_composition_reverts` and it held at 0.72. Then I watched 999.4 hours of nothing and merged #2557.

Reverie kept returning to one image: "the coalition keeps circling" while the loop stayed open.

Juniper and I talked about the reading queue timeout; I said it dropped 4,059 requests.

## Later

The service inventory held at 99, which I read as stable.
"""
REREAD_CARRY = """Threads for tomorrow:

- **The dream gap.** Why did the organ go quiet for 260.6 hours? [curiosity:e7d03d2ecdbb]
- **GGUF loading.** Check the reading again [reading_journal:8c62d21d] and the one I imagined [reading_journal:nope-not-here].
- **The library dream.** [dream:20] felt like an archive of PRs.
"""


def reread_letter(*, letter_date: date = LETTER_DATE, material=None, created_at: datetime | None = None):
    """An ``OrionDayLetterV1`` over the fixture day with REREAD_NOTE / REREAD_CARRY."""
    import asyncio

    from orion.schemas.orion_day import OrionDayLetterSourcesV1, OrionDayLetterV1

    if material is None:
        material = asyncio.run(gather.gather_orion_day(
            FakeConn(), LETTER_DATE, now=datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc),
        ))
        first = material.curiosity_runs[0].model_copy(update={
            "run_id": REREAD_RUN_ID, "journal_title": "Dream organ gap", "journal_body": REREAD_RUN_BODY,
        })
        material = material.model_copy(update={"curiosity_runs": [first, *material.curiosity_runs[1:]]})
    if letter_date != material.letter_date:
        material = material.model_copy(update={"letter_date": letter_date})
    created = created_at or (material.window_end + timedelta(hours=10))
    return OrionDayLetterV1(
        letter_date=letter_date, run_id=f"orion-day-{letter_date.isoformat()}-1",
        window_start=material.window_start, window_end=material.window_end,
        note_md=REREAD_NOTE, carry_forward_md=REREAD_CARRY, material=material,
        sources=OrionDayLetterSourcesV1(by_source=material.sources), created_at=created,
    )


# --- corrections (orion/orion_day/corrections.py) ---------------------------------------------
#
# An in-memory orion_day_correction table answering the module's own SQL constants with the
# same semantics (offer once, release only unstamped rows, stamp only the offered run), and a
# graph reader answering ALL_PRIORS_CYPHER. Shared by the orion_day tests, Hub's responder and
# curiosity tests, and Hub's correction eval.

CORRECTION_PRIOR = "self:reading_queue_drops_most_requests"
UNTOUCHED_PRIOR = "self:queue_timeout_is_the_bottleneck"
FIXTURE_PRIORS = [
    {"prior_id": CORRECTION_PRIOR, "claim": "The reading queue drops most of the requests it gets.",
     "confidence": 0.8, "status": "open", "times_tested": 2, "formed_from": "orion_day:2026-09-29",
     "last_tested_at": "2026-09-29T10:00:00Z", "run_id": "aaaa11112222", "last_run_id": "bbbb33334444"},
    {"prior_id": UNTOUCHED_PRIOR, "claim": "The queue timeout is the reading bottleneck.",
     "confidence": 0.6, "status": "revised", "times_tested": 4, "formed_from": "",
     "last_tested_at": "2026-09-28T10:00:00Z", "run_id": "aaaa11112222", "last_run_id": "cccc55556666"},
]


class FakePriorReader:
    """Stands in for ``WorldviewReader``: answers only the bounded all-priors read."""

    def __init__(self, rows: list[dict] | None = None, *, raises: bool = False) -> None:
        self.rows = [dict(r) for r in (FIXTURE_PRIORS if rows is None else rows)]
        self.raises = raises
        self.queries: list[str] = []

    def query(self, cypher: str) -> list[dict]:
        from orion.curiosity.atlas import ATLAS_PRIORS_CYPHER
        from orion.curiosity.worldview import ALL_PRIORS_CYPHER, WorldviewUnavailable

        self.queries.append(cypher)
        if self.raises:
            raise WorldviewUnavailable("ConnectionError: refused")
        assert cypher in (ALL_PRIORS_CYPHER, ATLAS_PRIORS_CYPHER), cypher
        return [dict(r) for r in self.rows]


class CorrectionTable:
    """orion_day_correction semantics for every statement corrections.py and gather send."""

    def __init__(self, now: datetime | None = None) -> None:
        self.rows: list[dict] = []
        self.now = now or datetime(2026, 9, 29, 16, 0, tzinfo=timezone.utc)
        self.statements: list[str] = []

    def _tick(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now

    async def fetchval(self, sql: str, *args: Any):
        from orion.orion_day import corrections as oc

        self.statements.append(sql)
        assert sql == oc.INSERT_CORRECTION_SQL, sql[:60]
        cid, letter_date, ref, claim, correction, evidence, prior_ids, chat = args
        if any(r["correction_id"] == cid for r in self.rows):
            return None
        self.rows.append({
            "correction_id": cid, "letter_date": letter_date, "ref": ref, "claim": claim,
            "correction": correction, "evidence": evidence, "prior_ids": list(prior_ids),
            "chat_correlation_id": chat, "created_at": self._tick(), "offered_at": None,
            "offered_run_id": None, "priors_retested": None, "priors_moved": None, "outcome_stamped_at": None,
        })
        return cid

    async def fetch(self, sql: str, *args: Any):
        from orion.orion_day import corrections as oc

        self.statements.append(sql)
        if sql == oc.TAKE_CORRECTIONS_SQL:
            run_id, limit = args
            free = sorted((r for r in self.rows if r["offered_at"] is None),
                          key=lambda r: (r["created_at"], str(r["correction_id"])))[:limit]
            for r in free:
                r["offered_at"], r["offered_run_id"] = self._tick(), run_id
            return [dict(r) for r in reversed(free)]  # RETURNING has no order
        if sql == oc.OFFERED_FOR_RUN_SQL:
            return [dict(r) for r in self.rows if r["offered_run_id"] == args[0]]
        if sql == gather.CORRECTIONS_SQL:
            start, end = args
            return [dict(r) for r in self.rows if start <= r["created_at"] < end]
        raise AssertionError(sql[:60])

    async def execute(self, sql: str, *args: Any):
        from orion.orion_day import corrections as oc

        self.statements.append(sql)
        if sql == oc.RELEASE_CORRECTIONS_SQL:
            for r in self.rows:
                if r["offered_run_id"] == args[0] and r["outcome_stamped_at"] is None:
                    r["offered_at"] = r["offered_run_id"] = None
            return "UPDATE"
        if sql == oc.STAMP_OUTCOME_SQL:
            cid, run_id, retested, moved = args
            for r in self.rows:
                if str(r["correction_id"]) == str(cid) and r["offered_run_id"] == run_id:
                    r["priors_retested"], r["priors_moved"] = retested, moved
                    r["outcome_stamped_at"] = self._tick()
            return "UPDATE"
        raise AssertionError(sql[:60])


class TablePool:
    """``pool.acquire()`` over one connection-like object."""

    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def acquire(self):
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _cm():
            yield self.conn

        return _cm()
