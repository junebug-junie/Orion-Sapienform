"""The `orion_lived_answers` felt-state hydration (orion/substrate/felt_state_reader.py)."""

from __future__ import annotations

from datetime import datetime, timezone

from orion.substrate.felt_state_reader import LIVED_ANSWERS_CTX_KEY, SubstrateFeltStateReader


class _Conn:
    def __init__(self, sink: list, rows):
        self._sink = sink
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params):
        self._sink.append((str(query), dict(params)))

        class _R:
            def __init__(self, rows):
                self._rows = rows

            def mappings(self):
                return self

            def all(self):
                return self._rows

        return _R(self._rows)


class _Engine:
    def __init__(self, rows):
        self.queries: list = []
        self._rows = rows

    def connect(self):
        return _Conn(self.queries, self._rows)


def _reader(rows) -> SubstrateFeltStateReader:
    reader = SubstrateFeltStateReader(enabled=False, database_url="postgresql://x", max_age_sec=600)
    reader._enabled = True
    reader._engine = _Engine(rows)
    return reader


def test_hydrate_lived_answers_latest_per_pinned_concept_id() -> None:
    rows = [
        {
            "concept_id": "self:lived:lived.who_matters",
            "content": "Juniper matters most.",
            "evidence_refs": ["chat_message:1"],
            "created_at": datetime(2026, 9, 18, tzinfo=timezone.utc),
        },
        {
            "concept_id": "self:lived:lived.who_matters",
            "content": "stale",
            "evidence_refs": [],
            "created_at": datetime(2026, 9, 1, tzinfo=timezone.utc),
        },
    ]
    reader = _reader(rows)
    ctx: dict = {}
    reader.hydrate(ctx, lanes=(LIVED_ANSWERS_CTX_KEY,))
    sql = next(q for q, _ in reader._engine.queries if "self_concept_history" in q)
    assert "concept_id IN" in sql
    assert "curiosity_self_inquiry" in sql
    answers = ctx[LIVED_ANSWERS_CTX_KEY]
    assert len(answers) == 1
    assert answers[0]["question_id"] == "lived.who_matters"
    assert answers[0]["content"] == "Juniper matters most."


def test_lived_answers_miss_does_not_set_ctx_key() -> None:
    reader = _reader([])
    ctx: dict = {}
    reader.hydrate(ctx, lanes=(LIVED_ANSWERS_CTX_KEY,))
    assert LIVED_ANSWERS_CTX_KEY not in ctx
