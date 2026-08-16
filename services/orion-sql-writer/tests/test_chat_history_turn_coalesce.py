from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_coalesce_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))
SPEC.loader.exec_module(worker)


def _compiled_conflict_sql(values: dict, *, incoming_wins: bool = True) -> str:
    """Render the real ON CONFLICT clause upsert_chat_history_row would emit."""
    from sqlalchemy.dialects import postgresql

    captured = {}

    class _CapturingSession:
        def execute(self, stmt):
            captured["stmt"] = stmt

    worker.upsert_chat_history_row(
        _CapturingSession(), values, incoming_wins=incoming_wins
    )
    # No literal_binds: it cannot render a JSONB value, and the '' argument to
    # nullif compiles to a bind param either way, so assertions match on
    # `nullif(excluded.<col>, ` rather than on the literal.
    return str(captured["stmt"].compile(dialect=postgresql.dialect())).lower()


def test_upsert_merges_on_conflict_instead_of_skipping():
    """The bug: a same-primary-key collision was treated as 'already have it'.

    chat_history_log is assembled from three events that each carry a different
    subset of columns, so DO NOTHING / skip loses real content. The statement
    must be DO UPDATE.
    """
    sql = _compiled_conflict_sql({"id": "abc", "correlation_id": "abc", "response": "reply"})

    assert "on conflict (id)" in sql
    assert "do update set" in sql
    assert "do nothing" not in sql


def test_a_non_empty_incoming_text_value_fills_but_an_empty_one_never_clobbers():
    """`''` counts as absent for text: prompt/response were seeded as empty
    strings, so a later event carrying the real text must still win -- while an
    empty incoming value must leave stored text alone."""
    sql = _compiled_conflict_sql({"id": "abc", "prompt": "hi", "response": ""})

    for col in ("prompt", "response"):
        assert f"{col} = coalesce(nullif(excluded.{col}, " in sql
        assert f"), chat_history_log.{col})" in sql


def test_jsonb_and_scalar_columns_coalesce_without_the_empty_string_test():
    """coalesce(spark_meta, '') would not typecheck against JSONB."""
    sql = _compiled_conflict_sql({"id": "abc", "spark_meta": {"mode": "orion"}})

    assert "coalesce(excluded.spark_meta, chat_history_log.spark_meta)" in sql
    assert "nullif(excluded.spark_meta" not in sql


def test_the_conflict_target_is_not_also_an_update_target():
    sql = _compiled_conflict_sql({"id": "abc", "correlation_id": "abc"})

    set_clause = sql.split("do update set", 1)[1]
    assert "correlation_id = coalesce" in set_clause
    # `id` is the conflict target; assigning it in SET is a no-op at best.
    assert " id = " not in set_clause


def test_upsert_refuses_a_row_with_no_id():
    """Without an id there is no conflict target, so the statement would insert
    a duplicate turn on every retry instead of merging."""
    import pytest

    with pytest.raises(ValueError):
        worker.upsert_chat_history_row(SimpleNamespace(execute=lambda s: None), {"prompt": "hi"})


def test_merge_spark_meta_telemetry_does_not_clobber_classify_novelty():
    existing = {
        "turn_change_appraisal": {"turn_change_status": "ok", "novelty_score": 0.91},
        "novelty": 0.91,
        "turn_effect": {"turn": {"novelty": 0.91}},
    }
    telemetry = {"novelty": 0.0, "phi": 0.8}
    merged = worker._merge_spark_meta(existing, telemetry, source="telemetry")
    assert merged["novelty"] == 0.91
    assert merged["turn_effect"] == existing["turn_effect"]


class TestMergePolicyPerProducer:
    """The turn event and the message events deliberately merge differently.

    Collapsing them into one policy was a real regression: the assistant message
    is published AFTER the turn on the WS path, so a last-writer-wins rule let a
    message clobber the turn's canonical source/client_meta/memory_* values.
    """

    def test_turn_path_lets_a_non_empty_incoming_value_win(self):
        sql = _compiled_conflict_sql(
            {"id": "abc", "source": "hub_orion"}, incoming_wins=True
        )

        assert "source = coalesce(nullif(excluded.source, " in sql
        assert "), chat_history_log.source)" in sql

    def test_message_path_keeps_what_is_already_stored(self):
        sql = _compiled_conflict_sql(
            {"id": "abc", "client_meta": {"origin": "x"}}, incoming_wins=False
        )

        # existing first -> a stored value is never overwritten by a message
        assert "client_meta = coalesce(chat_history_log.client_meta, excluded.client_meta)" in sql

    def test_message_path_still_fills_an_empty_column(self):
        """Fill-only must still FILL -- nullif keeps '' from counting as stored."""
        sql = _compiled_conflict_sql({"id": "abc", "prompt": "hi"}, incoming_wins=False)

        assert "prompt = coalesce(chat_history_log.prompt, nullif(excluded.prompt, " in sql

    def test_the_two_policies_are_actually_different(self):
        values = {"id": "abc", "response": "r"}

        turn = _compiled_conflict_sql(values, incoming_wins=True)
        message = _compiled_conflict_sql(values, incoming_wins=False)

        assert turn != message

    def test_turn_path_is_the_default(self):
        """_write_row calls it without the keyword; the turn is authoritative."""
        values = {"id": "abc", "source": "hub_orion"}

        assert _compiled_conflict_sql(values) == _compiled_conflict_sql(
            values, incoming_wins=True
        )
