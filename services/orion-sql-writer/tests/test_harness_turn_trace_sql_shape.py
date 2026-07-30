"""Shape checks for the HarnessTurnTraceSQL write path (no Postgres required).

Mirrors test_spark_telemetry_idempotency.py's approach: the on_conflict_do_update
statement is Postgres-dialect-specific, so these mock the session and either
inspect the constructed statement or compile it against a real postgres dialect,
rather than executing against sqlite (which can't compile ON CONFLICT ... DO
UPDATE the same way).

Ground truth: four independent bus kinds (harness.run.v1,
harness.verdict.molecule.v1, harness.turn.outcome.v1,
harness.post_turn.closure.v1) already PUBLISH unredacted on the bus today
(orion/harness/finalize.py) but were never durably persisted. Each fills in
one JSONB column of the same correlation_id-keyed row -- see
app/harness_turn_trace_persist.py's schema_version -> column dispatch.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

from sqlalchemy import create_engine

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

from app.harness_turn_trace_persist import (  # noqa: E402
    column_for_schema_version,
    upsert_harness_turn_trace,
)
from app.models.harness_turn_trace import HarnessTurnTraceSQL  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP  # noqa: E402
from app.worker import MODEL_MAP  # noqa: E402

RUN_ARTIFACT_PAYLOAD = {
    "schema_version": "harness.run.v1",
    "correlation_id": "corr-1",
    "final_text": "the reply",
    "draft_text": "the draft",
    "substrate_appraisal": {"surprise_level": 0.2},
    "reflection": {"alignment_verdict": "aligned"},
    "finalize_ran": True,
}

VERDICT_PAYLOAD = {
    "schema_version": "harness.verdict.molecule.v1",
    "correlation_id": "corr-1",
    "reflection": {"alignment_verdict": "aligned"},
    "cortex_trace_id": "trace-1",
}

OUTCOME_PAYLOAD = {
    "schema_version": "harness.turn.outcome.v1",
    "correlation_id": "corr-1",
    "final_hash": "abc123",
    "surprise_resolved": True,
}

CLOSURE_PAYLOAD = {
    "schema_version": "harness.post_turn.closure.v1",
    "correlation_id": "corr-1",
    "surprise_unresolved": False,
}


def test_column_for_schema_version_covers_all_four_kinds() -> None:
    assert column_for_schema_version("harness.run.v1") == "run_artifact"
    assert column_for_schema_version("harness.verdict.molecule.v1") == "verdict_molecule"
    assert column_for_schema_version("harness.turn.outcome.v1") == "outcome_molecule"
    assert column_for_schema_version("harness.post_turn.closure.v1") == "closure"
    assert column_for_schema_version("some.unknown.kind") is None
    assert column_for_schema_version(None) is None


def test_upsert_without_correlation_id_is_a_noop() -> None:
    sess = MagicMock()
    assert upsert_harness_turn_trace(sess, {"schema_version": "harness.run.v1"}) is False
    sess.execute.assert_not_called()
    sess.commit.assert_not_called()


def test_upsert_without_recognized_schema_version_is_a_noop() -> None:
    sess = MagicMock()
    payload = {"correlation_id": "corr-1", "schema_version": "unknown.kind.v1"}
    assert upsert_harness_turn_trace(sess, payload) is False
    sess.execute.assert_not_called()


def test_upsert_uses_on_conflict_and_only_touches_its_own_column() -> None:
    sess = MagicMock()
    assert upsert_harness_turn_trace(sess, RUN_ARTIFACT_PAYLOAD) is True

    sess.execute.assert_called_once()
    sess.commit.assert_called_once()
    stmt = sess.execute.call_args[0][0]
    compiled = str(stmt.compile(dialect=create_engine("postgresql://").dialect))
    assert "harness_turn_trace" in compiled
    assert "ON CONFLICT" in compiled.upper()
    # Only run_artifact + updated_at are in the SET clause -- verdict_molecule/
    # outcome_molecule/closure from a prior write must survive untouched.
    assert "run_artifact" in compiled
    set_clause = compiled.upper().split("DO UPDATE SET")[1]
    assert "VERDICT_MOLECULE" not in set_clause
    assert "OUTCOME_MOLECULE" not in set_clause
    assert "CLOSURE" not in set_clause.split("UPDATED_AT")[0].replace("CLOSURE_", "")


def test_each_source_kind_updates_its_own_column() -> None:
    for payload, expected_column in (
        (RUN_ARTIFACT_PAYLOAD, "run_artifact"),
        (VERDICT_PAYLOAD, "verdict_molecule"),
        (OUTCOME_PAYLOAD, "outcome_molecule"),
        (CLOSURE_PAYLOAD, "closure"),
    ):
        sess = MagicMock()
        upsert_harness_turn_trace(sess, payload)
        stmt = sess.execute.call_args[0][0]
        compiled = str(stmt.compile(dialect=create_engine("postgresql://").dialect))
        assert expected_column in compiled


def test_default_route_map_points_all_four_kinds_at_harness_turn_trace_sql() -> None:
    for kind in (
        "harness.run.v1",
        "harness.verdict.molecule.v1",
        "harness.turn.outcome.v1",
        "harness.post_turn.closure.v1",
    ):
        assert DEFAULT_ROUTE_MAP.get(kind) == "HarnessTurnTraceSQL"


def test_model_map_registers_harness_turn_trace_sql_with_no_fixed_schema() -> None:
    # schema_cls is None: the column is chosen at persist time from the
    # payload's own schema_version, not a single fixed pydantic model here --
    # see app/harness_turn_trace_persist.py.
    assert MODEL_MAP["HarnessTurnTraceSQL"] == (HarnessTurnTraceSQL, None)


def test_channels_are_in_settings_default_subscribe_list() -> None:
    from app.settings import Settings

    default_channels = Settings.model_fields["sql_writer_subscribe_channels"].default
    for channel in (
        "orion:harness:run:artifact",
        "orion:harness:verdict:artifact",
        "orion:substrate:turn_outcome",
        "orion:substrate:post_turn_closure",
    ):
        assert channel in default_channels


def test_env_example_route_map_json_includes_all_four_kinds() -> None:
    env_example = (SQL_WRITER_ROOT / ".env_example").read_text()
    for line in env_example.splitlines():
        if line.startswith("SQL_WRITER_ROUTE_MAP_JSON="):
            for kind in (
                "harness.run.v1",
                "harness.verdict.molecule.v1",
                "harness.turn.outcome.v1",
                "harness.post_turn.closure.v1",
            ):
                assert f'"{kind}":"HarnessTurnTraceSQL"' in line
            return
    raise AssertionError("SQL_WRITER_ROUTE_MAP_JSON not found in .env_example")


def test_generic_dispatch_does_not_contaminate_jsonb_with_envelope_extras() -> None:
    """Regression: worker._handle_envelope_body's generic path merges
    extra_sql_fields (node, source_message_id, envelope correlation_id) into
    every write via _write()'s data.update(extra_fields) -- none of those are
    real fields on any of the four harness molecules, so letting them through
    would silently corrupt the "this JSONB blob is exactly what was
    published" guarantee the whole Turn Trace feature depends on. The
    HarnessTurnTraceSQL branch must pass extra_fields={} explicitly, the same
    way the Dream/SparkTelemetrySQL/EndogenousRuntimeRecordSQL special cases
    already do a few lines above it.
    """
    worker_src = (SQL_WRITER_ROOT / "app" / "worker.py").read_text(encoding="utf-8")
    assert "elif sql_model is HarnessTurnTraceSQL:" in worker_src
    branch = worker_src.split("elif sql_model is HarnessTurnTraceSQL:", 1)[1].split("elif ", 1)[0]
    assert "_write(sql_model, None, data_to_process, {}, kind=env.kind)" in branch


def test_env_example_subscribe_channels_include_all_four_channels() -> None:
    env_example = (SQL_WRITER_ROOT / ".env_example").read_text()
    for line in env_example.splitlines():
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS="):
            for channel in (
                "orion:harness:run:artifact",
                "orion:harness:verdict:artifact",
                "orion:substrate:turn_outcome",
                "orion:substrate:post_turn_closure",
            ):
                assert channel in line
            return
    raise AssertionError("SQL_WRITER_SUBSCRIBE_CHANNELS not found in .env_example")
