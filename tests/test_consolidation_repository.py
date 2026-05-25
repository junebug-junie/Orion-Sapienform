from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from orion.consolidation import repository as consolidation_repository
from orion.schemas.consolidation_frame import (
    ConsolidationFrameV1,
    ExpectationV1,
    MotifObservationV1,
    SchemaCandidateV1,
    SparseTensorSliceV1,
)

NOW = datetime(2026, 5, 25, 15, 30, tzinfo=timezone.utc)
START = datetime(2026, 5, 25, 14, 30, tzinfo=timezone.utc)
MOTIF_ID = "motif:loaded_but_reliable:consolidation_policy.v1"


def _motif() -> MotifObservationV1:
    return MotifObservationV1(
        motif_id=MOTIF_ID,
        motif_kind="self_state_pattern",
        label="loaded_but_reliable",
        recurrence_count=3,
        support_score=0.6,
        confidence_score=0.7,
        evidence_frame_ids=["self.state:s1"],
    )


def _frame_payload() -> dict:
    return ConsolidationFrameV1(
        frame_id=(
            "consolidation.frame:2026-05-25T14:30:00+00:00:"
            "2026-05-25T15:30:00+00:00:consolidation_policy.v1"
        ),
        generated_at=NOW,
        window_start=START,
        window_end=NOW,
        motif_observations=[_motif()],
    ).model_dump(mode="json")


def _expectation_payload() -> dict:
    return ExpectationV1(
        expectation_id=f"expectation:{MOTIF_ID}:reliability_clear",
        trigger_motif_id=MOTIF_ID,
        expected_outcome_kind="reliability_clear",
        confidence_score=0.7,
        support_count=3,
    ).model_dump(mode="json")


def _schema_candidate_payload() -> dict:
    return SchemaCandidateV1(
        schema_candidate_id="schema.candidate:habit:loaded_but_reliable",
        candidate_kind="habit_candidate",
        label="loaded_but_reliable",
        source_motif_ids=[MOTIF_ID],
        support_score=0.6,
        confidence_score=0.7,
    ).model_dump(mode="json")


def _tensor_slice_payload() -> dict:
    return SparseTensorSliceV1(
        tensor_id=(
            "tensor:field_attention_self:2026-05-25T14:30:00+00:00:"
            "2026-05-25T15:30:00+00:00"
        ),
        tensor_kind="field_attention_self",
        axes=["time_bucket", "self_condition"],
        coordinates=[{"time_bucket": "2026-05-25T15:00:00+00:00", "self_condition": "loaded"}],
        values=[0.8],
    ).model_dump(mode="json")


def _fake_engine(rows_by_query: dict[str, list[dict]]):
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    executed_sql: list[str] = []

    def execute_side_effect(stmt, params=None):
        sql = str(stmt)
        executed_sql.append(sql)
        result = MagicMock()
        if "substrate_consolidation_frames" in sql:
            rows = rows_by_query.get("consolidation_frames", [])
            if "LIMIT 1" in sql:
                result.mappings.return_value.first.return_value = rows[0] if rows else None
            else:
                result.mappings.return_value = rows
        elif "substrate_expectations" in sql:
            result.mappings.return_value = rows_by_query.get("expectations", [])
        elif "substrate_schema_candidates" in sql:
            result.mappings.return_value = rows_by_query.get("schema_candidates", [])
        elif "substrate_tensor_slices" in sql:
            result.mappings.return_value = rows_by_query.get("tensor_slices", [])
        else:
            result.mappings.return_value = []
        return result

    conn.execute.side_effect = execute_side_effect
    return fake_engine, executed_sql


def test_load_recent_motifs_from_latest_frame() -> None:
    fake_engine, executed_sql = _fake_engine(
        {"consolidation_frames": [{"consolidation_frame_json": _frame_payload()}]}
    )

    motifs = consolidation_repository.load_recent_motifs(
        "postgresql://test:test@localhost/test",
        limit=10,
        engine=fake_engine,
    )

    assert len(motifs) == 1
    assert motifs[0].motif_id == MOTIF_ID
    assert all("INSERT" not in sql.upper() for sql in executed_sql)


def test_load_recent_motifs_empty_when_no_frame() -> None:
    fake_engine, executed_sql = _fake_engine({"consolidation_frames": []})

    motifs = consolidation_repository.load_recent_motifs(
        "postgresql://test:test@localhost/test",
        limit=10,
        engine=fake_engine,
    )

    assert motifs == []
    assert all("INSERT" not in sql.upper() for sql in executed_sql)


def test_load_expectations_for_motif() -> None:
    fake_engine, executed_sql = _fake_engine(
        {"expectations": [{"expectation_json": _expectation_payload()}]}
    )

    expectations = consolidation_repository.load_expectations_for_motif(
        "postgresql://test:test@localhost/test",
        MOTIF_ID,
        engine=fake_engine,
    )

    assert len(expectations) == 1
    assert expectations[0].trigger_motif_id == MOTIF_ID
    assert all("INSERT" not in sql.upper() for sql in executed_sql)


def test_load_schema_candidates() -> None:
    fake_engine, executed_sql = _fake_engine(
        {"schema_candidates": [{"schema_candidate_json": _schema_candidate_payload()}]}
    )

    candidates = consolidation_repository.load_schema_candidates(
        "postgresql://test:test@localhost/test",
        limit=5,
        engine=fake_engine,
    )

    assert len(candidates) == 1
    assert candidates[0].candidate_kind == "habit_candidate"
    assert all("INSERT" not in sql.upper() for sql in executed_sql)


def test_load_latest_tensor_slices() -> None:
    fake_engine, executed_sql = _fake_engine(
        {"tensor_slices": [{"tensor_json": _tensor_slice_payload()}]}
    )

    slices = consolidation_repository.load_latest_tensor_slices(
        "postgresql://test:test@localhost/test",
        limit=5,
        engine=fake_engine,
    )

    assert len(slices) == 1
    assert slices[0].tensor_kind == "field_attention_self"
    assert all("INSERT" not in sql.upper() for sql in executed_sql)
