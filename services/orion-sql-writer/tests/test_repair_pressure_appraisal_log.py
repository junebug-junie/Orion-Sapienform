"""Shape checks for the repair_pressure appraisal durable-log write path (no
Postgres required).

Built after a live-data check found the relational metacog trigger's evidence
source (repair_pressure_v2) had zero Postgres persistence anywhere -- only
ephemeral docker logs, wiped on every container restart, one real sample ever
observed. This is that persistence: every appraisal published on
`orion:repair_pressure:appraisal` gets logged here, gated or not, so the real
level/confidence distribution becomes checkable against a real window instead
of a single ephemeral log line.
"""
from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.repair_pressure_appraisal import RepairPressureAppraisalV1  # noqa: E402

from app.models.repair_pressure_appraisal import RepairPressureAppraisalLog  # noqa: E402
from app.worker import MODEL_MAP  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402

CHANNEL = "orion:repair_pressure:appraisal"


def _appraisal(**overrides) -> RepairPressureAppraisalV1:
    defaults = dict(
        correlation_id="corr-123",
        level=0.087,
        level_label="LOW",
        confidence=0.0,
        evidence=[],
    )
    defaults.update(overrides)
    return RepairPressureAppraisalV1(**defaults)


def _row_dict(appraisal: RepairPressureAppraisalV1) -> dict:
    """Mirror `_write_row`'s generic column-name filter."""
    data = appraisal.model_dump(mode="json")
    mapper = inspect(RepairPressureAppraisalLog)
    valid_keys = {attr.key for attr in mapper.attrs}
    return {k: v for k, v in data.items() if k in valid_keys}


def test_default_route_map_points_appraisal_at_its_sql_model() -> None:
    assert DEFAULT_ROUTE_MAP.get("repair_pressure.appraisal.v1") == "RepairPressureAppraisalLog"


def test_model_map_registers_appraisal_sql_with_its_schema() -> None:
    assert MODEL_MAP["RepairPressureAppraisalLog"] == (RepairPressureAppraisalLog, RepairPressureAppraisalV1)


def test_channel_is_in_settings_default_subscribe_list() -> None:
    default_channels = Settings.model_fields["sql_writer_subscribe_channels"].default
    assert CHANNEL in default_channels


def test_channel_is_in_env_example_subscribe_channels() -> None:
    env_example = (SERVICE_ROOT / ".env_example").read_text()
    for line in env_example.splitlines():
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS="):
            assert CHANNEL in line
            return
    raise AssertionError("SQL_WRITER_SUBSCRIBE_CHANNELS not found in .env_example")


def test_env_example_route_map_json_includes_appraisal() -> None:
    env_example = (SERVICE_ROOT / ".env_example").read_text()
    for line in env_example.splitlines():
        if line.startswith("SQL_WRITER_ROUTE_MAP_JSON="):
            assert '"repair_pressure.appraisal.v1":"RepairPressureAppraisalLog"' in line
            return
    raise AssertionError("SQL_WRITER_ROUTE_MAP_JSON not found in .env_example")


def test_table_name_is_dedicated_not_a_patch_onto_chat_history() -> None:
    """Standalone insert-only table -- avoids the row-creation-timing race a
    chat_history_log patch would hit, since repair_pressure computes
    pre-turn, before that row necessarily exists."""
    assert RepairPressureAppraisalLog.__tablename__ == "repair_pressure_appraisal_log"


def test_row_dict_constructs_appraisal_sql_without_raising() -> None:
    appraisal = _appraisal()
    row = RepairPressureAppraisalLog(**_row_dict(appraisal))
    assert row.correlation_id == "corr-123"
    assert row.level == 0.087
    assert row.confidence == 0.0


def test_write_and_read_roundtrip_sqlite() -> None:
    engine = create_engine("sqlite://")
    RepairPressureAppraisalLog.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)

    appraisal = _appraisal(
        level=0.9,
        level_label="HIGH",
        confidence=0.9,
        evidence=[{"evidence_kind": "trust_rupture", "score": 0.8, "confidence": 0.9}],
        behavior_applied="acknowledge_and_repair",
    )
    row = RepairPressureAppraisalLog(**_row_dict(appraisal))

    sess = Session()
    try:
        sess.add(row)
        sess.commit()
        fetched = sess.get(RepairPressureAppraisalLog, appraisal.id)
        assert fetched is not None
        assert fetched.level == 0.9
        assert fetched.confidence == 0.9
        assert fetched.evidence[0]["evidence_kind"] == "trust_rupture"
        assert fetched.behavior_applied == "acknowledge_and_repair"
    finally:
        sess.close()


def test_gated_and_ungated_appraisals_both_persist() -> None:
    """The whole point of this table: log every appraisal, not just the ones
    that cross the relational trigger's confidence floor -- otherwise the
    real distribution stays invisible the same way it was before this table
    existed."""
    engine = create_engine("sqlite://")
    RepairPressureAppraisalLog.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)
    sess = Session()
    try:
        below_floor = _appraisal(correlation_id="below", level=0.087, confidence=0.0)
        above_floor = _appraisal(correlation_id="above", level=0.9, confidence=0.9)
        sess.add(RepairPressureAppraisalLog(**_row_dict(below_floor)))
        sess.add(RepairPressureAppraisalLog(**_row_dict(above_floor)))
        sess.commit()
        assert sess.query(RepairPressureAppraisalLog).count() == 2
    finally:
        sess.close()
