from __future__ import annotations

from pathlib import Path


def test_endogenous_sql_bus_channels_are_cataloged() -> None:
    channels_yaml = Path(__file__).resolve().parents[1] / "orion" / "bus" / "channels.yaml"
    text = channels_yaml.read_text(encoding="utf-8")
    assert 'name: "orion:endogenous:runtime:record"' in text
    assert 'schema_id: "EndogenousRuntimeExecutionRecordV1"' in text
    assert 'name: "orion:endogenous:runtime:audit"' in text
    assert 'schema_id: "EndogenousRuntimeAuditV1"' in text
    assert 'name: "orion:calibration:profile:audit"' in text
    assert 'schema_id: "CalibrationProfileAuditV1"' in text
