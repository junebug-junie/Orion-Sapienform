from __future__ import annotations

from orion.substrate.appraisal import REPAIR_PRESSURE_CONTRACT_METADATA_KEY


def test_metadata_key_is_stable_string() -> None:
    assert REPAIR_PRESSURE_CONTRACT_METADATA_KEY == "repair_pressure_contract"
