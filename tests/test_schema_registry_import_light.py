from __future__ import annotations

import importlib
import sys


def test_schema_registry_import_does_not_load_substrate() -> None:
    """Registry must not eagerly import orion.substrate (Fuseki/SPARQL store stack)."""
    sys.modules.pop("orion.schemas.registry", None)
    sys.modules.pop("orion.schemas.pre_turn_appraisal", None)
    sys.modules.pop("orion.substrate", None)

    importlib.import_module("orion.schemas.registry")

    assert "orion.substrate" not in sys.modules
