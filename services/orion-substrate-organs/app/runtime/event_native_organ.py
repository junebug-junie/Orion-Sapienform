from __future__ import annotations

from orion.schemas.biometrics_projection import (
    ActiveNodePressureProjectionV1,
    NodeBiometricsProjectionV1,
)
from orion.schemas.grammar import GrammarEventV1
from orion.schemas.organ_emission import OrganEmissionV1

from orion.substrate.biometrics_loop.emission_validator import validate_organ_emission
from orion.substrate.biometrics_loop.pressure_organ import invoke_biometrics_pressure


def run_biometrics_pressure_organ(
    *,
    trigger_event: GrammarEventV1,
    node_bio: NodeBiometricsProjectionV1,
    active_pressure: ActiveNodePressureProjectionV1,
    catalog,
    stale_after_sec: int = 180,
    min_confidence: float = 0.60,
) -> OrganEmissionV1:
    emission = invoke_biometrics_pressure(
        trigger_event=trigger_event,
        node_bio=node_bio,
        active_pressure=active_pressure,
        catalog=catalog,
        stale_after_sec=stale_after_sec,
        min_confidence=min_confidence,
    )
    validate_organ_emission(emission)
    return emission
