"""Bridge a RepairPressureAppraisalV1 into an OrionSignalV1.

The bridge owns the mapping from appraisal dimensions to canonical signal
dimensions registered on the graph_cognition organ. It is pure: no I/O,
no clock skew beyond `emitted_at`.
"""

from __future__ import annotations

from datetime import datetime, timezone

from orion.signals.models import OrganClass, OrionSignalV1
from orion.signals.signal_ids import make_signal_id

from .models import RepairPressureAppraisalV1


ORGAN_ID = "graph_cognition"
SIGNAL_KIND = "repair_pressure"
TTL_MS = 15_000


_DIMENSION_KEYS = (
    "level",
    "specificity_demand",
    "trust_rupture",
    "coherence_gap",
    "repetition_failure",
    "operational_block",
    "explicit_repair_command",
    "assistant_accountability_demand",
    "confidence",
)


def repair_appraisal_to_signal(
    appraisal: RepairPressureAppraisalV1,
    *,
    observed_at: datetime | None = None,
) -> OrionSignalV1:
    """Project the appraisal onto the canonical graph_cognition/repair_pressure signal."""

    dimensions: dict[str, float] = {
        key: appraisal.dimensions[key] for key in _DIMENSION_KEYS
    }

    return OrionSignalV1(
        signal_id=make_signal_id(ORGAN_ID, appraisal.appraisal_id),
        organ_id=ORGAN_ID,
        organ_class=OrganClass.endogenous,
        signal_kind=SIGNAL_KIND,
        dimensions=dimensions,
        causal_parents=list(appraisal.causal_molecule_ids),
        source_event_id=appraisal.appraisal_id,
        observed_at=observed_at or appraisal.created_at,
        emitted_at=datetime.now(timezone.utc),
        ttl_ms=TTL_MS,
        summary=appraisal.summary,
        notes=list(appraisal.notes[:5]),
    )
