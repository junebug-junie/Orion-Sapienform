from __future__ import annotations

"""orion/metacog/service.py

Row-field assembly for MetacogEntryV1 (orion/schemas/metacog_entry.py).

As of 2026-09-24 (spec: docs/superpowers/specs/2026-09-24-metacog-capture-and-
transport-ewma-baseline-design.md, section B) severity, causal_density and
touches are derived from the TRIGGER's own upstream evidence via
``orion.metacog.evidence_map`` -- never from the metacog pipeline's own step
log or from the writing LLM's token logprobs.

Retired here, not kept as fallbacks:
- ``compute_severity(llm_uncertainty, non_ok_step_count)``: scored how sure the
  writing model felt about its own paraphrase. Live 2026-09-24 it ranked
  transport rows backwards (nominal median p95 15.8 s > critical 10.2 s).
- ``compute_causal_density(MetacogRealState)``: a blend of repair_pressure,
  substrate eventfulness (<= 0.25) and turn_effect; only ever produced 0 or
  0.25 in 114k rows.
- ``compute_touches(MetacogRealState)``: named which *global* state fields
  happened to be populated, not what the event touched.
"""

import logging

from orion.metacog.evidence_map import CRITICAL_FLOOR, EvidenceMapping, map_trigger
from orion.schemas.metacog_entry import MetacogCausalDensity, MetacogProvenance

logger = logging.getLogger("orion.metacog.service")

# Kept at the same numeric value as before (0.6) and now tied to the
# evidence_map band edge, so "causally dense" == "critical event".
IS_CAUSALLY_DENSE_THRESHOLD = CRITICAL_FLOOR

PIPELINE_STEP_LOG_MAX_LINES = 20
PIPELINE_STEP_LOG_MAX_CHARS = 200


def compute_causal_density(mapping: EvidenceMapping) -> MetacogCausalDensity:
    """causal_density.score is the event's own normalized magnitude."""
    return MetacogCausalDensity(
        label=mapping.density_label,
        score=mapping.magnitude,
        rationale=mapping.density_rationale,
    )


def compute_provenance(
    *,
    trigger_kind: str,
    touches: list[str],
    pipeline_steps: list[str] | None = None,
) -> MetacogProvenance:
    """`source` names what fired the entry; `impacts` are the services /
    channels / artifacts the event's own upstream names (evidence_map
    touches); `pipeline_steps` is the metacog pipeline's own step log, moved
    here out of what_changed.evidence -- it describes how the row was made,
    not what happened."""
    steps = [str(line)[:PIPELINE_STEP_LOG_MAX_CHARS] for line in (pipeline_steps or [])]
    return MetacogProvenance(
        source=f"cortex_exec.metacog_pipeline.{trigger_kind}",
        produces="metacog_entry",
        impacts=list(touches),
        pipeline_steps=steps[-PIPELINE_STEP_LOG_MAX_LINES:],
    )


__all__ = [
    "IS_CAUSALLY_DENSE_THRESHOLD",
    "compute_causal_density",
    "compute_provenance",
    "map_trigger",
]
