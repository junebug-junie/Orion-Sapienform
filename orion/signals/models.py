from datetime import datetime
from enum import Enum
from pydantic import BaseModel, ConfigDict, field_validator
from typing import List, Optional, Dict

class OrganClass(str, Enum):
    exogenous = "exogenous"   # root signal: hardware, user input, environment
    endogenous = "endogenous"  # derived from other organs' signals
    hybrid = "hybrid"      # partially derived, partially independent

class OrionSignalV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Identity
    signal_id: str
    # Deterministic when source_event_id is present:
    #   hashlib.sha256(f"{organ_id}:{source_event_id}".encode()).hexdigest()[:16]
    # If source_event_id is None, fall back to str(uuid4()) — not deterministic,
    # but avoids collision when the source event carries no stable ID.

    organ_id: str          # e.g. "biometrics", "collapse_mirror", "recall"
    organ_class: OrganClass

    # Signal type
    signal_kind: str
    # Organ-specific. Canonical values defined per organ in the registry.
    # Examples: "gpu_load", "cognitive_collapse", "recall_result", "chat_stance"

    # Dimensional representation — the hardened signal
    dimensions: Dict[str, float]
    # Keys follow conventions across organs:
    #   level:       current intensity              [0.0, 1.0]
    #   trend:       direction of change            [-1.0, 1.0]
    #   volatility:  rate of change                 [0.0, 1.0]
    #   valence:     positive/negative charge       [-1.0, 1.0]  (where applicable)
    #   confidence:  signal reliability             [0.0, 1.0]
    #   arousal:     activation level               [0.0, 1.0]  (spark/affect signals)
    #   coherence:   internal consistency           [0.0, 1.0]
    #   novelty:     deviation from baseline        [0.0, 1.0]
    #   salience:    attentional weight             [0.0, 1.0]
    # Per-drive pressure keys (autonomy organ only):
    #   pressure_coherence, pressure_continuity, pressure_relational,
    #   pressure_autonomy, pressure_capability, pressure_predictive   [0.0, 1.0]

    # Causal provenance
    causal_parents: List[str] = []
    # signal_ids of OrionSignalV1 records this was derived from.
    # Populated by adapters using the gateway's prior_signals window.
    # Empty for exogenous signals.

    source_event_id: Optional[str] = None
    # Original bus event ID / correlation_id from the source organ.
    # Preserved as otel span attribute for migration compatibility.

    # OTEL trace context
    otel_trace_id:       Optional[str] = None
    otel_span_id:        Optional[str] = None
    otel_parent_span_id: Optional[str] = None
    # otel_trace_id propagates across the causal chain:
    # exogenous signals start a new trace; endogenous signals inherit
    # the trace_id of their causal_parents (first parent's trace_id wins
    # if parents disagree — rare, logged as a note).

    # Temporal
    observed_at: datetime   # when the source event occurred
    emitted_at:  datetime   # when the gateway produced this signal
    ttl_ms: int = 15_000

    # Human-readable audit
    summary: Optional[str] = None
    notes:   List[str] = []     # max 5; gateway may append adapter warnings

    @field_validator("notes", mode="after")
    @classmethod
    def _cap_notes(cls, v: List[str]) -> List[str]:
        return list(v)[:5]

class OrionOrganRegistryEntry(BaseModel):
    organ_id: str
    organ_class: OrganClass
    service: str                      # e.g. "orion-biometrics"
    signal_kinds: List[str]           # canonical signal_kind values this organ emits
    canonical_dimensions: List[str]   # dimension keys this organ populates
    causal_parent_organs: List[str]   # organ_ids structurally upstream in the causal DAG
    bus_channels: List[str]           # bus channels the gateway subscribes to for this organ
    notes: List[str] = []