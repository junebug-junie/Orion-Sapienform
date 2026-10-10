"""cortex-exec's chat-stance curiosity reader vs the 2026-10-10 producer fields.

substrate-runtime now stores seeds with boundary_edge_refs / neighbor_node_refs /
projection_endpoint_node_refs (when non-empty) and unscored link-accepted seeds. The
reader validates with an extra="forbid" model; before this patch one unknown key dropped
the whole signal from chat. Deploy order is still cortex-exec first.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1
from orion.substrate.curiosity_seed_neighborhood import attach_seed_neighborhoods
from orion.substrate.evals.neighborhood_fixture import PROVENANCE, TEMPORAL, concept, edge, graph
from orion.core.schemas.cognitive_substrate import AssertionNodeV1
from orion.substrate.link_accepted_seeds import AcceptedLinkV1, link_accepted_seed, projection_edge_id
from orion.substrate.relational.adapters import curiosity_ctx
from orion.substrate.relational.adapters.curiosity_ctx import map_curiosity_ctx_to_substrate

CLAIM = "assertion-compat-1"


def _producer_rows() -> list[dict]:
    """What substrate-runtime stores when a link exists: a scored seed on one end
    (boundary edge) and a link-accepted seed on both ends (internal edge)."""
    store = graph(
        [concept("read-a").model_copy(update={"promotion_state": "proposed"}),
         concept("read-b").model_copy(update={"promotion_state": "proposed"}),
         AssertionNodeV1(node_id=CLAIM, anchor_scope="world", promotion_state="provisional", temporal=TEMPORAL,
                         provenance=PROVENANCE, predicate="associated_with", statement_key="k|c",
                         statement_text="claim", revision=1)],
        [edge(projection_edge_id(CLAIM), "read-a", "read-b", edge_role="semantic_projection",
              assertion_id=CLAIM, assertion_revision=1)])
    scored = FrontierInvocationSignalV1(
        signal_type="curiosity_candidate", anchor_scope="orion", target_zone="concept_graph",
        task_type_candidate="concept_expand", focal_node_refs=["read-a"], signal_strength=0.8,
        confidence=0.6, notes=["endogenous_seed", "source:attention_open_loop"])
    link = link_accepted_seed(AcceptedLinkV1(
        assertion_id=CLAIM, revision=1, subject_node_id="read-a", object_node_id="read-b",
        predicate="associated_with", statement_text="a goes with b", projection_edge_id=projection_edge_id(CLAIM)))
    stored, receipt = attach_seed_neighborhoods([scored, link], store=store)
    assert receipt["nonempty"] == 2
    return [s.model_dump(mode="json") for s in stored]


def test_producer_rows_use_only_fields_the_reader_model_knows():
    rows = _producer_rows()
    emitted = set().union(*rows)
    assert {"boundary_edge_refs", "neighbor_node_refs", "projection_endpoint_node_refs"} <= emitted
    assert emitted <= set(FrontierInvocationSignalV1.model_fields), emitted - set(FrontierInvocationSignalV1.model_fields)


def test_reader_keeps_every_producer_row_with_its_edges(caplog):
    rows = _producer_rows()
    with caplog.at_level(logging.INFO, logger=curiosity_ctx.logger.name):
        signals = curiosity_ctx._coerce(rows)
    assert signals is not None and len(signals) == 2
    assert signals[0].boundary_edge_refs == [projection_edge_id(CLAIM)]
    assert signals[1].focal_edge_refs == [projection_edge_id(CLAIM)]
    assert "dropped_unknown_fields" not in caplog.text


class _PreChangeSignal(BaseModel):
    """FrontierInvocationSignalV1 as it was before 2026-10-10 (an older reader build)."""

    model_config = ConfigDict(extra="forbid")
    signal_id: str = "x"
    signal_type: str
    anchor_scope: str
    subject_ref: str | None = None
    target_zone: str
    task_type_candidate: str
    focal_node_refs: list[str] = Field(default_factory=list)
    focal_edge_refs: list[str] = Field(default_factory=list)
    signal_strength: float
    evidence_summary: str = ""
    confidence: float
    notes: list[str] = Field(default_factory=list)


def test_reader_tolerates_fields_newer_than_its_own_model(monkeypatch):
    """The reader must not lose a signal because the producer is newer than it."""
    monkeypatch.setattr(curiosity_ctx, "FrontierInvocationSignalV1", _PreChangeSignal)
    signals = curiosity_ctx._coerce(_producer_rows())
    assert signals is not None and len(signals) == 2


def test_unscored_link_seed_does_not_lower_aggregate_confidence():
    rows = _producer_rows()
    record = map_curiosity_ctx_to_substrate({"curiosity_signals": rows})
    node = record.nodes[0]
    assert node.metadata["aggregate_confidence"] == 0.6  # the scored seed's, not (0.6 + 0.0) / 2
    assert node.metadata["gap_count"] == 2
