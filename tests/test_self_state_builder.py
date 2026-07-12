from datetime import datetime, timezone
from pathlib import Path

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.policy import load_attention_policy
from orion.self_state.builder import _attention_target_summary, build_self_state
from orion.self_state.policy import load_self_state_policy
from orion.schemas.field_attention_frame import FieldAttentionTargetV1
from orion.schemas.field_state import FieldStateV1

REPO = Path(__file__).resolve().parents[1]
ATTENTION_POLICY = load_attention_policy(REPO / "config" / "attention" / "field_attention_policy.v1.yaml")
SELF_POLICY = load_self_state_policy(REPO / "config" / "self_state" / "self_state_policy.v1.yaml")
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _synthetic_field() -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_exec_attention",
        node_vectors={
            "node:athena": {
                "execution_load": 1.0,
                "reasoning_load": 0.35,
                "availability": 1.0,
            },
        },
        capability_vectors={
            "capability:orchestration": {
                "execution_pressure": 1.0,
                "reliability_pressure": 0.0,
            }
        },
        recent_perturbations=["state_delta:exec_1", "state_delta:exec_2"],
    )


def test_builder_references_sources() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    assert state.source_field_tick_id == field.tick_id
    assert state.source_attention_frame_id == attention.frame_id


def test_execution_pressure_nonzero() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    assert "execution_pressure" in state.dimensions
    assert state.dimensions["execution_pressure"].score > 0.0


def test_agency_readiness_present() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    assert "agency_readiness" in state.dimensions


def test_dominant_attention_targets() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    ids = set(state.dominant_attention_targets)
    assert "node:athena" in ids or "capability:orchestration" in ids


def test_dominant_attention_target_details_carries_structured_data() -> None:
    # 2026-07-12, inner-state unification Phase 1: dominant_attention_targets
    # (bare ID strings) previously discarded pressure_score/dominant_channels/
    # reasons one hop after orion-attention-runtime computed them for real.
    # This must now survive, additively, alongside the bare-string list.
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)

    assert state.dominant_attention_target_details
    # Same target_ids, same order, as the existing bare-string list -- additive,
    # not a divergent second source of truth.
    assert [d.target_id for d in state.dominant_attention_target_details] == (
        state.dominant_attention_targets
    )
    for detail in state.dominant_attention_target_details:
        assert detail.target_kind in ("node", "capability", "channel", "edge", "field", "system")
        assert 0.0 <= detail.pressure_score <= 1.0

    top = state.dominant_attention_target_details[0]
    assert top.pressure_score > 0.0
    assert top.dominant_channel is not None
    assert top.reason is not None


def test_attention_target_summary_picks_first_channel_and_reason() -> None:
    # 2026-07-12 review finding: the end-to-end test above only checks
    # presence, not that the "top-1" pick is actually the first entry (the
    # documented contract -- upstream, weighted_pressure/_reasons_from_dominant
    # already sort descending before returning; this helper trusts that
    # ordering rather than re-sorting). A direct unit test on
    # _attention_target_summary() with an explicit, order-distinguishable
    # fixture is the only way to catch a regression that silently picked a
    # different entry (e.g. min instead of max, or last instead of first).
    target = FieldAttentionTargetV1(
        target_id="node:atlas",
        target_kind="node",
        salience_score=0.9,
        pressure_score=0.72,
        novelty_score=0.1,
        urgency_score=0.5,
        confidence_score=0.6,
        dominant_channels={"gpu_pressure": 0.85, "thermal_pressure": 0.40},
        reasons=["node gpu_pressure is elevated", "node thermal_pressure is elevated"],
    )

    summary = _attention_target_summary(target)

    assert summary.target_id == "node:atlas"
    assert summary.target_kind == "node"
    assert summary.pressure_score == 0.72
    assert summary.dominant_channel == "gpu_pressure"
    assert summary.reason == "node gpu_pressure is elevated"


def test_attention_target_summary_handles_no_channels_or_reasons() -> None:
    target = FieldAttentionTargetV1(
        target_id="field:recent_perturbations",
        target_kind="system",
        salience_score=0.2,
        pressure_score=0.2,
        novelty_score=0.0,
        urgency_score=0.0,
        confidence_score=0.0,
    )

    summary = _attention_target_summary(target)

    assert summary.dominant_channel is None
    assert summary.reason is None


def test_summary_labels_execution_loaded() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    assert "execution_loaded" in state.summary_labels


def test_self_state_id_stable() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    a = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    b = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    assert a.self_state_id == b.self_state_id


def test_dimension_reasons_reflect_real_evidence() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)

    execution_dim = state.dimensions["execution_pressure"]
    reasoning_dim = state.dimensions["reasoning_pressure"]

    # Reasons are not a fixed template repeated across every dimension.
    assert execution_dim.reasons != reasoning_dim.reasons
    assert "from field+attention channel synthesis" not in " ".join(
        execution_dim.reasons + reasoning_dim.reasons
    )

    # Reasons are derived from the same evidence already computed for
    # dominant_evidence, not an independent/duplicated computation.
    assert execution_dim.dominant_evidence
    for ev in execution_dim.dominant_evidence:
        assert any(ev in reason for reason in execution_dim.reasons)

    # A dimension with no contributing channel evidence this tick gets an
    # honest fallback, not a reused fixed string.
    social_dim = state.dimensions["social_pressure"]
    assert social_dim.dominant_evidence == []
    assert social_dim.reasons == ["no contributing channel evidence this tick"]


def _confidence_field() -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_confidence",
        node_vectors={
            "node:athena": {
                "availability": 0.90,
                "confidence": 0.85,
                "thermal_pressure": 0.30,
            },
        },
        capability_vectors={
            "capability:orchestration": {
                "pressure": 0.70,
                "reliability_pressure": 0.50,
            },
            "capability:transport": {
                "available_capacity": 0.80,
            },
        },
    )


def test_dimension_confidence_varies_across_dimensions() -> None:
    # Phase 1 (2026-07-12): per-dimension confidence used to be a single
    # global proxy (0.5 + 0.5*len(dominant_targets)/5) copy-pasted onto all
    # 13 dimensions, carrying zero per-dimension information. It must now be
    # a real function of how many channels contributed to each dimension
    # this tick and how much they agree.
    field = _confidence_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)

    confidences = {dim_id: dim.confidence for dim_id, dim in state.dimensions.items()}
    assert len(set(confidences.values())) > 1, confidences

    # coherence has 3 agreeing contributing channels (availability=0.90,
    # confidence=0.85, available_capacity=0.80); reliability_pressure has
    # only 1 (reliability_pressure=0.50). More agreeing evidence must mean
    # higher confidence.
    assert state.dimensions["coherence"].confidence > state.dimensions["reliability_pressure"].confidence

    # A dimension with zero contributing channels this tick (no channel in
    # merged_channels maps to continuity_pressure in this fixture) reports
    # honest zero confidence, not a borrowed global proxy.
    assert state.dimensions["continuity_pressure"].confidence == 0.0

    # Confidence values stay within the schema's valid range.
    for dim in state.dimensions.values():
        assert 0.0 <= dim.confidence <= 1.0


def test_composite_dimension_confidence_falls_back_to_overall_confidence() -> None:
    # field_intensity/agency_readiness are synthesized from attention
    # salience/other dimension scores, never from a direct
    # channel_dimension_map entry -- channel_dimension_confidence() can never
    # find a contributing channel for them, so a hardcoded 0.0 there would be
    # a permanent, structural false signal (not an honest "no evidence this
    # tick", since this is true every tick forever). Regression found by
    # code review (2026-07-12): both fall back to overall_confidence instead.
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)

    assert state.dimensions["field_intensity"].confidence == state.overall_confidence
    assert state.dimensions["agency_readiness"].confidence == state.overall_confidence
    assert state.overall_confidence > 0.0


def _circe_evidence_field() -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_evidence_channel_map",
        node_vectors={"node:circe": {"gpu_pressure": 1.0}},
        capability_vectors={"capability:llm_inference": {"pressure": 0.50}},
    )


def test_evidence_channel_map_restores_raw_channel_without_double_counting_score() -> None:
    # Regression found by code review (2026-07-12): orion-spark-introspector's
    # tissue-viz bypass (_hardware_resource_pressure/_execution_load_pressure)
    # parses dominant_evidence for exact raw channel names (gpu_pressure, etc.)
    # to avoid a generic diffused channel that can stick saturated (live
    # incident 2026-07-10). Phase 1's double-counting fix removed those raw
    # channels from channel_dimension_map so they stop feeding the SCORE --
    # which also silently removed them from dominant_evidence, breaking that
    # bypass. evidence_channel_map restores raw-channel visibility in
    # evidence without reintroducing the double-count into the score.
    field = _circe_evidence_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)

    resource_dim = state.dimensions["resource_pressure"]
    # Score reflects only the weighted diffusion (0.50), not the raw spike (1.0).
    assert resource_dim.score == 0.50
    # But the raw channel is still visible in evidence for downstream bypass logic.
    assert any(ev.startswith("gpu_pressure=") for ev in resource_dim.dominant_evidence)


def _mesh_provenance_field() -> FieldStateV1:
    # Mirrors what real orion-field-digester diffusion output looks like:
    # capability_provenance populated by apply_diffusion, recording which
    # node fed capability:llm_inference's "pressure" channel this tick.
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_mesh_provenance",
        node_vectors={"node:circe": {"gpu_pressure": 0.95}},
        capability_vectors={"capability:llm_inference": {"pressure": 0.60}},
        capability_provenance={"capability:llm_inference": {"pressure": "node:circe"}},
    )


def test_resource_pressure_reasons_name_the_contributing_node() -> None:
    # Phase 3 (2026-07-12, mesh embodiment): the concrete acceptance goal --
    # a live self-state tick's evidence names a specific non-Athena node,
    # e.g. "driven by pressure=0.60 (node: circe)" instead of an anonymous
    # capability-level pressure number.
    field = _mesh_provenance_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)

    resource_dim = state.dimensions["resource_pressure"]
    # dominant_evidence keeps its exact channel_name=value format, unchanged --
    # downstream consumers (orion-spark-introspector's hardware bypass) parse
    # it by exact shape.
    assert any(ev == "pressure=0.60" for ev in resource_dim.dominant_evidence)
    # reasons is where node attribution lives.
    assert any("(node: circe)" in reason for reason in resource_dim.reasons)


def test_no_action_outputs() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    payload = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW).model_dump()
    forbidden = ("proposal", "action", "policy_gate", "cortex", "selected_action")
    for key in payload:
        assert not any(f in key.lower() for f in forbidden)
