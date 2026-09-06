"""The attention schema surface: one thin shared shape, four projections.

docs/superpowers/specs/2026-09-04-attention-schema-surface-design.md (PR #2092).

Each adapter is a pure function of that process's own artifact; every test
here hand-builds the artifact and asserts the projected row, branch by
branch. No I/O, no bus, no database. The contract-level checks (both
registry maps, the channel catalog) live here too so a schema that resolves
in one map but not the other cannot ship green (the two-registries trap this
repo has hit before, tests/test_agent_trace_schema_registry.py).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from orion.curiosity.attention_schema import (
    AttendedPrior,
    attended_priors_cypher,
    build_attended_priors,
    to_attention_schema as curiosity_to_attention_schema,
)
from orion.curiosity.worldview import TurnOutcome
from orion.reverie.attention_schema import to_attention_schema as reverie_to_attention_schema
from orion.schemas.attention_frame import (
    AttentionFrameV1,
    CuriosityCandidateActionV1,
    CuriositySuppressionV1,
    OpenLoopV1,
    VoluntaryOverrideV1,
)
from orion.schemas.attention_schema import (
    ATTENTION_SCHEMA_CHANNEL,
    ATTENTION_SCHEMA_KIND,
    MAX_NARRATIVE_CHARS,
    AttentionSchemaV1,
    bind_correlation,
    clip,
)
from orion.schemas.attention_self_model import AttentionSelfModelV1
from orion.schemas.registry import SCHEMA_REGISTRY, resolve
from orion.schemas.reverie import ReverieChainTriggerV1, ReverieChainV1, SpontaneousThoughtV1
from orion.schemas.thought import CoalitionSnapshotV1
from orion.substrate.attention_frame import to_attention_schema as cortex_to_attention_schema
from orion.substrate.attention_self_model import to_attention_schema as substrate_to_attention_schema

REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 6, 0, 0, tzinfo=timezone.utc)


# --- contract ------------------------------------------------------------------


def test_schema_resolves_in_both_registry_maps() -> None:
    assert resolve("AttentionSchemaV1") is AttentionSchemaV1
    assert SCHEMA_REGISTRY["AttentionSchemaV1"].kind == ATTENTION_SCHEMA_KIND


def test_channel_is_catalogued_with_four_producers_and_the_sql_writer() -> None:
    catalog = yaml.safe_load((REPO_ROOT / "orion" / "bus" / "channels.yaml").read_text())
    entries = [c for c in catalog["channels"] if c.get("name") == ATTENTION_SCHEMA_CHANNEL]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["schema_id"] == "AttentionSchemaV1"
    assert entry["message_kind"] == ATTENTION_SCHEMA_KIND
    assert set(entry["producer_services"]) == {
        "orion-substrate-runtime",
        "orion-thought",
        "orion-hub",
        "orion-cortex-exec",
    }
    assert entry["consumer_services"] == ["orion-sql-writer"]


def test_schema_rejects_unknown_fields_and_out_of_range_confidence() -> None:
    with pytest.raises(Exception):
        AttentionSchemaV1(entry_id="x", process="reverie", attention_reason="r", extra=1)
    with pytest.raises(Exception):
        AttentionSchemaV1(entry_id="x", process="reverie", attention_reason="r", confidence=1.5)
    with pytest.raises(Exception):
        AttentionSchemaV1(entry_id="x", process="not_a_process", attention_reason="r")


def test_bind_correlation_makes_row_and_envelope_agree() -> None:
    """Review finding: sql-writer stamps the column from the envelope, so the
    two must be the same id or the row's own correlation is lost."""
    from uuid import UUID

    row = AttentionSchemaV1(entry_id="x", process="reverie", attention_reason="r",
                            correlation_id="7dcc3944-29bb-5d8f-915f-90f4e6968d47")
    bound, corr = bind_correlation(row)
    assert corr == UUID("7dcc3944-29bb-5d8f-915f-90f4e6968d47") and bound.correlation_id == str(corr)

    for raw in (None, "corr-1", ""):
        bound, corr = bind_correlation(AttentionSchemaV1(entry_id="x", process="reverie", attention_reason="r", correlation_id=raw))
        assert bound.correlation_id == str(corr)
        assert isinstance(corr, UUID)


def test_clip_normalises_whitespace_and_caps_with_an_ellipsis() -> None:
    assert clip("  a\n b   c ", 100) == "a b c"
    out = clip("x" * 50, 10)
    assert len(out) == 10 and out.endswith("…")
    assert clip(None, 5) == ""


# --- substrate_attention -------------------------------------------------------


def _self_model(**over) -> AttentionSelfModelV1:
    base = dict(
        generated_at=NOW,
        broadcast_lane_present=True,
        broadcast_selected_open_loop_id="open-loop-abc",
        broadcast_selected_description="unresolved prediction error in execution",
        attention_reason="bottom_up_salience",
        voluntary_override_absent_reason="goal_matched_no_loop",
        reason_narrative="Pure bottom-up dispatch: a goal was active but about none of the loops.",
        prediction_error_confidence=0.71,
        prediction_error_confidence_basis="1 - mean prediction_error across 4 domains",
        predicted_shift="stable",
    )
    base.update(over)
    return AttentionSelfModelV1(**base)


def test_substrate_projection_keeps_the_absent_override_cause_in_the_reason() -> None:
    row = substrate_to_attention_schema(_self_model())
    assert row.process == "substrate_attention"
    assert row.entry_id.startswith("substrate-")
    assert row.attended_id == "open-loop-abc"
    assert row.attended_label == "unresolved prediction error in execution"
    # PR #2106 split "no override" into its causes; the surface must not merge them back.
    assert row.attention_reason == "bottom_up_salience:goal_matched_no_loop"
    assert row.narrative_kind == "computed"
    assert row.confidence == pytest.approx(0.71)
    assert row.confidence_basis == "1 - mean prediction_error across 4 domains"
    assert row.predicted_next == "stable"


def test_substrate_projection_override_branch_prefers_branch_confidence() -> None:
    model = _self_model(
        attention_reason="top_down_override",
        voluntary_override_absent_reason=None,
        voluntary_override=VoluntaryOverrideV1(
            chosen_loop_id="open-loop-abc", beat_loop_id="open-loop-def",
            chosen_bottom_up=0.25, beat_bottom_up=0.75, applied_bias=1.0, effort_spent=1.0,
        ),
        confidence=0.4,
        confidence_basis="branch-conditional",
    )
    row = substrate_to_attention_schema(model)
    assert row.attention_reason == "top_down_override"
    assert row.confidence == pytest.approx(0.4)
    assert row.confidence_basis == "branch-conditional"


def test_substrate_projection_no_data_has_no_attended_id_and_no_invented_confidence() -> None:
    model = AttentionSelfModelV1(generated_at=NOW)
    row = substrate_to_attention_schema(model)
    assert row.attended_id is None
    assert row.attention_reason == "no_data"
    assert row.confidence is None and row.confidence_basis is None


def test_substrate_entry_id_is_deterministic_per_tick() -> None:
    a = substrate_to_attention_schema(_self_model())
    b = substrate_to_attention_schema(_self_model())
    c = substrate_to_attention_schema(_self_model(generated_at=NOW.replace(second=30)))
    assert a.entry_id == b.entry_id != c.entry_id


# --- cortex_turn ---------------------------------------------------------------


def _frame(**over) -> AttentionFrameV1:
    base = dict(
        generated_at=NOW,
        turn_id="turn-1",
        correlation_id="corr-1",
        open_loops=[OpenLoopV1(id="loop-1", description="Zephyr Bridge"), OpenLoopV1(id="loop-2", description="the deploy")],
    )
    base.update(over)
    return AttentionFrameV1(**base)


def test_cortex_projection_selected_action_uses_the_policy_rationale() -> None:
    frame = _frame(
        selected_action=CuriosityCandidateActionV1(
            action_type="ask", open_loop_id="loop-1", score=0.8, rationale="high novelty, low cost"
        ),
        deferred_items=["the deploy"],
    )
    row = cortex_to_attention_schema(frame, leg="harness_finalize_reflect")
    assert row.process == "cortex_turn"
    assert row.entry_id == "cortex-turn-1-harness_finalize_reflect"
    assert row.correlation_id == "corr-1"
    assert row.attended_id == "loop-1"
    assert row.attended_label == "Zephyr Bridge"
    assert row.attention_reason == "selected:ask"
    assert row.reason_narrative == "high novelty, low cost"
    assert row.narrative_kind == "computed"
    assert row.confidence == pytest.approx(0.8)
    assert row.predicted_next == "the deploy"


def test_cortex_projection_override_wins_over_selection() -> None:
    frame = _frame(
        selected_action=CuriosityCandidateActionV1(action_type="ask", open_loop_id="loop-2", score=0.6),
        voluntary_override=VoluntaryOverrideV1(
            chosen_loop_id="loop-2", beat_loop_id="loop-1",
            chosen_bottom_up=0.3, beat_bottom_up=0.7, applied_bias=0.9, effort_spent=0.5,
        ),
    )
    row = cortex_to_attention_schema(frame)
    assert row.attention_reason == "top_down_override"
    assert row.attended_id == "loop-2"
    assert "beat 'loop-1'" in row.reason_narrative


def test_cortex_projection_suppression_and_empty_branches() -> None:
    suppressed = cortex_to_attention_schema(_frame(
        suppressions=[CuriositySuppressionV1(reason="already_known", target_ref="loop-1", rationale="asked last turn", confidence=0.7)],
    ))
    assert suppressed.attention_reason == "suppressed:already_known"
    assert suppressed.attended_id == "loop-1"
    assert suppressed.confidence == pytest.approx(0.7)

    idle = cortex_to_attention_schema(_frame())
    assert idle.attention_reason == "open_loops_no_action" and idle.attended_id is None

    empty = cortex_to_attention_schema(_frame(open_loops=[]))
    assert empty.attention_reason == "no_open_loops"

    # action_type "none" is not a selection.
    none_action = cortex_to_attention_schema(_frame(
        selected_action=CuriosityCandidateActionV1(action_type="none", open_loop_id=None, score=0.0)
    ))
    assert none_action.attention_reason == "open_loops_no_action"


def test_cortex_entry_id_falls_back_without_turn_or_correlation_ids() -> None:
    row = cortex_to_attention_schema(_frame(turn_id=None, correlation_id=None))
    assert row.entry_id.startswith("cortex-") and len(row.entry_id) > len("cortex-")


def test_cortex_two_legs_of_one_turn_do_not_collide() -> None:
    """Review finding: a unified turn runs more than one brain-mode leg under
    one correlation id and no turn_id; without the leg in the key the writer
    kept whichever arrived first."""
    frame = _frame(turn_id=None)
    a = cortex_to_attention_schema(frame, leg="harness_finalize_reflect")
    b = cortex_to_attention_schema(frame, leg="orion_voice_finalize")
    assert a.entry_id != b.entry_id and a.correlation_id == b.correlation_id == "corr-1"
    # No leg at all: the generated_at stamp still separates distinct builds.
    c = cortex_to_attention_schema(frame)
    d = cortex_to_attention_schema(_frame(turn_id=None, generated_at=NOW.replace(second=1)))
    assert c.entry_id != d.entry_id


# --- reverie -------------------------------------------------------------------


def _thought(idx: int, *, interpretation: str = "the loop keeps recurring", hollow: bool = False, next_focus: str | None = None) -> SpontaneousThoughtV1:
    return SpontaneousThoughtV1(
        thought_id=f"th-{idx}",
        correlation_id="corr-r",
        coalition=CoalitionSnapshotV1(
            attended_node_ids=["n-1"], selected_open_loop_id="ol-1", open_loop_ids=["ol-1"], generated_at=NOW
        ),
        interpretation=interpretation,
        salience=0.6,
        hollow=hollow,
        next_focus=next_focus,
    )


def _chain(**over) -> ReverieChainV1:
    base = dict(chain_id="chain-1", created_at=NOW, theme_key="ol-1", thought_ids=["th-0", "th-1"],
                ema_salience=0.55, ema_summary="2 steps on ol-1; ema_salience=0.550", terminal_reason="max_steps")
    base.update(over)
    return ReverieChainV1(**base)


def test_reverie_projection_is_a_self_report_from_the_first_real_thought() -> None:
    thoughts = [_thought(0, interpretation="", hollow=True), _thought(1, interpretation="strained by an unresolved error", next_focus="the codebase anomaly")]
    row = reverie_to_attention_schema(_chain(), thoughts)
    assert row.process == "reverie"
    assert row.entry_id == "reverie-chain-1"
    assert row.correlation_id == "corr-r"
    assert row.attended_id == "ol-1"
    assert row.attended_label == "ol-1"
    assert row.attention_reason == "coalition_broadcast"
    assert row.reason_narrative == "strained by an unresolved error"
    assert row.narrative_kind == "self_report"
    assert row.confidence == pytest.approx(0.55)
    assert "ended max_steps" in (row.confidence_basis or "")
    assert row.predicted_next == "the codebase anomaly"


def test_reverie_projection_falls_back_to_the_computed_summary_when_no_thought_narrates() -> None:
    row = reverie_to_attention_schema(_chain(), [_thought(0, interpretation="")])
    assert row.reason_narrative == "2 steps on ol-1; ema_salience=0.550"
    assert row.narrative_kind == "computed"
    assert row.predicted_next is None


def test_reverie_projection_no_coalition_and_trigger_vocabulary() -> None:
    empty = reverie_to_attention_schema(_chain(theme_key="unknown", thought_ids=[]), [])
    assert empty.attended_id is None and empty.attention_reason == "no_coalition"
    assert empty.narrative_kind == "computed"

    triggered = reverie_to_attention_schema(
        _chain(trigger=ReverieChainTriggerV1(pressure_kind="prediction_error", magnitude=0.4)), [_thought(0)]
    )
    assert triggered.attention_reason == "trigger:prediction_error"


def test_reverie_projection_clips_a_long_interpretation_instead_of_failing() -> None:
    row = reverie_to_attention_schema(_chain(), [_thought(0, interpretation="x" * (MAX_NARRATIVE_CHARS + 500))])
    assert len(row.reason_narrative) == MAX_NARRATIVE_CHARS


# --- curiosity -----------------------------------------------------------------


def test_curiosity_cypher_refuses_a_non_hex_run_id() -> None:
    with pytest.raises(ValueError):
        attended_priors_cypher("abc'); MATCH (n) DETACH DELETE n; //")
    sql = attended_priors_cypher("3b2d038cf18e")
    assert "p.last_run_id = '3b2d038cf18e'" in sql and "PriorRevision" in sql


def test_curiosity_rows_become_priors_tested_first() -> None:
    rows = [
        {"prior_id": "new_one", "claim": "a new claim", "status": "open", "confidence": 0.55, "run_id": "r1", "last_run_id": None},
        {"prior_id": "held_one", "claim": "an old claim", "status": "revised", "confidence": 0.92, "run_id": "r0", "last_run_id": "r1",
         "from_confidence": 0.95, "to_confidence": 0.92},
        {"prior_id": "", "claim": "unreadable"},
    ]
    priors = build_attended_priors(rows, "r1")
    assert [p.prior_id for p in priors] == ["held_one", "new_one"]
    assert priors[0].touched == "tested" and priors[0].from_confidence == pytest.approx(0.95)
    assert priors[1].touched == "formed"


def test_curiosity_projection_tested_prior_is_the_live_run_shape() -> None:
    # Run 3b2d038cf18e, read live 2026-09-06: tested substrate_domain_isolation_two_paths 0.95 -> 0.92.
    priors = [AttendedPrior("substrate_domain_isolation_two_paths", "The 9 node:substrate.* concepts are isolated",
                            "revised", 0.92, "tested", 0.95, 0.92)]
    outcome = TurnOutcome(run_id="3b2d038cf18e", continue_line=False, continue_note="Question resolved: by-design isolation.",
                          reach_out=True, reach_out_why="worth telling")
    row = curiosity_to_attention_schema(run_id="3b2d038cf18e", outcome=outcome, priors=priors, correlation_id="corr-c", generated_at=NOW)
    assert row.process == "curiosity"
    assert row.entry_id == "curiosity-3b2d038cf18e"
    assert row.attended_id == "substrate_domain_isolation_two_paths"
    assert row.attention_reason == "tested_held_prior"
    assert "0.95 -> 0.92" in row.reason_narrative and "'revised'" in row.reason_narrative
    assert row.narrative_kind == "computed"
    assert row.confidence == pytest.approx(0.92)
    assert "self-assigned" in (row.confidence_basis or "")
    assert row.predicted_next == "Question resolved: by-design isolation."


def test_curiosity_projection_formed_none_and_unreadable_are_distinct_states() -> None:
    formed = curiosity_to_attention_schema(
        run_id="aaa111", outcome=None,
        priors=[AttendedPrior("p1", "a fresh claim", "open", 0.55, "formed"), AttendedPrior("p2", "another", "open", 0.5, "formed")],
        correlation_id=None, generated_at=NOW,
    )
    assert formed.attention_reason == "formed_new_prior" and formed.attended_id == "p1"
    assert "1 more prior touched" in formed.reason_narrative
    assert formed.predicted_next is None

    none = curiosity_to_attention_schema(run_id="aaa111", outcome=None, priors=[], correlation_id=None, generated_at=NOW)
    assert none.attention_reason == "no_prior_touched" and none.attended_id is None and none.confidence is None

    unreadable = curiosity_to_attention_schema(run_id="aaa111", outcome=None, priors=None, correlation_id=None, generated_at=NOW)
    assert unreadable.attention_reason == "graph_unreadable"
    assert unreadable.reason_narrative != none.reason_narrative
