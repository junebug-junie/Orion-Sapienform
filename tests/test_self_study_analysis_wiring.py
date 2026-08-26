"""Proof that the analysis verb is actually REACHABLE by Orion's own dispatch
loop, not merely implemented.

The gap this patch closes was never missing code: `self_review` /
`self_concept_reflect` have existed and been `autonomous_invocable` for
months, with 0 invocations in 72h and 0 journal entries ever, because nothing
in `proposal_policy.v1.yaml` or `execution_dispatch_policy.v1.yaml` referenced
them. So these tests assert the wire, at every joint.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orion.execution_dispatch.envelopes import build_cortex_request_envelope
from orion.execution_dispatch.policy import CortexRouteTemplateV1, load_execution_dispatch_policy
from orion.proposals.builder import build_proposal_frame
from orion.proposals.policy import load_proposal_policy
from orion.proposals.scoring import PRESSURE_DIMENSIONS
from orion.proposals.templates import _TEMPLATE_COPY, template_title_description
from orion.schemas.field_state import FieldStateV1
from orion.schemas.policy_decision_frame import PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1

REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
TEMPLATE_KEY = "analyze_self_study_source"
VERB = "skills.self_study.analyze.v1"


def _proposal_policy():
    return load_proposal_policy(REPO_ROOT / "config" / "proposals" / "proposal_policy.v1.yaml")


def _dispatch_policy():
    return load_execution_dispatch_policy(
        REPO_ROOT / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml"
    )


def _loaded_field() -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick",
        node_vectors={
            "node:self-study-test": {
                "execution_pressure": 0.8,
                "reasoning_pressure": 0.8,
                "reliability_pressure": 0.9,
                "pressure": 0.8,
                "staleness": 0.8,
                "repair_pressure": 0.8,
                "egress_confidence_deficit": 0.8,
            },
        },
        dimension_precision_ewma_n={dim: 128 for dim in PRESSURE_DIMENSIONS},
        dimension_precision_zscore={dim: 2.5 for dim in PRESSURE_DIMENSIONS},
        dimension_precision_ewma_var={dim: 1.0 for dim in PRESSURE_DIMENSIONS},
    )


# --- joint 1: the template exists and has copy -----------------------------


def test_the_template_is_in_the_proposal_policy() -> None:
    template = _proposal_policy().proposal_templates[TEMPLATE_KEY]
    assert template.required_policy_gate == "read_only"
    assert template.proposed_effect == "increase_observability"
    # No expected_signal on purpose -- there is no evidence any field dimension
    # predicts "there is something notable in that table right now".
    assert template.expected_signal is None


def test_the_template_has_human_copy() -> None:
    assert TEMPLATE_KEY in _TEMPLATE_COPY, "falls back to generic copy otherwise"
    title, description, tags = template_title_description(
        TEMPLATE_KEY, target_id="self_state:self_study"
    )
    assert title and description
    assert "read_only" in tags


# --- joint 2: it actually becomes a candidate ------------------------------


def test_the_template_becomes_a_real_candidate_on_a_loaded_field() -> None:
    policy = _proposal_policy()
    frame = build_proposal_frame(field=_loaded_field(), attention=None, policy=policy, now=NOW)
    keys = {c.proposal_id.split(":")[1] for c in frame.candidates}
    assert TEMPLATE_KEY in keys, (
        "template did not survive the candidate cut -- if this fails after adding "
        "templates, limits.max_candidates is the knob, not base_priority"
    )


def test_raising_max_candidates_did_not_displace_an_existing_template() -> None:
    """The cap went 10 -> 11 for exactly one added template. If a future patch
    adds templates without raising it, this catches the silent eviction."""
    policy = _proposal_policy()
    assert len(policy.proposal_templates) - policy.limits.max_candidates == 6
    frame = build_proposal_frame(field=_loaded_field(), attention=None, policy=policy, now=NOW)
    assert len(frame.candidates) <= policy.limits.max_candidates


# --- joint 3: the dispatch route resolves ----------------------------------


def test_the_dispatch_route_points_at_the_verb() -> None:
    route = _dispatch_policy().template_to_cortex[TEMPLATE_KEY]
    assert route.cortex_verb == VERB
    # summarize_only, not a new scope: builder.py::scope_allowed permits it
    # unconditionally, so no safety gate needed editing.
    assert route.allowed_scope == "summarize_only"
    assert route.rpc_timeout_sec == 60.0
    # Deliberately unset -- the verb picks the most overdue source itself.
    assert route.skill_args == {}


def test_the_verb_yaml_exists_and_matches_the_route() -> None:
    path = REPO_ROOT / "orion" / "cognition" / "verbs" / f"{VERB}.yaml"
    spec = yaml.safe_load(path.read_text())
    assert spec["name"] == VERB
    # services: [] is what routes this through executor.py's local-verb branch,
    # which is the ONLY branch that injects a live bus into VerbContext.meta --
    # without it the journal write silently degrades to "missing_bus".
    assert spec["services"] == []
    # 2x margin over the verb's own budget, same discipline as the other routes.
    assert _dispatch_policy().template_to_cortex[TEMPLATE_KEY].rpc_timeout_sec >= (
        spec["timeout_ms"] / 1000
    ) * 2


def test_the_verb_is_registered_under_that_exact_name() -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "services" / "orion-cortex-exec"))
    from app import verb_adapters  # noqa: F401 -- registers on import
    from orion.core.verbs.registry import registry

    assert registry.get(VERB) is not None


# --- joint 4: route skill_args reach the envelope --------------------------


def _candidate() -> ProposalCandidateV1:
    return ProposalCandidateV1(
        proposal_id=f"proposal:{TEMPLATE_KEY}:self_state",
        proposal_kind="summarize",
        title="t",
        description="d",
        target_id="self_state:self_study",
        target_kind="self_state",
        priority_score=0.34,
        urgency_score=0.3,
        confidence_score=0.9,
        risk_score=0.05,
        reversibility_score=1.0,
        motivating_dimensions={},
        proposed_effect="increase_observability",
        required_policy_gate="read_only",
        execution_intent={"mode": "descriptive_only"},
    )


def _decision() -> PolicyDecisionV1:
    return PolicyDecisionV1(
        decision_id="policy.decision:x",
        proposal_id=f"proposal:{TEMPLATE_KEY}:self_state",
        decision="approved_read_only",
        policy_gate="read_only",
        risk_score=0.05,
        reversibility_score=1.0,
        confidence_score=0.9,
        allowed_scope="summarize_only",
    )


def _envelope(route: CortexRouteTemplateV1, *, dry_run: bool = True) -> dict:
    return build_cortex_request_envelope(
        candidate=_candidate(),
        decision=_decision(),
        route=route,
        field_tick_id="field.tick:1",
        dry_run=dry_run,
    )


def test_a_route_without_skill_args_sends_none() -> None:
    env = _envelope(_dispatch_policy().template_to_cortex[TEMPLATE_KEY])
    assert "skill_args" not in env["context"]


def test_route_skill_args_reach_the_verb_context() -> None:
    route = CortexRouteTemplateV1(
        dispatch_kind="summarize",
        cortex_verb=VERB,
        allowed_scope="summarize_only",
        skill_args={"source": "vision_events", "window_hours": "12"},
    )
    env = _envelope(route)
    assert env["context"]["skill_args"] == {"source": "vision_events", "window_hours": "12"}


def test_config_cannot_declare_a_run_mode_at_all() -> None:
    """A skill's run mode is DERIVED from the dispatch runtime's own dry_run
    state. envelopes.py only forces it for maintenance-scoped routes, and
    `allowed_scope` / `cortex_verb` are independent unvalidated config fields --
    so a mutating verb declared under `summarize_only` (which scope_allowed
    admits unconditionally) plus `skill_args: {mode: execute}` would have
    reached that verb with mode=execute regardless of dry_run. Refused at the
    schema, not merely overridden downstream."""
    for forbidden in ("mode", "run_mode", "dry_run", "MODE"):
        with pytest.raises(ValidationError):
            CortexRouteTemplateV1(
                dispatch_kind="summarize",
                cortex_verb="skills.runtime.builder_prune.v1",
                allowed_scope="summarize_only",
                skill_args={forbidden: "execute"},
            )


def test_a_maintenance_route_still_gets_its_derived_mode_alongside_config_args() -> None:
    route = CortexRouteTemplateV1(
        dispatch_kind="maintain",
        cortex_verb="skills.runtime.image_prune.v1",
        allowed_scope="maintenance_bounded",
        skill_args={"note": "kept"},
    )
    assert _envelope(route, dry_run=True)["context"]["skill_args"] == {
        "note": "kept",
        "mode": "preview",
    }


def test_maintenance_routes_without_skill_args_still_get_their_mode() -> None:
    route = CortexRouteTemplateV1(
        dispatch_kind="maintain",
        cortex_verb="skills.runtime.image_prune.v1",
        allowed_scope="maintenance_bounded",
    )
    assert _envelope(route, dry_run=False)["context"]["skill_args"] == {"mode": "execute"}
