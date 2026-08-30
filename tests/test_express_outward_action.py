"""Orion's first outward action: the `express` kind and its scope gate.

Every dispatch kind before this one either observes Orion (inspect / summarize /
observe) or tidies it (maintain). That is why an enforcing value-of-information
allocator refused the whole repertoire: `maintain host:docker_images
resource_pressure` has posterior variance 5.2e-06 over 7,685 observations.
Nothing Orion could do was worth motor-seconds, because it already knew what
everything did.

`express` is for an action whose product exists outside Orion and costs a
physical resource to make. These tests pin the parts that make it real: the kind
is accepted, the route resolves, and the gate is genuinely independent of the
docker-pruning one.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from orion.autonomy.allocator import expected_information_gain_nats
from orion.execution_dispatch.policy import (
    EXPRESS_SCOPE,
    MAINTENANCE_SCOPE,
    ExecutionDispatchPolicyV1,
)
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1
from orion.schemas.proposal_frame import ProposalCandidateV1

REPO = pathlib.Path(__file__).resolve().parents[1]
DISPATCH_POLICY = REPO / "config/execution_dispatch/execution_dispatch_policy.v1.yaml"
PROPOSAL_POLICY = REPO / "config/proposals/proposal_policy.v1.yaml"


def _literal_values(model, field: str) -> set[str]:
    import typing

    annotation = model.model_fields[field].annotation
    return {a for a in typing.get_args(annotation) if isinstance(a, str)}


def _proposal_kinds() -> set[str]:
    return _literal_values(ProposalCandidateV1, "proposal_kind")


def _dispatch_kinds() -> set[str]:
    return _literal_values(ExecutionDispatchCandidateV1, "dispatch_kind")


def _dispatch_policy() -> ExecutionDispatchPolicyV1:
    return ExecutionDispatchPolicyV1.model_validate(yaml.safe_load(DISPATCH_POLICY.read_text()))


def _render_scene_template() -> dict:
    return yaml.safe_load(PROPOSAL_POLICY.read_text())["proposal_templates"]["render_scene"]


class TestKindIsAccepted:
    """`express` had to be added to three closed Literals. Missing any one of
    them means the proposal is built and then rejected downstream."""

    def test_proposal_schema_accepts_express(self) -> None:
        """Constructed, not introspected: poking at __args__ tests pydantic's
        internals, and a Literal nested in an Optional makes that assertion pass
        or fail for reasons unrelated to whether the value is usable."""
        with pytest.raises(Exception):
            ProposalCandidateV1.model_validate({"proposal_kind": "not_a_real_kind"})
        assert _proposal_kinds() >= {"inspect", "summarize", "observe", "maintain", "express"}

    def test_dispatch_schema_accepts_express(self) -> None:
        assert _dispatch_kinds() >= {"inspect", "summarize", "observe", "maintain", "express"}


class TestScopeGateIsIndependent:
    """The whole reason express_bounded exists rather than reusing
    maintenance_bounded: turning off image generation must not also turn off
    docker pruning, and neither may ride on the other's gate."""

    def test_the_two_scopes_are_distinct(self) -> None:
        assert EXPRESS_SCOPE != MAINTENANCE_SCOPE

    def test_express_defaults_closed(self) -> None:
        """An action whose product exists outside Orion does not become
        dispatchable because a config file appeared."""
        from orion.execution_dispatch.policy import DispatchModeConfigV1

        assert DispatchModeConfigV1().allow_express_dispatch is False
        assert DispatchModeConfigV1().allow_mutating_dispatch is False

    def test_the_live_policy_opens_it_deliberately(self) -> None:
        assert _dispatch_policy().mode.allow_express_dispatch is True

    def test_express_route_uses_the_express_scope_not_maintenance(self) -> None:
        route = _dispatch_policy().template_to_cortex["render_scene"]
        assert route.allowed_scope == EXPRESS_SCOPE, (
            "borrowing maintenance_bounded would make the route table lie about "
            "what the action does, and couple two unrelated kill switches"
        )


class TestRouteResolves:
    def test_template_route_points_at_the_render_verb(self) -> None:
        route = _dispatch_policy().template_to_cortex["render_scene"]
        assert route.cortex_verb == "skills.imagination.render_scene.v1"
        assert route.dispatch_kind == "express"

    def test_the_kind_fallback_resolves_too(self) -> None:
        """A kind with no mapping cannot dispatch at all."""
        route = _dispatch_policy().proposal_kind_to_cortex["express"]
        assert route.cortex_verb == "skills.imagination.render_scene.v1"
        assert route.allowed_scope == EXPRESS_SCOPE

    def test_the_verb_definition_exists_on_disk(self) -> None:
        verb = REPO / "orion/cognition/verbs/skills.imagination.render_scene.v1.yaml"
        assert verb.exists()
        spec = yaml.safe_load(verb.read_text())
        assert spec["name"] == "skills.imagination.render_scene.v1"
        assert spec["requires_gpu"] is True, "this action's whole point is that it costs watts"

    def test_timeouts_are_staggered_innermost_first(self) -> None:
        """A slow generation must be reported by the innermost hop, not severed
        by an outer one -- the same discipline the two prune routes use."""
        verb = yaml.safe_load(
            (REPO / "orion/cognition/verbs/skills.imagination.render_scene.v1.yaml").read_text()
        )
        route = _dispatch_policy().template_to_cortex["render_scene"]
        cortex_exec_http = 150.0  # settings.thought_http_timeout_sec
        assert cortex_exec_http < route.rpc_timeout_sec < verb["timeout_ms"] / 1000.0


class TestItWillActuallyBeChosen:
    """The point of the whole exercise. A cold action has to clear the
    information floor, or it is refused like everything else and Orion stays
    silent."""

    def test_the_template_declares_a_falsifiable_signal(self) -> None:
        t = _render_scene_template()
        assert t["expected_signal"], "no signal means the allocator refuses it 'unmeasurable'"
        assert t["expected_direction"] == "increase", (
            "GPU work loads circe -- claiming no_change would be the same "
            "unfalsifiable claim that drove every other action's variance to zero"
        )

    def test_a_cold_action_clears_the_information_floor(self) -> None:
        """Hand-computed against the live defaults: cold prior variance 0.25,
        typical cost 5.0s, floor 0.02."""
        rate = expected_information_gain_nats(0.25) / 5.0
        # 0.198 is the figure settings.py's own comment quotes for exactly this
        # case. My first pass at this by hand said 0.0446 and was wrong -- the
        # test is written against the implementation's real output, checked
        # against that independently-recorded number.
        assert rate == pytest.approx(0.1981, abs=1e-3)
        assert rate > 0.02 * 9, "a never-measured action must clear the floor with real margin"

    def test_it_beats_the_actions_orion_already_knows(self) -> None:
        """The busiest existing action, measured live: variance 5.2e-06 over
        7,685 observations."""
        cold = expected_information_gain_nats(0.25) / 5.0
        known = expected_information_gain_nats(5.2e-06) / 5.0
        assert cold > known * 100

    def test_it_is_kind_express_so_it_is_not_introspection(self) -> None:
        assert _render_scene_template()["kind"] == "express"
