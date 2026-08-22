"""Per-template cortex route override (2026-08-13).

Why this seam exists: `proposal_kind_to_cortex` is keyed on `proposal_kind`
alone, so an entire kind could address exactly ONE verb. `maintain` meant
`skills.runtime.builder_prune.v1` and could never mean anything else, which
capped Orion's mutating action repertoire at one BY CONSTRUCTION -- underneath
every scoring, budgeting, and starvation fix applied above it.

The most important test in this file is
`test_empty_template_map_is_byte_identical_to_the_old_lookup`: the whole
justification for shipping this is that it is inert until a route is declared.
"""

from __future__ import annotations

import pytest

from orion.execution_dispatch.builder import candidate_template_key, resolve_cortex_route
from orion.execution_dispatch.policy import CortexRouteTemplateV1, ExecutionDispatchPolicyV1
from orion.schemas.proposal_frame import ProposalCandidateV1


def _candidate(
    *, proposal_kind: str = "maintain", template: str | None = "prune_build_cache"
) -> ProposalCandidateV1:
    intent: dict[str, str] = {"mode": "bounded_action", "policy_gate": "operator_review"}
    if template is not None:
        intent["template"] = template
    return ProposalCandidateV1(
        proposal_id=f"proposal:{template or 'none'}:tick_abc:attention.frame:tick_abc:p.v1",
        proposal_kind=proposal_kind,
        title="t",
        description="d",
        target_kind="system",
        target_id="host:docker",
        # `proposed_effect` is a Literal on ProposalCandidateV1; "reduce_pressure"
        # is the member a disk-reclaiming maintenance action belongs under.
        proposed_effect="reduce_pressure",
        urgency_score=0.5,
        priority_score=0.5,
        confidence_score=0.5,
        risk_score=0.1,
        reversibility_score=1.0,
        required_policy_gate="operator_review",
        execution_intent=intent,
    )


def _route(verb: str, *, kind: str = "maintain") -> CortexRouteTemplateV1:
    return CortexRouteTemplateV1(
        dispatch_kind=kind,
        cortex_verb=verb,
        allowed_scope="maintenance_bounded",
    )


# ---------------------------------------------------------------------------
# The backward-compatibility guarantee. This is the load-bearing one.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,template",
    [
        ("maintain", "prune_build_cache"),
        ("inspect", "inspect_bus_channel_catalog"),
        ("summarize", "summarize_loaded_state"),
        ("observe", "watch_reliability"),
        # An externally-produced candidate with no template at all -- reverie
        # and cognitive-hop producers legitimately carry none.
        ("inspect", None),
    ],
)
def test_empty_template_map_is_byte_identical_to_the_old_lookup(kind, template):
    """With no template routes declared, resolution must equal the old code.

    The old code was exactly `policy.proposal_kind_to_cortex.get(kind)`. That
    expression is evaluated here directly and compared, rather than a
    remembered expectation, so this cannot drift from what it claims to check.
    """
    policy = ExecutionDispatchPolicyV1(
        proposal_kind_to_cortex={
            "maintain": _route("skills.runtime.builder_prune.v1"),
            "inspect": _route("substrate.inspect", kind="inspect"),
            "summarize": _route("substrate.summarize", kind="summarize"),
            "observe": _route("substrate.observe", kind="observe"),
        }
    )
    candidate = _candidate(proposal_kind=kind, template=template)

    old_behaviour = policy.proposal_kind_to_cortex.get(candidate.proposal_kind)
    assert resolve_cortex_route(candidate, policy) is old_behaviour


def test_a_kind_with_no_route_still_resolves_to_none():
    """The blocked path must stay reachable, or nothing is ever rejected."""
    policy = ExecutionDispatchPolicyV1(proposal_kind_to_cortex={})
    assert resolve_cortex_route(_candidate(), policy) is None


# ---------------------------------------------------------------------------
# The new capability.
# ---------------------------------------------------------------------------


def test_template_route_wins_over_the_kind_route():
    """Two maintenance actions can now coexist -- the point of the whole seam."""
    policy = ExecutionDispatchPolicyV1(
        proposal_kind_to_cortex={"maintain": _route("skills.runtime.builder_prune.v1")},
        template_to_cortex={"prune_dangling_images": _route("skills.runtime.image_prune.v1")},
    )
    image = _candidate(template="prune_dangling_images")
    cache = _candidate(template="prune_build_cache")

    assert resolve_cortex_route(image, policy).cortex_verb == "skills.runtime.image_prune.v1"
    # ...and the pre-existing one is untouched, still riding the kind route.
    assert resolve_cortex_route(cache, policy).cortex_verb == "skills.runtime.builder_prune.v1"


def test_template_route_applies_even_when_the_kind_has_no_route():
    """A template route must not require a kind route to exist as a fallback.

    Otherwise a brand-new proposal kind could never be routed by template
    alone, which would reintroduce the same cap one level up.
    """
    policy = ExecutionDispatchPolicyV1(
        proposal_kind_to_cortex={},
        template_to_cortex={"prune_dangling_images": _route("skills.runtime.image_prune.v1")},
    )
    assert (
        resolve_cortex_route(_candidate(template="prune_dangling_images"), policy).cortex_verb
        == "skills.runtime.image_prune.v1"
    )


def test_template_route_for_a_different_template_does_not_leak():
    """A declared override must apply ONLY to its own template.

    Without this, one override would silently capture every candidate of the
    same kind -- the exact failure the seam exists to remove, inverted.
    """
    policy = ExecutionDispatchPolicyV1(
        proposal_kind_to_cortex={"maintain": _route("skills.runtime.builder_prune.v1")},
        template_to_cortex={"prune_dangling_images": _route("skills.runtime.image_prune.v1")},
    )
    other = _candidate(template="some_other_maintenance_template")
    assert resolve_cortex_route(other, policy).cortex_verb == "skills.runtime.builder_prune.v1"


def test_candidate_with_no_template_ignores_the_template_map_entirely():
    policy = ExecutionDispatchPolicyV1(
        proposal_kind_to_cortex={"maintain": _route("skills.runtime.builder_prune.v1")},
        template_to_cortex={"prune_dangling_images": _route("skills.runtime.image_prune.v1")},
    )
    assert (
        resolve_cortex_route(_candidate(template=None), policy).cortex_verb
        == "skills.runtime.builder_prune.v1"
    )


# ---------------------------------------------------------------------------
# candidate_template_key
# ---------------------------------------------------------------------------


def test_template_key_reads_execution_intent():
    assert candidate_template_key(_candidate(template="prune_build_cache")) == "prune_build_cache"


@pytest.mark.parametrize("bad", [None, ""])
def test_template_key_treats_absent_or_blank_as_no_template(bad):
    """A blank string must not be looked up as a route key.

    `template_to_cortex.get("")` would be a silent miss today, but an empty key
    accidentally present in config would then match every template-less
    candidate. Treated as absent instead.
    """
    candidate = _candidate(template=bad)
    assert candidate_template_key(candidate) is None


def test_live_config_declares_exactly_the_shipped_template_routes():
    """Guard on the real shipped config, not a synthetic one.

    2026-08-13: this asserted `template_to_cortex == {}` -- the seam shipped
    inert on purpose, so a route landing here would be a deliberate, visible
    step with its own tests, not something riding along in an unrelated
    change (see git history for that original docstring).

    2026-08-16 (docs/superpowers/specs/2026-08-16-tension-driven-mutating-
    dispatch-design.md) is exactly that deliberate step: the first two
    template routes since the seam shipped, both with their own gating tests
    (see TestTensionRoutesGating below). This test now pins the live file to
    those three named entries -- still failing loudly if a fourth is ever
    added without a matching test, same purpose as before, updated contract.
    """
    from pathlib import Path

    from orion.execution_dispatch.policy import load_execution_dispatch_policy

    repo_root = Path(__file__).resolve().parents[1]
    policy = load_execution_dispatch_policy(
        repo_root / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml"
    )
    assert set(policy.template_to_cortex) == {
        "observe_tension_via_camera",
        "prune_dangling_images",
        "prune_stopped_containers",
    }
    assert (
        policy.template_to_cortex["observe_tension_via_camera"].cortex_verb
        == "skills.perception.look_at_camera.v1"
    )
    assert policy.template_to_cortex["observe_tension_via_camera"].allowed_scope == "inspect_only"
    assert (
        policy.template_to_cortex["prune_dangling_images"].cortex_verb
        == "skills.runtime.image_prune.v1"
    )
    assert (
        policy.template_to_cortex["prune_stopped_containers"].cortex_verb
        == "skills.runtime.docker_prune_stopped_containers.v1"
    )
    for key in ("prune_dangling_images", "prune_stopped_containers"):
        assert policy.template_to_cortex[key].allowed_scope == "maintenance_bounded"
