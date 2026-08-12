"""Regression test for the container-bringup skill activation gap.

Live incident (2026-08-12): `skills.docker.compose_service_bringup.v1` was registered in
`services/orion-cortex-exec/app/verb_adapters.py` (via `@verb(...)`) but had no matching
`orion/cognition/verbs/<name>.yaml`. `orion.cognition.verb_activation.is_active()` treats an
undiscovered verb (no yaml, not in an explicit allow list) as inactive regardless of code
registration, so the skill returned "Verb '...' is inactive on node athena." even though it
was fully implemented and policy-enabled. This asserts the yaml exists and the verb resolves
active on a representative node, so a future new skill missing its activation yaml fails a
fast local test instead of only surfacing at live dispatch time.
"""

from __future__ import annotations

from orion.cognition import verb_activation

VERB_NAME = "skills.docker.compose_service_bringup.v1"


def test_docker_compose_bringup_yaml_exists() -> None:
    verb_activation.refresh_verb_activation_cache()
    assert VERB_NAME in verb_activation.list_all_verbs(), (
        f"{VERB_NAME} has no orion/cognition/verbs/{VERB_NAME}.yaml -- "
        "verb_activation.is_active() will report it inactive on every node "
        "regardless of code registration or policy flags."
    )


def test_docker_compose_bringup_is_active_on_athena() -> None:
    verb_activation.refresh_verb_activation_cache()
    assert verb_activation.is_active(VERB_NAME, node_name="athena") is True


def test_docker_compose_bringup_is_active_with_no_node() -> None:
    # Hub's dispatch path may omit node_name; default allow/deny rules must still resolve.
    verb_activation.refresh_verb_activation_cache()
    assert verb_activation.is_active(VERB_NAME, node_name=None) is True
