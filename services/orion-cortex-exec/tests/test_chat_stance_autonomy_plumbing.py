from __future__ import annotations

import logging

from app import chat_stance


class _Lookup:
    def __init__(
        self,
        subject: str,
        availability: str,
        state=None,
        unavailable_reason: str | None = None,
        subquery_diagnostics: dict | None = None,
    ):
        self.subject = subject
        self.availability = availability
        self.state = state
        self.unavailable_reason = unavailable_reason
        self.subquery_diagnostics = subquery_diagnostics or {}


class _Repo:
    def __init__(self, lookups):
        self._lookups = lookups

    def list_latest(self, subjects, *, observer=None):
        return [self._lookups.get(s, _Lookup(s, "empty", None)) for s in subjects]

    def status(self):
        class _Status:
            backend = "graph"
            source_path = "graphdb:test"
            source_available = True

        return _Status()


def test_chat_stance_inputs_include_autonomy_summary_when_available(monkeypatch) -> None:
    from orion.autonomy.models import AutonomyStateV1

    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="orion",
        dominant_drive="coherence",
        drive_pressures={"coherence": 0.9},
        active_drives=["coherence"],
        tension_kinds=["scope_sprawl"],
        source="graph",
    )
    repo = _Repo({"orion": _Lookup("orion", "available", state), "relationship": _Lookup("relationship", "empty", None)})
    monkeypatch.setattr(chat_stance, "build_autonomy_repository", lambda **_: repo)

    ctx = {"user_message": "help me synthesize this"}
    built = chat_stance.build_chat_stance_inputs(ctx)

    assert built["autonomy"]["summary"]["stance_hint"] == "favor synthesis and reduction"
    assert ctx["chat_autonomy_state"]["subject"] == "orion"
    assert "chat_autonomy_summary" in ctx


def test_chat_stance_unavailable_autonomy_keeps_behavior_stable(monkeypatch) -> None:
    repo = _Repo({})
    monkeypatch.setattr(chat_stance, "build_autonomy_repository", lambda **_: repo)

    ctx = {"user_message": "can you help"}
    chat_stance.build_chat_stance_inputs(ctx)
    fb = chat_stance.fallback_chat_stance_brief(ctx)

    assert fb.answer_strategy == "DirectAnswer"
    assert fb.task_mode == "direct_response"


def test_chat_stance_inputs_include_mutation_adaptation_context_from_metadata(monkeypatch) -> None:
    repo = _Repo({})
    monkeypatch.setattr(chat_stance, "build_autonomy_repository", lambda **_: repo)
    ctx = {
        "user_message": "help me with current routing posture",
        "metadata": {
            "mutation_cognition_context": {
                "mutation_scope": "routing_threshold_patch_only",
                "live_ramp_active": True,
                "live_surface": {"value": 0.58},
                "latest_routing_adoption": {"adoption_id": "adopt-1", "proposal_id": "prop-1"},
                "latest_routing_rollback": {"rollback_id": "rb-1", "proposal_id": "prop-1"},
            }
        },
    }
    built = chat_stance.build_chat_stance_inputs(ctx)
    assert built["mutation_adaptation"]["mutation_scope"] == "routing_threshold_patch_only"
    assert built["mutation_adaptation"]["latest_routing_adoption"]["adoption_id"] == "adopt-1"
    assert built["mutation_adaptation"]["latest_routing_rollback"]["rollback_id"] == "rb-1"
    assert ctx["chat_mutation_cognition_context"]["live_ramp_active"] is True


def test_chat_stance_autonomy_debug_contains_unavailable_reason(monkeypatch) -> None:
    monkeypatch.setenv("AUTONOMY_GRAPH_TIMEOUT_SEC", "4.5")
    monkeypatch.delenv("AUTONOMY_SUBJECT_MAX_WORKERS", raising=False)
    monkeypatch.setenv("AUTONOMY_SUBQUERY_MAX_WORKERS", "1")
    repo = _Repo(
        {
            "orion": _Lookup(
                "orion",
                "unavailable",
                None,
                unavailable_reason="timeout",
                subquery_diagnostics={"identity": {"status": "timeout", "elapsed_ms": 4501.2, "row_count": 0}},
            )
        }
    )
    monkeypatch.setattr(chat_stance, "build_autonomy_repository", lambda **_: repo)

    ctx = {"user_message": "hello"}
    chat_stance.build_chat_stance_inputs(ctx)

    assert ctx["chat_autonomy_debug"]["orion"]["availability"] == "unavailable"
    assert ctx["chat_autonomy_debug"]["orion"]["unavailable_reason"] == "timeout"
    assert ctx["chat_autonomy_debug"]["orion"]["subqueries"]["identity"]["status"] == "timeout"
    assert ctx["chat_autonomy_debug"]["_runtime"]["backend"] == "graph"
    assert ctx["chat_autonomy_debug"]["_runtime"]["timeout_sec"] == 4.5
    assert ctx["chat_autonomy_debug"]["_runtime"]["subject_max_workers"] == 3
    assert ctx["chat_autonomy_debug"]["_runtime"]["subquery_max_workers"] == 1
    assert "repository_status" in ctx["chat_autonomy_debug"]["_runtime"]
    assert ctx["chat_autonomy_backend"] == "graph"
    assert "chat_autonomy_repository_status" in ctx


def test_autonomy_lookup_turn_log_distinguishes_empty_and_unavailable(monkeypatch, caplog) -> None:
    from orion.autonomy.models import AutonomyStateV1

    state = AutonomyStateV1(
        subject="juniper",
        model_layer="user-model",
        entity_id="user:juniper",
        source="graph",
    )
    repo = _Repo(
        {
            "orion": _Lookup("orion", "empty", None),
            "relationship": _Lookup("relationship", "unavailable", None, unavailable_reason="query_error"),
            "juniper": _Lookup("juniper", "available", state=state),
        }
    )
    monkeypatch.setattr(chat_stance, "build_autonomy_repository", lambda **_: repo)
    caplog.set_level(logging.INFO)

    chat_stance.build_chat_stance_inputs({"user_message": "hello"})

    assert "autonomy_lookup_turn" in caplog.text
    assert '"availability_counts": {"available": 1, "empty": 1, "partial": 0, "unavailable": 1}' in caplog.text
    assert '"selected_subject_availability": "unavailable"' in caplog.text


def test_triage_mode_not_overridden_by_autonomy_hint(monkeypatch) -> None:
    repo = _Repo({})
    monkeypatch.setattr(chat_stance, "build_autonomy_repository", lambda **_: repo)

    ctx = {
        "user_message": "I'm super pissed; workflows are offline and gpu is dead",
        "chat_social_summary": {"social_posture": ["technical", "strained"], "relationship_facets": ["shared build"]},
        "chat_social_bridge_summary": {"posture": ["operational_frustration", "technical"], "hazards": []},
        "chat_autonomy_summary": {
            "stance_hint": "favor synthesis and reduction",
            "response_hazards": ["avoid contradictory framing"],
        },
    }
    fb = chat_stance.fallback_chat_stance_brief(ctx)
    assert fb.task_mode == "triage"
    assert fb.identity_salience == "low"
    assert "triage_operational_blockers_first" in fb.response_priorities


def test_autonomy_graphdb_config_falls_back_to_concept_profile(monkeypatch) -> None:
    for key in (
        "GRAPHDB_QUERY_ENDPOINT",
        "GRAPHDB_URL",
        "GRAPHDB_REPO",
        "GRAPHDB_USER",
        "GRAPHDB_PASS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CONCEPT_PROFILE_GRAPHDB_URL", "http://orion-athena-graphdb:7200")
    monkeypatch.setenv("CONCEPT_PROFILE_GRAPHDB_REPO", "collapse")
    monkeypatch.setenv("CONCEPT_PROFILE_GRAPHDB_USER", "admin")
    monkeypatch.setenv("CONCEPT_PROFILE_GRAPHDB_PASS", "admin")

    cfg = chat_stance.resolve_autonomy_graphdb_config()

    assert cfg["endpoint"] == "http://orion-athena-graphdb:7200/repositories/collapse"
    assert cfg["repo"] == "collapse"
    assert cfg["user"] == "admin"
    assert cfg["password"] == "admin"
    assert cfg["source"] == "concept_profile_fallback"


def test_autonomy_graphdb_config_prefers_generic_vars(monkeypatch) -> None:
    monkeypatch.setenv("GRAPHDB_URL", "http://generic-graphdb:7200")
    monkeypatch.setenv("GRAPHDB_REPO", "generic-repo")
    monkeypatch.setenv("GRAPHDB_USER", "generic-user")
    monkeypatch.setenv("GRAPHDB_PASS", "generic-pass")
    monkeypatch.setenv("CONCEPT_PROFILE_GRAPHDB_URL", "http://orion-athena-graphdb:7200")
    monkeypatch.setenv("CONCEPT_PROFILE_GRAPHDB_REPO", "collapse")
    monkeypatch.setenv("CONCEPT_PROFILE_GRAPHDB_USER", "admin")
    monkeypatch.setenv("CONCEPT_PROFILE_GRAPHDB_PASS", "admin")

    cfg = chat_stance.resolve_autonomy_graphdb_config()

    assert cfg["endpoint"] == "http://generic-graphdb:7200/repositories/generic-repo"
    assert cfg["repo"] == "generic-repo"
    assert cfg["user"] == "generic-user"
    assert cfg["password"] == "generic-pass"
    assert cfg["source"] == "generic_graphdb"


def test_autonomy_timeout_prefers_autonomy_specific_env(monkeypatch) -> None:
    monkeypatch.setenv("GRAPHDB_TIMEOUT_SEC", "4.5")
    monkeypatch.setenv("AUTONOMY_GRAPH_TIMEOUT_SEC", "9.25")

    assert chat_stance.resolve_autonomy_graph_timeout_sec() == 9.25


def test_autonomy_subject_max_workers_env(monkeypatch) -> None:
    monkeypatch.delenv("AUTONOMY_SUBJECT_MAX_WORKERS", raising=False)
    assert chat_stance.resolve_autonomy_subject_max_workers() == 3

    monkeypatch.setenv("AUTONOMY_SUBJECT_MAX_WORKERS", "2")
    assert chat_stance.resolve_autonomy_subject_max_workers() == 2

    monkeypatch.setenv("AUTONOMY_SUBJECT_MAX_WORKERS", "0")
    assert chat_stance.resolve_autonomy_subject_max_workers() == 1


def test_autonomy_subquery_max_workers_env(monkeypatch) -> None:
    monkeypatch.delenv("AUTONOMY_SUBQUERY_MAX_WORKERS", raising=False)
    assert chat_stance.resolve_autonomy_subquery_max_workers() == 1

    monkeypatch.setenv("AUTONOMY_SUBQUERY_MAX_WORKERS", "3")
    assert chat_stance.resolve_autonomy_subquery_max_workers() == 3

    monkeypatch.setenv("AUTONOMY_SUBQUERY_MAX_WORKERS", "99")
    assert chat_stance.resolve_autonomy_subquery_max_workers() == 3

    monkeypatch.setenv("AUTONOMY_SUBQUERY_MAX_WORKERS", "0")
    assert chat_stance.resolve_autonomy_subquery_max_workers() == 1
