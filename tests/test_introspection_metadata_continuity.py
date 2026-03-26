from orion.spark.introspection_metadata import build_introspection_context


def test_introspection_context_preserves_workflow_identity_and_personality_metadata() -> None:
    continuity = build_introspection_context(
        spark_meta={
            "session_id": "session-1",
            "user_id": "user-1",
            "trace_verb": "self_review",
            "workflow": {"workflow_id": "self_review", "status": "completed"},
            "personality_file": "orion/cognition/personality/orion_identity.yaml",
        },
        trace_id="trace-123",
        correlation_id="corr-123",
    )
    assert continuity["correlation_id"] == "corr-123"
    assert continuity["trace_id"] == "trace-123"
    assert continuity["session_id"] == "session-1"
    assert continuity["workflow_id"] == "self_review"
    assert continuity["personality_file"] == "orion/cognition/personality/orion_identity.yaml"
