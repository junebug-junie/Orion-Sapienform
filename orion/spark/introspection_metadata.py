from __future__ import annotations

from typing import Any, Dict


def build_introspection_context(
    *,
    spark_meta: Dict[str, Any] | None,
    trace_id: str,
    correlation_id: str | None,
) -> Dict[str, Any]:
    spark_meta = spark_meta if isinstance(spark_meta, dict) else {}
    workflow = spark_meta.get("workflow") if isinstance(spark_meta.get("workflow"), dict) else {}
    workflow_id = (
        spark_meta.get("workflow_id")
        or workflow.get("workflow_id")
        or spark_meta.get("trace_verb")
    )
    personality_file = (
        spark_meta.get("personality_file")
        or (spark_meta.get("plan_metadata") or {}).get("personality_file")
    )
    session_id = spark_meta.get("session_id") or spark_meta.get("conversation_id") or spark_meta.get("thread_id")
    user_id = spark_meta.get("user_id")
    return {
        "correlation_id": correlation_id,
        "trace_id": trace_id,
        "session_id": session_id,
        "user_id": user_id,
        "workflow_id": workflow_id,
        "workflow_status": workflow.get("status") if isinstance(workflow, dict) else None,
        "trace_verb": spark_meta.get("trace_verb"),
        "identity_kernel_source": spark_meta.get("identity_kernel_source"),
        "personality_file": personality_file,
    }
