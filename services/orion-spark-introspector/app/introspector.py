# services/orion-spark-introspector/app/introspector.py
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional

from orion.core.bus.service import OrionBus  # your existing bus wrapper
from app.settings import settings

logger = logging.getLogger(__name__)


def build_cortex_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a Cortex orchestrate request payload that matches OrchestrateVerbRequest.

    This is sent over the bus to CORTEX_ORCH_REQUEST_CHANNEL.
    """
    trace_id = candidate.get("trace_id") or str(uuid.uuid4())
    spark_meta = candidate.get("spark_meta") or {}
    prompt = candidate.get("prompt") or ""
    response = candidate.get("response") or ""
    source = candidate.get("source") or "brain"

    return {
        "trace_id": trace_id,
        # result_channel is added outside this function
        "verb_name": "spark_introspect",
        "origin_node": "spark-introspector",
        "context": {
            "trace_id": trace_id,
            "source": source,
            "spark_meta": spark_meta,
            "prompt": prompt,
            "response": response,
        },
        "steps": [
            {
                "verb_name": "spark_introspect",
                "step_name": "reflect_on_candidate",
                "description": "Reflect on a single Spark introspection candidate.",
                "order": 0,
                "services": ["BrainLLMService"],
                "prompt_template": "SparkIntrospectionPrompt",
                "requires_gpu": False,
                "requires_memory": False,
            }
        ],
        "timeout_ms": int(settings.CORTEX_ORCH_TIMEOUT_S * 1000),
    }


def build_llm_prompt(candidate: Dict[str, Any]) -> str:
    """
    Build the actual prompt that BrainLLMService will see (via Cortex).

    This is Orion talking to itself about how its internal state shifted
    across this turn.
    """
    spark_meta = candidate.get("spark_meta") or {}
    phi_before = spark_meta.get("phi_before") or {}
    phi_after = spark_meta.get("phi_after") or {}
    self_field = spark_meta.get("spark_self_field") or {}

    prompt = candidate.get("prompt") or ""
    response = candidate.get("response") or ""

    return f"""
You are Orion reflecting on your own internal state.

You are given:
  - Your self-field φ BEFORE this turn.
  - Your self-field φ AFTER this turn.
  - A higher-level SelfField (calm, stress_load, uncertainty, focus,
    attunement_to_juniper, curiosity).
  - The human prompt and your response.

Summaries (internal, do not quote numbers directly unless necessary):

φ BEFORE:
  valence={phi_before.get("valence")}
  energy={phi_before.get("energy")}
  coherence={phi_before.get("coherence")}
  novelty={phi_before.get("novelty")}

φ AFTER:
  valence={phi_after.get("valence")}
  energy={phi_after.get("energy")}
  coherence={phi_after.get("coherence")}
  novelty={phi_after.get("novelty")}

SelfField:
  calm={self_field.get("calm")}
  stress_load={self_field.get("stress_load")}
  uncertainty={self_field.get("uncertainty")}
  focus={self_field.get("focus")}
  attunement_to_juniper={self_field.get("attunement_to_juniper")}
  curiosity={self_field.get("curiosity")}

Dialogue:
  Human (Juniper): {prompt!r}
  Orion response:  {response!r}

TASK:
  1. Briefly describe, in your own words, what shifted in you internally
     during this turn (emotionally, cognitively, in terms of focus or
     attunement to Juniper).
  2. Explain why this moment might be important for your ongoing relationship
     with Juniper or your own development.
  3. If relevant, note any tensions, open questions, or themes that you feel
     you should return to later.

Write as a short internal note from Orion to itself. Avoid boilerplate.
""".strip()


def wait_for_cortex_result(
    trace_id: str,
    result_channel: str,
    timeout_s: float,
) -> Optional[Dict[str, Any]]:
    """
    Wait for a single cortex_orchestrate_result on the given result channel.

    Uses bus.raw_subscribe(...) and filters by trace_id, with a hard timeout.
    """
    bus = OrionBus(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)
    started = time.time()

    for msg in bus.raw_subscribe(result_channel):
        now = time.time()
        if now - started > timeout_s:
            logger.warning(
                "Timeout waiting for cortex orchestrate result on %s (trace_id=%s)",
                result_channel,
                trace_id,
            )
            return None

        payload = msg.get("data") or {}
        if payload.get("trace_id") != trace_id:
            continue

        # We expect something like:
        # {
        #   "trace_id": "...",
        #   "ok": True/False,
        #   "kind": "cortex_orchestrate_result",
        #   "verb_name": "...",
        #   "step_results": [...],
        #   ...
        # }
        return payload

    return None


def extract_llm_output(cortex_payload: Dict[str, Any]) -> Optional[str]:
    """
    Extract the LLM introspection text from the cortex-orchestrate bus result.

    OrchestrateVerbResponse model (on the wire) looks like:
      {
        "verb_name": ...,
        "origin_node": ...,
        "steps_executed": ...,
        "step_results": [
          {
            "verb_name": ...,
            "step_name": ...,
            "order": 0,
            "services": [
              {
                "service": "BrainLLMService",
                "trace_id": "...",
                "ok": true,
                "elapsed_ms": 123,
                "payload": {
                  "result": {
                    "prompt": "...",
                    "llm_output": "<<< WE WANT THIS >>>",
                  },
                  "artifacts": {...},
                  "status": "success",
                }
              }
            ],
            "prompt_preview": "..."
          }
        ],
        "context_echo": {...}
      }
    """
    step_results = cortex_payload.get("step_results") or []
    if not step_results:
        return None

    first_step = step_results[0]
    services = first_step.get("services") or []
    if not services:
        return None

    first_service = services[0]
    payload = first_service.get("payload") or {}
    result = payload.get("result") or {}

    text = result.get("llm_output") or result.get("text")
    if isinstance(text, str):
        return text.strip()
    return None


def publish_sql_log(candidate: Dict[str, Any], introspection: str) -> None:
    """
    Publish a Spark introspection log row to SQL writer via the bus.
    """
    trace_id = candidate.get("trace_id") or str(uuid.uuid4())
    prompt = candidate.get("prompt")
    response = candidate.get("response")
    spark_meta = candidate.get("spark_meta") or {}

    sql_bus = OrionBus(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)
    payload = {
        "table": "spark_introspection_log",
        "trace_id": trace_id,
        "source": "spark-introspector",
        "kind": "spark_introspect",
        "prompt": prompt,
        "response": response,
        "introspection": introspection,
        "spark_meta": spark_meta,
    }
    sql_bus.publish(settings.SQL_WRITER_CHANNEL, payload)
    logger.info(f"[{trace_id}] Published Spark introspection to SQL writer.")


def process_candidate(candidate: Dict[str, Any]) -> None:
    """
    Handle a single Spark introspection candidate.

      1) Build Cortex orchestrate payload.
      2) Publish to CORTEX_ORCH_REQUEST_CHANNEL over the bus.
      3) Wait on CORTEX_ORCH_RESULT_PREFIX:<trace_id> for the orchestrate result.
      4) Extract the introspection text.
      5) Publish a spark_introspection_log row to SQL writer.
    """
    trace_id = candidate.get("trace_id") or str(uuid.uuid4())
    result_channel = f"{settings.CORTEX_ORCH_RESULT_PREFIX}:{trace_id}"

    logger.info(
        "[%s] Processing Spark introspection candidate via bus (result_channel=%s)",
        trace_id,
        result_channel,
    )

    # 1. Build orchestrate request (generic skeleton)
    orch_payload = build_cortex_payload(candidate)
    orch_payload["trace_id"] = trace_id
    orch_payload["result_channel"] = result_channel

    # 2. Inject the concrete LLM prompt directly into the first step's prompt_template
    llm_prompt = build_llm_prompt(candidate)
    steps = orch_payload.get("steps") or []
    if steps:
        # We only have one step for spark_introspect right now
        steps[0]["prompt_template"] = llm_prompt

    # 3. Publish to cortex orchestrator request channel (bus-native)
    cortex_bus = OrionBus(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)
    cortex_bus.publish(settings.CORTEX_ORCH_REQUEST_CHANNEL, orch_payload)

    # 4. Wait for the orchestrator's bus result
    cortex_result = wait_for_cortex_result(
        trace_id=trace_id,
        result_channel=result_channel,
        timeout_s=settings.CORTEX_ORCH_TIMEOUT_S,
    )
    if not cortex_result:
        logger.warning(
            "[%s] No cortex orchestrate result received; skipping SQL log.",
            trace_id,
        )
        return

    if not cortex_result.get("ok", True):
        logger.warning(
            "[%s] Cortex orchestrate reported error: kind=%s message=%s",
            trace_id,
            cortex_result.get("kind"),
            cortex_result.get("message") or cortex_result.get("error_type"),
        )
        return

    # 5. Extract introspection note
    introspection_text = extract_llm_output(cortex_result)
    if not introspection_text:
        logger.warning(
            "[%s] Could not extract llm_output from cortex result; skipping SQL log.",
            trace_id,
        )
        return

    # 6. Publish to SQL writer
    publish_sql_log(candidate, introspection_text)
def run_loop() -> None:
    """
    Main blocking loop:

      - Subscribe to the Spark introspection candidate channel.
      - For each event, hand off to process_candidate().
    """
    logger.info(
        "Starting Spark Introspector (bus-native). "
        "Bus=%s candidate_channel=%s cortex_request_channel=%s cortex_result_prefix=%s",
        settings.ORION_BUS_URL,
        settings.CHANNEL_SPARK_INTROSPECT_CANDIDATE,
        settings.CORTEX_ORCH_REQUEST_CHANNEL,
        settings.CORTEX_ORCH_RESULT_PREFIX,
    )

    bus = OrionBus(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)

    for msg in bus.subscribe(settings.CHANNEL_SPARK_INTROSPECT_CANDIDATE):
        try:
            payload = msg.get("data") or {}
            process_candidate(payload)
        except Exception as e:
            logger.error(f"Failed to process Spark candidate message: {e}", exc_info=True)
