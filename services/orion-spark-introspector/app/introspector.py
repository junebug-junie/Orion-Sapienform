# services/orion-spark-introspector/app/introspector.py
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

import httpx

from orion.core.bus.service import OrionBus  # <-- your existing class
from app.settings import settings

logger = logging.getLogger(__name__)


def build_cortex_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a single-step Cortex orchestration payload for a Spark introspection.

    This calls the 'spark_introspect' verb with a single step 'reflect_on_candidate',
    targeting BrainLLMService.
    """
    trace_id = candidate.get("trace_id") or str(uuid.uuid4())
    spark_meta = candidate.get("spark_meta") or {}
    prompt = candidate.get("prompt") or ""
    response = candidate.get("response") or ""
    source = candidate.get("source") or "brain"

    return {
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
        "timeout_ms": 10000,
    }


def build_llm_prompt(candidate: Dict[str, Any]) -> str:
    """
    Build the actual prompt that BrainLLMService will see via Cortex.

    This is Orion talking to itself about how its inner state changed.
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


def call_cortex_orchestrator(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a single orchestration request to the Cortex orchestrator over HTTP.
    """
    url = settings.CORTEX_ORCH_URL
    timeout = httpx.Timeout(settings.CONNECT_TIMEOUT, read=settings.READ_TIMEOUT)

    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


def extract_llm_output(orch_result: Dict[str, Any]) -> Optional[str]:
    """
    Extract the LLM introspection text from the orchestrator result.

    Adapt this if your orchestrator uses a slightly different shape.
    """
    steps = orch_result.get("steps") or []
    if not steps:
        return None

    first = steps[0]
    if not isinstance(first, dict):
        return None

    result = first.get("result") or {}
    text = result.get("llm_output") or result.get("text")
    return text.strip() if isinstance(text, str) else None


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
    Handle a single Spark introspection candidate:

      1) Build Cortex orchestration payload.
      2) Call /orchestrate.
      3) Extract LLM introspection text.
      4) Publish a spark_introspection_log row to SQL writer.
    """
    trace_id = candidate.get("trace_id") or "unknown"
    logger.info(f"[{trace_id}] Processing Spark introspection candidate...")

    # 1. Build orchestrator payload
    orch_payload = build_cortex_payload(candidate)

    # 2. Inject the concrete prompt into step.args so BrainLLMService sees it
    llm_prompt = build_llm_prompt(candidate)
    if orch_payload["steps"]:
        orch_payload["steps"][0]["args"] = {"llm_prompt": llm_prompt}

    try:
        # 3. Call Cortex orchestrator
        orch_result = call_cortex_orchestrator(orch_payload)
        logger.info(f"[{trace_id}] Spark introspection completed via Cortex.")

        # 4. Extract introspection note
        llm_text = extract_llm_output(orch_result)
        if not llm_text:
            logger.warning(f"[{trace_id}] No llm_output found in Cortex result; skipping SQL log.")
            return

        # 5. Publish to SQL writer
        publish_sql_log(candidate, llm_text)

    except Exception as e:
        logger.error(f"[{trace_id}] Spark introspection failed: {e}", exc_info=True)


def run_loop() -> None:
    """
    Main blocking loop:

      - Subscribe to the Spark introspection candidate channel.
      - For each event, hand off to process_candidate().
    """
    logger.info(
        f"Starting Spark Introspector. "
        f"Bus={settings.ORION_BUS_URL} "
        f"channel={settings.CHANNEL_SPARK_INTROSPECT_CANDIDATE} "
        f"cortex={settings.CORTEX_ORCH_URL}"
    )

    bus = OrionBus(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)

    # Using your OrionBus.subscribe API
    for msg in bus.subscribe(settings.CHANNEL_SPARK_INTROSPECT_CANDIDATE):
        try:
            # msg["data"] is already parsed JSON dict
            payload = msg.get("data") or {}
            process_candidate(payload)
        except Exception as e:
            logger.error(f"Failed to process Spark candidate message: {e}", exc_info=True)
