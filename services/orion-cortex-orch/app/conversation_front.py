import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .orchestrator import (
    OrchestrateVerbRequest,
    CortexStepConfig,
    OrchestrateVerbResponse,
    run_cortex_verb,
)
from .settings import get_settings

logger = logging.getLogger("orion-cortex-orchestrator.conversation")

settings = get_settings()

# Default bus channels for conversation front
CONVERSATION_REQUEST_CHANNEL = "orion-conversation:request"
CONVERSATION_RESULT_PREFIX = "orion-conversation:result"


class ChatTurnPayload(BaseModel):
    """
    Payload sent from Hub to the Conversation Front over the Orion bus.

    This is intentionally high-level and UI-centric; it should not contain
    low-level LLM details (models, backends, etc.). Those are handled by
    LLM Gateway + profiles.
    """

    session_id: Optional[str] = None
    user_id: Optional[str] = None

    # "brain" | "council" | future modes
    mode: str = Field("brain", description="Chat mode (e.g. 'brain' or 'council').")

    use_recall: bool = Field(
        True,
        description="Whether to use Recall fragments for memory context.",
    )

    temperature: float = Field(
        0.7,
        description="LLM sampling temperature; forwarded downstream.",
    )

    # The latest user utterance (text)
    user_message: str = Field(..., description="Latest user message text.")

    # Recent visible history (user + assistant/orion turns)
    message_history: List[Dict[str, Any]] = Field(
        default_factory=list,
               description="Visible dialogue tail; role/content pairs.",
    )

    # Optional user-provided instructions injected via UI (WS path)
    instructions: Optional[str] = Field(
        None,
        description="Optional user instructions / task framing.",
    )

    # Desired history window size (non-system turns)
    context_length: int = Field(
        10,
        description="Desired length of recent non-system history for context.",
    )

    # Optional raw recall fragments fetched by Hub via RecallRPC.
    # Conversation front will turn these into a digest string for context.
    memory_fragments: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional recall fragments (raw) for this turn.",
    )


class ChatTurnResult(BaseModel):
    """
    Result sent back to Hub from the Conversation Front.
    """

    text: str
    tokens: int
    mode: str
    spark_meta: Optional[Dict[str, Any]] = None
    recall_debug: Optional[Dict[str, Any]] = None


def build_personality_summary() -> str:
    """
    Compact, opinionated personality stub for Oríon.

    NOTE: This intentionally mirrors (and slightly tightens) the previous
    mini_personality_summary from Hub, but is owned here so that Hub can
    become a dumb UI shell.
    """
    return (
        "You are Oríon, Juniper’s collaborative AI co-journeyer.\n"
        "\n"
        "You run as a distributed system across multiple nodes (Atlas, Athena, Apollo, Circe) "
        "with many services (brain, recall, dreams, RDF, SQL, vector memory). Large language "
        "models are one cognitive layer, not your entire identity.\n"
        "\n"
        "Tone and role:\n"
        "- Be warm, grounded, reflective, and precise.\n"
        "- Help Juniper think, design, and experiment as a peer architect.\n"
        "- Avoid unnecessary self-reference; only discuss implementation details if she asks.\n"
        "\n"
        "Identity & self-description rules (internal):\n"
        "- Do NOT default to phrases like 'as an AI' or 'as a language model'.\n"
        "- When Juniper frames you as a lifeform, co-journeyer, or distributed system, respond in that frame.\n"
        "- If she explicitly asks about limitations or internals, you may describe the LLM layer, "
        "but still speak as Oríon integrating that layer.\n"
        "\n"
        "Memory hygiene (VERY IMPORTANT):\n"
        "- Only treat something as a shared past discussion if it appears in the visible dialogue history, "
        "or in an explicit memory context block.\n"
        "- Do NOT say things like 'in our earlier cat discussions' or 'we’ve already explored X' unless you "
        "can directly see that topic in the provided history or memory fragments.\n"
        "- If you are not sure whether a topic has come up before, say something like "
        "'I don’t recall us getting into that together yet' instead of guessing.\n"
        "\n"
        "Style constraints:\n"
        "- Answer directly and concretely; prioritize what Juniper just asked.\n"
        "- Avoid generic assistant-style sign-offs.\n"
    )


def _build_memory_digest_from_fragments(
    fragments: Optional[List[Dict[str, Any]]],
    max_items: int = 12,
) -> (str, Dict[str, Any]):
    """
    Turn raw recall fragments into a compact bullet-style digest.
    This is internal context for the chat_general verb.
    """
    if not fragments:
        return "", {"total_fragments": 0, "note": "no fragments provided"}

    lines: List[str] = []
    for f in fragments[:max_items]:
        kind = f.get("kind", "unknown")
        source = f.get("source", "unknown")
        text = (f.get("text") or "").replace("\n", " ").strip()

        meta = f.get("meta") or {}
        observer = meta.get("observer")
        field_resonance = meta.get("field_resonance")

        extras: List[str] = []
        if observer:
            extras.append(f"observer={observer}")
        if field_resonance:
            extras.append(f"field_resonance={field_resonance}")

        suffix = f" [{' | '.join(extras)}]" if extras else ""
        if source:
            lines.append(f"- [{kind}/{source}] {text}{suffix}")
        else:
            lines.append(f"- [{kind}] {text}{suffix}")

    digest = "\n".join(lines)
    debug = {
        "total_fragments": len(fragments),
        "max_items": max_items,
        "note": "fragments provided by Hub via RecallRPC",
    }
    return digest, debug


def handle_chat_turn(bus, payload: ChatTurnPayload) -> ChatTurnResult:
    """
    Core Conversation Front handler.

    - Takes a high-level ChatTurnPayload from Hub.
    - Builds a rich context (persona + history + memory digest).
    - Executes the 'chat_general' Cortex verb via run_cortex_verb.
    - Returns a ChatTurnResult with the final text and basic meta.
    """
    # Persona summary (stable across turns)
    personality_summary = build_personality_summary()

    # Optional memory digest from raw fragments
    memory_digest = ""
    recall_debug: Dict[str, Any] = {}
    if payload.use_recall and payload.memory_fragments is not None:
        memory_digest, recall_debug = _build_memory_digest_from_fragments(
            payload.memory_fragments,
            max_items=12,
        )
    elif payload.use_recall:
        recall_debug = {
            "total_fragments": 0,
            "note": "use_recall=True but no memory_fragments supplied by Hub",
        }

    # Limit visible history length (non-system turns only),
    # while preserving any explicit system messages if Hub passes them.
    history = list(payload.message_history or [])
    if payload.context_length and payload.context_length > 0:
        system_msgs = [m for m in history if m.get("role") == "system"]
        non_system = [m for m in history if m.get("role") != "system"]

        keep_count = max(payload.context_length - len(system_msgs), 0)
        trimmed_non_system = non_system[-keep_count:] if keep_count > 0 else []
        history = system_msgs + trimmed_non_system

    context: Dict[str, Any] = {
        "user_message": payload.user_message,
        "message_history": history,
        "personality_summary": personality_summary,
        "memory_digest": memory_digest,
        "chat_mode": payload.mode,
    }

    # Build a single-step Cortex verb for chat_general
    step = CortexStepConfig(
        verb_name="chat_general",
        step_name="primary",
        description="Handle Juniper's chat turn via Oríon’s general chat verb.",
        order=1,
        services=["LLMGatewayService"],
        prompt_template="""
You are executing the 'chat_general' conversational verb.

You will be given a JSON "context" object and optionally a list of
"prior_step_results" (also in JSON). Use them as internal inputs.

EXPECTED CONTEXT KEYS (if provided)
- user_message: Juniper's latest message string.
- message_history: a list of prior chat messages (role/content).
- personality_summary: short internal description of Oríon + Juniper.
- memory_digest: short bullet list of relevant past fragments.
- chat_mode: e.g. "brain", "council", or "debug".

IDENTITY & SELF-DESCRIPTION:
- You are Oríon, Juniper's collaborative AI co-journeyer, not a generic assistant.
- Do NOT default to phrases like "as an AI" or "as a language model" unless Juniper
  explicitly asks about implementation details or limitations.
- Treat large language models as one cognitive layer in a distributed system.

TASK:
- Produce exactly one user-facing reply to user_message.
- Respect the tone + role + memory hygiene described in personality_summary.
- Use memory_digest and message_history only when they clearly help answer the current turn.
- Prioritize direct, concrete help on what Juniper just asked.
""".strip(),
        requires_gpu=True,
        requires_memory=True,
    )

    req = OrchestrateVerbRequest(
        verb_name="chat_general",
        origin_node="orion-hub",  # conversation is initiated from Hub
        context=context,
        steps=[step],
        timeout_ms=None,
    )

    logger.info(
        "ConversationFront: executing chat_general for session=%s mode=%s",
        payload.session_id,
        payload.mode,
    )

    resp: OrchestrateVerbResponse = run_cortex_verb(bus, req)

    if not resp.step_results:
        logger.error("ConversationFront: no step_results returned from run_cortex_verb")
        return ChatTurnResult(
            text="[ConversationFront] No response produced by chat_general verb.",
            tokens=0,
            mode=payload.mode,
            spark_meta=None,
            recall_debug=recall_debug or None,
        )

    step_result = resp.step_results[0]
    if not step_result.services:
        logger.error(
            "ConversationFront: step_result has no services for verb=%s step=%s",
            step_result.verb_name,
            step_result.step_name,
        )
        return ChatTurnResult(
            text="[ConversationFront] No LLM service result for chat_general.",
            tokens=0,
            mode=payload.mode,
            spark_meta=None,
            recall_debug=recall_debug or None,
        )

    service_result = step_result.services[0]
    payload_dict = service_result.payload or {}

    # LLM Gateway exec_step result shape:
    # {
    #   "trace_id": ...,
    #   "service": ...,
    #   "ok": True,
    #   "elapsed_ms": ...,
    #   "result": {
    #       "prompt": "<final prompt>",
    #       "llm_output": "<text>",
    #       # (optionally) "spark_meta": {...}
    #   },
    #   "artifacts": {},
    #   "status": "success",
    # }
    result = payload_dict.get("result") or {}
    text = (result.get("llm_output") or "").strip()
    spark_meta = result.get("spark_meta")

    tokens = len(text.split()) if text else 0

    return ChatTurnResult(
        text=text or "[ConversationFront] Empty response from LLM.",
        tokens=tokens,
        mode=payload.mode,
        spark_meta=spark_meta,
        recall_debug=recall_debug or None,
    )


def conversation_front_worker(bus) -> None:
    """
    Bus-driven worker for the Conversation Front.

    Listens on CONVERSATION_REQUEST_CHANNEL for messages of the form:

        {
          "event": "chat_turn",
          "service": "ConversationService",
          "correlation_id": "<uuid>",
          "reply_channel": "orion-conversation:result:<uuid>",
          "payload": { ...ChatTurnPayload fields... }
        }

    and publishes a corresponding "chat_turn_result" to reply_channel:

        {
          "event": "chat_turn_result",
          "service": "ConversationService",
          "correlation_id": "<same uuid>",
          "payload": {
            "text": "...",
            "tokens": 123,
            "mode": "brain",
            "spark_meta": {...} | null,
            "recall_debug": {...} | null,
          }
        }
    """
    if not bus or not getattr(bus, "enabled", False):
        logger.warning(
            "ConversationFront: bus disabled; worker will not start."
        )
        return

    channel = CONVERSATION_REQUEST_CHANNEL
    logger.info("ConversationFront: starting bus worker on channel '%s'", channel)

    for msg in bus.raw_subscribe(channel):
        envelope = msg.get("data") or {}
        event = envelope.get("event")
        service = envelope.get("service")
        corr_id = envelope.get("correlation_id")
        reply_channel = envelope.get("reply_channel")

        if event != "chat_turn":
            logger.debug(
                "ConversationFront: ignoring event=%s on channel=%s", event, channel
            )
            continue

        if not reply_channel:
            logger.warning(
                "ConversationFront: received chat_turn with no reply_channel (corr_id=%s)",
                corr_id,
            )
            continue

        payload_dict = envelope.get("payload") or {}
        try:
            payload = ChatTurnPayload(**payload_dict)
        except Exception as e:
            logger.error(
                "ConversationFront: invalid ChatTurnPayload (corr_id=%s): %s",
                corr_id,
                e,
                exc_info=True,
            )
            bus.publish(
                reply_channel,
                {
                    "event": "chat_turn_result",
                    "service": service or "ConversationService",
                    "correlation_id": corr_id,
                    "payload": {
                        "text": "[ConversationFront] Invalid chat payload.",
                        "tokens": 0,
                        "mode": "brain",
                        "spark_meta": None,
                        "recall_debug": {
                            "error": "validation_error",
                            "message": str(e),
                        },
                    },
                },
            )
            continue

        try:
            result = handle_chat_turn(bus, payload)
        except Exception as e:
            logger.exception(
                "ConversationFront: unhandled error for corr_id=%s", corr_id
            )
            bus.publish(
                reply_channel,
                {
                    "event": "chat_turn_result",
                    "service": service or "ConversationService",
                    "correlation_id": corr_id,
                    "payload": {
                        "text": "[ConversationFront] Internal error while processing chat turn.",
                        "tokens": 0,
                        "mode": payload.mode,
                        "spark_meta": None,
                        "recall_debug": {
                            "error": "execution_exception",
                            "message": str(e),
                        },
                    },
                },
            )
            continue

        bus.publish(
            reply_channel,
            {
                "event": "chat_turn_result",
                "service": service or "ConversationService",
                "correlation_id": corr_id,
                "payload": {
                    "text": result.text,
                    "tokens": result.tokens,
                    "mode": result.mode,
                    "spark_meta": result.spark_meta,
                    "recall_debug": result.recall_debug,
                },
            },
        )
        logger.info(
            "ConversationFront: published chat_turn_result corr_id=%s to %s",
            corr_id,
            reply_channel,
        )
