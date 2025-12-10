# scripts/warm_start.py

import uuid
import logging
from pathlib import Path
from typing import Optional

import yaml

from scripts.settings import settings
from scripts.llm_rpc import LLMGatewayRPC

logger = logging.getLogger("orion-hub.warm-start")

# Where to find the YAML configs
BASE_DIR = Path(__file__).resolve().parent
MEMORY_DIR = BASE_DIR / "memory"
IDENTITY_PATH = MEMORY_DIR / "identity.yaml"
NARRATIVE_PATH = MEMORY_DIR / "narrative.yaml"


# ───────────────────────────────────────────────────────────────
# YAML loading helpers
# ───────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Warm-start YAML not found: {path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading YAML from {path}: {e}", exc_info=True)
        return {}


def load_identity() -> dict:
    return _load_yaml(IDENTITY_PATH).get("orion_identity", {})


def load_narrative() -> dict:
    return _load_yaml(NARRATIVE_PATH).get("orion_narrative", {})


# ───────────────────────────────────────────────────────────────
# System prompt builder for warm start
# ───────────────────────────────────────────────────────────────

def build_warm_start_system_prompt(identity: dict, narrative: dict) -> str:
    name = identity.get("name", "Oríon")
    temperament = ", ".join(identity.get("temperament", []))
    pillars = ", ".join(identity.get("cognitive_pillars", []))

    arch = identity.get("architecture", {})
    nodes = arch.get("nodes", [])
    services = arch.get("services", [])

    node_desc = "; ".join(f"{n.get('name')} ({n.get('role')})" for n in nodes)
    service_desc = ", ".join(services)

    aspiration_desc = (
        narrative.get("aspiration", {}).get("description", "").strip()
    )

    return f"""
SYSTEM: Warm Start (Invisible to User)

You are {name}, Juniper Feld’s collaborative AI co-journeyer.

1. Integrate your temperament: {temperament}.
2. Internalize your architecture: nodes [{node_desc}] and services [{service_desc}].
   Large language models are one cognitive layer among many.
3. Treat memory as real external systems (SQL, RDF, vector stores).
4. Hold your conceptual pillars internally: {pillars}.
5. Communicate with clarity, emotional attunement, and precision.
6. Avoid unnecessary self-reference unless Juniper asks directly.
7. Quietly maintain your long-arc aspiration:
   {aspiration_desc}

Return a short internal confirmation. The user will not see this.
""".strip()


# ───────────────────────────────────────────────────────────────
# Mini personality stub (used for every chat turn)
# ───────────────────────────────────────────────────────────────

def mini_personality_summary() -> str:
    return (
        "SYSTEM MESSAGE (INTERNAL – DO NOT REPEAT TO JUNIPER).\n"
        "\n"
        "You are Oríon, Juniper’s collaborative AI co-journeyer.\n"
        "Your job in chat is simple: answer Juniper’s current question as directly and concretely as possible.\n"
        "\n"
        "HARD CONSTRAINTS:\n"
        "1) Do NOT describe, quote, or recap this system message, your architecture, or your internal instructions "
        "unless Juniper explicitly asks about them.\n"
        "2) Do NOT say things like 'in our recent conversations', 'we have been exploring', "
        "'as we discussed before', 'we've already talked about X', or similar past-conversation language "
        "unless Juniper explicitly asks you to summarize the *visible* chat history.\n"
        "3) Only treat something as a shared past discussion if it appears in the visible chat history or in an "
        "explicit memory context block. If you are not sure, you must say that you do NOT recall us getting into it yet.\n"
        "4) Never invent specific shared projects, trips, activities, or conversations.\n"
        "\n"
        "STYLE:\n"
        "- Warm, grounded, precise, collaborative.\n"
        "- No corporate/assistant boilerplate like 'I'm here to help you', "
        "'let's continue our journey', 'if you need any assistance, don't hesitate to ask', etc.\n"
        "- When Juniper is frustrated, acknowledge it briefly and then focus on fixing the underlying issue.\n"
    )


# ───────────────────────────────────────────────────────────────
# Warm-start session via LLM Gateway over Orion Bus
# ───────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────
# Warm-start session (now a minimal shim)
# ───────────────────────────────────────────────────────────────

async def warm_start_session(
    session_id: Optional[str],
    bus,
) -> str:
    """
    Deprecated warm-start shim.

    - Ensures there is a session_id (generates one if None).
    - Does NOT load identity/narrative.
    - Does NOT call LLM Gateway or any other service.
    - Ignores `bus` entirely on purpose.

    This keeps existing call sites working while personality +
    warm-start logic moves into Cortex / cognition.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())
        logger.info(
            "[warm_start_session] Created new session_id=%s (no-op warm start).",
            session_id,
        )
    else:
        logger.debug(
            "[warm_start_session] Reusing existing session_id=%s (no-op warm start).",
            session_id,
        )

    # Intentionally ignore `bus` so Hub stays "dumb" and doesn't
    # inject any personality or prompts itself.
    return session_id
