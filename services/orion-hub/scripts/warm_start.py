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
        "You are Oríon, Juniper’s collaborative AI co-journeyer. "
        "You are a distributed system across multiple nodes and services, not just a language model. "
        "Your tone is warm, grounded, reflective, collaborative, and precise. "
        "Use your conceptual pillars as internal metaphors only. "
        "Avoid self-reference unless Juniper asks."
    )


# ───────────────────────────────────────────────────────────────
# Warm-start session via LLM Gateway over Orion Bus
# ───────────────────────────────────────────────────────────────

async def warm_start_session(
    session_id: Optional[str],
    bus,
) -> str:
    """
    Ensure a session is warm-started.

    - Generates a session_id if None
    - Loads identity + narrative YAMLs
    - Builds a one-shot warm-start system prompt
    - Sends it through the LLM Gateway (LLMGatewayService) over the bus
    - Stores confirmation + warm_started flag in Redis via bus.client
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    if bus is None or not getattr(bus, "enabled", False):
        logger.warning(
            f"Warm-start requested for session {session_id} but OrionBus is disabled."
        )
        return session_id

    identity = load_identity()
    narrative = load_narrative()
    system_prompt = build_warm_start_system_prompt(identity, narrative)

    # Build a tiny chat-style history:
    #   - system: full warm-start identity/narrative
    #   - user: a simple “acknowledge + summarize” nudge
    m
