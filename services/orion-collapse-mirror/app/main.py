# services/orion-collapse-mirror/app/main.py

from __future__ import annotations

from fastapi import FastAPI
from dotenv import load_dotenv
import asyncio
import traceback
import json
import re
import redis.asyncio as redis
from uuid import uuid4
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import ValidationError

from app import routes
from app.settings import settings
from orion.core.bus.service import OrionBus
from orion.schemas.collapse_mirror import CollapseMirrorEntry
from app.exec_worker import start_collapse_mirror_exec_worker

# ───────────────────────────────────────────────
# 🪞 Orion Collapse Mirror — Event Transformer
# ───────────────────────────────────────────────

load_dotenv()

app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
)

# Register routes (e.g. /api/log/collapse)
app.include_router(routes.router, prefix="/api")

# Global synchronous bus (used for publishing + exec worker)
bus: OrionBus | None = None


# ───────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────

def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort: extract the first valid JSON object from a string.
    Handles cases like:
      "Here is the generated JSON:\\n\\n{ ... }"
    """
    if not text or "{" not in text:
        return None

    starts = [m.start() for m in re.finditer(r"\{", text)]
    for s in starts:
        depth = 0
        in_str = False
        esc = False

        for i in range(s, len(text)):
            ch = text[i]

            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[s : i + 1].strip()
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass
                    break

    return None


def _coerce_to_entry_payload(data: Any) -> Any:
    """
    Accept:
      - direct CollapseMirrorEntry dict
      - accidental "exec result" shapes containing llm_output
    Return something that CollapseMirrorEntry can validate, or original.
    """
    if not isinstance(data, dict):
        return data

    # allow pings / non-entry messages without spewing stacktraces
    if data.get("kind") == "ping":
        return None

    # If someone mistakenly publishes an LLM result wrapper to intake,
    # try to pull JSON from llm_output.
    if isinstance(data.get("llm_output"), str):
        extracted = _extract_first_json_object(data["llm_output"])
        return extracted or data

    result = data.get("result")
    if isinstance(result, dict) and isinstance(result.get("llm_output"), str):
        extracted = _extract_first_json_object(result["llm_output"])
        return extracted or data

    return data


# ───────────────────────────────────────────────
# Health Endpoints
# ───────────────────────────────────────────────

@app.get("/health")
def health():
    """Quick healthcheck endpoint."""
    return {"ok": True, "service": settings.SERVICE_NAME, "version": settings.SERVICE_VERSION}


@app.get("/")
def read_root():
    return {"message": f"{settings.SERVICE_NAME} is alive"}


# ───────────────────────────────────────────────
# Async Intake Listener
# ───────────────────────────────────────────────

async def listen_for_intake():
    """
    Listens to raw collapse intake messages,
    enriches with ID, timestamp, and provenance,
    then republishes to the triage channel.
    """
    listen_channel = settings.CHANNEL_COLLAPSE_INTAKE
    publish_channel = settings.CHANNEL_COLLAPSE_TRIAGE
    print(f"📡 Connecting to Redis bus → {settings.ORION_BUS_URL}")
    print(f"👂 Listening on {listen_channel} → publishing to {publish_channel}")

    try:
        client = redis.from_url(settings.ORION_BUS_URL, decode_responses=True)
        async with client.pubsub() as pubsub:
            await pubsub.subscribe(listen_channel)
            print(f"✅ Subscribed to {listen_channel}. Awaiting messages...")

            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue

                raw = message.get("data")
                if raw is None:
                    continue

                try:
                    print(f"👂 Received intake payload: {raw}")
                    data = json.loads(raw)

                    entry_payload = _coerce_to_entry_payload(data)
                    if entry_payload is None:
                        # ping or ignorable
                        continue

                    entry = CollapseMirrorEntry.model_validate(entry_payload)

                    # If you allow noop in schema, just ignore it here.
                    try:
                        if str(getattr(entry, "type", "")).strip().lower() == "noop":
                            continue
                    except Exception:
                        pass

                    # Enrich event with ID, timestamp, provenance
                    collapse_id = f"collapse_{uuid4().hex}"
                    enriched = {
                        "id": collapse_id,
                        "service_name": settings.SERVICE_NAME,
                        "timestamp": datetime.utcnow().isoformat(),
                        **entry.model_dump(),
                    }

                    if bus:
                        bus.publish(publish_channel, enriched)
                        print(f"📤 Published enriched collapse {collapse_id} → {publish_channel}")
                    else:
                        print("⚠️ Bus not initialized; skipping publish.")

                except ValidationError as ve:
                    print(f"⚠️ Intake payload failed schema validation (dropping): {ve}")
                except Exception as e:
                    print(f"❌ Error processing intake message: {e}")
                    traceback.print_exc()

    except asyncio.CancelledError:
        print("🛑 Intake listener task cancelled. Shutting down gracefully.")
    except Exception as e:
        print(f"💥 Fatal intake listener error: {e}")
        traceback.print_exc()


# ───────────────────────────────────────────────
# Lifecycle Events
# ───────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global bus
    if settings.ORION_BUS_ENABLED:
        bus = OrionBus(url=settings.ORION_BUS_URL, enabled=True)
        print(f"🚀 {settings.SERVICE_NAME} starting up (v{settings.SERVICE_VERSION})")

        # ✅ Start the exec-step worker (threaded, blocking subscribe)
        start_collapse_mirror_exec_worker(bus)

        # ✅ Start the async intake listener
        asyncio.create_task(listen_for_intake())
    else:
        print("⚠️ OrionBus disabled — intake listener not started.")


@app.on_event("shutdown")
async def shutdown_event():
    print(f"👋 {settings.SERVICE_NAME} shutting down.")


# ───────────────────────────────────────────────
# Local Run (development only)
# ───────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
