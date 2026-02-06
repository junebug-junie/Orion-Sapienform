from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import requests

from app.settings import settings
from app.services.llm_client import get_llm_client
from app.storage.pg import pg_conn


logger = logging.getLogger("topic-foundry.readiness")


def check_postgres() -> Dict[str, Any]:
    try:
        with pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return {"ok": True, "detail": "ok"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Readiness postgres check failed: %s", exc)
        return {"ok": False, "detail": str(exc)}


def check_embedding() -> Dict[str, Any]:
    url = settings.topic_foundry_embedding_url.rstrip("/")
    payload = {
        "doc_id": "ready-probe",
        "text": "ping",
        "embedding_profile": "default",
        "include_latent": False,
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("embedding response missing vector")
        if not all(isinstance(val, (int, float)) for val in embedding):
            raise RuntimeError("embedding vector is not numeric")
        return {"ok": True, "detail": f"ok len={len(embedding)}"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Readiness embedding check failed: %s", exc)
        return {"ok": False, "detail": str(exc)}


def check_model_dir() -> Dict[str, Any]:
    try:
        base = Path(settings.topic_foundry_model_dir)
        base.mkdir(parents=True, exist_ok=True)
        probe = base / ".ready_probe"
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
        return {"ok": True, "detail": "ok"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Readiness model dir check failed: %s", exc)
        return {"ok": False, "detail": str(exc)}


def readiness_payload() -> Dict[str, Any]:
    checks = {
        "pg": check_postgres(),
        "embedding": check_embedding(),
        "model_dir": check_model_dir(),
        "llm": get_llm_client().probe(),
    }
    ok = all(check["ok"] for check in checks.values())
    return {
        "ok": ok,
        "checks": checks,
        "service": settings.service_name,
        "version": settings.service_version,
        "node": settings.node_name,
    }
