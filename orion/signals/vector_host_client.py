"""Minimal synchronous client for orion-vector-host's ``/embedding`` endpoint.

Deliberately standalone -- does not reuse ``orion.spark.concept_induction.embedder``.
That client belongs to the concept-induction subsystem; borrowing it here would
couple an unrelated signal-adapter concern to that package's lifecycle and config.
This module talks to vector-host directly and knows nothing else.

Fails open: any network/parse error returns ``None`` rather than raising, since
adapters run inline in orion-signal-gateway's single event loop and must never
block the shared loop for longer than a short bounded timeout, let alone crash it.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import requests

logger = logging.getLogger("orion.signals.vector_host_client")

ORION_VECTOR_HOST_URL_ENV = "ORION_VECTOR_HOST_URL"
_DEFAULT_VECTOR_HOST_URL = "http://orion-athena-vector-host:8320"
DEFAULT_TIMEOUT_SEC = 0.6


def vector_host_url() -> str:
    return os.environ.get(ORION_VECTOR_HOST_URL_ENV) or _DEFAULT_VECTOR_HOST_URL


def embed_text(
    text: str,
    *,
    base_url: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> Optional[List[float]]:
    """Embed ``text`` via vector-host. Returns ``None`` on any failure -- never raises."""
    if not text or not text.strip():
        return None
    url_base = base_url or vector_host_url()
    if not url_base:
        return None
    url = f"{url_base.rstrip('/')}/embedding"
    try:
        response = requests.post(
            url,
            json={"doc_id": "signal-chat-stance", "text": text, "include_latent": False},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logger.warning("vector_host_embed_failed url=%s error=%s", url, exc)
        return None
    embedding = data.get("embedding") if isinstance(data, dict) else None
    if not isinstance(embedding, list) or not embedding:
        return None
    try:
        return [float(v) for v in embedding]
    except (TypeError, ValueError):
        return None
