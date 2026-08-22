"""Resolve whether a route can actually see, and build multimodal payloads.

Two rules shape this module.

**Runtime truth beats config truth.** ``LLMProfile.supports_vision`` is a claim
written in YAML. ``/props``'s ``modalities.vision`` is what the worker will
actually accept. On 2026-08-14 they disagreed: the chat lane's profile said
``supports_vision: false`` while its weights and chat template were VL-capable
and only the ``--mmproj`` launch flag was missing, and nothing anywhere read the
flag at all. Trusting the config would either send images at a blind worker or
refuse them at a sighted one. So the gate reads the live worker and treats an
unreachable worker as blind -- fail closed, never fail open.

**Bytes enter at the last possible moment.** Attachments travel the whole saga
as ``AttachmentRefV1`` URLs. This is the only place they become base64, one hop
before the model, so nothing upstream ever stores or replicates them.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

import httpx

from app.settings import settings

logger = logging.getLogger("orion.llm.gateway.vision")

# Mirrors the Hub's accepted set. A model is handed only what we can name.
ALLOWED_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Magic-byte prefixes for the formats we accept. The gateway re-checks these
# against the FETCHED bytes rather than trusting the ref's declared mime, which
# is what it hands to the model.
_MAGIC = (
    ("image/png", (b"\x89PNG\r\n\x1a\n",)),
    ("image/jpeg", (b"\xff\xd8",)),
    ("image/gif", (b"GIF87a", b"GIF89a")),
    ("image/webp", (b"RIFF",)),
)


def sniff_mime(data: bytes) -> Optional[str]:
    for mime, prefixes in _MAGIC:
        if any(data.startswith(p) for p in prefixes):
            if mime == "image/webp" and not (len(data) >= 12 and data[8:12] == b"WEBP"):
                continue
            return mime
    return None


class AttachmentFetchError(RuntimeError):
    """An attachment could not be turned into model input."""


@dataclass(frozen=True)
class VisionCapability:
    """What a route can do, as reported by the worker itself."""

    vision: bool
    source: str  # "props" | "props-unreachable" | "disabled"
    detail: Optional[str] = None

    def as_debug(self) -> Dict[str, Any]:
        return {"vision": self.vision, "source": self.source, "detail": self.detail}


# ───────────────────────────────────────────────────────────────
# Capability probe (TTL-cached)
# ───────────────────────────────────────────────────────────────

_probe_lock = threading.Lock()
_probe_cache: Dict[str, tuple[float, VisionCapability]] = {}


def _probe_ttl() -> float:
    return float(getattr(settings, "llm_gateway_vision_props_cache_ttl_sec", 300.0))


def clear_capability_cache() -> None:
    """Test seam; also useful after a worker restart flips --mmproj."""
    with _probe_lock:
        _probe_cache.clear()


def resolve_vision_capability(base_url: str, *, now: Optional[float] = None) -> VisionCapability:
    """Ask the worker what modalities it has. Cached; fails closed."""
    if not bool(getattr(settings, "llm_gateway_vision_enabled", True)):
        return VisionCapability(vision=False, source="disabled", detail="LLM_GATEWAY_VISION_ENABLED=false")

    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return VisionCapability(vision=False, source="props-unreachable", detail="no base url")

    clock = time.monotonic() if now is None else now
    with _probe_lock:
        cached = _probe_cache.get(base)
        if cached and clock - cached[0] < _probe_ttl():
            return cached[1]

    try:
        with httpx.Client(
            timeout=float(getattr(settings, "llm_gateway_vision_props_timeout_sec", 5.0)),
        ) as client:
            response = client.get(f"{base}/props")
        response.raise_for_status()
        props = response.json()
        modalities = props.get("modalities") if isinstance(props, dict) else None
        vision = bool(isinstance(modalities, dict) and modalities.get("vision"))
        capability = VisionCapability(
            vision=vision,
            source="props",
            detail=None if vision else "worker reports modalities.vision=false",
        )
    except Exception as exc:  # noqa: BLE001 -- any failure means "assume blind"
        logger.warning("[LLM-GW] vision /props probe failed for %s: %s", base, exc)
        # Deliberately NOT cached: a transient blip should not blind a sighted
        # worker for a whole TTL.
        return VisionCapability(vision=False, source="props-unreachable", detail=str(exc)[:200])

    with _probe_lock:
        _probe_cache[base] = (clock, capability)
    return capability


# ───────────────────────────────────────────────────────────────
# Attachment fetch
# ───────────────────────────────────────────────────────────────

def _allowed_hosts() -> set[str]:
    raw = str(getattr(settings, "llm_gateway_attachment_allowed_hosts", "") or "")
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def resolve_attachment_url(attachment: Any) -> str:
    """Build the fetch URL from the content address and a TRUSTED base.

    The ref's own ``source_url`` is **not used**. It round-trips through the
    browser (`_attachments_from_payload` re-validates the shape but never
    re-derives the URL), so by the time it reaches here it is client-controlled
    in full -- scheme, host, port, and path. A host-only allowlist does not
    contain that: the allowlisted value is the Tailscale node IP that also
    fronts the bus, Postgres, the Hub, and every other Orion service, so any
    ``http://<node>:<port>/<path>`` would have passed and the gateway would have
    base64'd an internal endpoint's response straight into the model prompt.

    Instead the URL is reconstructed as ``<configured base>/<sha256>``, with the
    sha256 regex-validated. The only caller-supplied component is a 64-char hex
    string, so there is no path or authority to inject into.
    """
    sha256 = str(getattr(attachment, "sha256", "") or "").strip().lower()
    if not _SHA256_RE.match(sha256):
        raise AttachmentFetchError("attachment sha256 is not a 64-char hex digest")

    # Percepts resolve from a DIFFERENT store than user uploads. A camera frame
    # of a private home and a file someone dragged into chat have different
    # lifetimes and different blast radii, and Hub -- which serves the chat
    # store -- is also the process holding the docker socket. Selecting the base
    # by kind here is what keeps that separation real rather than promised.
    kind = str(getattr(attachment, "kind", "image") or "image")
    if kind == "percept":
        base_key, env_name = "llm_gateway_percept_base_url", "LLM_GATEWAY_PERCEPT_BASE_URL"
    else:
        base_key, env_name = "llm_gateway_attachment_base_url", "LLM_GATEWAY_ATTACHMENT_BASE_URL"

    base = str(getattr(settings, base_key, "") or "").strip().rstrip("/")
    if not base:
        # Fail closed, and specifically do NOT fall back to the other store:
        # an unset percept base must not silently start reading chat uploads.
        raise AttachmentFetchError(
            f"{env_name} is not set; refusing to fetch {kind} attachments"
        )

    parsed = urlparse(base)
    if parsed.scheme not in ("http", "https"):
        raise AttachmentFetchError(f"attachment base scheme {parsed.scheme!r} is not http(s)")
    host = (parsed.hostname or "").lower()
    if not host:
        raise AttachmentFetchError("attachment base url has no host")

    # Defence in depth: the base is operator config, but a typo that points it
    # somewhere unexpected should still fail closed.
    allowed = _allowed_hosts()
    if not allowed:
        raise AttachmentFetchError(
            "LLM_GATEWAY_ATTACHMENT_ALLOWED_HOSTS is empty; refusing to fetch attachments"
        )
    if host not in allowed:
        raise AttachmentFetchError(f"attachment host {host!r} is not allowlisted")

    return f"{base}/{sha256}"


def fetch_attachment_data_uri(attachment: Any) -> str:
    """Fetch an attachment's bytes and return a ``data:`` URI.

    Three checks on the fetched bytes, in increasing order of strength:

    1. The declared mime must be one we accept.
    2. The bytes must *sniff* as that same mime -- the declared value is what
       gets handed to the model, so it is not taken on trust.
    3. **The bytes must hash to the ref's sha256.** This is the real guarantee:
       the store is content-addressed, so if the content does not hash to the
       requested address, something served the wrong thing and we refuse. It
       subsumes the old byte-count check (which a ref declaring ``bytes: 0``
       skipped entirely) and holds even if the transport were somehow diverted.
    """
    mime = str(getattr(attachment, "mime", "") or "").strip().lower()
    if mime not in ALLOWED_IMAGE_MIMES:
        raise AttachmentFetchError(f"attachment mime {mime!r} is not an accepted image type")

    sha256 = str(getattr(attachment, "sha256", "") or "").strip().lower()
    url = resolve_attachment_url(attachment)

    max_bytes = int(getattr(settings, "llm_gateway_attachment_max_bytes", 8_388_608))
    declared = int(getattr(attachment, "bytes", 0) or 0)
    if declared > max_bytes:
        raise AttachmentFetchError(f"attachment declares {declared} bytes, gateway limit is {max_bytes}")

    timeout = float(getattr(settings, "llm_gateway_attachment_fetch_timeout_sec", 10.0))
    try:
        # httpx, not requests: httpx is already a declared gateway dependency and
        # is used throughout this service. It also does NOT follow redirects by
        # default, which closes the "allowlisted host 302s elsewhere" bypass.
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                # Cap the read itself; a lying Content-Length must not let the
                # gateway buffer an unbounded body.
                chunks: List[bytes] = []
                total = 0
                for chunk in response.iter_bytes(65536):
                    total += len(chunk)
                    if total > max_bytes:
                        raise AttachmentFetchError(f"attachment body exceeded {max_bytes} bytes")
                    chunks.append(chunk)
        data = b"".join(chunks)
    except AttachmentFetchError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AttachmentFetchError(f"attachment fetch failed: {exc}") from exc

    if not data:
        raise AttachmentFetchError("attachment fetch returned no bytes")

    actual = hashlib.sha256(data).hexdigest()
    if actual != sha256:
        raise AttachmentFetchError(
            f"attachment content address mismatch: asked for {sha256[:12]}, got {actual[:12]}"
        )

    sniffed = sniff_mime(data)
    if sniffed != mime:
        raise AttachmentFetchError(
            f"attachment bytes sniff as {sniffed or 'unknown'}, ref declares {mime}"
        )

    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


# ───────────────────────────────────────────────────────────────
# Multimodal message assembly
# ───────────────────────────────────────────────────────────────

def build_multimodal_messages(
    serialized: Sequence[Dict[str, str]],
    attachments: Sequence[Any],
) -> List[Dict[str, Any]]:
    """Attach images to the last user turn of an already-serialized message list.

    Takes the *output* of the existing text serializer rather than replacing it,
    so the text half of every message is produced by exactly the same code path
    as a turn with no attachments. Only the last user message changes shape, and
    only when there is at least one usable image.

    Raises AttachmentFetchError if no attachment could be fetched -- the caller
    decides whether that is fatal, because silently sending a text-only turn
    after the user attached an image is the failure this whole feature exists
    to avoid.
    """
    messages: List[Dict[str, Any]] = [dict(m) for m in serialized]
    if not attachments:
        return messages

    target_index = None
    for index in range(len(messages) - 1, -1, -1):
        if str(messages[index].get("role") or "").lower() == "user":
            target_index = index
            break
    if target_index is None:
        raise AttachmentFetchError("no user message to attach images to")

    parts: List[Dict[str, Any]] = [
        {"type": "text", "text": str(messages[target_index].get("content") or "")}
    ]
    failures: List[str] = []
    for attachment in attachments:
        try:
            parts.append(
                {"type": "image_url", "image_url": {"url": fetch_attachment_data_uri(attachment)}}
            )
        except AttachmentFetchError as exc:
            failures.append(str(exc))
            logger.warning("[LLM-GW] dropping attachment: %s", exc)

    if len(parts) == 1:
        raise AttachmentFetchError("; ".join(failures) or "no usable attachments")

    messages[target_index] = {"role": messages[target_index].get("role", "user"), "content": parts}
    return messages
