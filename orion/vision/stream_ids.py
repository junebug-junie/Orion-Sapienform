"""Camera identifiers must be names, never source URLs.

The Reolink RTSP URL, password included, was stored as ``stream_id`` in
~480k rows of ``substrate_perception_embedding_baseline`` because
orion-vision-edge put its capture SOURCE into ``camera_id`` and the substrate
perception listener keyed on ``camera_id`` first
(docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md,
"Two things found in passing"). This module is the guard every consumer that
persists a camera identifier runs it through.
"""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit


def is_url_like(value: Any) -> bool:
    """Anything that could be a source URL: a scheme separator, or an ``@``
    (userinfo in a malformed ``rtsp:/user:pass@host``). Camera names have
    neither."""
    return isinstance(value, str) and ("://" in value or ":/" in value or "@" in value)


def strip_userinfo(value: str) -> str:
    """``rtsp://user:pass@host:554/path?user=u&password=p`` -> ``rtsp://host:554/path``.

    Drops the query string and fragment too: Reolink/HTTP-FLV URLs carry
    credentials as query parameters.

    Falls back to dropping everything up to the last ``@`` before the path if
    the URL does not parse, so a malformed URL still cannot leak a password.
    """
    try:
        parts = urlsplit(value)
        netloc = parts.netloc.rsplit("@", 1)[-1]
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    except Exception:
        scheme, _, rest = value.partition("://")
        host_and_path = rest.rsplit("@", 1)[-1].split("?", 1)[0]
        return f"{scheme}://{host_and_path}"


def safe_camera_name(*candidates: Any, default: str = "unknown") -> str:
    """First candidate that is a plain name (no ``://``); else the first URL
    with its userinfo stripped; else ``default``.

    Prefer passing the camera name (``stream_id``) alongside the raw
    ``camera_id`` so a URL-shaped ``camera_id`` is skipped rather than kept.
    """
    first_url: Optional[str] = None
    for c in candidates:
        if c is None:
            continue
        s = str(c).strip()
        if not s:
            continue
        if not is_url_like(s):
            return s
        if first_url is None:
            first_url = s
    if first_url is not None:
        return strip_userinfo(first_url)
    return default
