"""Public source validation, shared by ingress, backfill and reentry.

Preserve World Pulse's stripped path/query normalization. Never fetch a body
here. DNS is checked at acceptance and again before the reader runs.
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit, urlunsplit

from pydantic import HttpUrl, TypeAdapter

_HTTP = TypeAdapter(HttpUrl)


def normalize_source_url(value: str) -> str:
    raw = str(value).strip()
    if not raw or any(ord(c) < 32 or c.isspace() for c in raw) or "\\" in raw:
        raise ValueError("invalid_source_url")
    try:
        parsed = _HTTP.validate_python(raw)
        parts = urlsplit(str(parsed))
        host = parts.hostname or ""
        if parts.username or parts.password or "%" in host:
            raise ValueError("credentials_or_scoped_host")
        # Block private names even if an operator resolver maps them publicly.
        host = host.rstrip(".").lower()
        if "." not in host and ":" not in host:
            raise ValueError("internal_hostname")
        if host.endswith((".localhost", ".local", ".internal", ".home", ".lan", ".onion", ".ts.net")):
            raise ValueError("internal_hostname")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and not _public(address):
            raise ValueError("non_public_address")
        # Fragments do not identify a different HTTP resource for active dedup.
        return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path, parts.query, ""))
    except Exception as exc:
        raise ValueError("source must be a public HTTP(S) URL") from exc


def _public(address) -> bool:
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(address.is_global and not address.is_multicast and (mapped is None or mapped.is_global))


async def validate_source_url(value: str) -> str:
    normalized = normalize_source_url(value)
    parsed = urlsplit(normalized)
    try:
        addresses = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(
                parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            ), timeout=5.0,
        )
        if not addresses or any(not _public(ipaddress.ip_address(row[4][0])) for row in addresses):
            raise ValueError("non_public_dns_answer")
    except (OSError, ValueError, asyncio.TimeoutError) as exc:
        raise ValueError("source must resolve only to public addresses") from exc
    return normalized
