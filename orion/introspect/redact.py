"""Log-safe exception text for introspect responders.

Responders log DB/search failures with their detail for diagnosis; driver
messages can embed the connection DSN, so credentials are stripped first.
"""
from __future__ import annotations

from orion.core.redact import redact_secrets


def safe_exception_detail(exc: BaseException, limit: int = 300) -> str:
    """Keep useful SQL/schema detail while redacting credential-bearing DSNs."""
    detail = str(exc).replace("\n", " ").replace("\r", " ")
    return redact_secrets(detail)[:limit]
