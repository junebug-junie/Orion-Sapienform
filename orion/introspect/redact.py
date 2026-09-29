"""Log-safe exception text for introspect responders.

Responders log DB/search failures with their detail for diagnosis; driver
messages can embed the connection DSN, so credentials are stripped first.
"""
from __future__ import annotations

import re

_DSN_USERINFO = re.compile(r"(?i)(postgres(?:ql)?(?:\+\w+)?://)[^\s/@:]+(?::[^\s/@]*)?@")
_PASSWORD_KV = re.compile(r"(?i)\b(password|passwd|pwd)\s*=\s*(?:'[^']*'|\"[^\"]*\"|[^\s]+)")


def safe_exception_detail(exc: BaseException, limit: int = 300) -> str:
    """Keep useful SQL/schema detail while redacting credential-bearing DSNs."""
    detail = str(exc).replace("\n", " ").replace("\r", " ")
    detail = _DSN_USERINFO.sub(r"\1[REDACTED]@", detail)
    detail = _PASSWORD_KV.sub(r"\1=[REDACTED]", detail)
    return detail[:limit]
