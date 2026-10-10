"""Strip credentials from text that leaves the process (logs, traces, UI).

Two layers: URL userinfo (``scheme://user:pass@host``) for any DSN, and the
literal values of credential-named env vars, which catches a bare password
printed by ``echo $X_PASSWORD`` that no pattern could recognise.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Iterable, Mapping

REDACTED = "[REDACTED]"

# Shell-variable userinfo ("$USER:$PASS@") is left alone: it names the
# credential without revealing it, and is useful when reading a trace.
_URL_USERINFO = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)(?!\$)[^\s/@:'\"]+(?::[^\s/@'\"]*)?@")
_PASSWORD_KV = re.compile(r"(?i)\b(password|passwd|pwd)\s*=\s*(?:'[^']*'|\"[^\"]*\"|[^\s]+)")
_SECRET_ENV_NAME = re.compile(r"(?i)(PASSWORD|PASSWD|SECRET|TOKEN$|API_KEY|ADMIN_KEY|_DSN$|_PAT$|PRIVATE_KEY)")
# Shorter values ("true", "1", a port) would redact ordinary text.
_MIN_SECRET_LEN = 8


def secret_env_values(env: Mapping[str, str]) -> tuple[str, ...]:
    """Values of credential-named env vars, longest first so no value is half-replaced."""
    values = {v for k, v in env.items() if _SECRET_ENV_NAME.search(k) and v and len(v) >= _MIN_SECRET_LEN}
    return tuple(sorted(values, key=len, reverse=True))


@lru_cache(maxsize=1)
def _process_secret_values() -> tuple[str, ...]:
    return secret_env_values(os.environ)


def redact_secrets(text: str, secret_values: Iterable[str] | None = None) -> str:
    # Patterns first so a DSN keeps its host; literals then catch bare values.
    text = _URL_USERINFO.sub(rf"\1{REDACTED}@", text)
    text = _PASSWORD_KV.sub(rf"\1={REDACTED}", text)
    values = _process_secret_values() if secret_values is None else secret_values
    for value in values:
        if value in text:
            text = text.replace(value, REDACTED)
    return text


def redact_secrets_deep(obj: Any, secret_values: Iterable[str] | None = None) -> Any:
    """Copy of a JSON-shaped value with every string passed through redact_secrets."""
    values = tuple(_process_secret_values() if secret_values is None else secret_values)
    if isinstance(obj, str):
        return redact_secrets(obj, values)
    if isinstance(obj, dict):
        return {k: redact_secrets_deep(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_secrets_deep(v, values) for v in obj]
    return obj
