"""Strip credentials from text that leaves the process (logs, traces, UI).

Two layers: patterns (``scheme://user:pass@host``, ``*PASSWORD=value``) and
the literal values of credential-named env vars, which catch a bare secret
printed by ``echo $X_PASSWORD`` that no pattern could recognise. The values
come from this process's environment plus every env file a spawner hands to a
child (``remember_secret_env``): the FCC subprocess gets its credentials from
``~/.fcc/.env``, which never enters ``os.environ``.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any, Iterable, Mapping

REDACTED = "[REDACTED]"

# Anchored at the start of a scheme-character run, not at \b: a \b start
# rescans every boundary inside a long dotted/hyphenated run (quadratic).
# The password may contain '@'; greedy matching stops at the last one before
# the host. Shell-variable userinfo ("$USER:$PASS@") names the credential
# without revealing it, so it is left readable.
_URL_USERINFO = re.compile(
    r"(?i)(?<![a-z0-9+.\-])([a-z][a-z0-9+.\-]*://)(?!\$)[^\s/@:'\"]*(?::[^\s/'\"]*)?@"
)
# Prefixed names too (PGPASSWORD=, ORION_X_PASSWORD=). An unquoted value stops
# before closing syntax so a reply's JSON or code keeps its brackets.
_PASSWORD_KV = re.compile(
    r"(?i)(?<![a-z0-9])(\w*(?:password|passwd|pwd))\s*=\s*(?:'[^']*'|\"[^\"]*\"|[^\s\"'),;}\]]+)"
)
_SECRET_ENV_NAME = re.compile(r"(?i)(PASSWORD|PASSWD|SECRET|TOKEN$|API_KEY|ADMIN_KEY|_DSN$|_PAT$|PRIVATE_KEY)")
# Shorter values ("true", "1", a port) would redact ordinary text.
_MIN_SECRET_LEN = 8

_lock = threading.Lock()
_remembered: set[str] = set()
_cached: tuple[str, ...] | None = None


def secret_env_values(env: Mapping[str, str]) -> tuple[str, ...]:
    """Values of credential-named env vars, longest first so no value is half-replaced."""
    values = {v for k, v in env.items() if _SECRET_ENV_NAME.search(k) and v and len(v) >= _MIN_SECRET_LEN}
    return tuple(sorted(values, key=len, reverse=True))


def remember_secret_env(env: Mapping[str, str]) -> None:
    """Register an env handed to a child process so its secrets are redacted too."""
    global _cached
    new = set(secret_env_values(env)) - _remembered
    if new:
        with _lock:
            _remembered.update(new)
            _cached = None


def _process_secret_values() -> tuple[str, ...]:
    global _cached
    cached = _cached
    if cached is None:
        with _lock:
            values = set(secret_env_values(os.environ)) | _remembered
            cached = _cached = tuple(sorted(values, key=len, reverse=True))
    return cached


def redact_secrets(text: str, secret_values: Iterable[str] | None = None) -> str:
    # Patterns first so a DSN keeps its host; literals then catch bare values.
    if "://" in text:
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
