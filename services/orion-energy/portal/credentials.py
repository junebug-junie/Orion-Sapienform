"""The RMP login, read from an owner-only file outside the repo at attempt time.

Kept out of the environment on purpose: env values show up in `docker inspect` and
crash dumps. The password is never logged and never part of any repr.
"""

from __future__ import annotations

import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

USERNAME_KEY = "RMP_USERNAME"
PASSWORD_KEY = "RMP_PASSWORD"


class CredentialsFileTooOpen(PermissionError):
    """Group or other can read the file; refuse rather than use a leaked password."""


class CredentialsIncomplete(ValueError):
    """The file exists but lacks a non-empty RMP_USERNAME or RMP_PASSWORD."""


@dataclass(frozen=True)
class PortalCredentials:
    username: str = field(repr=False)
    password: str = field(repr=False)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def load_credentials(path: Path) -> Optional[PortalCredentials]:
    """None only when the file is absent (manual reauth mode).

    Values are stripped of surrounding whitespace and one pair of matching quotes, so a
    password that really begins and ends with the same quote or with spaces cannot be used.
    Raises CredentialsFileTooOpen, CredentialsIncomplete, OSError, or UnicodeDecodeError --
    callers report the class name only, since messages can quote file bytes.
    """
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return None
    if stat.S_IMODE(mode) & 0o077:
        raise CredentialsFileTooOpen(f"{path} must be chmod 600")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = _unquote(value.strip())
    username, password = values.get(USERNAME_KEY, ""), values.get(PASSWORD_KEY, "")
    if not username or not password:
        raise CredentialsIncomplete(f"{path} needs non-empty {USERNAME_KEY} and {PASSWORD_KEY}")
    return PortalCredentials(username=username, password=password)
