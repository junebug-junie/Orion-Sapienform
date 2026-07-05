"""Read FCC operator env for Hub agent-claude model label catalog."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

FCC_MODEL_ENV_KEYS: tuple[str, ...] = (
    "MODEL",
    "MODEL_OPUS",
    "MODEL_SONNET",
    "MODEL_HAIKU",
)


def expand_env_path(raw: str) -> Path:
    return Path(os.path.expanduser(str(raw or "").strip() or "~/.fcc/.env"))


def load_fcc_env(path: Path | str) -> Dict[str, str]:
    p = Path(path)
    if not p.is_file():
        return {}
    out: Dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def model_labels_from_env(env: Dict[str, str]) -> List[str]:
    labels: List[str] = []
    for key in FCC_MODEL_ENV_KEYS:
        if str(env.get(key) or "").strip():
            labels.append(key)
    return labels


def resolve_auth_token(env: Dict[str, str], *, override: str = "") -> str:
    token = str(override or "").strip()
    if token:
        return token
    return str(env.get("ANTHROPIC_AUTH_TOKEN") or "").strip()


def catalog_from_settings(*, env_path: str, auth_override: str = "") -> dict:
    from scripts.settings import settings

    path = expand_env_path(env_path or settings.HUB_FCC_ENV_PATH)
    env = load_fcc_env(path)
    return {
        "enabled": bool(settings.HUB_AGENT_CLAUDE_ENABLED),
        "env_path": str(path),
        "labels": model_labels_from_env(env),
        "default_label": model_labels_from_env(env)[0] if model_labels_from_env(env) else None,
        "auth_token_configured": bool(resolve_auth_token(env, override=auth_override or settings.HUB_FCC_AUTH_TOKEN)),
    }
