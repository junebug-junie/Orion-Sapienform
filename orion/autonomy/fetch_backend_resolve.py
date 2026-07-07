from __future__ import annotations

import logging
import os
from functools import partial
from typing import Awaitable, Callable

from orion.autonomy.episode_fetch import default_fetch_backend
from orion.autonomy.fcc_env import expand_env_path, load_fcc_env
from orion.autonomy.fetch_backends import firecrawl_search_backend

logger = logging.getLogger(__name__)


def resolve_firecrawl_api_key() -> str:
    key = str(os.getenv("FIRECRAWL_API_KEY") or "").strip()
    if key:
        return key
    path = expand_env_path(os.getenv("ORION_FCC_ENV_PATH", "~/.fcc/.env"))
    return str(load_fcc_env(path).get("FIRECRAWL_API_KEY") or "").strip()


def resolve_fetch_backend() -> Callable[..., Awaitable[dict]]:
    backend = str(os.getenv("ORION_EPISODE_FETCH_BACKEND", "auto")).strip().lower()
    if backend == "stub":
        return default_fetch_backend

    api_key = resolve_firecrawl_api_key()
    if backend in {"auto", "firecrawl", ""} and api_key:
        return partial(firecrawl_search_backend, api_key=api_key)
    if backend == "firecrawl" and not api_key:
        logger.warning("episode_fetch_firecrawl_requested_without_api_key")
    return default_fetch_backend
