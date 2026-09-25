#!/usr/bin/env python3
"""Fail loudly if AI Town's live LLM route resolves to the chat lane.

Juniper's direct deep/FCC conversations use the gateway `chat` route
(circe-worker-1 post-Atlas decommission). AI Town NPC dialogue must use
`quick_background` (circe-worker-fast-1) instead -- same physical host,
different lane, background priority so town dialogue never blocks chat.

Why this exists: that 2026-07-10 fix only changed `wire_llm_gateway.sh`'s
*default*. This world's live Convex `LLM_MODEL` env var was set to "chat"
(-> circe) before that fix ever landed and was never updated afterward --
Convex env vars are a persisted, out-of-band store that doesn't re-derive
from the repo's defaults. Confirmed live 2026-07-30: that stale value
survived undetected for weeks (through at least one `compact_convex_data.sh`
export/restore cycle, which faithfully preserves whatever was already set)
until circe went offline and every NPC conversation in the town silently
stalled for 10+ hours. Nothing had ever checked the *live* value against
policy -- only the script default had been fixed, not the deployed state.

This script checks the live state directly: it reads AI Town's actual
deployed `LLM_MODEL`/`LLM_API_URL` from Convex, confirms the gateway lists
that route (`/v1/models`), and resolves the route to its orion-gpu-pool work
class in config/gpu_pool.yaml -- the authority on placement since the
2026-09-24 GPU pool cutover (the gateway no longer pins a route to a worker,
so served_by is decided per call). It refuses to pass if the route's class is
`chat`: that is what would let town dialogue claim Juniper's chat GPU.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from orion.autonomy.fcc_env import expand_env_path, load_fcc_env as _load_fcc_env_file  # noqa: E402

DEFAULT_FCC_ENV_PATH = os.path.expanduser("~/.fcc/.env")

# Gateway route keys / GPU-pool work classes reserved for Juniper's chat.
FORBIDDEN_CHAT_MODEL_IDS = frozenset({"chat"})
FORBIDDEN_WORK_CLASSES = frozenset({"chat"})


def load_fcc_env(path: str) -> None:
    """Populate os.environ from the fcc env file, AITOWN_* keys always winning.

    Reuses orion.autonomy.fcc_env (a generic top-level lib, not an
    embodiment-internal one) for parsing/quote-stripping rather than
    hand-rolling it -- and matches services/orion-embodiment/app/worker.py's
    `_load_fcc_env` precedence exactly: AITOWN_* keys must track ~/.fcc/.env
    updates unconditionally (a stale process env must lose), everything else
    only fills in what's unset. A check script whose whole job is catching
    silent drift must not itself be fooled by a stale AITOWN_CONVEX_URL/
    AITOWN_ADMIN_KEY left in someone's shell.
    """
    for key, value in _load_fcc_env_file(expand_env_path(path)).items():
        if key.startswith("AITOWN_"):
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


def convex_query(base_url: str, admin_key: str, path: str, *, timeout_sec: float = 10.0):
    """Minimal self-hosted Convex admin query call (POST /api/query).

    Deliberately not shared with orion/embodiment/aitown_client.py: that
    module belongs to orion-embodiment, and this script has no other reason
    to depend on that service. The call shape (admin-key auth header, JSON
    body) is derived directly from Convex's own CLI
    (node_modules/convex/dist/cli.bundle.cjs) -- see wire_llm_gateway.sh's
    `npx convex env set`/`env list`, which do the same thing through the CLI.
    """
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/query",
        data=json.dumps({"path": path, "args": {}, "format": "json"}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Convex {admin_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if body.get("status") == "error":
        raise RuntimeError(str(body.get("errorMessage") or body))
    return body.get("value")


def fetch_gateway_models(gateway_url: str, *, timeout_sec: float = 10.0) -> list:
    url = f"{gateway_url.rstrip('/')}/v1/models"
    with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("data", [])


def route_listed(models: list, model_id: str) -> bool:
    return any(entry.get("id") == model_id for entry in models)


def resolve_work_class(model_id: str, config_path: Optional[str] = None) -> Optional[str]:
    """The orion-gpu-pool work class this gateway route runs as, or None if the route is not in
    config/gpu_pool.yaml (the gateway refuses such routes)."""
    from orion.gpu_pool.config import load_pool_config

    route = load_pool_config(config_path).routes.get(model_id)
    return route.work_class if route else None


def check_not_circe(work_class: Optional[str], *, model_id: str, allow_circe: bool) -> Optional[str]:
    """Return an error message if the check fails, else None."""
    if model_id in FORBIDDEN_CHAT_MODEL_IDS and not allow_circe:
        return (
            f"AI Town's LLM_MODEL={model_id!r} is the chat lane -- reserved for "
            "Juniper's direct deep/FCC turns. Fix: re-run "
            "services/orion-ai-town/scripts/wire_llm_gateway.sh (defaults to "
            "quick_background)."
        )
    if work_class is None:
        return (
            f"AI Town's configured model {model_id!r} is not a route in config/gpu_pool.yaml "
            "(or the gateway does not list it) -- cannot confirm it's safe. Refusing to pass."
        )
    if work_class.lower() in FORBIDDEN_WORK_CLASSES and not allow_circe:
        return (
            f"AI Town's LLM_MODEL={model_id!r} runs as GPU-pool class {work_class!r} -- Juniper's "
            "chat GPU. AI Town must use quick_background (class fast). Fix: re-run "
            "services/orion-ai-town/scripts/wire_llm_gateway.sh."
        )
    return None


def main() -> int:
    load_fcc_env(os.environ.get("AITOWN_FCC_ENV_PATH", DEFAULT_FCC_ENV_PATH))
    allow_circe = os.environ.get("AITOWN_ALLOW_CIRCE", "").strip() == "1"

    convex_url = os.environ.get("AITOWN_CONVEX_URL", "").strip()
    admin_key = os.environ.get("AITOWN_ADMIN_KEY", "").strip()
    if not convex_url or not admin_key:
        print(
            "check_llm_route_not_circe: AITOWN_CONVEX_URL/AITOWN_ADMIN_KEY not set "
            f"(checked env and {DEFAULT_FCC_ENV_PATH}) -- cannot reach Convex.",
            file=sys.stderr,
        )
        return 1

    try:
        rows = convex_query(convex_url, admin_key, "_system/cli/queryEnvironmentVariables")
    except Exception as exc:  # noqa: BLE001 -- fail loudly with a clean message, not a traceback
        print(f"check_llm_route_not_circe: could not query Convex at {convex_url}: {exc}", file=sys.stderr)
        return 1

    by_name = {row["name"]: row["value"] for row in rows}
    llm_model = by_name.get("LLM_MODEL")
    gateway_url = by_name.get("LLM_API_URL")
    if not llm_model or not gateway_url:
        print(
            "check_llm_route_not_circe: AI Town's Convex deployment has no "
            "LLM_MODEL/LLM_API_URL set -- run wire_llm_gateway.sh first.",
            file=sys.stderr,
        )
        return 1

    try:
        models = fetch_gateway_models(gateway_url)
    except Exception as exc:  # noqa: BLE001 -- fail loudly with a clean message, not a traceback
        print(f"check_llm_route_not_circe: could not reach gateway at {gateway_url}: {exc}", file=sys.stderr)
        return 1
    work_class = resolve_work_class(llm_model) if route_listed(models, llm_model) else None
    error = check_not_circe(work_class, model_id=llm_model, allow_circe=allow_circe)
    if error:
        print(f"check_llm_route_not_circe FAIL: {error}", file=sys.stderr)
        return 1

    print(f"check_llm_route_not_circe: OK -- LLM_MODEL={llm_model!r} gpu-pool class={work_class!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
