"""Idempotent bootstrap for Orion's AI Town body.

Dry-run by default. ``--write`` persists only ``AITOWN_ORION_PLAYER_ID`` and
``AITOWN_ORION_AGENT_ID`` into ``~/.fcc/.env`` in place (all other lines/secrets
byte-for-byte untouched). The persona is a privacy-filtered projection of
Orion's live self-model; if the projection is empty the guard falls back to a
minimal safe persona and reports ``persona_source=fallback`` (never a hollow
persona treated as success).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional

_HERE = Path(__file__).resolve()
_SERVICE_ROOT = _HERE.parents[1]
_REPO_ROOT = _HERE.parents[3]
for _p in (str(_SERVICE_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from orion.autonomy.fcc_env import expand_env_path, load_fcc_env  # noqa: E402
from orion.embodiment import aitown_client  # noqa: E402
from orion.embodiment.persona import build_orion_town_persona  # noqa: E402

_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

_RESTART_REMINDER = """\
# ~/.fcc/.env changed → restart consumers of AITOWN_ORION_*
docker compose --env-file services/orion-embodiment/.env -f services/orion-embodiment/docker-compose.yml up -d --build
# and, so FCC MCP telekinesis shares the same body:
docker compose --env-file services/orion-harness-governor/.env -f services/orion-harness-governor/docker-compose.yml up -d
"""


def upsert_env_keys(path: Path, updates: Dict[str, str]) -> None:
    """Replace/add only ``updates`` keys in ``path`` in place; leave everything else byte-for-byte."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.is_file() else []
    seen: set[str] = set()
    out: list[str] = []
    for raw in lines:
        m = _KEY.match(raw.rstrip("\n"))
        if m and m.group(1) in updates:
            seen.add(m.group(1))
            out.append(f"{m.group(1)}={updates[m.group(1)]}\n")
        else:
            out.append(raw if raw.endswith("\n") or not raw else raw + "\n")
    for key, val in updates.items():
        if key not in seen:
            out.append(f"{key}={val}\n")
    path.write_text("".join(out), encoding="utf-8")


def _load_fcc(fcc_env_path: str) -> None:
    for k, v in load_fcc_env(expand_env_path(fcc_env_path)).items():
        os.environ.setdefault(k, v)


def _fetch_self_model(base_url: str) -> Dict[str, Optional[str]]:
    """Best-effort self-model fetch; on any failure return empties so the guard falls back."""
    empties: Dict[str, Optional[str]] = {
        "identity_summary": None,
        "anchor_strategy": None,
        "dominant_drive": None,
        "snapshot_id": None,
        "generated_at": None,
    }
    if not base_url:
        return empties
    url = f"{base_url.rstrip('/')}/self-model/latest"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, OSError):
        return empties
    if not isinstance(data, dict):
        return empties
    return {
        "identity_summary": data.get("identity_summary") or data.get("overall_condition"),
        "anchor_strategy": data.get("anchor_strategy") or data.get("trajectory_condition"),
        "dominant_drive": data.get("dominant_drive"),
        "snapshot_id": data.get("snapshot_id"),
        "generated_at": data.get("generated_at"),
    }


def _player_resolves(player_id: str) -> bool:
    if not player_id:
        return False
    try:
        players = aitown_client.list_players() or []
    except aitown_client.AitownClientError:
        return False
    return any(str(p.get("id")) == player_id for p in players)


def _poll_new_body(timeout: float = 30.0) -> tuple[Optional[str], Optional[str]]:
    """Wait for the player named "Orion" to appear. ``join`` creates a player (no
    town-AI agent), so a missing agent id is expected — Orion is driven externally."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            aitown_client.heartbeat_world()  # keep the engine running so join is processed
            players = aitown_client.list_players() or []
        except aitown_client.AitownClientError:
            players = []
        player = next((p for p in players if str(p.get("name") or "").lower() == "orion"), None)
        if player is not None:
            return str(player.get("id")), None
        time.sleep(2.0)
    return None, None


def _build_persona(self_state_url: str, spritesheet: str):
    sm = _fetch_self_model(self_state_url)
    persona = build_orion_town_persona(
        identity_summary=sm["identity_summary"],
        anchor_strategy=sm["anchor_strategy"],
        dominant_drive=sm["dominant_drive"],
        snapshot_id=sm["snapshot_id"],
        generated_at=sm["generated_at"],
        spritesheet=spritesheet,
    )
    print(f"persona_source={persona.persona_source} snapshot_id={persona.provenance.get('snapshot_id')}")
    return persona


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap Orion's AI Town body (dry-run by default).")
    parser.add_argument("--write", action="store_true", help="persist AITOWN_ORION_* into ~/.fcc/.env")
    parser.add_argument("--force", action="store_true", help="recreate body even if one already resolves")
    parser.add_argument("--resync-identity", action="store_true", help="rebuild persona and update description")
    args = parser.parse_args(argv)

    fcc_env_path = os.environ.get("EMBODIMENT_FCC_ENV_PATH", "/root/.fcc/.env")
    _load_fcc(fcc_env_path)

    self_state_url = os.environ.get("EMBODIMENT_SELF_STATE_URL", "")
    spritesheet = os.environ.get("EMBODIMENT_ORION_SPRITE", "f1")

    player_id = str(os.environ.get("AITOWN_ORION_PLAYER_ID") or "").strip()
    agent_id = str(os.environ.get("AITOWN_ORION_AGENT_ID") or "").strip()

    if player_id and _player_resolves(player_id) and not args.force:
        if args.resync_identity:
            # This deployment (monolithic world:* layout) has no input to update an
            # existing player's description; identity is set at join time. Rebuild the
            # persona for inspection but do not mutate the live body (avoids duplicates).
            _build_persona(self_state_url, spritesheet)
            print(
                f"resync unsupported on this town: player={player_id} already embodied; "
                "recreate with --force to apply a new persona"
            )
            return 0
        print(f"already embodied: player={player_id} agent={agent_id or '?'}")
        return 0

    persona = _build_persona(self_state_url, spritesheet)
    try:
        aitown_client.heartbeat_world()  # wake an inactive world before queuing the join
        # `join` spawns a named player driven externally by Orion's mind (no town-AI
        # agent). `createAgent` only clones a canned Descriptions[] character, so it
        # cannot produce an "Orion" identity on this deployment.
        aitown_client.join_player(
            name="Orion",
            character=persona.spritesheet,
            description=persona.identity_blurb,
        )
    except aitown_client.AitownClientError as exc:
        print(f"join failed: {exc}")
        return 1

    new_player_id, new_agent_id = _poll_new_body()
    if not new_player_id:
        print("timed out waiting for the new Orion body to appear")
        return 1
    player_id, agent_id = new_player_id, new_agent_id or agent_id

    print(f"AITOWN_ORION_PLAYER_ID={player_id}")
    print(f"AITOWN_ORION_AGENT_ID={agent_id or ''}")
    print(f"provenance.snapshot_id={persona.provenance.get('snapshot_id')}")

    if args.write:
        updates = {"AITOWN_ORION_PLAYER_ID": player_id}
        if agent_id:
            updates["AITOWN_ORION_AGENT_ID"] = agent_id
        upsert_env_keys(expand_env_path(fcc_env_path), updates)
        print(f"wrote AITOWN_ORION_* to {fcc_env_path}")
        print()
        print(_RESTART_REMINDER)
    else:
        print("dry-run: pass --write to persist AITOWN_ORION_* to ~/.fcc/.env")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
