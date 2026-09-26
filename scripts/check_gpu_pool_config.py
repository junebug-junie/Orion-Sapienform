#!/usr/bin/env python3
"""Static gate for config/gpu_pool.yaml (docs/superpowers/specs/2026-09-24-gpu-pool-design.md).

Fails on:
  - schema / cross-reference errors (unknown cards, roles, classes, evictions, duplicate ports)
  - a llama.cpp worker in orion-llamacpp-host's compose files whose LLM_ROLE is not a pool role,
    or whose announce-port default differs from that role's port (the pool would read the
    worker as "silent" and never grant it)
  - any class that may borrow the chat role on a card that is not lendable: the ONLY borrow
    path onto chat's card is the operator lend flag
  - service roles whose declared VRAM does not fit their card, alone or after a swap
  - a big-model class listing an 8B role (nothing spills down to gpu3)
  - stage 4 (docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md):
    a swap seat with neither a load/unload bridge nor a launch on itself and every role it evicts;
    a launch naming an unknown actuator; a launch role on a card with no index; and a launch whose
    compose file/service/profile/LLM_ROLE/port/cuda_env does not match what it names

Exit 0 = clean. Model-dependent VRAM for LLM roles is checked live by the pool against the
discovered profile, not here: the YAML deliberately carries no model names.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion.gpu_pool.config import check_launch, check_vram, load_pool_config  # noqa: E402

COMPOSE = [
    ROOT / "services/orion-llamacpp-host/docker-compose.atlas-workers.yml",
    ROOT / "services/orion-llamacpp-host/docker-compose.dsv41.yml",
]
ROLE_RE = re.compile(r"-\s*LLM_ROLE=([\w-]+)")
PORT_RE = re.compile(r"-\s*LLM_ANNOUNCE_PORT=\$\{[A-Z0-9_]+:-(\d+)\}")
SMALL_ROLES = {"metacog", "fast"}
BIG_CLASSES = {"chat", "agent"}


def main() -> int:
    problems: list[str] = []
    try:
        cfg = load_pool_config()
    except Exception as exc:  # noqa: BLE001
        print(f"check_gpu_pool_config: config invalid: {exc}")
        return 1

    for path in COMPOSE:
        text = path.read_text()
        roles, ports = ROLE_RE.findall(text), PORT_RE.findall(text)
        if len(roles) != len(ports):
            problems.append(f"{path.name}: {len(roles)} LLM_ROLE vs {len(ports)} LLM_ANNOUNCE_PORT")
        for role, port in zip(roles, ports):
            spec = cfg.roles.get(role)
            if spec is None:
                problems.append(f"{path.name}: LLM_ROLE={role} is not a role in config/gpu_pool.yaml")
            elif spec.port != int(port):
                problems.append(f"{path.name}: {role} announces port {port}, pool expects {spec.port}")

    for cls, spec in cfg.classes.items():
        for role in spec.roles:
            if cfg.owns(cls, role):
                continue
            for card in cfg.roles[role].cards:
                if role == "chat" and not cfg.cards[card].lendable:
                    problems.append(f"class {cls} can borrow chat on non-lendable {card}")
        if cls in BIG_CLASSES and SMALL_ROLES & set(spec.roles):
            problems.append(f"class {cls} lists a small-model role {sorted(SMALL_ROLES & set(spec.roles))}")

    services = {r: s.vram_gb for r, s in cfg.roles.items() if s.kind == "service" and s.vram_gb}
    problems += check_vram(cfg, services)
    problems += check_launch(cfg, ROOT)

    for p in problems:
        print(f"check_gpu_pool_config: {p}")
    if not problems:
        print(f"check_gpu_pool_config: ok ({len(cfg.cards)} cards, {len(cfg.roles)} roles, "
              f"{len(cfg.classes)} classes, "
              f"{sum(1 for r in cfg.roles.values() if r.launch)} launch blocks, digest {cfg.digest})")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
