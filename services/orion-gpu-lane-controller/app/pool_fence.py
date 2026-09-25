"""Stage 4.2: the pool-generation fence for gpu2 (GPU2_AUTHORITY=pool).

Replaces the durable-runs ``/elastic/status`` callback as the thing that says "this transition is
still the current intent". Two checks, both local to circe:

- **generation**: the pool issues a generation per card set; the controller refuses anything
  <= the last one it accepted, and persists the accepted one *before* touching a container, so a
  controller restart cannot let an older, replayed request through.
- **launch_digest**: the pool sends the digest of its own copy of ``config/gpu_pool.yaml``'s launch
  blocks for the role; the controller recomputes it from its own checkout (the read-only ``/repo``
  mount -- the same files its ``docker compose`` calls read). A mismatch means one side is on a
  stale checkout and must not start anything.

Spec: docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md
("Stage 4 bridge (Option A)").
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any

from orion.gpu_pool.config import PoolConfig, launch_digest, load_pool_config

from .settings import settings

# Stage-4 bridge verbs (config/gpu_pool.yaml roles.<seat>.swap.load/unload) -> gpu2.transition target.
# Deleted in stage 5, when the controller builds the compose call from the role's `launch` block.
BRIDGE_TARGETS = {"gpu2/agent": "agent-burst", "gpu2/restore": "diffusion"}
# gpu2.transition target -> the pool role name it puts on the card (for `observed`).
TARGET_ROLES = {"agent-burst": "agent-gpu2", "diffusion": "diffusion"}
MAX_RECORDED_ACTIONS = 50

_file_lock = threading.Lock()


class Refusal(Exception):
    """A request the actuator will not act on; ``str(exc)`` is the result's ``reason``."""


def config_path() -> Path:
    return Path(settings.GPU_LANE_REPO_ROOT) / "config" / "gpu_pool.yaml"


def load_config() -> PoolConfig:
    """Re-read on every request: the digest must follow the checkout the compose calls read."""
    return load_pool_config(config_path())


def card_set(cards: list[str]) -> str:
    return ",".join(sorted(set(cards)))


def resolve(cfg: PoolConfig, *, role: str, action: str, cards: list[str], digest: str | None) -> str | None:
    """The gpu2.transition target for a pool request, or raise Refusal. ``digest=None`` skips the
    digest check (``status`` is a read and must still answer from a diverged checkout)."""
    spec = cfg.roles.get(role)
    if spec is None:
        raise Refusal("unknown_role")
    if spec.launch is None or spec.launch.actuator != settings.GPU_POOL_ACTUATOR_NAME:
        raise Refusal("role_not_on_this_actuator")
    if set(cards) != set(spec.cards):
        raise Refusal("cards_mismatch")
    if digest is not None and digest != launch_digest(cfg, role):
        raise Refusal("launch_digest_mismatch")
    if action == "status":
        return None
    if spec.swap is None or not spec.swap.bridged:
        # Stage 4 only bridges seats with swap.load/unload; generic launch actuation is stage 5.
        raise Refusal("not_a_bridge_role")
    verb = spec.swap.load if action == "load" else spec.swap.unload
    target = BRIDGE_TARGETS.get(verb or "")
    if target is None:
        raise Refusal("bridge_verb_unsupported")
    return target


# --- persisted fence state -------------------------------------------------------------------

def _empty() -> dict[str, Any]:
    return {"generations": {}, "actions": {}, "order": [], "in_flight": None}


def read_state() -> dict[str, Any]:
    path = Path(settings.GPU2_POOL_FENCE_STATE_PATH)
    with _file_lock:
        if not path.exists():
            return _empty()
        data = json.loads(path.read_text())  # a corrupt file raises: fail closed, never act on it
    state = _empty()
    state.update({k: data[k] for k in state if k in data})
    return state


def write_state(state: dict[str, Any]) -> None:
    """Atomic replace + fsync: the generation must be durable before any container is touched."""
    path = Path(settings.GPU2_POOL_FENCE_STATE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with _file_lock:
        with open(tmp, "w") as fh:
            json.dump(state, fh, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)


def last_generation(state: dict[str, Any], cards: list[str]) -> int:
    return int(state["generations"].get(card_set(cards), 0))


def record(state: dict[str, Any], action_id: str, result: dict[str, Any]) -> None:
    """Keep the terminal result of an accepted action so a replayed action_id gets it back."""
    state["actions"][action_id] = result
    order = [a for a in state["order"] if a != action_id] + [action_id]
    for old in order[:-MAX_RECORDED_ACTIONS]:
        state["actions"].pop(old, None)
    state["order"] = order[-MAX_RECORDED_ACTIONS:]


def recover_interrupted() -> dict[str, Any] | None:
    """On boot: an action still marked in flight was cut off by a controller restart. Record it as
    failed (restored unknown) so a pool replay or ``status`` adopts that instead of re-running it;
    the observed containers in the ``status`` reply tell the pool what is actually on the card."""
    state = read_state()
    flight = state.get("in_flight")
    if not flight:
        return None
    # A load cut off mid-way may already have stopped diffusion: report restored=False (the pool
    # then marks the card `fault` for an operator) rather than None, which reads "nothing evicted".
    restored = False if flight.get("action") == "load" else None
    result = {**flight, "status": "failed", "phase": None, "restored": restored,
              "reason": "interrupted_by_controller_restart", "elapsed_ms": None, "observed": {}}
    record(state, flight["action_id"], result)
    state["in_flight"] = None
    write_state(state)
    return result


async def authority(req, *, require_drained=True):
    # require_drained=False is the pre-flight and the rollback check. Rollback returns the card to
    # its previous residents, so a checkout edited mid-load must not block it: identity and
    # generation only. (The durable admissions require_drained guarded do not exist under pool.)
    return await asyncio.to_thread(_authority, req, require_drained)


def _authority(req, check_digest=True):
    """Pool-mode replacement for gpu2.authority(): the transition in progress must still be the
    newest generation accepted for gpu2, and the checkout must still match the digest it was
    accepted under. Drain/idle *safety* stays in gpu2.transition; whether-to-act (thermal, visual
    baseline, lease recall) is pool policy and is not re-asked here."""
    state = read_state()
    flight = state.get("in_flight") or {}
    if flight.get("action_id") != req.operation_id or flight.get("generation") != req.generation:
        raise RuntimeError("stale_or_unknown_intent")
    if last_generation(state, flight["cards"]) != req.generation:
        raise RuntimeError("stale_or_unknown_intent")
    if not check_digest:
        return {"can_transition": True, "activation_eligible": True}
    cfg = load_config()
    if launch_digest(cfg, flight["role"]) != flight["launch_digest"]:
        raise RuntimeError("launch_digest_changed")
    return {"can_transition": True, "activation_eligible": True}
