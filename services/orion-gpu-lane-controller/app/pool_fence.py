"""Stage 4.2: the pool-generation fence -- the only actuation authority since stage 4.6.

The thing that says "this transition is still the current intent". Two checks, both local to circe:

- **generation**: the pool issues a generation per card set; the controller refuses anything
  <= the last one it accepted, and persists the accepted one *before* touching a container, so a
  controller restart cannot let an older, replayed request through.
- **launch_digest**: the pool sends the digest of its own copy of ``config/gpu_pool.yaml``'s launch
  blocks for the role; the controller recomputes it from its own checkout (the read-only ``/repo``
  mount -- the same files its ``docker compose`` calls read). A mismatch means one side is on a
  stale checkout and must not start anything.

Spec: docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md
("Stage 4 bridge (Option A)").

Stage 5.2 (docs/superpowers/specs/2026-09-29-gpu-pool-stage5-world-diffusion-generic-actuation.md,
Decision 2): ``resolve()`` returns a ``LaunchPlan`` for a swap seat,
built only from this checkout's ``launch`` blocks (compose file, service, profile, env var names) plus
the card ``index`` and a profile from the role's ``launch.profiles`` allow-list. The request names a
role and a profile, never a container, path or env value. Executed by app/launch_exec.py. Stage 5.6
deleted the stage-4 gpu2 bridge (``swap.load``/``swap.unload`` verbs, BRIDGE_TARGETS, gpu2.py): a
LaunchPlan is the only thing a load/unload resolves to.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orion.gpu_pool.config import DrainSpec, PoolConfig, launch_digest, load_pool_config

from .settings import settings

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


@dataclass(frozen=True)
class RolePlan:
    """One role's compose call, entirely from this checkout's ``launch`` block."""
    role: str
    kind: str                      # "llm" (idle check = llama.cpp /slots) | "service"
    compose: str                   # repo-relative
    env_file: str | None           # repo-relative
    service: str
    compose_profile: str | None
    env: dict[str, str]            # compose interpolation vars: cuda_env (+ profile_var)
    unset: tuple[str, ...] = ()    # vars removed from the process env (profile_var when no profile)
    drain: DrainSpec | None = None
    ready: str = "/health"         # HTTP path
    base_url: str = ""             # http://<pool host address>:<role port>
    timeout_sec: float = 600.0     # readiness wait after `up` (one budget for resume + ready)

    def env_text(self) -> str:
        return " ".join(f"{k}={v}" for k, v in sorted(self.env.items()))


@dataclass(frozen=True)
class LaunchPlan:
    """A swap seat plus the roles it evicts (drained + stopped before it starts; restarted in
    reverse stop order on a failed load's rollback, in YAML order on unload)."""
    seat: RolePlan
    evicts: tuple[RolePlan, ...] = field(default_factory=tuple)
    profile: str | None = None


def _role_plan(cfg: PoolConfig, name: str, profile: str | None) -> RolePlan:
    spec = cfg.roles[name]
    launch = spec.launch
    if launch is None or launch.actuator != settings.GPU_POOL_ACTUATOR_NAME:
        raise Refusal(f"no_launch_block:{name}")
    # Card index, never a request value: the config validator guarantees an index on every card of
    # a launch role, and the 5.1 gate that the compose device entry is ${cuda_env}.
    env = {launch.cuda_env: ",".join(str(cfg.cards[c].index) for c in spec.cards)}
    unset: tuple[str, ...] = ()
    if profile is not None:
        env[launch.profile_var] = profile
    elif launch.profile_var is not None:
        # No profile -> compose's own default, never a value inherited from the controller's env.
        unset = (launch.profile_var,)
    return RolePlan(role=name, kind=spec.kind, compose=launch.compose, env_file=launch.env_file,
                    service=launch.service, compose_profile=launch.compose_profile, env=env, unset=unset,
                    drain=launch.drain, ready=launch.ready, base_url=cfg.url(name),
                    timeout_sec=float(launch.timeout_sec))


def build_plan(cfg: PoolConfig, role: str, profile: str | None) -> LaunchPlan:
    """The generic launch plan for a swap seat. Evicted roles restart on their compose default
    model (no profile): the seat's profile choice is the seat's alone."""
    return LaunchPlan(seat=_role_plan(cfg, role, profile),
                      evicts=tuple(_role_plan(cfg, r, None) for r in cfg.evicted_by(role)),
                      profile=profile)


def role_plans(cfg: PoolConfig) -> dict[str, RolePlan]:
    """Every role whose launch names this actuator, for ``observed`` (replaces TARGET_ROLES)."""
    return {name: _role_plan(cfg, name, None) for name, spec in cfg.roles.items()
            if spec.launch is not None and spec.launch.actuator == settings.GPU_POOL_ACTUATOR_NAME}


def resolve(cfg: PoolConfig, *, role: str, action: str, cards: list[str], digest: str | None,
            profile: str | None = None) -> LaunchPlan | None:
    """What to run for a pool request, or raise Refusal: a LaunchPlan for a swap seat's load/unload,
    or None for ``status``. ``digest=None`` skips the digest check (``status`` is a read and must still
    answer from a diverged checkout)."""
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
    if profile is not None and profile not in spec.launch.profiles:
        # The allow-list is this checkout's YAML: a bus message can only pick among its entries.
        raise Refusal("profile_not_allowed")
    if spec.swap is None:
        # Residents are started only as a seat's evictions (restore), never loaded on their own:
        # a direct load could put them on a card a loaded seat still holds.
        raise Refusal("not_a_swap_seat")
    return build_plan(cfg, role, profile)


# --- persisted fence state -------------------------------------------------------------------

def _empty() -> dict[str, Any]:
    return {"generations": {}, "actions": {}, "order": [], "in_flight": None}


def read_state() -> dict[str, Any]:
    path = Path(settings.GPU_POOL_FENCE_STATE_PATH)
    with _file_lock:
        if not path.exists():
            return _empty()
        data = json.loads(path.read_text())  # a corrupt file raises: fail closed, never act on it
    state = _empty()
    state.update({k: data[k] for k in state if k in data})
    return state


def write_state(state: dict[str, Any]) -> None:
    """Atomic replace + fsync: the generation must be durable before any container is touched."""
    path = Path(settings.GPU_POOL_FENCE_STATE_PATH)
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
    # generation only.
    return await asyncio.to_thread(_authority, req, require_drained)


def _authority(req, check_digest=True):
    """The action in progress must still be the newest generation accepted for its card set,
    and the checkout must still match the digest it was accepted under. Drain/idle *safety* stays in
    launch_exec; whether-to-act (thermal, lease recall) is pool policy and is not re-asked here."""
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
