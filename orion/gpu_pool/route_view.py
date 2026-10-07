"""Per-route view of GPU pool state: status, model, ctx and vision for each gateway route.

GPU pool stage 6.3 (spec docs/superpowers/specs/2026-09-30-gpu-pool-stage6-telemetry-reducers-lockdown.md,
"Lockdown"). Until now every reader asked orion-llm-gateway's ``GET /routes`` compatibility view,
which the gateway generated from pool state anyway. This module is that generator, moved where
every reader can call it on the pool state it reads itself (``orion:gpu_pool:state`` RPC, with
``include_config=True`` so the pool's own route/class/role table travels with the state -- no
reader needs a copy of ``config/gpu_pool.yaml``). The gateway's ``/routes`` now calls it too,
until PR 6.5 deletes that endpoint.

Semantics are unchanged from the gateway view (``status`` up/down/operator_closed/unknown,
``model`` the discovered file on the role a call for that route would land on right now). One
rule is load-bearing: **no pool state means every route is ``unknown`` with no model, ctx or
vision** -- never a guessed default.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from orion.gpu_pool.config import PoolConfig
from orion.llm.routes import BACKGROUND_LLM_ROUTES, LLM_ROUTE_DISPLAY_ORDER, SYSTEM_LLM_ROUTES

logger = logging.getLogger("orion.gpu_pool.route_view")

LLAMACPP_BACKEND = "llamacpp"  # every pool llm role is a llama.cpp server
UP_STATUSES = frozenset({"confirmed", "static"})

# Where a view came from, so a reader (and its tests) can tell "the pool said down" from
# "nobody could ask the pool".
SOURCE_POOL = "gpu_pool"
SOURCE_UNAVAILABLE = "gpu_pool_unavailable"


def config_from_state(state: Mapping[str, Any] | None) -> PoolConfig | None:
    """The pool's parsed config, as it sent it with ``include_config=True``. None when absent
    (a periodic broadcast, or a pool that predates the field) or unparseable."""
    if not isinstance(state, Mapping):
        return None
    raw = state.get("config")
    if not isinstance(raw, Mapping):
        return None
    try:
        return PoolConfig.model_validate(dict(raw))
    except Exception:  # noqa: BLE001 -- a newer pool's config shape must not break a reader
        logger.warning("gpu_pool_route_view_config_unparseable", exc_info=True)
        return None


def _definitional_priority(route_id: str) -> str | None:
    """The Hub picker filters on this; it is what the route IS, not the pool's queue priority."""
    if route_id in BACKGROUND_LLM_ROUTES:
        return "background"
    if route_id in SYSTEM_LLM_ROUTES:
        return "system"
    return None


def _catalog_route_ids(cfg: PoolConfig | None) -> list[str]:
    if cfg is None:
        return list(LLM_ROUTE_DISPLAY_ORDER)
    ordered = [r for r in LLM_ROUTE_DISPLAY_ORDER if r in cfg.routes]
    return ordered + sorted(r for r in cfg.routes if r not in ordered)


def _role_serves_class(cfg: PoolConfig, work_class: str, role: str, cards: dict[str, dict[str, Any]]) -> bool:
    spec = cfg.roles.get(role)
    if spec is None or spec.operator_only:
        return False
    if cfg.owns(work_class, role):
        return True
    # A borrower only reaches a lendable card while the operator has it lent.
    return all(bool((cards.get(card) or {}).get("lent")) for card in cfg.lendable_cards(role))


def _unknown_entry(route_id: str, *, served_by: str | None = None, upstream: str | None = None) -> dict[str, Any]:
    return {
        "id": route_id,
        "served_by": served_by,
        "backend": LLAMACPP_BACKEND if served_by else None,
        "status": "unknown",
        "latency_ms": None,
        "last_checked_at": None,
        "model": None,
        "vision": None,
        "n_ctx": None,
        "priority": _definitional_priority(route_id),
        "reserved_free_slots": None,
        "upstream": upstream,
        "gate_open": None,
        "role": None,
    }


def _entry(route_id: str, *, cfg: PoolConfig, role: str, status: str, discovered: dict[str, Any] | None,
           checked_at: str | None, gate_open: bool | None = None) -> dict[str, Any]:
    live = status == "up" and discovered is not None
    return {
        "id": route_id,
        "served_by": f"{cfg.host.name}-worker-{role}",  # the same label the pool puts on a grant
        "backend": LLAMACPP_BACKEND,
        "status": status,
        "latency_ms": None,
        "last_checked_at": checked_at,
        # Full path when the pool knows it (durable-runs compared against a full activation path).
        "model": (discovered.get("model_path") or discovered.get("model_file")) if live else None,
        "vision": discovered.get("vision") if live else None,
        "n_ctx": discovered.get("ctx_per_slot") if live else None,
        "priority": _definitional_priority(route_id),
        "reserved_free_slots": None,
        "upstream": (discovered or {}).get("url") or cfg.url(role),
        "gate_open": gate_open,
        # The pool role a call for this route would land on right now (the one whose model is shown).
        "role": role if live else None,
    }


def build_route_view(state: Mapping[str, Any] | None, cfg: PoolConfig | None = None) -> dict[str, Any]:
    """``{"source", "routes": [...]}`` from pool state. ``cfg`` defaults to the config the pool sent
    with the state; the gateway passes its own copy.

    With no state every route is ``unknown`` (``source=gpu_pool_unavailable``). With state but no
    config (a broadcast frame, or an old pool) the same: without the route -> class -> role table
    the view cannot say which role a route lands on, and guessing is exactly what this replaces."""
    if cfg is None:
        cfg = config_from_state(state)
    if not isinstance(state, Mapping) or cfg is None:
        routes = []
        for route_id in _catalog_route_ids(cfg):
            if cfg is not None:
                first = cfg.classes[cfg.routes[route_id].work_class].roles[0]
                routes.append(_unknown_entry(route_id, served_by=f"{cfg.host.name}-worker-{first}",
                                             upstream=cfg.url(first)))
            else:
                routes.append(_unknown_entry(route_id))
        return {"source": SOURCE_UNAVAILABLE, "routes": routes}

    roles = {r.get("role"): r for r in (state.get("roles") or []) if isinstance(r, Mapping)}
    roles = {k: dict(v) for k, v in roles.items()}
    cards = {c.get("card"): dict(c) for c in (state.get("cards") or []) if isinstance(c, Mapping)}
    generated_at = state.get("generated_at")

    def up(role: str) -> bool:
        return (roles.get(role) or {}).get("status") in UP_STATUSES

    def checked(role: str) -> str | None:
        value = (roles.get(role) or {}).get("checked_at") or generated_at
        return str(value) if value is not None else None

    routes: list[dict[str, Any]] = []
    for route_id in _catalog_route_ids(cfg):
        work_class = cfg.routes[route_id].work_class
        if route_id == "chat-burst" and "chat" in cfg.roles:
            # Borrowable only while Juniper has gpu0 lent.
            lent = all(bool((cards.get(c) or {}).get("lent")) for c in cfg.roles["chat"].cards)
            status = ("up" if up("chat") else "down") if lent else "operator_closed"
            routes.append(_entry(route_id, cfg=cfg, role="chat", status=status, discovered=roles.get("chat"),
                                 checked_at=checked("chat"), gate_open=lent))
            continue
        if route_id == "agent-burst" and "agent-gpu2" in cfg.roles:
            status = "up" if up("agent-gpu2") else "down"
            routes.append(_entry(route_id, cfg=cfg, role="agent-gpu2", status=status,
                                 discovered=roles.get("agent-gpu2"), checked_at=checked("agent-gpu2")))
            continue
        candidates = cfg.classes[work_class].roles
        chosen = next((r for r in candidates if up(r) and _role_serves_class(cfg, work_class, r, cards)), None)
        role = chosen or candidates[0]
        routes.append(_entry(route_id, cfg=cfg, role=role, status="up" if chosen else "down",
                             discovered=roles.get(role), checked_at=checked(role)))
    return {"source": SOURCE_POOL, "routes": routes}


def route_entry(view: Mapping[str, Any] | None, route_id: str | None) -> dict[str, Any] | None:
    """One route's entry out of a view, or None."""
    if not isinstance(view, Mapping) or not route_id:
        return None
    for entry in view.get("routes") or []:
        if isinstance(entry, Mapping) and entry.get("id") == route_id:
            return dict(entry)
    return None


async def fetch_route_view(bus: Any, *, source: str, timeout_sec: float = 2.0) -> dict[str, Any]:
    """One ``orion:gpu_pool:state`` RPC (with the pool's config) turned into a route view.

    Never raises: no bus, a timeout or an undecodable reply all give the all-``unknown`` view."""
    from orion.gpu_pool.placement import fetch_pool_state

    if bus is None:
        return build_route_view(None)
    state = await fetch_pool_state(bus, source=source, timeout_sec=timeout_sec, include_config=True)
    return build_route_view(state)
