from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Optional

from orion.schemas.embodiment import EmbodimentIntentV1


@dataclass(frozen=True)
class ResolveResult:
    status: str  # "actuated" | "resolved_noop" | "denied"
    destination: Optional[dict[str, float]]
    ref_player_id: Optional[str]
    reason: str


def _pos(player: dict[str, Any]) -> Optional[dict[str, float]]:
    p = player.get("position")
    if isinstance(p, dict) and "x" in p and "y" in p:
        return {"x": float(p["x"]), "y": float(p["y"])}
    return None


def _match(player: dict[str, Any], ref: str) -> bool:
    return str(player.get("id")) == ref or str(player.get("name") or "").lower() == ref.lower()


def _others(players: list[dict[str, Any]], orion_player_id: str) -> list[dict[str, Any]]:
    return [p for p in players if str(p.get("id")) != orion_player_id and _pos(p) is not None]


def _nearest(players: list[dict[str, Any]], origin: dict[str, float]) -> Optional[dict[str, Any]]:
    best, best_d = None, math.inf
    for p in players:
        pos = _pos(p)
        d = (pos["x"] - origin["x"]) ** 2 + (pos["y"] - origin["y"]) ** 2
        if d < best_d:
            best, best_d = p, d
    return best


def _resolve_target(
    intent: EmbodimentIntentV1, orion_player_id: str, players: list[dict[str, Any]]
) -> tuple[Optional[dict[str, Any]], str]:
    orion = next((p for p in players if str(p.get("id")) == orion_player_id), None)
    origin = _pos(orion) if orion else {"x": 0.0, "y": 0.0}
    others = _others(players, orion_player_id)
    if intent.ref:
        target = next((p for p in others if _match(p, intent.ref)), None)
        return target, (f"target={intent.ref}" if target else f"no player matching {intent.ref}")
    target = _nearest(others, origin or {"x": 0.0, "y": 0.0})
    return target, ("nearest player" if target else "no other players")


def resolve_destination(
    intent: EmbodimentIntentV1,
    *,
    orion_player_id: str,
    players: list[dict[str, Any]],
    locations: Optional[dict[str, dict[str, float]]] = None,
    wander_radius: float = 3.0,
    rng: Optional[random.Random] = None,
) -> ResolveResult:
    locations = locations or {}
    rng = rng or random.Random()

    if intent.kind == "idle":
        return ResolveResult("resolved_noop", None, None, "idle")

    if intent.kind in ("approach_player", "start_conversation"):
        target, why = _resolve_target(intent, orion_player_id, players)
        if target is None:
            return ResolveResult("denied", None, None, why)
        return ResolveResult("actuated", _pos(target), str(target.get("id")), why)

    if intent.kind == "go_to_location":
        loc = locations.get(intent.ref or "")
        if not loc:
            return ResolveResult("denied", None, None, f"unknown location {intent.ref!r}")
        return ResolveResult("actuated", {"x": float(loc["x"]), "y": float(loc["y"])}, None, f"location {intent.ref}")

    if intent.kind == "wander":
        orion = next((p for p in players if str(p.get("id")) == orion_player_id), None)
        origin = _pos(orion) if orion else {"x": 0.0, "y": 0.0}
        dx = rng.uniform(-wander_radius, wander_radius)
        dy = rng.uniform(-wander_radius, wander_radius)
        return ResolveResult(
            "actuated", {"x": origin["x"] + dx, "y": origin["y"] + dy}, None, "wander offset"
        )

    return ResolveResult("denied", None, None, f"unhandled kind {intent.kind}")
