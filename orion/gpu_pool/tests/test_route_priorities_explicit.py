"""Every pool route names its priority (spec 2026-10-06-thermal-controller-redesign, D4 / C6).

RouteSpec.priority keeps its "system" default on purpose (the shorthand validator and tests that
build RouteSpec without one rely on it). The cost of that default was real: ``agent: agent`` and
``quick: fast`` meant ``system`` without saying so, and system work is shed under heat -- human
turns on those routes waited out cooling incidents. So the check lives on the YAML text, read
with plain yaml (not PoolConfig, whose validator expands the shorthand and fills the default).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from orion.gpu_pool.config import PRIORITIES

POOL_YAML = Path(__file__).resolve().parents[3] / "config" / "gpu_pool.yaml"


def implicit_priority_routes(doc: dict) -> list[str]:
    """``section.route`` for every route that is shorthand, lacks ``priority`` or names an unknown one."""
    bad: list[str] = []
    for section in ("routes", "hold_routes"):
        for name, spec in (doc.get(section) or {}).items():
            if not isinstance(spec, dict) or "priority" not in spec or spec["priority"] not in PRIORITIES:
                bad.append(f"{section}.{name}")
    return bad


def test_every_pool_route_has_an_explicit_priority():
    doc = yaml.safe_load(POOL_YAML.read_text(encoding="utf-8"))
    assert doc.get("routes"), "config/gpu_pool.yaml has no routes: section"
    assert implicit_priority_routes(doc) == []


@pytest.mark.parametrize("routes, hold_routes, expected", [
    ({"agent": "agent"}, {}, ["routes.agent"]),                                   # shorthand
    ({"quick": {"class": "fast"}}, {}, ["routes.quick"]),                         # long form, no priority
    ({"x": {"class": "fast", "priority": "sometimes"}}, {}, ["routes.x"]),        # not a priority
    ({}, {"diffusion": {"class": "diffusion"}}, ["hold_routes.diffusion"]),
    ({"chat": {"class": "chat", "priority": "interactive"}}, {}, []),
])
def test_the_check_catches_implicit_priorities(routes, hold_routes, expected):
    assert implicit_priority_routes({"routes": routes, "hold_routes": hold_routes}) == expected


def test_turn_routes_are_interactive_and_their_plain_twins_stay_system():
    """D4: the turn routes are the fix; the plain routes keep shedding their background callers."""
    doc = yaml.safe_load(POOL_YAML.read_text(encoding="utf-8"))
    routes = doc["routes"]
    for turn, plain, cls in (("metacog_turn", "metacog", "metacog"), ("quick_turn", "quick", "fast"),
                             ("agent_turn", "agent", "agent")):
        assert routes[turn] == {"class": cls, "priority": "interactive"}
        assert routes[plain] == {"class": cls, "priority": "system"}
