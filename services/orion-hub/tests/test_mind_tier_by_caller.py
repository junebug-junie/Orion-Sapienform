"""Tiers attach to callers, not lanes (body-economy design Organ 3b, 2026-10-23).

orion-mind lifts a run to the interactive (never heat-shed) tier only when it is positively marked
as Juniper's turn (utterance_origin="juniper"; engine.carries_live_turn). This file pins the Hub
half of that contract:

1. Every call site that builds a turn which can reach Mind passes the origin this table says.
   A new call site fails here until someone decides who is waiting on it. A Juniper path missing
   the mark would be silently demoted to sheddable -- the dangerous direction.
2. build_chat_request (cortex-orch brain-mode path) writes the mark into request metadata, which
   cortex-orch copies onto MindRunRequestV1 (services/orion-cortex-orch/tests/
   test_mind_utterance_origin_tier.py).

The unified-turn half (execute_unified_turn -> stance_inputs -> orion-thought
build_light_mind_request) is pinned by test_turn_orchestrator_utterance_origin.py and
services/orion-thought/tests/test_mind_utterance_origin.py.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cortex_request_builder.py"
_SPEC = importlib.util.spec_from_file_location("hub_cortex_request_builder_tier", _MODULE_PATH)
assert _SPEC and _SPEC.loader
hub_builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hub_builder)

# (file relative to repo root, enclosing function) -> utterance_origin literal passed (None = unset).
# "juniper": Juniper typed it and is waiting -> interactive.
# "orion": Orion's own curiosity / self-inquiry / self-sense-eval / urgent runs -> system.
# None: Orion-started work with no human waiting -> system. Deliberately NOT "orion": in
#   execute_unified_turn "orion" also turns on the curiosity role-teach splice (a "hire Cursor"
#   nudge in the motor prompt) and the "self-authored investigation subject" stance note, which are
#   wrong for a world-pulse read, an outreach message or a collapse-mirror reply.
EXPECTED_ORIGIN = {
    # Juniper unless the payload is a relayed external social room (is_third_party_room).
    ("services/orion-hub/scripts/api_routes.py", "handle_chat_request", "execute_unified_turn"): ["if:None|else:juniper"],
    ("services/orion-hub/scripts/api_routes.py", "handle_chat_request", "build_chat_request"): ["juniper"],
    ("orion/hub/turn_orchestrator.py", "run_unified_turn", "execute_unified_turn"): ["juniper"],
    # run_unified_turn hardcodes "juniper" (above): every caller of it inherits Juniper's mark.
    ("services/orion-hub/scripts/websocket_handler.py", "websocket_endpoint", "run_unified_turn"): [None],  # juniper inside
    ("services/orion-hub/scripts/websocket_handler.py", "websocket_endpoint", "build_chat_request"): ["juniper"],
    ("services/orion-hub/scripts/curiosity_investigation.py", "_generate", "execute_unified_turn"): ["orion"],
    ("services/orion-hub/scripts/reading_turn_listener.py", "_execute", "execute_unified_turn"): [None],
    ("services/orion-hub/scripts/endogenous_outreach.py", "_attempt_unified_turn", "execute_unified_turn"): [None],
    ("services/orion-hub/scripts/collapse_mirror_chat_reply.py", "_run_chat_lane_turn", "execute_unified_turn"): [None],
    ("services/orion-hub/scripts/memory_graph_suggest.py", "suggest_with_escalation", "build_chat_request"): [None],
    ("orion/social/scenario_replay.py", "_run_fixture", "build_chat_request"): [None],
    # Direct stance-request builders: the origin rides stance_inputs, not a kwarg.
    ("orion/hub/turn_orchestrator.py", "execute_unified_turn", "StanceReactRequestV1"): [None],
    ("services/orion-hub/scripts/unified_turn_stub.py", "run_orion_unified_turn_stub", "StanceReactRequestV1"): [None],
}
_TARGETS = ("execute_unified_turn", "run_unified_turn", "build_chat_request", "build_cortex_chat_request",
            "StanceReactRequestV1")
_SCAN_ROOTS = ("services/orion-hub/scripts", "services/orion-hub/app", "orion")


def _call_name(node: ast.Call) -> str | None:
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return None


def _origin_literal(node: ast.Call) -> object:
    for kw in node.keywords:
        if kw.arg == "utterance_origin":
            if isinstance(kw.value, ast.Constant):
                return kw.value.value
            v = kw.value
            if isinstance(v, ast.IfExp) and isinstance(v.body, ast.Constant) and isinstance(v.orelse, ast.Constant):
                return f"if:{v.body.value}|else:{v.orelse.value}"
            return "<non-literal>"
    return None


def _scan() -> dict[tuple[str, str, str], list[object]]:
    found: dict[tuple[str, str, str], list[object]] = {}
    for root in _SCAN_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if "/tests/" in rel or "/evals/" in rel:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))

            def visit(node: ast.AST, func: str) -> None:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        visit(child, child.name)
                        continue
                    if isinstance(child, ast.Call):
                        name = _call_name(child)
                        # build_chat_request's own body calls build_cortex_chat_request: not a caller.
                        if name in _TARGETS and not (
                            name == "build_cortex_chat_request" and rel.endswith("cortex_request_builder.py")
                        ):
                            found.setdefault((rel, func, name), []).append(_origin_literal(child))
                    visit(child, func)

            visit(tree, "<module>")
    return found


def _expected_for(key: tuple[str, str, str]) -> list[object] | None:
    return EXPECTED_ORIGIN.get(key)


def test_every_turn_call_site_is_classified() -> None:
    found = _scan()
    unclassified = sorted(k for k in found if _expected_for(k) is None)
    assert not unclassified, (
        "New turn call site(s) that can reach Mind. Decide who is waiting and add them to "
        f"EXPECTED_ORIGIN (pass utterance_origin=\"juniper\" only if Juniper typed it): {unclassified}"
    )


def test_every_call_site_passes_its_expected_origin() -> None:
    found = _scan()
    wrong = {k: v for k, v in found.items() if _expected_for(k) is not None and set(v) != set(_expected_for(k))}
    assert not wrong, f"utterance_origin drifted at: {wrong}"


def test_no_aliased_import_of_a_turn_builder() -> None:
    """`import execute_unified_turn as x` would hide a call site from the scan."""
    aliased = []
    for root in _SCAN_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name in _TARGETS and alias.asname and alias.asname != alias.name:
                            aliased.append(f"{path.relative_to(REPO_ROOT)}: {alias.name} as {alias.asname}")
    assert not aliased, aliased


def test_every_juniper_path_in_the_table_still_exists() -> None:
    """A renamed Juniper call site must not silently drop out of the gate."""
    found = _scan()
    for (rel, func, name), origins in EXPECTED_ORIGIN.items():
        if not any("juniper" in str(o) for o in origins):
            continue
        assert (rel, func, name) in found, f"Juniper call site missing: {rel}::{func} -> {name}"


def _build(payload: dict, origin: str | None) -> dict:
    req, _debug, _ = hub_builder.build_chat_request(
        payload=payload, session_id="s", user_id="u", trace_id="t", default_mode="brain",
        auto_default_enabled=False, source_label="hub_ws", prompt="hello", utterance_origin=origin,
    )
    return dict(req.metadata or {})


def test_juniper_chat_with_mind_toggle_carries_the_mark() -> None:
    meta = _build({"mode": "brain", "context": {"metadata": {"mind_enabled": True}}}, "juniper")
    assert meta.get("mind_enabled") is True
    assert meta.get("utterance_origin") == "juniper"


def test_unmarked_caller_carries_no_mark() -> None:
    meta = _build({"mode": "brain", "context": {"metadata": {"mind_enabled": True}}}, None)
    assert "utterance_origin" not in meta


def test_client_payload_cannot_claim_juniper() -> None:
    meta = _build({"mode": "brain", "context": {"metadata": {"mind_enabled": True, "utterance_origin": "juniper"}}}, None)
    assert "utterance_origin" not in meta


def test_external_social_room_message_is_not_juniper() -> None:
    payload = {"mode": "brain", "chat_profile": "social_room",
               "external_room": {"platform": "callsyne", "room_id": "r1"}}
    assert "utterance_origin" not in _build(payload, "juniper")


def test_hub_direct_social_room_is_still_juniper() -> None:
    payload = {"mode": "brain", "chat_profile": "social_room", "social_room_mode": "hub_direct",
               "external_room": {"platform": "hub", "room_id": "hub-direct"}}
    assert _build(payload, "juniper").get("utterance_origin") == "juniper"


def test_unified_http_turn_from_external_room_is_not_juniper() -> None:
    assert hub_builder.is_third_party_room(
        {"chat_profile": "social_room", "external_room": {"platform": "callsyne", "room_id": "r"}}) is True
    assert hub_builder.is_third_party_room({"mode": "orion"}) is False
    assert hub_builder.is_third_party_room(
        {"chat_profile": "social_room", "social_room_mode": "hub_direct"}) is False
