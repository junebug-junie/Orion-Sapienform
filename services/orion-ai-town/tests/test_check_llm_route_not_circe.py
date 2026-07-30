"""Gate tests for check_llm_route_not_circe.py's resolution/assertion logic.

See scripts/check_llm_route_not_circe.py's module docstring for why this
exists: AI Town's live Convex LLM_MODEL silently drifted onto a
circe-hosted worker for weeks with nothing checking the *live* value
against policy. These tests cover the pure resolve/assert logic directly
(no live Convex/gateway needed); the script's __main__ wiring is smoke-
tested manually per services/orion-ai-town/README.md.
"""

from __future__ import annotations

import importlib.util
import stat
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _SERVICE / "scripts" / "check_llm_route_not_circe.py"

_spec = importlib.util.spec_from_file_location("check_llm_route_not_circe", _SCRIPT_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

resolve_served_by = _mod.resolve_served_by
check_not_circe = _mod.check_not_circe

_MODELS = [
    {"id": "chat", "served_by": "circe-worker-1"},
    {"id": "agent", "served_by": "circe-worker-1"},
    {"id": "metacog", "served_by": "atlas-worker-2"},
    {"id": "quick", "served_by": "atlas-worker-fast-1"},
]


def test_script_exists_and_is_executable():
    assert _SCRIPT_PATH.exists()
    assert _SCRIPT_PATH.stat().st_mode & stat.S_IXUSR


def test_resolve_served_by_finds_matching_model():
    assert resolve_served_by(_MODELS, "quick") == "atlas-worker-fast-1"
    assert resolve_served_by(_MODELS, "chat") == "circe-worker-1"


def test_resolve_served_by_returns_none_for_unknown_model():
    assert resolve_served_by(_MODELS, "not-a-real-route") is None


def test_check_not_circe_passes_for_atlas_worker():
    assert check_not_circe("atlas-worker-fast-1", model_id="quick", allow_circe=False) is None


def test_check_not_circe_fails_for_circe_worker():
    error = check_not_circe("circe-worker-1", model_id="chat", allow_circe=False)
    assert error is not None
    assert "circe" in error.lower()
    assert "chat" in error


def test_check_not_circe_is_case_insensitive():
    error = check_not_circe("Circe-Worker-1", model_id="chat", allow_circe=False)
    assert error is not None


def test_check_not_circe_allows_explicit_override():
    assert check_not_circe("circe-worker-1", model_id="chat", allow_circe=True) is None


def test_check_not_circe_fails_when_model_missing_from_gateway():
    error = check_not_circe(None, model_id="ghost-route", allow_circe=False)
    assert error is not None
    assert "ghost-route" in error


def test_check_not_circe_does_not_false_positive_on_similar_names():
    # A hypothetical "circe-adjacent" or "not-circe" label should not match --
    # the policy is specifically about circe-hosted workers.
    assert check_not_circe("atlas-circe-relay", model_id="quick", allow_circe=False) is None
