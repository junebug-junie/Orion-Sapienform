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

route_listed = _mod.route_listed
resolve_work_class = _mod.resolve_work_class
check_not_circe = _mod.check_not_circe

_MODELS = [
    {"id": "chat", "served_by": "circe-worker-1"},
    {"id": "agent", "served_by": "circe-worker-agent-1"},
    {"id": "metacog", "served_by": "circe-worker-2"},
    {"id": "quick", "served_by": "circe-worker-fast-1"},
    {"id": "quick_background", "served_by": "circe-worker-fast-1"},
]


def test_script_exists_and_is_executable():
    assert _SCRIPT_PATH.exists()
    assert _SCRIPT_PATH.stat().st_mode & stat.S_IXUSR


def test_routes_resolve_to_their_gpu_pool_class():
    assert resolve_work_class("quick_background") == "fast"
    assert resolve_work_class("chat") == "chat"
    assert resolve_work_class("not-a-real-route") is None
    assert route_listed(_MODELS, "quick_background") and not route_listed(_MODELS, "ghost")


def test_check_not_circe_passes_for_the_fast_class():
    assert check_not_circe("fast", model_id="quick_background", allow_circe=False) is None


def test_check_not_circe_fails_for_the_chat_class_even_under_another_route_name():
    assert "chat GPU" in check_not_circe("chat", model_id="harness", allow_circe=False)
    assert "chat lane" in check_not_circe("chat", model_id="chat", allow_circe=False)


def test_check_not_circe_allows_explicit_override():
    assert check_not_circe("chat", model_id="chat", allow_circe=True) is None


def test_check_not_circe_fails_when_route_unknown():
    error = check_not_circe(None, model_id="ghost-route", allow_circe=False)
    assert error and "Refusing to pass" in error
