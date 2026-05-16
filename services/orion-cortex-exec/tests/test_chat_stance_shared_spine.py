from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_guard = Path(__file__).resolve().parent / "_exec_import_guard.py"
_spec = importlib.util.spec_from_file_location("_exec_guard_boot_shared_spine", _guard)
assert _spec and _spec.loader
_guard_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard_mod)

ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def test_exec_app_import_installs_shared_chat_stance_spine(monkeypatch) -> None:
    monkeypatch.delenv("CHAT_STANCE_SHARED_PROJECTION_SPINE_DISABLED", raising=False)
    _guard_mod.ensure_orion_cortex_exec_app()

    import app
    from app import chat_stance
    from app import chat_stance_shared_spine

    assert app is not None
    assert getattr(chat_stance, "_CHAT_STANCE_SHARED_PROJECTION_SPINE") is True
    assert chat_stance._unified_beliefs_for_stance is chat_stance_shared_spine.shared_unified_beliefs_for_stance


def test_shared_chat_stance_spine_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("CHAT_STANCE_SHARED_PROJECTION_SPINE_DISABLED", "true")
    _guard_mod.ensure_orion_cortex_exec_app()

    import app
    from app import chat_stance
    from app import chat_stance_shared_spine

    assert app is not None
    assert chat_stance._unified_beliefs_for_stance is not chat_stance_shared_spine.shared_unified_beliefs_for_stance
