"""Guards the 2026-09-07 route move: semantic/stance off the saturated `quick`
lane, onto `metacog` (confirmed idle live at the time of the patch). appraisal
was already on metacog. See docs/superpowers/pr-reports/ for the incident this
fixes -- semantic_synthesis calls on `quick` were timing out and driving
mind_quality=fallback_contract_only on ~97% of live chat turns.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(SERVICE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

_guard_path = Path(__file__).resolve().parent / "_mind_import_guard.py"


def _mind_prep() -> None:
    spec = importlib.util.spec_from_file_location("_mind_guard_lazy", _guard_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_orion_mind_app()


def test_semantic_and_stance_routes_default_to_metacog(monkeypatch) -> None:
    _mind_prep()
    for key in (
        "MIND_SEMANTIC_MODEL_ROUTE",
        "MIND_APPRAISAL_MODEL_ROUTE",
        "MIND_STANCE_MODEL_ROUTE",
    ):
        monkeypatch.delenv(key, raising=False)
    settings_module = importlib.import_module("app.settings")
    importlib.reload(settings_module)
    settings = settings_module.settings
    assert settings.MIND_SEMANTIC_MODEL_ROUTE == "metacog"
    assert settings.MIND_APPRAISAL_MODEL_ROUTE == "metacog"
    assert settings.MIND_STANCE_MODEL_ROUTE == "metacog"
