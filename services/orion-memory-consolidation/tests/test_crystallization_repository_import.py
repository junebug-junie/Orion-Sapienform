from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_repository_imports_without_psycopg2(monkeypatch):
    """orion-memory-consolidation ships only asyncpg; the crystallization
    repository (and thus the gate's insert_crystallization import) must load
    without psycopg2 available. Regression for the live gate crash where the
    module-level `import psycopg2` broke consolidate_window on window close.
    """
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "psycopg2" or name.startswith("psycopg2."):
            raise ImportError("psycopg2 is intentionally unavailable in this test")
        return real_import(name, *args, **kwargs)

    for key in list(sys.modules):
        if key == "psycopg2" or key.startswith("psycopg2."):
            monkeypatch.delitem(sys.modules, key, raising=False)
    monkeypatch.delitem(sys.modules, "orion.memory.crystallization.repository", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    module = importlib.import_module("orion.memory.crystallization.repository")
    assert hasattr(module, "insert_crystallization")


def test_apply_schema_still_uses_psycopg2_lazily(monkeypatch):
    """The sync DDL helper is allowed to require psycopg2, but only when called."""
    monkeypatch.delitem(sys.modules, "orion.memory.crystallization.repository", raising=False)
    module = importlib.import_module("orion.memory.crystallization.repository")
    assert callable(module.apply_memory_crystallizations_schema)
