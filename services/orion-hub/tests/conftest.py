"""Ensure Hub ``scripts`` package resolves to ``services/orion-hub/scripts`` (not repo ``scripts/``)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

_HUB_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _HUB_ROOT.parents[1]
_OTHER_SERVICE_ROOTS = tuple(
    p
    for p in _REPO_ROOT.glob("services/orion-*")
    if p.is_dir() and p.resolve() != _HUB_ROOT.resolve()
)


def _ensure_hub_paths() -> None:
    # Drop cached imports so the corrected sys.path wins (repo-root `scripts/` namespace vs Hub package).
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    # Remove stale entries so we can prepend Hub before cwd (`''`) and duplicate PYTHONPATH entries.
    for p in (str(_REPO_ROOT), str(_HUB_ROOT), *(str(p) for p in _OTHER_SERVICE_ROOTS)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_HUB_ROOT))


# api_routes.py binds SUBSTRATE_REVIEW_QUEUE_STORE and
# SUBSTRATE_REVIEW_TELEMETRY_STORE at import time, resolving their Postgres URL
# from these three keys in order (see _resolve_control_plane_postgres_url).
# On a developer or operator box where the service .env is present, that meant
# `pytest services/orion-hub/tests` connected the control-plane stores to LIVE
# Postgres and every `.record(...)` in a test wrote a real row. Verified
# 2026-09-01: substrate_review_telemetry held 1562 rows, of which 1560 were
# `selection_reason="test"` seeds from test_substrate_standalone_page.py and
# test_phase20_policy_comparison.py -- enough to bury the 2 genuine signals and
# make the mutation scheduler's starvation report cite a store full of fiction.
# pytest_configure runs before any test module is imported, which is the only
# point early enough to change what those module-level stores bind to.
_CONTROL_PLANE_POSTGRES_ENV_KEYS = (
    "SUBSTRATE_CONTROL_PLANE_POSTGRES_URL",
    "SUBSTRATE_POLICY_POSTGRES_URL",
    "DATABASE_URL",
)


def _detach_control_plane_from_live_postgres() -> None:
    # pop, not `= ""`. A present-but-empty key still satisfies os.environ.setdefault,
    # which test_grammar_atlas_api.py:39 relies on to install its own DSN -- setting
    # these to "" made that setdefault a no-op and 503'd 3 tests, and also silently
    # turned that file's monkeypatch.delenv("DATABASE_URL") into a no-op, so the test
    # could no longer tell "config removed" from "config never existed".
    for key in _CONTROL_PLANE_POSTGRES_ENV_KEYS:
        os.environ.pop(key, None)
    # SubstratePolicyProfileStore falls back to a HARDCODED shared path
    # (/tmp/orion_substrate_policy.sqlite3, policy_profiles.py) when both its URL
    # and sqlite path are unset, so merely detaching Postgres relocates the test
    # writes rather than isolating them -- that file accumulates across every run,
    # session and process on the box, cross-contaminating
    # pair_mode="previous_vs_current" comparisons.
    os.environ.setdefault(
        "SUBSTRATE_POLICY_SQL_DB_PATH",
        str(Path(tempfile.mkdtemp(prefix="orion-hub-tests-policy-")) / "policy.sqlite3"),
    )


def pytest_configure() -> None:
    _detach_control_plane_from_live_postgres()
    _ensure_hub_paths()


@pytest.fixture(autouse=True)
def _hub_service_isolation() -> None:
    # Deliberately does NOT re-pop the control-plane keys. pytest_configure runs
    # before any test module is imported, which is early enough for the
    # module-level store binding in api_routes -- and a test module is then free
    # to os.environ.setdefault its own DSN at collection time.
    # test_grammar_atlas_api.py:39 does exactly that for a DIFFERENT store, and
    # popping again per-test would undo it and 503 that suite.
    _ensure_hub_paths()
    yield
