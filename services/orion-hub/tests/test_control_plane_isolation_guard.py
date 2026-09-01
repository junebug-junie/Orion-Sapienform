"""This suite must never bind Orion's control plane to a live database.

`api_routes` builds SUBSTRATE_REVIEW_QUEUE_STORE, SUBSTRATE_REVIEW_TELEMETRY_STORE
and SUBSTRATE_POLICY_STORE at *import* time from the ambient environment, so a
developer or operator box with the service `.env` in scope had `pytest
services/orion-hub/tests` writing straight into production. Measured 2026-09-01:
substrate_review_telemetry held 1562 rows, 1560 of them `selection_reason="test"`
seeds from test_substrate_standalone_page.py (12 per run) -- enough to bury the 2
genuine signals and make the mutation scheduler's own starvation report cite a
store full of fiction.

conftest.pytest_configure is the fix; without a guard nothing fails if a later
edit drops it, which is exactly how the leak lasted 130 runs.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
hub_scripts_pkg = HUB_ROOT / "scripts" / "__init__.py"
if (
    "scripts" not in sys.modules
    or not str(getattr(sys.modules.get("scripts"), "__file__", "")).startswith(str(HUB_ROOT))
):
    spec = importlib.util.spec_from_file_location(
        "scripts", str(hub_scripts_pkg), submodule_search_locations=[str(HUB_ROOT / "scripts")]
    )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["scripts"] = module
        spec.loader.exec_module(module)

from scripts import api_routes  # noqa: E402


def test_control_plane_env_keys_are_absent_not_blank() -> None:
    """Absent, not "". A present-but-empty key defeats os.environ.setdefault,
    which test_grammar_atlas_api.py relies on to install its own DSN."""
    for key in ("SUBSTRATE_CONTROL_PLANE_POSTGRES_URL", "SUBSTRATE_POLICY_POSTGRES_URL", "DATABASE_URL"):
        assert key not in os.environ, f"{key} is set during tests: {os.environ.get(key)!r}"


def test_control_plane_stores_are_not_postgres_backed() -> None:
    """The property that actually matters -- env is only the mechanism."""
    for name in ("SUBSTRATE_REVIEW_QUEUE_STORE", "SUBSTRATE_REVIEW_TELEMETRY_STORE", "SUBSTRATE_POLICY_STORE"):
        store = getattr(api_routes, name)
        assert store.source_kind() != "postgres", f"{name} is bound to live Postgres"


def test_policy_store_is_not_on_the_shared_hardcoded_path() -> None:
    """SubstratePolicyProfileStore falls back to /tmp/orion_substrate_policy.sqlite3
    when both its URL and sqlite path are unset -- shared across every run,
    session and process on the box, which cross-contaminates
    pair_mode="previous_vs_current" comparisons."""
    path = os.environ.get("SUBSTRATE_POLICY_SQL_DB_PATH") or ""
    assert path, "SUBSTRATE_POLICY_SQL_DB_PATH must be pinned for tests"
    assert path != "/tmp/orion_substrate_policy.sqlite3"
