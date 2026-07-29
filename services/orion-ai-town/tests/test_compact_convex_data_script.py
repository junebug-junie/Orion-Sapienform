"""Gate tests for the Convex data-compaction script (structural, no Docker).

These check the script's shape (flags, safety steps present) without running
it -- actually exercising it requires a live Docker/Convex deployment, which
is exactly what services/orion-ai-town/README.md's own "Smoke" section
covers manually. See scripts/compact_convex_data.sh's header comment for why
this exists: self-hosted Convex retains unbounded document-revision history
with no built-in compaction (confirmed live 2026-07-29: 23.5GB file, VACUUM
recovered only ~5%, a full export/reset/reimport cycle took it to 240MB).
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
_SCRIPT = _SERVICE / "scripts" / "compact_convex_data.sh"


def test_script_exists_and_is_executable():
    assert _SCRIPT.exists()
    mode = _SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR


def test_script_supports_check_and_force_flags():
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "--check" in text
    assert "--force" in text
    assert "CHECK_ONLY" in text
    assert "FORCE" in text


def test_script_is_threshold_gated_not_blind():
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "AITOWN_COMPACT_THRESHOLD_BYTES" in text
    assert "below threshold" in text


def test_script_exports_before_touching_anything():
    text = _SCRIPT.read_text(encoding="utf-8")
    export_idx = text.index("npx convex export")
    stop_idx = text.index('"${COMPOSE[@]}" stop backend')
    assert export_idx < stop_idx, "export must happen before the backend is stopped"


def test_script_backs_up_before_removing_the_db_file():
    text = _SCRIPT.read_text(encoding="utf-8")
    backup_idx = text.index("db.sqlite3.pre-compact.bak")
    rename_idx = text.index("mv /data/db.sqlite3 /data/db.sqlite3.pre-compact-")
    assert backup_idx < rename_idx, "must snapshot before renaming/removing the live file"


def test_script_restores_env_vars_and_redeploys_functions_after_reset():
    # Regression: a live run found the reset wipes deployed function code and
    # `npx convex env` vars along with the data, since they live in the same
    # SQLite file. Both must be captured before and restored after.
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "npx convex env list" in text
    assert "npx convex env set --from-file" in text
    assert "npx convex dev --once" in text


def test_script_heartbeats_world_back_to_running():
    # Regression: without this, the world stays "inactive" after a reset
    # until someone happens to reload the frontend tab.
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "world:heartbeatWorld" in text


def test_script_has_bash_syntax_ok():
    import subprocess

    result = subprocess.run(
        ["bash", "-n", str(_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
