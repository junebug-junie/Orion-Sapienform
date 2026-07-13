from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from sync_local_env_from_example import (  # noqa: E402
    NEVER_SYNC_KEYS,
    example_value_is_host_placeholder,
    should_sync_key,
    sync_file,
)


def test_orion_bus_url_never_synced() -> None:
    assert "ORION_BUS_URL" in NEVER_SYNC_KEYS
    assert should_sync_key("ORION_BUS_URL", all_keys=False) is False
    assert should_sync_key("ORION_BUS_URL", all_keys=True) is False


def test_placeholder_bus_url_skipped_even_if_all_keys() -> None:
    assert example_value_is_host_placeholder("ORION_BUS_URL", "redis://100.x.x.x:6379/0")
    assert example_value_is_host_placeholder("ORION_BUS_URL", "redis://bus-core:6379/0")
    assert not example_value_is_host_placeholder("ORION_BUS_URL", "redis://100.92.216.81:6379/0")


def test_sync_file_does_not_clobber_local_bus_url(tmp_path: Path) -> None:
    svc = tmp_path / "orion-thought"
    svc.mkdir()
    (svc / ".env_example").write_text(
        "ORION_BUS_URL=redis://100.x.x.x:6379/0\nSTANCE_REACT_TIMEOUT_SEC=120\n",
        encoding="utf-8",
    )
    (svc / ".env").write_text(
        "ORION_BUS_URL=redis://100.92.216.81:6379/0\nSTANCE_REACT_TIMEOUT_SEC=12\n",
        encoding="utf-8",
    )
    result = sync_file(svc / ".env", svc / ".env_example", dry_run=False, all_keys=True)
    text = (svc / ".env").read_text(encoding="utf-8")
    # ORION_BUS_URL is in NEVER_SYNC_KEYS: excluded entirely, regardless of divergence.
    assert "ORION_BUS_URL=redis://100.92.216.81:6379/0" in text
    assert not any("ORION_BUS_URL" in c for c in result.updated + result.diverged)
    # STANCE_REACT_TIMEOUT_SEC diverges (12 vs 120) but force defaults to False, so it
    # must NOT be silently overwritten — this is the bug this module now guards against.
    assert "STANCE_REACT_TIMEOUT_SEC=12" in text
    assert "STANCE_REACT_TIMEOUT_SEC=120" not in text
    assert any("STANCE_REACT_TIMEOUT_SEC" in c for c in result.diverged)
    assert not any("STANCE_REACT_TIMEOUT_SEC" in c for c in result.updated)


def test_sync_file_default_does_not_overwrite_diverged_key(tmp_path: Path) -> None:
    """Regression test for the graphiti-adapter incident: an existing local value that
    differs from .env_example (an intentional deployment-specific override) must be
    left alone by default, and reported as diverged rather than silently reset."""
    svc = tmp_path / "orion-graphiti-adapter"
    svc.mkdir()
    (svc / ".env_example").write_text("GRAPHITI_BACKEND=orion_postgres\n", encoding="utf-8")
    (svc / ".env").write_text("GRAPHITI_BACKEND=graphiti_core\n", encoding="utf-8")

    result = sync_file(svc / ".env", svc / ".env_example", dry_run=False, all_keys=False)

    text = (svc / ".env").read_text(encoding="utf-8")
    assert "GRAPHITI_BACKEND=graphiti_core" in text
    assert not result.updated
    assert any("GRAPHITI_BACKEND" in c for c in result.diverged)
    assert "local='graphiti_core'" in result.diverged[0]
    assert "example='orion_postgres'" in result.diverged[0]


def test_sync_file_force_overwrites_diverged_key(tmp_path: Path) -> None:
    svc = tmp_path / "orion-graphiti-adapter"
    svc.mkdir()
    (svc / ".env_example").write_text("GRAPHITI_BACKEND=orion_postgres\n", encoding="utf-8")
    (svc / ".env").write_text("GRAPHITI_BACKEND=graphiti_core\n", encoding="utf-8")

    result = sync_file(
        svc / ".env", svc / ".env_example", dry_run=False, all_keys=False, force=True
    )

    text = (svc / ".env").read_text(encoding="utf-8")
    assert "GRAPHITI_BACKEND=orion_postgres" in text
    assert not result.diverged
    assert any("GRAPHITI_BACKEND" in c for c in result.updated)


def test_sync_file_missing_key_auto_added_default_and_force(tmp_path: Path) -> None:
    for force in (False, True):
        svc = tmp_path / f"orion-graphiti-adapter-{force}"
        svc.mkdir()
        (svc / ".env_example").write_text(
            "GRAPHITI_BACKEND=orion_postgres\nCRYSTALLIZER_NEW_KEY=example_value\n",
            encoding="utf-8",
        )
        (svc / ".env").write_text("GRAPHITI_BACKEND=orion_postgres\n", encoding="utf-8")

        result = sync_file(
            svc / ".env", svc / ".env_example", dry_run=False, all_keys=False, force=force
        )

        text = (svc / ".env").read_text(encoding="utf-8")
        assert "CRYSTALLIZER_NEW_KEY=example_value" in text
        assert any("CRYSTALLIZER_NEW_KEY" in c for c in result.updated)
        assert not result.diverged
