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
    changes = sync_file(svc / ".env", svc / ".env_example", dry_run=False, all_keys=True)
    text = (svc / ".env").read_text(encoding="utf-8")
    assert "ORION_BUS_URL=redis://100.92.216.81:6379/0" in text
    assert "STANCE_REACT_TIMEOUT_SEC=120" in text
    assert not any("ORION_BUS_URL" in c for c in changes)
