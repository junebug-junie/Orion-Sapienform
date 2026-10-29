from __future__ import annotations

from orion import self_inventory
from orion.dig.__main__ import _unfamiliar


def test_junk_folders_are_not_services(tmp_path):
    for name in ("orion-hub", "__pycache__", ".cache"):
        (tmp_path / "services" / name).mkdir(parents=True)
    assert [r["name"] for r in self_inventory.services(tmp_path)] == ["orion-hub"]


def test_a_checkout_older_than_the_layout_is_reported_as_different_not_empty(tmp_path):
    # live 2026-10-27: the 2025-08-01 repo was conjourney-api/ + emergence/, no services/ yet
    (tmp_path / "conjourney-api").mkdir()
    (tmp_path / "emergence").mkdir()
    (tmp_path / "README.md").write_text("x")
    inv = self_inventory.inventory(tmp_path)
    assert not any(inv.values())
    note = "\n".join(_unfamiliar(inv, tmp_path, "2025-08-01"))
    assert "NOT the same as nothing existing" in note
    assert "conjourney-api/  emergence/  README.md" in note


def test_a_familiar_checkout_gets_no_note(tmp_path):
    (tmp_path / "services" / "orion-hub").mkdir(parents=True)
    assert _unfamiliar(self_inventory.inventory(tmp_path), tmp_path, "now") == []
