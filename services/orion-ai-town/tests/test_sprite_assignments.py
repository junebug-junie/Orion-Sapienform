"""Girl names get girl sheets. Boy names get boy sheets. Orion keeps f1.

Juniper's join used to pick a random f1-f8 every session. Nico (he/him) sat on
the short ginger tank. Cards are the source of truth; the tracked patches must
match.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

_SERVICE = Path(__file__).resolve().parents[1]
_GEN = _SERVICE / "scripts" / "generate_descriptions.py"
_CARDS = _SERVICE / "cards" / "town_cards.yaml"

_spec = importlib.util.spec_from_file_location("gen_descriptions_sprites", _GEN)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def _cards():
    return yaml.safe_load(_CARDS.read_text(encoding="utf-8"))
_HUMAN = _SERVICE / "patches" / "orion-human-juniper.patch"
_CHARACTER = _SERVICE / "patches" / "orion-character.patch"

# Presentation locked 2026-08-29 against 32x32folk.png down-facing frames.
# f6 = long magenta-red hair (female redhead). f2 = buzz cut / teal (boy).
# f1 = short grey / blue shirt (Orion's current body).
EXPECTED = {
    "juniper_feld": "f6",
    "mara_vale": "f3",
    "sofia_bell": "f8",
    "nico_sable": "f2",
    "cam_lin": "f1",
    "orion": "f1",
}


def test_card_sprites_match_presentation():
    sprites = _cards()["sprites"]
    assert sprites == EXPECTED


def test_human_juniper_patch_pins_female_redhead():
    patch = _HUMAN.read_text(encoding="utf-8")
    assert "+      character: 'f6'," in patch
    assert "-      character: characters[Math.floor(Math.random() * characters.length)].name," in patch
    for line in patch.splitlines():
        if "Math.random()" in line:
            assert line.startswith("-")
        if "character: 'f6'" in line:
            assert line.startswith("+")


def test_character_patch_gives_nico_a_boy_sheet():
    patch = _CHARACTER.read_text(encoding="utf-8")
    assert "+    name: 'Nico Sable'," in patch
    # The added Nico block must be f2, not the old f7 ginger tank.
    added = "\n".join(
        line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    idx = added.index("name: 'Nico Sable'")
    nico_block = added[idx : idx + 200]
    assert "character: 'f2'" in nico_block
    assert "character: 'f7'" not in nico_block


def test_character_patch_keeps_girl_and_orion_sheets():
    patch = _CHARACTER.read_text(encoding="utf-8")
    added = "\n".join(
        line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    for name, sheet in (
        ("Mara Vale", "f3"),
        ("Sofia Bell", "f8"),
        ("Cam Lin", "f1"),
    ):
        idx = added.index(f"name: '{name}'")
        block = added[idx : idx + 200]
        assert f"character: '{sheet}'" in block, f"{name} should be {sheet}"


def test_patch_juniper_pins_sprite_from_cards(tmp_path, monkeypatch):
    constants = tmp_path / "constants.ts"
    world = tmp_path / "world.ts"
    constants.write_text("export const DEFAULT_NAME = 'Me';\n", encoding="utf-8")
    world.write_text(
        "    return await insertInput(ctx, world._id, 'join', {\n"
        "      name,\n"
        "      character: characters[Math.floor(Math.random() * characters.length)].name,\n"
        "      description: `old`,\n"
        "    });\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gen, "CONSTANTS_TS", constants)
    monkeypatch.setattr(gen, "WORLD_TS", world)

    cards = _cards()
    changed = gen.patch_juniper(cards)
    assert "convex/world.ts" in changed
    out = world.read_text(encoding="utf-8")
    assert "character: 'f6'," in out
    assert "Math.random()" not in out
