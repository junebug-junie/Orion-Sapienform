from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

_SERVICE = Path(__file__).resolve().parents[1]
_GEN = _SERVICE / "scripts" / "generate_descriptions.py"
_CARDS = _SERVICE / "cards" / "town_cards.yaml"

_spec = importlib.util.spec_from_file_location("gen_descriptions", _GEN)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def _cards():
    return yaml.safe_load(_CARDS.read_text(encoding="utf-8"))


def test_cards_have_all_expected_ids():
    ids = {c["id"] for c in _cards()["characters"]}
    assert set(gen.NPC_ORDER).issubset(ids)
    assert {"juniper_feld", "orion"}.issubset(ids)


def test_compose_identity_is_rich_and_collapsed():
    by = {c["id"]: c for c in _cards()["characters"]}
    ident = gen.compose_identity(by["mara_vale"])
    assert len(ident) > 400  # rich, not a one-liner
    assert "\n" not in ident  # whitespace collapsed
    assert "description of your logs" in ident  # signature line folded in


def test_compose_presence_blurb_orion_uses_they():
    by = {c["id"]: c for c in _cards()["characters"]}
    blurb = gen.compose_presence_blurb(by["orion"])
    assert "synthetic mind" in blurb
    assert "A line they often use:" in blurb


def test_render_descriptions_emits_eight_valid_sprites():
    ts = gen.render_descriptions(_cards())
    assert ts.count("    character: '") == len(gen.NPC_ORDER)
    for cid, sprite in _cards()["sprites"].items():
        if cid in gen.NPC_ORDER:
            assert f"character: '{sprite}'" in ts


def test_ts_escaping_neutralizes_template_literal_injection():
    """Backtick / ${...} / backslash must be escaped for a TS backtick literal."""
    out = gen._ts_backtick("back`tick and ${interp} and \\ slash")
    assert "`" not in out.replace("\\`", "")  # every backtick is escaped
    assert "${" not in out.replace("\\${", "")  # every interpolation is escaped
    assert "\\\\ slash" in out  # backslash doubled


def test_ts_single_escaping():
    assert gen._ts_single("O'Brien") == "'O\\'Brien'"


def test_signature_intro_pronoun_agreement():
    assert gen._signature_intro("she/her") == "she often uses"
    assert gen._signature_intro("he/him") == "he often uses"
    assert gen._signature_intro("they/them") == "they often use"
    # Compound pronouns take the first token; unknown/empty default to they.
    assert gen._signature_intro("he/they") == "he often uses"
    assert gen._signature_intro("they/she") == "they often use"
    assert gen._signature_intro(None) == "they often use"


def test_juniper_blurb_present_in_world_ts():
    """Drift guard: the composed Juniper blurb must be spliced into world.ts."""
    world = (gen.WORLD_TS).read_text(encoding="utf-8")
    by = {c["id"]: c for c in _cards()["characters"]}
    blurb = gen.compose_presence_blurb(by["juniper_feld"])
    assert blurb in world
    assert f"DEFAULT_NAME = '{by['juniper_feld']['name']}'" in (
        gen.CONSTANTS_TS
    ).read_text(encoding="utf-8")
