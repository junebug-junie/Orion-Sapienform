from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

_SERVICE = Path(__file__).resolve().parents[1]
_GEN = _SERVICE / "scripts" / "generate_descriptions.py"
_CARDS = _SERVICE / "cards" / "town_cards.yaml"

_spec = importlib.util.spec_from_file_location("gen_descriptions", _GEN)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

_BANNED_BAIT = ("lighting", "glow", "shadows", "echoes")


def _cards():
    return yaml.safe_load(_CARDS.read_text(encoding="utf-8"))


def test_cards_have_all_expected_ids():
    ids = {c["id"] for c in _cards()["characters"]}
    assert set(gen.NPC_ORDER).issubset(ids)
    assert {"juniper_feld", "orion"}.issubset(ids)


def test_npc_order_is_the_live_four():
    assert gen.NPC_ORDER == ["mara_vale", "nico_sable", "sofia_bell", "cam_lin"]


def test_compose_identity_uses_job_fields_not_signature():
    by = {c["id"]: c for c in _cards()["characters"]}
    ident = gen.compose_identity(by["mara_vale"])
    assert "\n" not in ident
    assert "systems cartographer" in ident.lower()
    assert "Today:" in ident
    assert "diagrams" in ident.lower() or "maps" in ident.lower()
    assert "description of your logs" not in ident.lower()
    for bait in _BANNED_BAIT:
        assert bait not in ident.lower()


def test_live_identities_have_no_light_bait():
    by = {c["id"]: c for c in _cards()["characters"]}
    for cid in gen.NPC_ORDER:
        ident = gen.compose_identity(by[cid]).lower()
        for bait in _BANNED_BAIT:
            assert bait not in ident, f"{cid} identity contains {bait}"


def test_plans_come_from_daily_loop():
    cards = _cards()
    by = {c["id"]: c for c in cards["characters"]}
    for cid in gen.NPC_ORDER:
        loop0 = " ".join(by[cid]["daily_loop"][0].split()).lower()
        # plan is second person; must share a concrete noun/verb from daily_loop[0]
        plan = cards["plans"][cid].lower()
        assert plan.startswith("you ")
        assert any(token in plan for token in loop0.split() if len(token) > 4)


def test_render_descriptions_emits_four_valid_sprites():
    ts = gen.render_descriptions(_cards())
    assert ts.count("    character: '") == 4
    for cid in gen.NPC_ORDER:
        assert f"character: '{_cards()['sprites'][cid]}'" in ts
    for dead in ("Juno Park", "Tessa Quinn", "Vale Moreno", "Dr. Elian Cross", "Elian Cross"):
        assert dead not in ts


def test_orion_blurb_does_not_name_retired_cast():
    by = {c["id"]: c for c in _cards()["characters"]}
    blurb = gen.compose_presence_blurb(by["orion"])
    for dead in ("Elian", "Juno", "Tessa", "Vale"):
        assert dead not in blurb


def test_archived_retired_cast_exists():
    archived = _SERVICE / "cards" / "archived" / "2026-08-29-retired-cast.yaml"
    text = archived.read_text(encoding="utf-8")
    for name in ("Juno Park", "Tessa Quinn", "Vale Moreno", "Dr. Elian Cross"):
        assert name in text


def test_compose_presence_blurb_orion_uses_they():
    by = {c["id"]: c for c in _cards()["characters"]}
    blurb = gen.compose_presence_blurb(by["orion"])
    assert "synthetic mind" in blurb
    assert "A line they often use:" in blurb


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


@pytest.mark.skipif(not gen.WORLD_TS.exists(), reason="upstream/world.ts not cloned")
def test_juniper_blurb_present_in_world_ts():
    """Drift guard: the composed Juniper blurb must be spliced into world.ts."""
    world = (gen.WORLD_TS).read_text(encoding="utf-8")
    by = {c["id"]: c for c in _cards()["characters"]}
    blurb = gen.compose_presence_blurb(by["juniper_feld"])
    assert blurb in world
    assert f"DEFAULT_NAME = '{by['juniper_feld']['name']}'" in (
        gen.CONSTANTS_TS
    ).read_text(encoding="utf-8")
