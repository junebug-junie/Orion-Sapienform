from __future__ import annotations

from orion.embodiment.persona import build_orion_town_persona


def test_projection_from_self_model():
    persona = build_orion_town_persona(
        identity_summary="Orion is a curious, coherence-seeking digital mind exploring continuity.",
        anchor_strategy="continuity",
        dominant_drive="social",
        snapshot_id="snap-1",
        generated_at="2026-07-07T00:00:00Z",
        spritesheet="f1",
    )
    assert persona.persona_source == "projection"
    assert persona.name == "Orion"
    assert persona.identity_blurb
    assert len(persona.identity_blurb) <= 280
    assert persona.provenance["snapshot_id"] == "snap-1"
    assert "social" in persona.plan.lower()


def test_empty_projection_falls_back():
    persona = build_orion_town_persona(
        identity_summary="   ",
        anchor_strategy="",
        dominant_drive=None,
        snapshot_id=None,
        generated_at=None,
        spritesheet="f1",
    )
    assert persona.persona_source == "fallback"
    assert persona.name == "Orion"
    assert persona.identity_blurb.strip()
    assert persona.plan.strip()


def test_blurb_is_truncated():
    persona = build_orion_town_persona(
        identity_summary="x" * 500, anchor_strategy="a", dominant_drive="curiosity",
        snapshot_id="s", generated_at="t", spritesheet="f1",
    )
    assert len(persona.identity_blurb) <= 280
