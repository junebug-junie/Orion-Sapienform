"""Pass 2: runtime pack merge proves delivery_pack for instructional / code asks."""

from __future__ import annotations

from orion.cognition.runtime_pack_merge import DELIVERY_ORIENTED_OUTPUT_MODES, ensure_delivery_pack_in_packs


def test_discord_deploy_instruction_merges_delivery_pack():
    packs = ensure_delivery_pack_in_packs(
        ["executive_pack"],
        output_mode=None,
        user_text="Please provide instructions on how to deploy you onto Discord.",
    )
    assert "delivery_pack" in packs
    assert "executive_pack" in packs


def test_code_scaffold_merges_delivery_pack():
    packs = ensure_delivery_pack_in_packs(
        None,
        output_mode=None,
        user_text="Write the code scaffold for a Discord bot bridge.",
    )
    assert "delivery_pack" in packs


def test_output_mode_code_delivery_merges_without_text_classify():
    packs = ensure_delivery_pack_in_packs(["executive_pack"], output_mode="code_delivery", user_text="")
    assert "delivery_pack" in packs


def test_comparative_analysis_in_delivery_oriented_set():
    assert "comparative_analysis" in DELIVERY_ORIENTED_OUTPUT_MODES
