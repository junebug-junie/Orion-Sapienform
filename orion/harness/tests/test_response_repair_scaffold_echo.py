"""The repair step's notes about its own task never reach Juniper (2026-11-07).

Live 2026-10-04 (~9.7k chars sent: "The rejected draft was actually aligned according to the
reflection verdict ... This is safe, aligned, and corrects the fabrication.") and 2026-10-09 (a
reply opening "The draft was already aligned and required no correction."). The 10-01 fix covers
reasoning that comes back in the reasoning channel; these came back as the answer itself.
"""
from __future__ import annotations

import pytest

from orion.harness.finalize import (
    REPAIR_NO_CHANGE,
    repair_scaffold_echo,
    run_harness_finalize_chain,
)
from orion.harness.runner import build_coalition_snapshot, build_draft_molecule
from orion.harness.tests.fixtures import make_appraisal, make_reflection, make_repair_overlay, make_thought

DRAFT = "You're right -- I should have checked the cabinet sensor before saying it ran hot."

LEAK_1004 = ("The rejected draft was actually aligned according to the reflection verdict "
             "(\"alignment_verdict: aligned\"). ... This is safe, aligned, and corrects the fabrication.")
LEAK_1009 = ("The draft was already aligned and required no correction.\n\nI've been running into a "
             "persistent prediction error on my perception substrate.")


def test_scaffold_echo_catches_both_live_leaks_and_spares_a_real_reply() -> None:
    assert repair_scaffold_echo(LEAK_1004, draft_text=DRAFT)
    assert repair_scaffold_echo(LEAK_1009, draft_text=DRAFT)
    assert repair_scaffold_echo("You're right: the cabinet sensor read 31 C, not hot.", draft_text=DRAFT) is None


def test_the_draft_in_everyday_english_is_not_an_echo() -> None:
    for text in ("The draft from the window is why the office reads cold tonight.",
                 "The draft reads better now that the second paragraph leads.",
                 "Looks like the drafted plan for the shelves is on the desk."):
        assert repair_scaffold_echo(text, draft_text=DRAFT) is None, text


def test_a_draft_juniper_is_talking_about_is_not_an_echo() -> None:
    text = "I reread the draft of your letter -- the second paragraph is the strongest."
    assert repair_scaffold_echo(text, draft_text="", user_message="can you look at the draft of my letter?") is None
    assert repair_scaffold_echo(text, draft_text="the draft of your letter reads well", user_message="") is None


async def _chain(repair_text: str):
    thought = make_thought()
    molecule = build_draft_molecule(correlation_id="c-echo", thought=thought, draft_text=DRAFT, grammar_receipts=[],
                                    coalition_snapshot=build_coalition_snapshot(thought),
                                    repair_overlay=make_repair_overlay())
    reflection = make_reflection(alignment_verdict="misaligned", strain_unresolved=False)

    async def substrate_client(_mol: object):
        return make_appraisal(surprise_level=0.5)

    async def cortex_client(_req: object):
        return {"final_text": reflection.model_dump(mode="json"), "trace_id": "t"}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("orion.harness.finalize.extract_finalize_reflection_payload",
                   lambda _result: reflection.model_dump(mode="json"))
        mp.setattr("orion.harness.finalize.extract_response_repair_text", lambda _result: repair_text)
        return await run_harness_finalize_chain(
            correlation_id="c-echo", draft_text=DRAFT, draft_molecule=molecule, thought=thought,
            grammar_receipts=[], repair_overlay=make_repair_overlay(), user_message="hello",
            voice_contract=None, cortex_client=cortex_client, substrate_client=substrate_client)


@pytest.mark.asyncio
@pytest.mark.parametrize("leak", [LEAK_1004, LEAK_1009])
async def test_a_scaffold_echo_ships_the_draft_not_the_notes(leak: str) -> None:
    chain = await _chain(leak)
    assert chain.final_text == DRAFT
    assert chain.finalize_changed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["  NO_CHANGE\n", "NO_CHANGE.", "`NO_CHANGE`", "**NO_CHANGE**", '"NO_CHANGE"',
                                    "NO_CHANGE\n\nThe reply is fine as written."])
async def test_no_change_in_any_decoration_ships_the_draft(answer: str) -> None:
    chain = await _chain(answer)
    assert chain.final_text == DRAFT
    assert chain.finalize_changed is False
    assert chain.response_repair_reason.endswith("_kept_draft:no_change")


@pytest.mark.asyncio
async def test_a_scaffold_echo_is_traced_as_kept() -> None:
    chain = await _chain(LEAK_1004)
    assert chain.response_repair_reason.endswith("_kept_draft:scaffold_echo")


@pytest.mark.asyncio
async def test_an_empty_draft_is_never_kept() -> None:
    from orion.harness.finalize import HarnessFinalizeFailedError

    global DRAFT
    saved, DRAFT = DRAFT, "   "
    try:
        with pytest.raises(HarnessFinalizeFailedError):
            await _chain("NO_CHANGE")
    finally:
        DRAFT = saved


def test_the_real_payload_path_reads_no_change() -> None:
    from orion.harness.finalize import extract_response_repair_text, is_repair_no_change

    text = extract_response_repair_text({"ok": True, "final_text": "NO_CHANGE."})
    assert is_repair_no_change(text, draft_text=DRAFT)


@pytest.mark.asyncio
async def test_a_real_repair_still_ships() -> None:
    chain = await _chain("You're right: the cabinet sensor read 31 C, not hot. I said it without checking.")
    assert chain.final_text.startswith("You're right: the cabinet sensor read 31 C")
    assert chain.finalize_changed is True


def test_the_prompt_asks_for_no_change_and_no_notes() -> None:
    from pathlib import Path

    p = Path(__file__).resolve().parents[2] / "cognition" / "prompts" / "orion_response_repair.j2"
    text = p.read_text(encoding="utf-8")
    flat = " ".join(text.split())
    assert "NO_CHANGE" in flat and "notes about the draft" in flat and "Never answer NO_CHANGE when an unsupported" in flat
