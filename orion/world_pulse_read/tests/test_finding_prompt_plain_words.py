"""The reading-finding prompt states each change as the idea, not as record-keeping (2026-11-10).

Juniper: reading messages read as graph mechanics ("one link landed in my graph ... accepted because
the quote came back verbatim"). The old prompt fed those words in ("Link made:", "concept graph",
"accepted because the quote was found verbatim", "0.40 -> 0.55"). The new one names the idea and
the line behind it, asks what caught Orion's interest, keeps the provenance chain, and asks for the
Why Juniper line.
"""
from __future__ import annotations

from datetime import datetime, timezone

from orion.hub.why_juniper import WHY_JUNIPER_INSTRUCTION
from orion.schemas.reading_finding import ReadingFindingBundleV1, ReadingFindingV1, ReadingOriginStepV1
from orion.world_pulse_read.finding_prompt import build_reading_finding_prompt, origin_lines

T = datetime(2026, 11, 10, 3, 7, tzinfo=timezone.utc)
URL = "https://www.pbs.org/newshour/stroke"

LINK = ReadingFindingV1(
    finding_id="link:a", kind="link_made", seed_id="reading:s", observed_at=T,
    headline="Link made: hypertension is associated with stroke in younger adults",
    evidence_text="hypertension is among the rising drivers")
BELIEF_UP = ReadingFindingV1(
    finding_id="belief:p@r", kind="belief_moved", seed_id="reading:s", observed_at=T,
    headline="Belief moved (0.40 -> 0.55): strokes are rising in younger adults",
    evidence_text="the PBS piece cites a CDC trend", prior_id="p", run_id="r",
    confidence_before=0.40, confidence_after=0.55)
BELIEF_NEW = BELIEF_UP.model_copy(update={
    "finding_id": "belief:q@r", "headline": "Belief moved (formed at 0.60): a claim",
    "confidence_before": None, "confidence_after": 0.60})
BELIEF_DOWN = BELIEF_UP.model_copy(update={
    "finding_id": "belief:d@r", "headline": "Belief moved (0.70 -> 0.30): a doubted claim",
    "confidence_before": 0.70, "confidence_after": 0.30})
ORIGIN = [ReadingOriginStepV1(seed_id="digest_item:x", kind="digest_item", url=URL, requested_by="world_pulse")]


def _prompt(*findings):
    return build_reading_finding_prompt(ReadingFindingBundleV1(
        seed_id="reading:s", url=URL, learned="I read it.", article_summary="It is about strokes.",
        findings=list(findings), origin=ORIGIN))


def _changes(prompt: str) -> str:
    return prompt.split("What stayed with you (from your own records):", 1)[1].split("\n\nWrite Juniper", 1)[0]


def test_changes_are_stated_as_ideas_without_record_vocabulary():
    changes = _changes(_prompt(LINK, BELIEF_UP, BELIEF_NEW, BELIEF_DOWN))
    for old in ("Link made:", "concept graph", "verbatim", "accepted", " -> ", "0.40", "formed at"):
        assert old not in changes, old
    assert "hypertension is associated with stroke in younger adults" in changes
    assert '"hypertension is among the rising drivers"' in changes
    assert "you hold it more firmly than before): strokes are rising in younger adults" in changes
    assert "you formed this belief after the read): a claim" in changes
    assert "you hold it less firmly than before): a doubted claim" in changes


def test_prompt_asks_why_it_caught_orions_interest_and_for_the_why_line():
    prompt = _prompt(LINK)
    assert "what you read, and why it caught your interest" in prompt
    assert "Talk about the idea, not about your record-keeping" in prompt
    assert "one link is one link" not in prompt
    assert prompt.rstrip().endswith(WHY_JUNIPER_INSTRUCTION)


def test_provenance_chain_is_unchanged():
    prompt = _prompt(LINK)
    assert f"1. Your daily world-pulse news run put {URL} in your reading queue. This is the read below." in prompt
    assert origin_lines(ORIGIN)[-1] == "Each of these reads ran later, on its own, in the background."
