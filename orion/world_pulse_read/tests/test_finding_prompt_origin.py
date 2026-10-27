"""The reading-finding prompt states how a read started from the queue's rows, not by assertion."""
from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.reading_finding import ReadingFindingBundleV1, ReadingFindingV1, ReadingOriginStepV1
from orion.world_pulse_read.finding_prompt import build_reading_finding_prompt, origin_lines
from orion.world_pulse_read.origin import origins_from_rows

T = datetime(2026, 10, 26, 2, 8, tzinfo=timezone.utc)
N1X = "https://www.servethehome.com/nvidia-rtx-spark-n1x-launch-overview-nvidia-arms-windows/"
GB10 = "https://www.servethehome.com/nvidia-outlines-gb10-soc-architecture-at-hot-chips-2025/"
DGX = "https://www.servethehome.com/the-nvidia-dgx-spark-is-a-tiny-128gb-ai-mini-pc-made-for-scale-out-clustering-arm/"


def _row(leaf, depth, seed_id, kind, url, requested_by="world_pulse", ctx="world_pulse", why="", parent=None):
    return {"leaf": leaf, "depth": depth, "seed_id": seed_id, "kind": kind, "url": url, "title": "",
            "requested_by": requested_by, "invocation_context": ctx, "why_now": why, "parent_request_id": parent}


# The live chain behind the 2026-10-26 DGX Spark message, as ORIGIN_SQL returns it (depth DESC).
LIVE_ROWS = [
    _row("reading:dgx", 2, "digest_item:n1x", "digest_item", N1X, requested_by=None, ctx=None),
    _row("reading:dgx", 1, "reading:gb10", "reading", GB10, parent="req-n1x"),
    _row("reading:dgx", 0, "reading:dgx", "reading", DGX, parent="req-gb10"),
]


def test_live_chain_reads_oldest_first_and_names_each_choice():
    origin = origins_from_rows(LIVE_ROWS)["reading:dgx"]
    assert [s.seed_id for s in origin] == ["digest_item:n1x", "reading:gb10", "reading:dgx"]
    assert origin_lines(origin) == [
        f"1. Your daily world-pulse news run put {N1X} in your reading queue.",
        f"2. After that read, you chose to read {GB10} next (a link you came across while working on it).",
        f"3. After that read, you chose to read {DGX} next (a link you came across while working on it). "
        "This is the read below.",
        "Your reading pipeline ran each of these reads on its own timer.",
    ]


def test_starts_name_who_queued_it_and_whose_note_it_is():
    # A chat-turn request is Orion's own recommend_reading call (requested_by is "juniper" for
    # every one of them), so it is never rendered as Juniper asking.
    chat = ReadingOriginStepV1(seed_id="r", kind="reading", url="u", requested_by="juniper",
                               invocation_context="unified_chat", why_now="Orca paper")
    assert origin_lines([chat])[0] == ('1. During a chat with Juniper, you queued u to read (your note: "Orca paper"). '
                                       "This is the read below.")
    op = chat.model_copy(update={"invocation_context": "operator", "why_now": "read this"})
    assert origin_lines([op])[0] == '1. Juniper asked you to read u (her note: "read this"). This is the read below.'
    c = ReadingOriginStepV1(seed_id="r", kind="reading", url="u", requested_by="orion", invocation_context="curiosity")
    assert origin_lines([c])[0] == "1. A curiosity run of yours queued u to read. This is the read below."
    legacy = ReadingOriginStepV1(seed_id="r", kind="reading", url="u")
    assert "the record does not say who asked" in origin_lines([legacy])[0]
    assert origin_lines([]) == ["How this read started was not recorded."]
    long = chat.model_copy(update={"why_now": "x" * 4000})
    assert len(origin_lines([long])[0]) < 320


def test_a_chain_whose_start_is_missing_is_not_rendered_as_a_request():
    # Live 2026-10-27: two rows name a parent request that is not in the table. A hop inherits its
    # parent's requester and note, so rendering it as a start would invent who asked.
    orphan = ReadingOriginStepV1(seed_id="reading:o", kind="reading", url="u", requested_by="juniper",
                                 invocation_context="unified_chat", why_now="parent's note", parent_request_id="gone")
    lines = origin_lines([orphan])
    assert lines[0] == "1. This continues a chain from an earlier read whose record is missing."
    assert lines[1].startswith("2. After that read, you chose to read u next") and "parent's note" not in " ".join(lines)


def test_prompt_carries_the_chain_and_no_longer_asserts_self_direction():
    bundle = ReadingFindingBundleV1(
        seed_id="reading:dgx", url=DGX, learned="I read it.", article_summary="It is a box.",
        findings=[ReadingFindingV1(finding_id="link:a", kind="link_made", seed_id="reading:dgx", observed_at=T,
                                   headline="Link made: GB10 is associated with GB10 (Blackwell Arm).")],
        origin=origins_from_rows(LIVE_ROWS)["reading:dgx"],
    )
    prompt = build_reading_finding_prompt(bundle)
    assert "Nobody asked you to read it" not in prompt and "it changed you" not in prompt
    assert "why that change matters to you" not in prompt
    assert f"2. After that read, you chose to read {GB10} next" in prompt
    assert "one link is one link" in prompt
    assert prompt.rstrip().endswith("reply with exactly the single word PASS and nothing else.")
