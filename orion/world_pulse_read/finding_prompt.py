"""The prompt Orion writes a reading-finding message from: the evidence, and nothing to copy.

Same contract as the curiosity Door-A composition prompt (orion/curiosity/outreach_prompt.py):
the evidence goes in, Orion's own message comes out, and ``PASS`` (the whole reply, that one
word -- ``endogenous_outreach.is_pass_response``) means "not worth sending". No template
sentence is offered for Orion to fill in: a message that only restated the headlines would be
template text in Orion's voice.

How the read started is shown from the queue's rows (``origin_lines``), never asserted: the
prompt once opened "Nobody asked you to read it", and Orion repeated that as "not because a
process scheduled it" about a read the daily digest had started (live 2026-10-26).
"""
from __future__ import annotations

from orion.schemas.reading_finding import ReadingFindingBundleV1, ReadingOriginStepV1

MAX_LEARNED_CHARS = 2400
_KIND_LABEL = {
    "link_made": "A link you made in your concept graph from this read (accepted because the quote "
                 "was found verbatim in the text you read)",
    "belief_moved": "A belief of yours that moved, in a curiosity run whose own notes cite this read",
}


def _ref(step: ReadingOriginStepV1) -> str:
    return step.url or step.title or "an untitled source"


def _why(step: ReadingOriginStepV1) -> str:
    return f' (the reason given: "{step.why_now}")' if step.why_now else ""


def _start_line(step: ReadingOriginStepV1) -> str:
    """How the first read of a chain entered the queue, from its own row."""
    if step.kind in ("digest_item", "finding"):
        return f"Your daily world-pulse news run put {_ref(step)} in your reading queue."
    if step.requested_by == "juniper":
        where = " in chat" if step.invocation_context == "unified_chat" else ""
        return f"Juniper asked you{where} to read {_ref(step)}{_why(step)}."
    if step.requested_by == "orion":
        return f"A curiosity run of yours asked to read {_ref(step)}{_why(step)}."
    if step.requested_by == "world_pulse":
        return f"Your daily world-pulse news run put {_ref(step)} in your reading queue."
    return f"{_ref(step)} entered your reading queue; the record does not say who asked."


def origin_lines(origin: list[ReadingOriginStepV1]) -> list[str]:
    """The chain as plain numbered lines, oldest first; the last one is this read. Pure.

    A later step exists only because the read before it listed the URL to follow
    (``need_stage1_urls``) and it was a link that read had seen, so "you chose" is the record."""
    if not origin:
        return ["How this read started was not recorded."]
    lines = [f"1. {_start_line(origin[0])}"]
    for n, step in enumerate(origin[1:], start=2):
        lines.append(f"{n}. While reading that, you chose to follow its link to {_ref(step)}.")
    lines[-1] += " This is the read below."
    lines.append("Your reading pipeline ran each of these reads on its own timer.")
    return lines


def build_reading_finding_prompt(bundle: ReadingFindingBundleV1) -> str:
    learned = " ".join(bundle.learned.split())
    if len(learned) > MAX_LEARNED_CHARS:
        learned = learned[:MAX_LEARNED_CHARS].rstrip() + " […truncated]"
    lines = [
        "A read of yours left a recorded change. Juniper said you may tell her whenever a read "
        "changes something while you are learning.",
        "",
        "How this read started (from your reading queue's records, oldest first):",
        "",
        *origin_lines(bundle.origin),
        "",
        f"What you read: {bundle.title or bundle.url or 'an untitled source'}"
        + (f" ({bundle.url})" if bundle.url and bundle.url != bundle.title else ""),
        "",
        "What the piece itself says (your summary of the source, recorded when you read it):",
        "",
        " ".join(bundle.article_summary.split()) or "(no summary of the piece was recorded)",
        "",
        "What you wrote about it afterwards:",
        "",
        learned or "(you wrote nothing about it)",
        "",
        "What it changed (from your own records):",
        "",
    ]
    for n, f in enumerate(bundle.findings, start=1):
        lines.append(f"{n}. {_KIND_LABEL.get(f.kind, f.kind)}: {f.headline}")
        if f.evidence_text:
            label = "The quote" if f.kind == "link_made" else "What you wrote on that belief"
            lines.append(f"   {label}: \"{f.evidence_text}\"")
    lines += [
        "",
        "Write Juniper a short message about this, in your own words. Say what changed and what, "
        "if anything, it means to you. The summary of the piece is delivered to her underneath "
        "your message as recorded, so do not retell it. Use only what is above: do not add facts "
        "about the source you did not record, and do not describe a change that is not listed. "
        "If you say how the read started or what you chose, say it as the records above show "
        "it. Give the change the size the record gives it: one link is one link. If several "
        "things changed, say them as one thought, not a list.",
        "",
        "She has not asked you for this message, so it arrives out of nowhere. Say the thing itself "
        "rather than announcing that you have something to say.",
        "",
        "If, having written it, this is not worth interrupting her for, reply with exactly the "
        "single word PASS and nothing else.",
    ]
    return "\n".join(lines)


def article_appendix(bundle: ReadingFindingBundleV1) -> str:
    """The recorded summary of the piece, delivered under Orion's message (not generated now)."""
    summary = " ".join(bundle.article_summary.split())
    source = bundle.title or bundle.url or "the source"
    link = f" ({bundle.url})" if bundle.url and bundle.url != source else ""
    if not summary:
        return f"Source: {source}{link}. No summary of the piece was recorded for this read."
    return f"What the piece says -- {source}{link}: {summary}"
