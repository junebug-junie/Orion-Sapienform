"""The prompt Orion writes a reading-finding message from: the evidence, and nothing to copy.

Same contract as the curiosity Door-A composition prompt (orion/curiosity/outreach_prompt.py):
the evidence goes in, Orion's own message comes out, and ``PASS`` (the whole reply, that one
word -- ``endogenous_outreach.is_pass_response``) means "not worth sending". No template
sentence is offered for Orion to fill in: a message that only restated the headlines would be
template text in Orion's voice.
"""
from __future__ import annotations

from orion.schemas.reading_finding import ReadingFindingBundleV1

MAX_LEARNED_CHARS = 2400
_KIND_LABEL = {
    "link_made": "A link you made in your concept graph from this read (accepted because the quote "
                 "was found verbatim in the text you read)",
    "belief_moved": "A belief of yours that moved, in a curiosity run whose own notes cite this read",
}


def build_reading_finding_prompt(bundle: ReadingFindingBundleV1) -> str:
    learned = " ".join(bundle.learned.split())
    if len(learned) > MAX_LEARNED_CHARS:
        learned = learned[:MAX_LEARNED_CHARS].rstrip() + " […truncated]"
    lines = [
        "You read something on your own, and it changed you. Nobody asked you to read it. "
        "Juniper said you may tell her whenever a read changes something while you are learning.",
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
        "Write Juniper a short message about this, in your own words. Say what changed in you "
        "because of this read and why that change matters to you. The summary of the piece is "
        "delivered to her underneath your message as recorded, so do not retell it. Use only what is "
        "above: do not add facts about the source you did not record, and do not describe a "
        "change that is not listed. If several things changed, say them as one thought, not a "
        "list.",
        "",
        "She has not asked you anything, so this arrives out of nowhere. Say the thing itself "
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
