"""The prompt Orion writes a reading-finding message from: the evidence, and nothing to copy.

Same contract as the curiosity Door-A composition prompt (orion/curiosity/outreach_prompt.py):
the evidence goes in, Orion's own message comes out, and ``PASS`` means "not worth sending"
(the prompt asks for the whole reply to be that one word; ``endogenous_outreach.is_pass_response``
also takes a line holding only PASS after Orion's reasoning, which is how most declines arrived
live through 2026-10-28 and were delivered to Juniper as messages). No template
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


MAX_WHY_CHARS = 200


def _ref(step: ReadingOriginStepV1) -> str:
    return step.url or step.title or "an untitled source"


def _note(step: ReadingOriginStepV1) -> str:
    """The requester's own note. Juniper writes it only on an operator request; a chat-turn or
    curiosity request is Orion's own tool call (orion/world_pulse_read/tools.py), so it is Orion's."""
    if not step.why_now:
        return ""
    why = step.why_now if len(step.why_now) <= MAX_WHY_CHARS else step.why_now[:MAX_WHY_CHARS].rstrip() + "…"
    whose = "her" if step.invocation_context == "operator" else "your"
    return f' ({whose} note: "{why}")'


def _start_line(step: ReadingOriginStepV1) -> str:
    """How the first read of a chain entered the queue, from its own row."""
    if step.kind in ("digest_item", "finding") or step.requested_by == "world_pulse":
        return f"Your daily world-pulse news run put {_ref(step)} in your reading queue."
    if step.invocation_context == "operator":
        return f"Juniper asked you to read {_ref(step)}{_note(step)}."
    if step.invocation_context == "unified_chat":
        # requested_by is "juniper" for every chat-turn tool call, including Orion's own.
        return f"During a chat with Juniper, you queued {_ref(step)} to read{_note(step)}."
    if step.invocation_context == "curiosity":
        return f"A curiosity run of yours queued {_ref(step)} to read{_note(step)}."
    return f"{_ref(step)} entered your reading queue; the record does not say who asked."


def _hop_line(step: ReadingOriginStepV1) -> str:
    return f"After that read, you chose to read {_ref(step)} next (a link you came across while working on it)."


def origin_lines(origin: list[ReadingOriginStepV1]) -> list[str]:
    """The chain as plain numbered lines, oldest first; the last one is this read. Pure.

    A later step exists only because the read before it listed the URL (``need_stage1_urls``)
    and the URL was one that read's work had seen (``_hop_allowlist``: its own text, a page its
    Stage 2 fetched, or a URL a fetch reached), so "you chose" is the record. A first step that
    still has a parent is a chain whose start is missing (or past MAX_ORIGIN_DEPTH): it is a hop,
    and its inherited requester fields say nothing about how it began."""
    if not origin:
        return ["How this read started was not recorded."]
    if origin[0].parent_request_id:
        steps = ["This continues a chain from an earlier read whose record is missing."]
        steps += [_hop_line(s) for s in origin]
    else:
        steps = [_start_line(origin[0])] + [_hop_line(s) for s in origin[1:]]
    lines = [f"{n}. {text}" for n, text in enumerate(steps, start=1)]
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
