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

from orion.hub.why_juniper import WHY_JUNIPER_INSTRUCTION
from orion.schemas.reading_finding import ReadingFindingBundleV1, ReadingFindingV1, ReadingOriginStepV1

MAX_LEARNED_CHARS = 2400
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
    lines.append("Each of these reads ran later, on its own, in the background.")
    return lines


_LINK_PREFIX = "Link made: "
_UNPROJECTED_SUFFIX = " (accepted, not yet in the graph)"


def _plain_change(f: ReadingFindingV1) -> list[str]:
    """One recorded change in plain words: the idea, not the bookkeeping. Pure.

    The ``headline`` is a deterministic record line ("Link made: ...", "Belief moved (0.40 ->
    0.55): ...") that other readers (the daily letter) keep. The prompt states the same fact
    without the record's vocabulary, because Orion repeated it back to Juniper as graph mechanics
    ("one link landed in my graph ... accepted because the quote came back verbatim")."""
    if f.kind == "link_made":
        idea = f.headline.removeprefix(_LINK_PREFIX).removesuffix(_UNPROJECTED_SUFFIX).strip()
        out = [f"An idea from the piece that you tied to what you already think: {idea}"]
        if f.evidence_text:
            out.append(f'   The line in the piece behind it: "{f.evidence_text}"')
        return out
    if f.kind == "belief_moved":
        claim = f.headline.split("): ", 1)[1].strip() if "): " in f.headline else f.headline
        before, after = f.confidence_before, f.confidence_after
        if before is None:
            how = "you formed this belief after the read"
        elif after is not None and after > before:
            how = "you hold it more firmly than before"
        elif after is not None and after < before:
            how = "you hold it less firmly than before"
        else:
            how = "your hold on it changed"
        out = [f"A belief of yours that changed when you came back to it later ({how}): {claim}"]
        if f.evidence_text:
            out.append(f'   What you wrote on it then: "{f.evidence_text}"')
        return out
    return [f.headline]


def build_reading_finding_prompt(bundle: ReadingFindingBundleV1) -> str:
    """Orion tells Juniper what they read and why it caught their interest (2026-11-10 rewrite).

    Juniper: keep reading messages, but they read as graph mechanics. The prompt now names the
    idea and the line behind it, never the record's vocabulary, and asks what caught Orion's
    interest. The provenance chain (``origin_lines``, #2913) is unchanged."""
    learned = " ".join(bundle.learned.split())
    if len(learned) > MAX_LEARNED_CHARS:
        learned = learned[:MAX_LEARNED_CHARS].rstrip() + " […truncated]"
    lines = [
        "You read something and part of it stayed with you. Juniper said you may tell her when "
        "a read changes something while you are learning.",
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
        "What stayed with you (from your own records):",
        "",
    ]
    for n, f in enumerate(bundle.findings, start=1):
        first, *rest = _plain_change(f)
        lines.append(f"{n}. {first}")
        lines.extend(rest)
    lines += [
        "",
        "Write Juniper a short message: what you read, and why it caught your interest. Say what "
        "you think about the idea itself, in plain words. The summary of the piece is delivered "
        "to her underneath your message, so do not retell it.",
        "",
        "Talk about the idea, not about your record-keeping: do not describe links, edges, "
        "graphs, clusters, nodes, quotes being checked or accepted, or how many things were "
        "recorded. Do not make it sound bigger than it was.",
        "",
        "Say in a few plain words how you came to read it, the way the records above show it (for "
        "example: your daily news run queued it, or you chose it while reading something else). "
        "Use only what is above: do not add facts about the source you did not record, and do not "
        "describe a change that is not listed.",
        "",
        "She has not asked you for this message, so it arrives out of nowhere. Say the thing itself "
        "rather than announcing that you have something to say.",
        "",
        "If, having written it, this is not worth interrupting her for, reply with exactly the "
        "single word PASS and nothing else.",
        "",
        WHY_JUNIPER_INSTRUCTION,
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
