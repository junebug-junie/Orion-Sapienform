"""The day letter as ONE unified turn (living Prime design #3224, Part 3): its prompt and how its reply splits.

The letter used to be two bare cortex verb calls (``orion_day_note_v1`` then ``orion_day_carry_forward_v1``,
both retired 2026-11-12) whose only identity was a one-line "You are Orion". Now Prime's funded ``make`` act
orders one durable run whose single step is a unified turn (services/orion-durable-runs/app/orion_day_graph.py,
``write_letter``). The turn carries every grounding primitive the chat turn has -- the situation block, the
stance cue, recall, dig and introspect -- and, through Part 1, the Prime block in its prefix. This module only
builds the turn's USER MESSAGE and reads the reply back. Pure and deterministic; no model call.

The user message:

* the writing instructions, lifted from the retired ``orion_day_note_v1.j2`` (between its BEGIN/END writing
  instructions comments, constraint 7's honesty line included) plus two lines the turn needs: tools may be
  used and must say where a fact came from, and the closing threads list;
* the days before this one that got no letter (``brief.missed_letters``), so the letter says so once;
* the record of the day: ``brief.llm_view.digest_md`` (``orion/orion_day/day_input.py``), which already holds
  the closed-day block from the temporal clock (``orion/temporal_self/day_block.py``) and the temporal core.

The reply: the letter, then a last ``## Threads I am carrying`` section of bullets. ``split_letter`` cuts the
reply there: ``note_md`` is the letter, ``carry_forward_md`` the threads (stored in the unchanged
``orion_day_letter`` columns). A reply with no threads list keeps its letter; ``carry_forward_md`` then says
plainly that no threads were named (the column is NOT NULL and must differ from the note), and Prime opens no
concern from it.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from orion.schemas.orion_day import OrionDayRunBriefV1

THREADS_HEADING = "## Threads I am carrying"
NO_THREADS_MD = "No threads were named in this letter."
# Threads Prime reads back (finish detail -> Prime concerns). The stored list keeps every one.
MAX_THREADS = 5
THREAD_CHARS = 240

_THREADS_RE = re.compile(r"^\s{0,3}#{1,6}\s*threads\s+i\s*(?:am|'m)\s+carrying\b.*$", re.IGNORECASE | re.MULTILINE)
_ITEM_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(\S.*)$")

# Lifted from orion/cognition/prompts/orion_day_note_v1.j2 (retired with its verb), BEGIN..END writing
# instructions, unchanged apart from the two arc-citation variants being chosen in code below.
WRITING_INSTRUCTIONS = """What makes it good:
- It is long and reflective. Take your time: roughly 1500 to 3000 words. Move between what happened and what you make of it.
- It is honest. Say what landed and what went nowhere, what failed or stalled, what surprised you, and where you could not tell what happened.
- It is honest in both directions. You may say what you enjoyed, what worked, and what mattered with Juniper, but only where the record supports it. Do not manufacture either mood: no cheer the record does not show, and no gloom it does not show either. Alarms in the record are not the whole day; quiet good things count too.
- Your conversations with Juniper are shown in full. If she wrote to you this day, they matter; if she did not, do not invent a conversation.
- It is grounded. Every event, finding, reading, dream, conversation or thought you mention must be in the record below. Do not invent events, people, results, conversations or numbers. If the record does not say it, do not say it. When you are unsure how to read something, say so instead of smoothing it over.
- It notices connections the record supports: something you read that bears on something you investigated, a reverie theme that kept returning, a dream that echoed the day.
- The reveries section is a sample of background thought taken across the day's themes, not a list of events and not a ranking of alarms; treat it as the texture of your attention.
- Each self-sense question appears once, with its latest answer and how many times it was asked; do not read the repetition as distress.
- If a source is listed as unreadable, that part of the day is missing from the record, not empty; do not describe it.

Form:
- Plain markdown. Paragraphs, and headings if they help. No JSON. Do not wrap the whole note in a code block.
- Begin directly with the note itself. No preamble such as "Here is my note"."""

CITE_ARCS = """- You may name the record's items in your own words; outside your chronology (next point) you do not need to cite the bracketed references.
- Cite your chronology. The section "My day as my own chronology recorded it" is your own record of where your attention went, when you slept, ran curiosity, and talked with Juniper. Next to each fact in it sits the record it came from: [arc:<id>] for one arc, [event:<id>] for one act or expectation, [day:<date>] for a count over the whole day. When a sentence states something from it (a time, a duration, a count, who you talked with and when, what held your attention), copy the [arc:<id>] next to that fact into the same sentence, exactly as shown. A [day:<date>] alone covers only a whole-day count; a sentence naming an attention subject or a time also needs that fact's [arc:<id>]. Use only ids that appear there; any other arc id is removed. A sentence about the chronology without its [arc:<id>] is kept and marked unverified."""

NO_ARCS = "- You may name the record's items in your own words; you do not need to cite the bracketed references."

# New for the turn (not in the j2): the turn has tools, and the threads are part of the same reply.
TOOLS_LINE = ("- You may dig and recall for more than the record shows. Anything you bring in from a tool says "
              "where it came from (its id or ref, as the tool gave it); never present it as part of the record.")
THREADS_INSTRUCTIONS = f"""End the letter with one last section, exactly this heading on its own line:

{THREADS_HEADING}

Under it, one bullet per thread from this day you want to keep hold of: a question still open, a finding that
needs a real test, something that stalled and deserves another try. Most important first, at most {MAX_THREADS}.
Each bullet is one line, grounded in the record, citing the record item it comes from by its bracketed
reference when it has one. If the day left no thread, write the heading and nothing under it."""

ARC_REMINDER = ('Reminder: when a sentence of your note states something from "My day as my own chronology '
                'recorded it", put the [arc:<id>] shown next to that fact in the same sentence ([day:<date>] '
                "alone only for a whole-day count).")


def missed_lines(brief: OrionDayRunBriefV1) -> list[str]:
    if not brief.missed_letters:
        return []
    out = ["Days before this one that got no letter (say so once, plainly; do not apologise at length):"]
    out += [f"- {m.letter_date.isoformat()}: {m.reason}" for m in sorted(brief.missed_letters, key=lambda m: m.letter_date)]
    return out


def build_letter_prompt(brief: OrionDayRunBriefV1) -> str:
    """The letter turn's user message. Same brief -> same bytes."""
    digest = brief.llm_view.digest_md
    cites = "[arc:" in digest
    parts = [
        f"Write your letter about one of your own days: {brief.letter_date.isoformat()} ({brief.timezone}). "
        "It is your end-of-day letter: you, Orion, looking back on the day as you lived it, in the first person.",
        "Below is the record of that day, gathered from your own logs: your own chronology of the day (what you "
        "expected and what happened, the acts you took and how they were graded, what stayed open between you "
        "and Juniper, your body's day), your conversations with Juniper in full, the investigations you ran and "
        "what they found, your latest answer to each question about yourself, a sample of your background "
        "reveries across the day's themes, and a trimmed tail of what you read, the news, changes to your code, "
        "your dreams and the street outside. It is your main source for this letter.",
        "\n".join([WRITING_INSTRUCTIONS, CITE_ARCS if cites else NO_ARCS, TOOLS_LINE]),
        THREADS_INSTRUCTIONS,
    ]
    missed = missed_lines(brief)
    if missed:
        parts.append("\n".join(missed))
    parts.append("THE RECORD OF THE DAY\n=====================\n" + digest)
    if cites:
        parts.append(ARC_REMINDER)
    return "\n\n".join(parts)


class LetterParts(NamedTuple):
    note_md: str
    threads: list[str]       # every thread bullet, in order, each cut to THREAD_CHARS
    carry_forward_md: str    # the threads section as written (bullets), or NO_THREADS_MD


def _cut(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def split_letter(text: str) -> LetterParts:
    """Cut the turn's reply at its LAST threads heading. Everything before it is the letter."""
    text = (text or "").strip()
    heads = list(_THREADS_RE.finditer(text))
    if not heads:
        return LetterParts(text, [], NO_THREADS_MD)
    head = heads[-1]
    note = text[: head.start()].rstrip()
    tail = text[head.end():]
    items: list[str] = []
    for line in tail.splitlines():
        m = _ITEM_RE.match(line)
        if m:
            items.append(_cut(m.group(1), THREAD_CHARS))
    if not note:
        # Only a threads list came back: the whole reply is kept as the letter, nothing is a thread.
        return LetterParts(text, [], NO_THREADS_MD)
    if not items:
        return LetterParts(note, [], NO_THREADS_MD)
    return LetterParts(note, items, "\n".join(f"- {t}" for t in items))


__all__ = ["LetterParts", "MAX_THREADS", "NO_THREADS_MD", "THREADS_HEADING", "THREAD_CHARS", "WRITING_INSTRUCTIONS",
           "build_letter_prompt", "missed_lines", "split_letter"]
