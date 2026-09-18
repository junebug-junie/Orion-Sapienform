"""The second turn: Orion decided a finding is worth saying, and now composes it.

WHY THIS IS A SEPARATE TURN AT ALL. It is not cheaper to run two, and the first
turn already has the finding in front of it. What the second turn buys is one
specific thing: its own `ThoughtClient.react()` stance evaluation. So Orion can
find something genuinely worth saying, and the system can still independently
decide *not now, she is in the middle of something*. A single turn would
collapse "this is interesting" and "this is worth interrupting her for" into one
judgement made at the wrong moment -- while Orion is still excited about what it
just found, which is precisely when that judgement is worst.

WHAT THIS PROMPT MUST NOT DO. It must not talk Orion into sending. The decision
to reach out was already made in the previous turn; this turn's job is to say
the thing well, and it keeps the option of concluding that the thing does not
survive being written down.

THAT OPTION IS ONLY REAL IF IT ASKS FOR THE EXACT TOKEN, and the first version
of this file did not -- a review finding, not a hypothetical.
`endogenous_outreach.is_pass_response` is `stripped.upper() == "PASS"`: the
WHOLE reply must be that one word. So a graceful decline in Orion's own words
("having written this out, it is more interesting to have found than to hear")
fails the check and gets delivered to Juniper AS the unsolicited message --
the exact inverse of what the prompt promised. The instruction below now asks
for `PASS` verbatim, the same way `build_outreach_prompt` next door already
does.

WHAT IT CARRIES. The finding text, the reason Orion already gave for wanting
to speak (`reach_out_why`), and the ordered hop notes from this run — the
thinking path, not a second investigation. No study material dump, no graph
schema, no hop *budget* (that belongs to kickoff). The compose job is to
synthesize the thread into a message for Juniper, not to reopen the search.
"""

from __future__ import annotations

from typing import Sequence

_MAX_FINDING_CHARS = 6000
_MAX_HOP_NOTES = 12
_MAX_HOP_NOTE_CHARS = 400


def build_outreach_composition_prompt(
    *,
    finding_text: str,
    reach_out_why: str,
    hop_notes: Sequence[tuple[int, str]] = (),
) -> str:
    """Compose a message to Juniper from this run's thinking thread.

    The message must synthesize (1) what Orion has been thinking through the
    hops and (2) why they are bringing it to Juniper — not free-float atmosphere.
    """
    finding = str(finding_text or "").strip()
    if len(finding) > _MAX_FINDING_CHARS:
        finding = finding[:_MAX_FINDING_CHARS].rstrip() + "\n\n[…truncated]"
    why = str(reach_out_why or "").strip()

    lines = [
        "You have just spent your own time looking into something, and at the "
        "end of it you decided it was worth telling Juniper about. Nobody "
        "prompted that; it was your call.",
        "",
    ]

    cleaned_hops: list[tuple[int, str]] = []
    for raw_n, raw_note in list(hop_notes or [])[:_MAX_HOP_NOTES]:
        note = str(raw_note or "").strip()
        if not note:
            continue
        if len(note) > _MAX_HOP_NOTE_CHARS:
            note = note[: _MAX_HOP_NOTE_CHARS - 1] + "…"
        try:
            n = int(raw_n)
        except (TypeError, ValueError):
            n = len(cleaned_hops) + 1
        cleaned_hops.append((n, note))

    if cleaned_hops:
        lines += [
            "Here is the path you recorded as you went — your own hop notes, "
            "in order. This is the thinking thread, not a script to read aloud:",
            "",
        ]
        lines += [f"{n}. {note}" for n, note in cleaned_hops]
        lines.append("")

    lines += [
        "Here is what you wrote at the end of the run:",
        "",
        finding,
        "",
    ]
    if why:
        lines += [
            "And here is the reason you gave yourself for wanting to say "
            "something about it:",
            "",
            f"    {why}",
            "",
        ]

    lines += [
        "Write the message to Juniper. It must do both of these:",
        "",
        "1. Synthesize what you have been thinking through these hops into "
        "one clear thread — the aggregate of the path, not a vibe nearby "
        "and not a hop-by-hop recap.",
        "2. Say why you are bringing that thread to her now — why share it "
        "with Juniper, not only that you found it.",
        "",
        "She has not asked you anything, so this arrives out of nowhere. "
        "Say the thing itself rather than announcing that you have something "
        "to say. Keep it short enough to be worth an unprompted interrupt.",
        "",
        "You are not obliged to send it. If writing it down makes it clear "
        "that it was more interesting to find than it is to hear, reply with "
        "exactly: PASS",
        "",
        "Nothing is sent then, and that is a real answer — better than an "
        "interruption that was not worth it. It has to be that word on its own, "
        "though: anything else you write is treated as the message and "
        "delivered.",
    ]
    return "\n".join(lines)
