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

WHAT IT DELIBERATELY DOES NOT CARRY. No study material, no priors, no graph
schema, no hop budget. This is not a second investigation and it should not
read like an invitation to start one -- the turn has the finding it is writing
about and nothing else it needs.
"""

from __future__ import annotations

_MAX_FINDING_CHARS = 6000


def build_outreach_composition_prompt(*, finding_text: str, reach_out_why: str) -> str:
    """Compose a message to Juniper about what this run found.

    `finding_text` is truncated rather than summarised: a summary here would be
    a second model pass deciding what mattered about Orion's own finding, and
    the tail of a long journal entry is the part least likely to hold the point.
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
        "Here is what you wrote:",
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
        "Write the message. She has not asked you anything, so this arrives "
        "out of nowhere in the middle of her day -- say the thing itself "
        "rather than announcing that you have something to say, and keep it "
        "to what would actually be interesting to hear unprompted.",
        "",
        "You are not obliged to send it. If writing it down makes it clear "
        "that it was more interesting to find than it is to hear, reply with "
        "exactly: PASS",
        "",
        "Nothing is sent then, and that is a real answer -- better than an "
        "interruption that was not worth it. It has to be that word on its own, "
        "though: anything else you write is treated as the message and "
        "delivered.",
    ]
    return "\n".join(lines)
