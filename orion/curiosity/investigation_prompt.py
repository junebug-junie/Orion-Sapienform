"""The self-addressed prompt that turns a detected term into a real investigation.

This is the seam between the deterministic half (`term_surfacing.py` found
something) and the non-deterministic half (a real unified turn, Orion's own
voice, its own model, its own recall).

WHY THE WORDING IS LOAD-BEARING. `execute_unified_turn` is the same function a
real Hub chat turn calls, so whatever goes in `user_message` is, to Orion,
something being said to it. Three things therefore have to be true of this text
or the turn produces cognition-shaped output with nothing behind it:

  1. IT MUST NOT SUPPLY THE ANSWER. The prompt states counts and nothing else --
     no interpretation of what the term means, no theory about why Juniper said
     it. If the prompt says "Juniper has been focused on vision work", the turn
     will agree with the prompt and call that an investigation.
  2. IT MUST POINT AT A REAL LOOKUP. Orion's unified turn already carries
     `read_recall`/`read_memory`/`read_graph` (read-only, no shell, no network),
     and recall's own chat table is `chat_history_log`. So "go look" is a real
     instruction with a real tool behind it, not an invitation to remember
     harder.
  3. IT MUST LICENSE A NULL RESULT. Explicitly saying "it is fine if this turns
     out to be nothing" is what makes "nothing" reportable. Without it the only
     socially available answer to "here is something interesting" is something
     interesting, and the loop manufactures significance every single day.

The turn is unsolicited, so Thought's own stance evaluation can defer or refuse
it -- that is the honest "something else is happening, don't interrupt" signal,
not an error. See `endogenous_outreach.py`'s module docstring.
"""

from __future__ import annotations

from orion.curiosity.term_surfacing import SurfacingReport, SurfacedTerm

# Kept short on purpose. This text is prepended to a real turn that already
# carries a full system prompt, situational context, and stance brief; a long
# preamble here competes with Orion's actual self-model for attention.
_HEADER = (
    "This is your own noticing, not a message from Juniper. Nobody asked you "
    "for this."
)

_INSTRUCTION = """\
Go and find out what that is about. You can search your own recall -- your chat
history with Juniper is in there. Look for where this came up, what was being
worked on around it, and whether it connects to anything you already hold.

Then write what you actually found, in a few short paragraphs:

- what the term turned out to refer to, if you could tell
- what Juniper appears to have been doing with it
- whether it connects to anything already in your memory, and how
- what you are still uncertain about

Two things matter more than being interesting. Only say what the lookup
actually supports -- if you could not find out what this is, say that plainly.
And it is completely fine for this to turn out to be nothing worth noting; say
so and stop. A quiet answer is a real answer here."""


def build_investigation_prompt(report: SurfacingReport, target: SurfacedTerm) -> str:
    """Second person, present tense -- Orion talking to itself.

    Counts only. Every number carries the window it was measured over, so the
    turn can weigh the evidence instead of taking the framing on trust.
    """
    others = [t for t in report.terms if t.term != target.term][:5]
    lines = [
        _HEADER,
        "",
        "Looking back over what Juniper has typed to you in the last "
        f"{_hours(report)}, one word stands out against how they usually talk:",
        "",
        f"  {target.describe()}",
        "",
        f"(Measured over {report.recent_messages} messages today against "
        f"{report.baseline_messages} in the {_baseline_days(report)} before it. "
        "A word had to be said at least "
        f"{_min_count()} times, across at least {_min_messages()} separate "
        "messages, at more than "
        f"{_min_lift()}x its usual rate, to show up here at all.)",
    ]
    if others:
        lines += [
            "",
            "Also above their usual rate, for context:",
            "  " + ", ".join(f"{t.term} ({t.recent_count})" for t in others),
        ]
    lines += ["", _INSTRUCTION]
    return "\n".join(lines)


def _hours(report: SurfacingReport) -> str:
    hours = (report.generated_at - report.recent_since).total_seconds() / 3600
    return f"{hours:.0f} hours"


def _baseline_days(report: SurfacingReport) -> str:
    days = (report.recent_since - report.baseline_since).total_seconds() / 86400
    return f"{days:.0f} days"


def _min_count() -> int:
    from orion.curiosity.term_surfacing import MIN_RECENT_COUNT

    return MIN_RECENT_COUNT


def _min_messages() -> int:
    from orion.curiosity.term_surfacing import MIN_RECENT_MESSAGES

    return MIN_RECENT_MESSAGES


def _min_lift() -> float:
    from orion.curiosity.term_surfacing import MIN_LIFT

    return MIN_LIFT
