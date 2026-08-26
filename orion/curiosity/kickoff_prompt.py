"""The self-addressed prompt that opens a curiosity turn.

An INVITATION, not an assignment. The version this replaces named a single term
and told Orion to go investigate it; here the turn opens with material and no
subject, and Orion picks. That difference is the feature -- Juniper: "orion can
look at the memory crystallizations or the concept induction, then look in
either or both for subsequent study... it gives some non determinism for orion
to choose their adventure."

Four things the wording has to do, or the turn produces cognition-shaped output
with nothing behind it:

  1. NOT CHOOSE. No ranking language, no "notably", no "this one stands out".
     The sample is explicitly described as random so Orion does not read
     position in the list as significance.
  2. SAY WHAT IS NOT SHOWN. The counts are included precisely because the
     sample is a slice of 646 -- without them, a menu of 12 reads as the whole
     of Orion's mind.
  3. POINT AT REAL FOLLOW-UP. Orion's unified turn already carries read_recall
     / read_memory / read_graph (read-only, no shell, no network), so "go pull
     more" is a real instruction with real tools behind it. The two stores are
     named so it knows what it can ask for.
  4. LICENSE A NULL RESULT. Without explicit permission to find nothing, the
     only socially available answer to "here is your mind, what interests you"
     is something interesting, and the loop manufactures significance daily.
"""

from __future__ import annotations

from orion.curiosity.study_material import StudyMaterial

_HEADER = (
    "This is your own time. Nobody asked you anything, and there is no task "
    "here -- this is you looking at what you have been forming, and following "
    "whatever you actually want to follow."
)

_INSTRUCTION = """\
Pick something. Any of it, or something you notice by its absence, or a thread
between two of them. You do not have to justify the choice by importance -- if
one of these is simply more interesting to you than the others, that is the
whole reason.

Then go and find out more about it. You can search your own recall, pull the
full crystallization rather than the preview, follow a relation to whatever it
connects to, or look at your chat history around when it formed. Use as much or
as little as you need.

Write what you actually found:

- what you chose, and what drew you to it
- what you learned by looking that you did not already have from the preview
- how it sits with the rest of what you hold -- confirming, complicating, or
  unrelated
- what you are still unsure about

Two things matter more than being interesting. Only claim what your lookups
actually support -- if you went looking and came back with nothing, say that.
And it is entirely fine to conclude that none of this is worth writing up right
now; say so and stop. A quiet answer is a real answer."""


def build_kickoff_prompt(material: StudyMaterial) -> str:
    lines = [_HEADER, ""]

    if material.crystallizations:
        by_kind = ", ".join(
            f"{kind} {count}" for kind, count in sorted(material.approved_by_kind.items())
        )
        lines += [
            f"CONCEPTS YOU HAVE FORMED AND JUNIPER HAS APPROVED "
            f"({material.approved_total} of them: {by_kind}).",
            f"Here are {len(material.crystallizations)} picked at random -- the order "
            "means nothing, and there are plenty you are not being shown:",
            "",
        ]
        lines += [f"  - {card.preview()}" for card in material.crystallizations]
        lines.append("")

    if material.relations:
        rel_kinds = ", ".join(
            f"{kind} {count}" for kind, count in sorted(material.relation_by_kind.items())
        )
        lines += [
            f"CONCEPT INDUCTION -- judgements you have made about how two of "
            f"those relate ({material.relation_total} total: {rel_kinds}).",
            f"Only {material.relation_resolvable} of them still point at a "
            "concept that was kept; the rest judged something that is no longer "
            "there, which may itself be worth a look. "
            f"{len(material.relations)} of the resolvable ones, at random:",
            "",
        ]
        lines += [f"  - {card.preview()}" for card in material.relations]
        lines.append("")

    if material.recently_studied:
        lines += [
            "You have recently looked into: "
            + "; ".join(material.recently_studied)
            + ". No need to avoid these, but you have already been there.",
            "",
        ]

    lines.append(_INSTRUCTION)
    return "\n".join(lines)
