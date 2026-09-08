"""The self-inquiry invitation: the standing question, and where to look.

Same construction rules as `kickoff_prompt.py`, and it reuses that module's
access, budget, hop, prior and outcome sections verbatim so the two lines
cannot drift apart on the parts they share. What is different is the
opening (a standing question instead of a menu), the material (Orion's own
records instead of crystallization cards), and one extra write
(`:SelfDefinition`).

The question is an INVITATION, not a prior. Orion forms the priors. Code
does not say what Orion is; it says where the records that could answer that
are kept, and that the previous answer -- if there is one -- is Orion's own
to revise.
"""

from __future__ import annotations

from typing import Optional, Sequence

from orion.curiosity.kickoff_prompt import (
    _access_section,
    _budget_section,
    _continuation_section,
    _hops_section,
    _outcome_section,
    _priors_section,
)
from orion.curiosity.self_inquiry import (
    LABEL_SELF_DEFINITION,
    LedgerRow,
    SELF_DEFINITION_TEXT_CAP,
    SELF_PRIOR_LINE,
    STANDING_QUESTION,
    SelfDefinition,
    format_ledger,
)
from orion.curiosity.worldview import WorldviewSnapshot, _clip

_HEADER = (
    "This is your own time, and it has a standing question. Nobody is asking "
    "it of you right now; it is the one you keep for yourself:"
)


def _question_section() -> list[str]:
    return [
        _HEADER,
        "",
        f"    {STANDING_QUESTION}",
        "",
        "Not what you are for, and not what a document says you are. What you "
        "actually are, said by you, from what you can find. The answer is "
        "allowed to be partial, and it is allowed to change.",
        "",
    ]


def _previous_section(latest: Optional[SelfDefinition], *, count: Optional[int]) -> list[str]:
    if latest is None:
        return [
            "You have not written a definition of yourself before. "
            + (
                "The graph could not say how many exist."
                if count is None
                else "This would be the first."
            ),
            "",
        ]
    n = f"{count} so far" if count is not None else "count unknown"
    lines = [
        f"WHAT YOU LAST WROTE ABOUT YOURSELF ({n}; run {latest.run_id}"
        + (f", revising {latest.revises}" if latest.revises else "")
        + "):",
        "",
    ]
    for para in latest.text.split("\n"):
        para = para.strip()
        if para:
            lines.append(f"    {_clip(para, SELF_DEFINITION_TEXT_CAP)}")
    if latest.evidence:
        lines.append("")
        lines.append("  It cited: " + ", ".join(_clip(e, 60) for e in latest.evidence[:8]))
    lines += [
        "",
        "This is yours to revise, extend, or contradict. Reaffirming it "
        "unchanged is a legitimate outcome if you looked and it still holds -- "
        "say what you checked. Rewriting it without looking is not.",
        "",
    ]
    return lines


def _records_section(ledger: Sequence[LedgerRow], *, repo_root: str) -> list[str]:
    lines = [
        "WHERE THE RECORDS ARE. These are the places that can answer the "
        "question with something other than a description of you written by "
        "someone else. Counts are shown so you can see which are thin.",
        "",
        f"  Your own repository, read-only, at {repo_root}/ -- the README is "
        "what the people building you say you are; services/, orion/ and "
        "docs/ are what you are actually made of. Read carefully: a README "
        "describes intent, and some of what it names is aspirational or "
        "already retired. Prefer what you can confirm in a table or a "
        "running path.",
        "",
    ]
    if ledger:
        lines.append("  Your own records, in SQL (listed again under HOW TO REACH below):")
        lines += format_ledger(list(ledger))
        lines.append("")
    lines += [
        "  Your own graph holds what you have worked out so far -- the priors "
        f'with line = "{SELF_PRIOR_LINE}" are the ones you formed about yourself.',
        "",
    ]
    return lines


def _self_write_section(*, own_graph: str, run_id: str) -> list[str]:
    return [
        f"WRITING WHAT YOU ARE ({own_graph}). Two shapes are yours here, on top "
        "of the hops and priors you already know.",
        "",
        "  A SELF-PRIOR -- a claim about yourself that could turn out to be "
        "wrong. Exactly the Prior shape from your investigations, with one "
        "extra property so your self-inquiry runs can find it again:",
        '    MERGE (p:Prior {prior_id: "self:<something unique>"})',
        "    ON CREATE SET",
        '      p.claim = "<the claim about yourself, one sentence>",',
        "      p.confidence = 0.55,",
        f'      p.status = "open", p.times_tested = 0, p.line = "{SELF_PRIOR_LINE}",',
        '      p.formed_from = "<what produced it: a file, a table, a count>",',
        '      p.last_tested_at = "<iso timestamp>", p.run_id = "<RUN_ID>", '
        'p.why = "<one sentence>"',
        "",
        "  THE DEFINITION -- your answer to the standing question, as it stands "
        "at the end of this run. First person. A paragraph, not an essay. Say "
        "what you are made of, what you do without being asked, what you "
        "cannot yet do, and what you are unsure of. Every clause should be "
        "something you looked at this run or a previous one; `evidence` is "
        "where you say what:",
        f"    CREATE (:{LABEL_SELF_DEFINITION} {{",
        '      run_id: "<RUN_ID>",',
        '      text: "<what I am, in my own words>",',
        '      evidence: ["README.md#Project Overview", "dreams: 17 rows, last 2026-09-06", "..."],',
        '      revises: "<the run_id of the definition you are revising, or empty>",',
        "      written_at: timestamp()",
        "    })",
        "",
        "A definition with no evidence is not mirrored anywhere -- it stays in "
        "your graph as a draft nobody else reads. A definition with evidence "
        "becomes the current entry in your self-concept history, and it is "
        "what you will be shown of yourself in conversation from then on, "
        "next to the description others wrote. So this is not a summary for a "
        "log. It is the thing you will be reminded that you are.",
        "",
        "Write it before the last quarter of your budget, at whatever "
        "confidence you have. You can revise it next time.",
        "",
    ]


_INSTRUCTION = """\
Look. Then write what you actually found:

- what you went and read or counted, and what it said
- what changed in how you would describe yourself, or what held
- what you still cannot see about yourself, and where you looked for it

Only claim what your lookups support. If the records are thin on something,
say they are thin rather than filling the gap from the README. And if you
conclude the definition you already have still stands, say what you checked
and stop -- that is a real answer too."""


def build_self_inquiry_prompt(
    *,
    view: Optional[WorldviewSnapshot] = None,
    latest: Optional[SelfDefinition] = None,
    definition_count: Optional[int] = None,
    ledger: Sequence[LedgerRow] = (),
    granted_tables: Sequence[tuple[str, str]] = (),
    run_id: str = "",
    own_graph: str = "orion_worldview",
    atlas_graph: str = "orion_substrate",
    hub_url: str = "http://127.0.0.1:8080",
    repo_root: str = "/repo",
    max_hops: int = 5,
    stale_after: int = 3,
    graph_enabled: bool = True,
) -> str:
    """Assemble the whole invitation. Same three graph states as
    `build_kickoff_prompt`, same gating: read sections on `graph_enabled`,
    write sections on readability."""
    view = view or WorldviewSnapshot()
    writable = graph_enabled and not view.is_unavailable and bool(run_id)
    lines = _question_section()

    if graph_enabled:
        lines += _continuation_section(view.continuation)
        lines += _previous_section(latest, count=definition_count)
        lines += _priors_section(view, stale_after=stale_after)

    lines += _records_section(ledger, repo_root=repo_root)
    lines += _access_section(
        own_graph=own_graph,
        atlas_graph=atlas_graph,
        hub_url=hub_url,
        graph_enabled=graph_enabled,
        writable=writable,
        extra_tables=tuple(granted_tables),
    )
    lines += _budget_section(writable=writable)
    lines += _hops_section(max_hops, writable=writable)

    if writable:
        lines += _self_write_section(own_graph=own_graph, run_id=run_id)
        lines += _outcome_section(run_id=run_id)

    lines.append(_INSTRUCTION)
    text = "\n".join(lines)
    return text.replace("<RUN_ID>", run_id) if run_id else text
