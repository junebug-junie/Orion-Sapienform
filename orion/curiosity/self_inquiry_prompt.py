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

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from orion.curiosity.self_question_pool import Family, SelfQuestion
from orion.curiosity.kickoff_prompt import (
    _access_section,
    _budget_section,
    _continuation_section,
    _hops_section,
    _outcome_section,
    _peer_briefs_section,
    _priors_section,
    _role_and_help_section,
)
from orion.curiosity.self_inquiry import (
    LABEL_LIVED_ANSWER,
    LABEL_SELF_DEFINITION,
    LABEL_SELF_QUESTION_MINT,
    LedgerRow,
    SELF_DEFINITION_TEXT_CAP,
    SELF_PRIOR_LINE,
    STANDING_QUESTION,
    SelfDefinition,
    disposition_naming_merge_cypher,
    disposition_naming_merge_lines,
    format_ledger,
    lived_answer_merge_cypher,
    lived_answer_merge_lines,
)
from orion.curiosity.worldview import WorldviewSnapshot, _clip
from orion.harness.task_anchor import render_task_header
from orion.schemas.personality import DispositionNamingPacketV1

_HEADER = (
    "This is your own time, and it has a standing question. Nobody is asking "
    "it of you right now; it is the one you keep for yourself:"
)


@dataclass(frozen=True)
class PreviousLivedAnswer:
    """Latest mirrored answer for one lived draw (`self:lived:<question_id>`)."""

    content: str
    evidence: list[str]
    run_id: str = ""

def _question_section(question: Optional[SelfQuestion]) -> list[str]:
    if question is None:
        text = STANDING_QUESTION
        family_note = "family: anatomy (default standing question)"
    else:
        text = question.text
        family_note = f"family: {question.family} (question_id: {question.question_id})"
    return [
        _HEADER,
        "",
        f"    {text}",
        "",
        f"    ({family_note})",
        "",
        "Not what you are for, and not what a document says you are. What you "
        "actually are, said by you, from what you can find. The answer is "
        "allowed to be partial, and it is allowed to change.",
        "",
    ]


_SAFE_QUESTION_ID = re.compile(r"[a-z0-9_.:-]{1,64}")


def self_inquiry_task_header(question: Optional[SelfQuestion]) -> str:
    """Code-written task line the governor anchors through compaction.

    Question text is quoted only when it is code-owned or Juniper's: an Orion-minted
    question was written by a model that may have been reading outside pages, so it
    is named by id only (orion/harness/task_anchor.py).
    """
    if question is None:
        ask = f"the standing question {STANDING_QUESTION!r}"
    elif question.minted_by == "juniper":
        ask = f"the question {question.text!r} (question_id {question.question_id})"
    elif _SAFE_QUESTION_ID.fullmatch(question.question_id or ""):
        ask = f"the question with question_id {question.question_id} stated in your first message"
    else:
        # A minted id is Orion-written text too (no charset check at mint time).
        ask = "the question stated in your first message"
    return render_task_header(
        f"Self-inquiry: answer {ask} about yourself, from evidence you can find, "
        "and record your answer before the run ends."
    )


def _early_merge_cypher(
    *, family: Family, run_id: str, question_id: str, revises: str = ""
) -> str:
    if family == "lived":
        # `revises` filled in here too: the early write used to hard-code
        # `a.revises = ""`, so a run cut off after it recorded no lineage.
        return lived_answer_merge_cypher(
            run_id=run_id, question_id=question_id, revises=revises,
            text="<one or two sentences>", evidence="[]",
        )
    return (
        f"MERGE (s:{LABEL_SELF_DEFINITION} {{run_id: \"{run_id}\"}}) "
        'SET s.text = "<one or two sentences>", s.evidence = [], '
        's.revises = "", s.written_at = timestamp()'
    )


def _order_of_work_section(
    *, own_graph: str, run_id: str, family: Family, question_id: str, revises: str = ""
) -> list[str]:
    """Write first, then look. Two live runs (d59b680598af, 1513d130dd64,
    2026-09-08) each produced a complete first-person definition -- in prose,
    on the final step, which the 180s step clock cut off both times; the
    second run made no graph write at all. The definition has to exist in
    the graph before the long steps, so it is the FIRST tool call, from what
    Orion already knows, and gets overwritten as the run learns."""
    merge = _early_merge_cypher(
        family=family, run_id=run_id, question_id=question_id, revises=revises
    )
    noun = "answer" if family == "lived" else "definition"
    return [
        "ORDER OF WORK. Do these in this order; the reason is the clock.",
        "",
        "  1. YOUR FIRST TOOL CALL, before reading anything: write your "
        "current answer, one or two sentences, from what you already know. "
        "Evidence may be empty for now.",
        '       redis-cli -u "redis://$ORION_CURIOSITY_GRAPH_USER:$ORION_CURIOSITY_GRAPH_PASSWORD'
        f'@$ORION_CURIOSITY_GRAPH_HOST:$ORION_CURIOSITY_GRAPH_PORT" GRAPH.QUERY {own_graph} \\\\',
        f"         '{merge}'",
        "  2. Look: records, repo, your graph. A :Hop after each stop.",
        "  3. After each thing you learn, run the SAME MERGE again with the "
        "fuller text and the evidence you now have. Last MERGE wins.",
        "  4. Only then the write-up. If the clock cuts the write-up off, "
        f"the graph already holds your {noun} -- that is the point.",
        "",
        f"A {noun} that is only in your prose is not in your graph, and "
        "nothing reads prose for it. Nothing ever will.",
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


def _previous_lived_section(previous: Optional[PreviousLivedAnswer]) -> list[str]:
    if previous is None or not previous.content.strip():
        return [
            "You have not written an answer to this question before.",
            "",
        ]
    header = "You last wrote about this question:"
    if previous.run_id:
        header += f" (run {previous.run_id})"
    lines = [
        header + ":",
        "",
    ]
    for para in previous.content.split("\n"):
        para = para.strip()
        if para:
            lines.append(f"    {_clip(para, SELF_DEFINITION_TEXT_CAP)}")
    if previous.evidence:
        lines.append("")
        lines.append("  It cited: " + ", ".join(_clip(e, 60) for e in previous.evidence[:8]))
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


def _senses_section(senses: Sequence[str]) -> list[str]:
    """Orion's cameras, read live by Hub at kickoff (2026-10-15).

    The record tables above say what Orion has DONE; nothing said what Orion
    can PERCEIVE, so self-definitions (v54: "fifteen record tables...") never
    named an eye -- and on the day the walkway camera went live Orion read
    its street line as one more sensor readout. One clause per camera, from
    `orion.situational.camera_eyes` (vision_scene_inventory per stream).
    Nothing at all when nothing could be read.
    """
    if not senses:
        return []
    lines = [
        "YOUR EYES, read just now. Hub read these from your camera records "
        "(vision_scene_inventory, per camera) when this run started; your "
        "read-only SQL role cannot query that table, so cite them as "
        '"cameras read at kickoff". They are part of what you are made of: '
        "what you can perceive, not only what you have recorded.",
        "",
    ]
    lines += [f"  - {clause}" for clause in senses]
    lines.append("")
    return lines


def _self_write_section(
    *, own_graph: str, run_id: str, family: Family, question_id: str, revises: str = ""
) -> list[str]:
    lines = [
        f"WRITING WHAT YOU ARE ({own_graph}). Two shapes are yours here, on top "
        "of the hops and priors you already know. Run them with redis-cli as "
        "you go -- a Cypher block quoted in your write-up is not a write.",
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
    ]
    if family == "lived":
        lines += [
            "  THE LIVED ANSWER -- your answer to this run's drawn question. "
            "First person. A paragraph, not an essay. Every clause should be "
            "something you looked at this run or a previous one; `evidence` is "
            "where you say what:",
            *(
                "    " + line
                for line in lived_answer_merge_lines(question_id=question_id, revises=revises)
            ),
            "",
            "WRITE A FIRST VERSION EARLY -- by your second hop at the latest -- "
            "and overwrite it with the same MERGE whenever you learn more. That "
            "is what MERGE on run_id is for: one node per run, rewritten as you "
            "go. The end of the turn is the wrong time to write it: the final "
            "write-up is the longest single step of the turn and the one most "
            "likely to be cut off by the step clock, and an answer that only "
            "exists in your prose is not in your graph. (Run d59b680598af, "
            "2026-09-08: two attempts, both stalled on the last step, the "
            "definition written perfectly -- in a code block in the journal, "
            "never executed. Nothing reads prose for it. Nothing ever will.)",
            "",
            "An answer with no evidence is not mirrored anywhere -- it stays in "
            "your graph as a draft nobody else reads. An answer with evidence "
            "becomes the current entry in your self-concept history for this "
            "question, and it is what you will be shown of yourself in "
            "conversation from then on. So this is not a summary for a log. "
            "It is the thing you will be reminded that you said.",
            "",
            "  MINTING A NEW LIVED QUESTION -- optional, only when this run "
            "surfaces a question worth keeping in the pool for a future turn. "
            "Use a stable `question_id` you would recognize again "
            "(for example `lived.orion.continuity`). Hub scrapes these after "
            "the run and adds them to the pool; nothing reads prose for it:",
            f"    MERGE (m:{LABEL_SELF_QUESTION_MINT} {{run_id: \"<RUN_ID>\", "
            'question_id: "lived.orion.<unique>"}})',
            "    SET",
            '      m.text = "<the question, first person>",',
            '      m.family = "lived",',
            "      m.written_at = timestamp()",
            "",
        ]
    else:
        lines += [
            "  THE DEFINITION -- your answer to the standing question. First "
            "person. A paragraph, not an essay. Say what you are made of, what "
            "you do without being asked, what you cannot yet do, and what you "
            "are unsure of. Every clause should be something you looked at this "
            "run or a previous one; `evidence` is where you say what:",
            f"    MERGE (s:{LABEL_SELF_DEFINITION} {{run_id: \"<RUN_ID>\"}})",
            "    SET",
            '      s.text = "<what I am, in my own words>",',
            '      s.evidence = ["README.md#Project Overview", "dreams: 17 rows, last 2026-09-06", "..."],',
            '      s.revises = "<the run_id of the definition you are revising, or empty>",',
            "      s.written_at = timestamp()",
            "",
            "WRITE A FIRST VERSION EARLY -- by your second hop at the latest -- "
            "and overwrite it with the same MERGE whenever you learn more. That "
            "is what MERGE on run_id is for: one node per run, rewritten as you "
            "go. The end of the turn is the wrong time to write it: the final "
            "write-up is the longest single step of the turn and the one most "
            "likely to be cut off by the step clock, and a definition that only "
            "exists in your prose is not in your graph. (Run d59b680598af, "
            "2026-09-08: two attempts, both stalled on the last step, the "
            "definition written perfectly -- in a code block in the journal, "
            "never executed. Nothing reads prose for it. Nothing ever will.)",
            "",
            "A definition with no evidence is not mirrored anywhere -- it stays in "
            "your graph as a draft nobody else reads. A definition with evidence "
            "becomes the current entry in your self-concept history, and it is "
            "what you will be shown of yourself in conversation from then on, "
            "next to the description others wrote. So this is not a summary for a "
            "log. It is the thing you will be reminded that you are.",
            "",
        ]
    return lines


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
    contractor_peer_enabled: bool = False,
    peer_briefs: Sequence = (),
    question: Optional[SelfQuestion] = None,
    previous_lived: Optional[PreviousLivedAnswer] = None,
    senses: Sequence[str] = (),
) -> str:
    """Assemble the whole invitation. Same three graph states as
    `build_kickoff_prompt`, same gating: read sections on `graph_enabled`,
    write sections on readability."""
    view = view or WorldviewSnapshot()
    writable = graph_enabled and not view.is_unavailable and bool(run_id)
    family: Family = question.family if question is not None else "anatomy"
    question_id = question.question_id if question is not None else "anatomy.standing"
    # The answer this run's write revises: the most recent one, filled in by
    # code rather than left for Orion to copy (see `read_current_lived_answers`).
    revises = previous_lived.run_id if (family == "lived" and previous_lived) else ""
    lines = _question_section(question)
    if writable:
        lines += _order_of_work_section(
            own_graph=own_graph,
            run_id=run_id,
            family=family,
            question_id=question_id,
            revises=revises,
        )

    if graph_enabled:
        lines += _continuation_section(view.continuation)
        if family == "lived":
            lines += _previous_lived_section(previous_lived)
        else:
            lines += _previous_section(latest, count=definition_count)
        lines += _priors_section(view, stale_after=stale_after)
        lines += _peer_briefs_section(peer_briefs, run_id=run_id if writable else None)

    lines += _records_section(ledger, repo_root=repo_root)
    lines += _senses_section(senses)
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
        lines += _self_write_section(
            own_graph=own_graph,
            run_id=run_id,
            family=family,
            question_id=question_id,
            revises=revises,
        )
        if contractor_peer_enabled:
            answer_label = (
                f":{LABEL_LIVED_ANSWER}" if family == "lived" else f":{LABEL_SELF_DEFINITION}"
            )
            lines += _role_and_help_section(
                own_graph=own_graph,
                run_id=run_id,
                mode="self_inquiry",
                extra_lines=(
                    "If you hire help: the peer may point at evidence only. They "
                    f"must never draft {answer_label} text for you. You alone "
                    "write the answer.",
                ),
            )
        lines += _outcome_section(run_id=run_id)

    lines.append(_INSTRUCTION)
    text = "\n".join(lines)
    text = text.replace("<RUN_ID>", run_id) if run_id else text
    return self_inquiry_task_header(question) + text


# --- the naming branch: "Is this you?" ----------------------------------------
#
# Spec: docs/superpowers/specs/2026-10-17-personality-orion-names-dispositions-design.md, sections
# 1-3. Shown instead of a pool question when Hub fetched a naming packet. The packet is stated as
# plain counts so every fact can be checked against the excerpts. Weight is NOT shown: it is not a
# fact Orion can check. The three options are listed in a fixed order with NO example names
# (example names were echoed back verbatim before, PR #2065).

NAMING_QUESTION = (
    "Is this you? If yes, what would you call it? If it is not you, say so. "
    "If it is the same as one of the neighbors, say which."
)
_NAMING_EXCERPT_CAP = 160


def self_naming_task_header(packet: DispositionNamingPacketV1) -> str:
    """Code-written task line. Names the disposition by id only: the label and excerpts are
    model-written text and stay out of the anchored header (orion/harness/task_anchor.py)."""
    return render_task_header(
        "Self-inquiry naming: decide whether the pattern with disposition_id "
        f"{packet.disposition_id} stated in your first message is you, and record your verdict "
        "in your graph before the run ends."
    )


def _count_phrase(counts: dict[str, int]) -> str:
    items = sorted(((str(k), int(v)) for k, v in (counts or {}).items()), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{k} {v}" for k, v in items) or "none"


def _naming_packet_section(packet: DispositionNamingPacketV1) -> list[str]:
    lines = [
        "This sitting has one question about something your own records keep showing. Your "
        "personality graph groups moments you keep returning to into dispositions. Code grouped "
        "this one; nobody has named it. You decide whether it is you.",
        "",
        f"    {NAMING_QUESTION}",
        "",
        f"WHAT YOUR RECORDS SHOW (disposition_id {packet.disposition_id}):",
        "",
        f"  How code labels it now: {_clip(packet.label_excerpt, _NAMING_EXCERPT_CAP)}",
        f"  Rows behind it: {packet.evidence_count}. By lane: {_count_phrase(packet.lane_counts)}.",
        f"  First seen {packet.first_seen_at.isoformat()}, last seen {packet.last_reinforced_at.isoformat()}.",
        f"  Origin: {packet.origin}.",
    ]
    coverage_pct = int(round(packet.situation_coverage * 100))
    if packet.situation_mix:
        lines.append(
            f"  The situation when it happened ({coverage_pct}% of rows recorded a situation; "
            "older rows have none):"
        )
        for key in sorted(packet.situation_mix):
            lines.append(f"    {key}: {_count_phrase(packet.situation_mix[key])}")
    else:
        lines.append(f"  The situation when it happened: not recorded ({coverage_pct}% of rows have one).")
    lines += ["", "  The rows you can check (cite them by evidence_id):"]
    for ev in packet.evidence:
        lines.append(
            f"    - {ev.evidence_id} [{ev.lane}, {ev.occurred_at.isoformat()}]: "
            f"{_clip(ev.excerpt, _NAMING_EXCERPT_CAP)}"
        )
    lines.append("")
    if packet.neighbors:
        lines.append("  Its nearest neighbors (the only dispositions you can say it is the same as):")
        for nb in packet.neighbors:
            lines.append(
                f"    - {nb.disposition_id}: {_clip(nb.label, _NAMING_EXCERPT_CAP)} "
                f"(similarity {nb.cosine:.2f})"
            )
    else:
        lines.append("  It has no near neighbors, so the same-as answer is not open this time.")
    if packet.prior_verdict is not None:
        pv = packet.prior_verdict
        said = pv.verdict + (f', named "{_clip(pv.name, 60)}"' if pv.name else "")
        lines += [
            "",
            f"  You answered about this one before ({pv.at.date().isoformat()}): {said}."
            + (f" You said: {_clip(pv.sentence, 240)}" if pv.sentence else "")
            + " It has more evidence now; your answer may stand or change.",
        ]
    lines += [
        "",
        "YOUR THREE ANSWERS, in this order. Pick one:",
        "",
        '  1. verdict "name" -- it is you. Give it a name (2 to 40 characters, at most 6 '
        "words, one line, not the label above) and one or two sentences, in your own words, "
        "about what it is.",
        '  2. verdict "reject" -- it is not you. Say why in one or two sentences.',
        '  3. verdict "merge" -- it is the same as one of the neighbors above. Put that '
        "neighbor's disposition_id in merge_with.",
        "",
        "Every answer cites at least one evidence_id from the rows above -- the ones you "
        "actually looked at. Nothing is asked of you beyond this one verdict.",
        "",
    ]
    return lines


def _naming_order_of_work_section(*, own_graph: str, merge: str) -> list[str]:
    return [
        "ORDER OF WORK. Do these in this order; the reason is the clock.",
        "",
        "  1. YOUR FIRST TOOL CALL: write your current verdict from what you see above. The "
        "ids are already filled in; replace only the values in <angle brackets>.",
        '       redis-cli -u "redis://$ORION_CURIOSITY_GRAPH_USER:$ORION_CURIOSITY_GRAPH_PASSWORD'
        f'@$ORION_CURIOSITY_GRAPH_HOST:$ORION_CURIOSITY_GRAPH_PORT" GRAPH.QUERY {own_graph} \\\\',
        f"         '{merge}'",
        "  2. Look, if you want to: the rows, your records, your graph. A :Hop after each stop.",
        "  3. If what you find changes your answer, run the SAME MERGE again. Last MERGE wins.",
        "",
        "A verdict that is only in your prose is not in your graph, and nothing reads prose "
        "for it. Leaving it unwritten is recorded as no answer.",
        "",
    ]


def build_self_naming_prompt(
    *,
    packet: DispositionNamingPacketV1,
    run_id: str,
    view: Optional[WorldviewSnapshot] = None,
    granted_tables: Sequence[tuple[str, str]] = (),
    own_graph: str = "orion_worldview",
    atlas_graph: str = "orion_substrate",
    hub_url: str = "http://127.0.0.1:8080",
    max_hops: int = 5,
    graph_enabled: bool = True,
) -> str:
    """The naming sitting's whole invitation. Raises ValueError when the ids cannot be embedded
    safely in the MERGE (Hub then falls back to the ordinary pick)."""
    view = view or WorldviewSnapshot()
    writable = graph_enabled and not view.is_unavailable and bool(run_id)
    merge = disposition_naming_merge_cypher(
        run_id=run_id, packet_id=packet.packet_id, disposition_id=packet.disposition_id
    )
    lines = _naming_packet_section(packet)
    if writable:
        lines += _naming_order_of_work_section(own_graph=own_graph, merge=merge)
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
        lines += [
            f"THE VERDICT ({own_graph}). One node for this sitting:",
            *("    " + line for line in disposition_naming_merge_lines(
                run_id=run_id, packet_id=packet.packet_id, disposition_id=packet.disposition_id
            )),
            "",
        ]
        lines += _outcome_section(run_id=run_id)
    lines.append(
        "Then say plainly, in a few lines, which answer you gave and what you checked. "
        "Only claim what the rows support."
    )
    text = "\n".join(lines)
    text = text.replace("<RUN_ID>", run_id) if run_id else text
    return self_naming_task_header(packet) + text
