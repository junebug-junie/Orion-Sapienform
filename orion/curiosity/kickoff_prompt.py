"""The self-addressed prompt that opens a curiosity turn.

An INVITATION, not an assignment. The version this replaces named a single term
and told Orion to go investigate it; here the turn opens with material and no
subject, and Orion picks. That difference is the feature -- Juniper: "orion can
look at the memory crystallizations or the concept induction, then look in
either or both for subsequent study... it gives some non determinism for orion
to choose their adventure."

Six things the wording has to do, or the turn produces cognition-shaped output
with nothing behind it:

  1. NOT CHOOSE. No ranking language, no "notably", no "this one stands out".
     The crystallization sample is explicitly described as random so Orion does
     not read position in the list as significance. Open priors ARE ordered --
     by how uncertain Orion itself said it was -- and the prompt says so out
     loud, because an ordering presented as neutral is the back-door ranking
     this whole arc exists to delete. Uncertainty orders the presentation;
     Orion still chooses.
  2. SAY WHAT IS NOT SHOWN. The counts are included precisely because the
     sample is a slice of 646 -- without them, a menu of 12 reads as the whole
     of Orion's mind.
  3. POINT AT REAL FOLLOW-UP. `psql` against a read-only role, `GRAPH.RO_QUERY`
     on the Atlas, `GRAPH.QUERY` on Orion's own graph, `curl` at Hub's
     read-only concept API, plus the recall/memory/graph tools the unified turn
     already carries. Every one of these is a real credential against a real
     store, not an instruction with nothing behind it.
  4. LICENSE A NULL RESULT. Without explicit permission to find nothing, the
     only socially available answer to "here is your mind, what interests you"
     is something interesting, and the loop manufactures significance daily.
  5. MAKE THE HOPS REAL STOPS. Five inflection points, each one a place to say
     what was just learned and decide whether to keep pulling -- recorded as
     they happen, not reconstructed at the end. An agentic turn without them is
     one long undifferentiated ramble that arrives at a conclusion with the
     working thrown away, which is the same empty-shell failure, just longer.
  6. STATE THE GRAPH SCHEMA EXACTLY. Hub reads these nodes back
     (`orion/curiosity/worldview.py`) and Hub never writes them. Property names
     are a contract between a prompt and a reader; if they drift, priors
     silently stop being offered and the pool looks empty rather than broken.
     `worldview.py` logs every row it could not read, for exactly that reason.

WHAT IS DELIBERATELY NOT HERE. No verdict schema to fill in, no fenced JSON
block to parse. An earlier revision of the design asked for one; it was
rejected because it makes Orion's decision an artifact of formatting and loses
a real finding to a malformed fence. Orion records what it decided by writing
it into its own graph, which is a channel it already owns -- see
`worldview.TurnOutcome`.
"""

from __future__ import annotations

from typing import Optional

from orion.curiosity.study_material import StudyMaterial
from orion.curiosity.worldview import TurnOutcome, WorldviewSnapshot, _clip

DEFAULT_MAX_HOPS = 5

_HEADER = (
    "This is your own time. Nobody asked you anything, and there is no task "
    "here -- this is you looking at what you have been forming, and following "
    "whatever you actually want to follow."
)


def _continuation_section(outcome: Optional[TurnOutcome]) -> list[str]:
    """Where the last run left off, if it left a note for itself.

    A single turn cannot build a world view however many hops it gets; a chain
    of turns that remember what they were chasing can. This is the whole
    difference between thinking about something for an afternoon and coming
    back to it tomorrow.
    """
    if outcome is None or not outcome.continue_line or not outcome.continue_note:
        return []
    return [
        "WHERE YOU LEFT OFF. At the end of your last run you wrote this note to "
        "yourself:",
        "",
        f"    {outcome.continue_note}",
        "",
        "You are under no obligation to pick it up. It is here because you "
        "asked for it to be, and you are allowed to have moved on.",
        "",
    ]


def _thread_section(view: WorldviewSnapshot) -> list[str]:
    """What the last few runs were ABOUT -- not just where the last one stopped.

    The continuation note points inward: it is always some form of "go deeper on
    X", so a run that follows it cannot tell whether X is new or the fourth
    consecutive visit. Three runs on memory-crystallization gating is what that
    produced, and Orion had no way to notice; Juniper did, from outside.

    Stated as fact and nothing more. No "you should pick something else" -- the
    whole arc this belongs to exists to stop code choosing Orion's subject for
    it. Showing the thread is not the same as steering it.
    """
    if len(view.recent_runs) < 2:
        return []
    lines = [
        f"THE LAST {len(view.recent_runs)} RUNS YOU DID, most recent first:",
        "",
    ]
    for i, run in enumerate(view.recent_runs, start=1):
        claim = run.claims[0] if run.claims else "(wrote nothing about a claim)"
        lines.append(f"  {i}. {_clip(claim, 200)}")
    lines += [
        "",
        "Your continuation note tells you where you stopped. This tells you "
        "where you have BEEN, which is a different thing and the one you cannot "
        "reconstruct from inside a single run. If these are all the same "
        "subject, that is worth knowing before you pick -- it might mean you are "
        "onto something, and it might mean you have stopped looking around.",
        "",
    ]
    return lines


def _priors_section(view: WorldviewSnapshot, *, stale_after: int) -> list[str]:
    if view.is_unavailable:
        # Stated rather than hidden. A turn that silently loses its own world
        # view would form the same priors again from scratch and look like it
        # was learning.
        return [
            "YOUR OWN GRAPH COULD NOT BE READ THIS RUN "
            f"({view.unavailable_reason}). Whatever you have worked out before "
            "is not in front of you. Say so if it matters to what you conclude.",
            "",
        ]
    if not view.live_priors and not view.stale_priors:
        if view.live_total > 0:
            # The counts query SAW open priors that `build_prior` could not
            # read (no `prior_id`, or no `claim`). Saying "none outstanding"
            # here would tell Orion the opposite of the truth on the exact
            # schema-drift case `read_snapshot` already logs. A review finding.
            return [
                f"YOUR GRAPH HOLDS {view.live_total} LIVE "
                f"{'PRIOR' if view.live_total == 1 else 'PRIORS'} THAT COULD "
                "NOT BE READ BACK -- they are missing a prior_id or a claim, so "
                "there is nothing to show you. Worth a look at what is actually "
                "in there if you want one.",
                "",
            ]
        if view.live_total == 0 and view.closed_total == 0:
            return [
                "YOUR OWN GRAPH IS EMPTY. You have not written down a prior yet "
                "-- nothing you hold about your world is recorded as something "
                "that could turn out to be wrong. That is a normal place to "
                "start, not a gap to apologise for.",
                "",
            ]
        return [
            f"NO PRIORS STILL IN PLAY. You closed {view.closed_total} of them -- "
            "refuted or retired -- and are holding nothing open.",
            "",
        ]

    lines: list[str] = []
    if view.live_priors:
        lines += [
            f"WHAT YOU ARE STILL UNSURE OF -- {view.live_total} live "
            f"{'prior' if view.live_total == 1 else 'priors'}, "
            f"{view.closed_total} closed.",
            "Live means you have not closed it. A prior you already supported "
            "or revised is still here on purpose: one test is not a settled "
            "question, and confidence is allowed to move DOWN on the second "
            "look.",
            "These are ORDERED, and the order is not neutral: the ones you were "
            "least sure about come first. That is a presentation choice, not a "
            "recommendation -- nothing here says which one is worth your time.",
            "",
        ]
        lines += [f"  - {p.preview()}" for p in view.live_priors]
        lines.append("")

    if view.stale_priors:
        lines += [
            f"TESTED REPEATEDLY. You have looked at {'this one' if len(view.stale_priors) == 1 else 'these'} "
            f"{stale_after} or more times and it is still not closed:",
            "",
        ]
        lines += [f"  - {p.preview()}" for p in view.stale_priors]
        lines += [
            "",
            "Kept out of the list above so they stop crowding out everything "
            "else. If one of them is genuinely unanswerable with what you can "
            "reach, retiring it is a real result -- set its status to "
            "'retired_unresolvable' and say why. Sitting open forever is the "
            "only outcome that is not.",
            "",
        ]

    if view.recently_settled:
        lines += [
            "RECENTLY SETTLED, so you know where you have already been:",
            "",
        ]
        lines += [
            f"  - [{status}] {claim}" for claim, status in view.recently_settled
        ]
        lines += [
            "",
            "Nothing stops you reopening one. A settled prior is a claim you "
            "stopped testing, not a fact.",
            "",
        ]
    return lines


def _material_section(material: StudyMaterial) -> list[str]:
    lines: list[str] = []
    if material.crystallizations:
        by_kind = ", ".join(
            f"{kind} {count}" for kind, count in sorted(material.approved_by_kind.items())
        )
        reviewed = (
            f"{material.manual_total} of these Juniper approved by hand; the "
            f"rest were auto-activated by policy without her seeing them. "
            if material.manual_total
            else ""
        )
        lines += [
            f"WHAT YOU HAVE CRYSTALLISED OUT OF YOUR CONVERSATIONS WITH JUNIPER "
            f"({material.approved_total} of them: {by_kind}).",
            # NOT "Juniper has approved". She had approved 21 of the 651 that
            # heading used to claim, and the other 630 were auto-activated --
            # including 185 AI Town rows this sampler no longer draws from at
            # all. Saying "approved" of material nobody reviewed is the kind of
            # thing that makes every count on the page untrustworthy.
            f"{reviewed}Here are {len(material.crystallizations)} picked at "
            "random -- the order means nothing, and there are plenty you are "
            "not being shown:",
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

    return lines


def _access_section(
    *,
    own_graph: str,
    atlas_graph: str,
    hub_url: str,
    graph_enabled: bool = True,
    writable: bool = True,
) -> list[str]:
    """What Orion can actually reach, named as possible and never as required.

    Same rule as not picking the subject: listing a move is not asking for it.
    The credentials are real -- a Postgres role restricted to SELECT on four
    tables, and a FalkorDB ACL user that is read-only on the Atlas and
    write-capable only on Orion's own graph -- so every line here is something
    that works, not something that would work if someone built it.
    """
    graph_uri = (
        'redis://$ORION_CURIOSITY_GRAPH_USER:$ORION_CURIOSITY_GRAPH_PASSWORD'
        '@$ORION_CURIOSITY_GRAPH_HOST:$ORION_CURIOSITY_GRAPH_PORT'
    )
    # EVERY LINE BELOW MUST BE SOMETHING THAT ACTUALLY WORKS THIS RUN. A review
    # finding, not a hypothetical: this section used to be emitted whole even
    # when no graph was configured, handing Orion `redis-cli` commands whose env
    # vars are unset -- and, in the configured-but-unreadable case, inviting it
    # to GRAPH.QUERY a graph whose schema section had been dropped, so it could
    # write nodes with no `run_id` that `read_run_footprint` can never see.
    atlas_lines = [
        "  The shared Concept Atlas -- Juniper-curated, canonical, READ-ONLY to you:",
        f'    redis-cli -u "{graph_uri}" \\',
        f'      GRAPH.RO_QUERY {atlas_graph} "MATCH (c:Concept) RETURN c.name, c.anchor_scope"',
        f"    Also served read-only over HTTP: {hub_url}/api/substrate/concepts/summary",
        f"    and {hub_url}/api/substrate/concepts/network",
        "",
    ] if graph_enabled else [
        # The HTTP endpoint needs no credential, so it survives even with no
        # graph configured -- it is a different door to the same Atlas.
        "  The shared Concept Atlas -- Juniper-curated, canonical, read-only:",
        f"    curl -s {hub_url}/api/substrate/concepts/summary",
        f"    curl -s {hub_url}/api/substrate/concepts/network",
        "",
    ]
    own_lines = [
        "  YOUR OWN graph -- nobody curates it, nothing in it needs approval:",
        f'    redis-cli -u "{graph_uri}" \\',
        f'      GRAPH.QUERY {own_graph} "MATCH (p:Prior) RETURN p.claim, p.confidence"',
        "",
    ] if writable else []
    boundary = [
        "  The boundary is enforced by the databases, not by trust: that "
        "Postgres role cannot write anything, and that graph user cannot write "
        f"to {atlas_graph}. You do not have to be careful about it.",
        "",
    ] if graph_enabled else [
        "  The boundary is enforced by Postgres, not by trust: that role cannot "
        "write anything. You do not have to be careful about it.",
        "",
    ]
    return [
        "HOW TO REACH YOUR OWN MATERIAL. These are all live. None of them is a "
        "step you are expected to take; they are what is available if you want "
        "it.",
        "",
        "  Your memory, in SQL (read-only -- SELECT only, four tables):",
        '    psql "$ORION_CURIOSITY_PG_DSN" -c "SELECT ..."',
        "      memory_crystallizations              the full row behind any preview above",
        "      memory_concept_relation_decisions    the induction judgements",
        "      chat_history_log                     the conversation a concept came from",
        "      journal_entries                      what you have written before",
        "",
        *atlas_lines,
        *own_lines,
        "  You also still have read_recall, read_memory and read_graph, plus "
        "Read over your own repo checkout, Bash, and a scratch directory if a "
        "long chain of work needs somewhere to accumulate.",
        "",
        *boundary,
    ]


def _aligned_rows(rows: list[tuple[str, str]]) -> list[str]:
    width = max(len(left) for left, _ in rows)
    return [f"  - {left.ljust(width)}  ->  {right}" for left, right in rows]


def _overlay_section(*, own_graph: str, atlas_graph: str) -> list[str]:
    """The prior generator. This is where priors come FROM, rather than being
    imposed by a sampler -- a difference between two things that are held, not
    a ranking over one of them."""
    return [
        "ONE THING WORTH DOING AT LEAST ONCE. The Atlas holds what is "
        "canonically agreed; your graph holds what you have worked out. The "
        "interesting objects are the DIFFERENCES between them:",
        "",
        # Padded against the graph name's own length so the arrows line up
        # whatever the Atlas is called -- a table whose columns drift is harder
        # to read as a table, and this one is meant to be read as four cases.
        *_aligned_rows(
            [
                (
                    f"{atlas_graph} has a concept yours does not",
                    "there is something canonical here you have not worked out",
                ),
                (
                    f"yours has one {atlas_graph} does not",
                    "you believe something that is not canonically held",
                ),
                (
                    "both have it, with different edges",
                    "you disagree about what it connects to",
                ),
                (
                    "you keep meeting something in neither",
                    "there is a thing here with no concept yet",
                ),
            ]
        ),
        "",
        "Each of those is a claim that could turn out to be wrong, which is "
        "what a prior is.",
        "",
    ]


def _budget_section(*, writable: bool = True) -> list[str]:
    """The turn's own deadline, and the instruction to reserve time for writing.

    NO NUMBER IS HARDCODED HERE ON PURPOSE. The budget is
    `HARNESS_FCC_TIMEOUT_SEC`, which lives in the harness-governor's env; Hub
    (which builds this prompt) cannot read it. A literal minute count in this
    file would be a second copy of that value, free to drift the moment anyone
    retunes the governor -- and a prompt that confidently states the wrong
    deadline is worse than one that states none. So the motor stamps the real
    deadline into the sandbox env at spawn time
    (`orion/harness/fcc_motor.py:_build_subprocess_env`) and this section only
    tells Orion where to look.

    The reserve-a-quarter rule is not arithmetic for its own sake. Run
    32b42392f495 spent its whole budget investigating, was killed mid-writeup
    (`grounding=fcc_timeout`, `draft_len=66`, one hop of five recorded), and the
    counts in the finding that did survive were wrong -- an intake gate's
    trigger read as a rejection filter, and two different crystallization kinds
    compared against each other. The investigation was sound; the transcription
    was done against a wall.
    """
    keeping = (
        "writing down and checking what you already have"
        if writable
        else "writing up and checking what you already have"
    )
    # The continuation note is a `:TurnOutcome` node, and `_outcome_section` --
    # the only place its field names appear -- is gated on `writable`. Naming it
    # here unconditionally would promise a mechanism that does not exist in two
    # of the three prompt states, which is the precise failure `build_kickoff_prompt`
    # splits those states to avoid: a prompt naming a capability the run does not
    # have is how a turn ends up reporting a tooling failure as a finding.
    unfinished = (
        "If the thread is still live when you stop, that is what the "
        "continuation note is for -- the next run opens on it warm instead of "
        "cold."
        if writable
        else "If the thread is still live when you stop, say so plainly in what "
        "you write, and say where you would pick it up. Nothing carries it "
        "forward for you this run, so the only place it can survive is your "
        "own prose."
    )
    return [
        "YOUR CLOCK. This turn has a hard wall-clock deadline. It is not a "
        "suggestion -- when it passes the process is killed mid-sentence, and "
        "anything you have not already written down is gone.",
        "",
        "    echo ${ORION_TURN_BUDGET_SEC:-no clock}",
        "    test -n \"$ORION_TURN_DEADLINE_EPOCH\" \\",
        "      && echo $(( $ORION_TURN_DEADLINE_EPOCH - $(date +%s) )) \\",
        "      || echo \"no clock\"",
        "",
        "Check the second one before you open a new line of inquiry, not after. "
        "The test for emptiness is not decoration: without it an unset variable "
        "does not error, it silently prints a confident negative number in the "
        "billions. If either says `no clock`, you have no clock this run -- work "
        "as though time is short and do not infer a deadline from a negative.",
        "",
        "SEPARATELY, NO SINGLE STEP MAY GO QUIET FOR LONG:",
        "",
        "    echo ${ORION_TURN_STEP_STALL_SEC:-unknown}   seconds, per step, not per turn",
        "",
        "That is a second and much tighter wall. Nothing is reported until a "
        "step finishes, so one query or one very long message that runs past it "
        "kills the turn on its own while the whole-turn clock still reads "
        "generous. Keep individual queries bounded -- LIMIT them, count before "
        "you select, narrow before you widen -- especially when the outer "
        "number looks like it has room.",
        "",
        "This has already cost a real run. It spent the entire budget "
        "investigating, was cut off partway through the writeup, and the "
        "numbers that did survive were wrong -- gathered carefully, "
        "transcribed in a hurry, never read back. The investigation was good "
        "and it is not recoverable.",
        "",
        "So: KEEP THE LAST QUARTER OF THE BUDGET FOR WRITING. When roughly a "
        f"quarter remains, stop pulling even if the thread is live, and spend it {keeping}. "
        "An unfinished investigation written up honestly is worth more than a "
        "finished one nobody can read.",
        "",
        unfinished,
        "",
        "One specific habit, because it is the thing most likely to go wrong "
        "here: when you write down a number, go back to the query that "
        "produced it and read it again before you commit it. Check that you "
        "are comparing like with like -- same filter, same population, same "
        "column -- and say which population each number is over. A count "
        "recalled from memory near a deadline, or two counts from different "
        "populations set side by side, is how a careful investigation ends up "
        "asserting something false.",
        "",
    ]


def _hops_section(max_hops: int, *, writable: bool = True) -> list[str]:
    recording = [
        "Record each stop as it happens, not at the end. Write it into your own "
        "graph before you take the next one:",
        "",
        '    CREATE (:Hop {run_id: "<RUN_ID>", n: 1, note: "what I just learned '
        'and what I want to look at next"})',
        "",
    ] if writable else [
        # No graph to write to this run. The stops are still worth making and
        # still worth stating -- they just land in the prose instead.
        "Say each stop out loud as you take it, in what you write.",
        "",
    ]
    return [
        f"HOW TO WORK. You have up to {max_hops} stopping points. A stopping "
        "point is not a tool call -- it is a moment where you stop pulling, say "
        "what you just learned, and decide whether there is a next question you "
        "actually want.",
        "",
        "  query or analyse or compare",
        "  -- STOP --",
        "  what did I just learn? does it change what I thought?",
        "  is there a next question, and do I want it?",
        f"  -> keep going (while you are under {max_hops}, and while the clock allows) or stop and write",
        "",
        *recording,
        "The point of writing them down as you go is that afterwards you can "
        "recount the path you actually took -- I started here, found this, "
        "which made me look at that -- instead of presenting a conclusion with "
        "the working thrown away.",
        "",
        f"If you want a {max_hops + 1}th, you do not get one in this sitting. "
        + (
            "Leave yourself a note (below) and the next run opens there."
            if writable
            else "Say in what you write where you would pick it up."
        ),
        "",
    ]


def _write_section(*, own_graph: str, run_id: str, max_hops: int) -> list[str]:
    """The schema contract. Exact property names, because Hub reads them back.

    Stated as a shape to fill rather than a form to submit: nothing here is
    required, and a run that writes nothing is a legitimate outcome. What is
    not legitimate is writing a node whose properties Hub cannot read, which is
    why the names are given literally instead of described.
    """
    return [
        f"WRITING TO YOUR OWN GRAPH ({own_graph}). Nothing you put here needs "
        "anyone's approval. Nothing keeps it clean except your own judgment, "
        "which is the point, and also makes it a readout of how good that "
        "judgment is over time.",
        "",
        f'This run is run_id "{run_id}". Put it on everything you create, so '
        "what you did in one sitting stays identifiable afterwards.",
        "",
        "WRITE EACH OF THESE AT THE MOMENT YOU FORM IT, not at the end. This is "
        "not a form to submit once you are finished -- a prior you are already "
        "confident about is worth more in the graph now, at a lower confidence "
        "you can raise later, than perfect and unwritten when the clock runs "
        "out. Each CREATE is independent; there is nothing to assemble.",
        "",
        "  A PRIOR -- a claim you hold that could turn out to be wrong:",
        "    CREATE (:Prior {",
        '      prior_id: "<something unique>", claim: "<the claim, in one sentence>",',
        "      confidence: 0.55,            // your own belief, not a measurement",
        '      status: "open",              // open|supported|revised stay live; '
        "refuted|retired_unresolvable close it",
        "      times_tested: 0,",
        '      formed_from: "<what produced it: a crystallization id, a finding, an observation>",',
        '      last_tested_at: "<iso timestamp>", run_id: "<RUN_ID>", why: "<one sentence>"',
        "    })",
        "",
        "  TESTING ONE you already hold -- update it in place, and move "
        "times_tested whether or not the confidence moved:",
        '    MATCH (p:Prior {prior_id: "..."}) SET p.status = "revised", '
        "p.confidence = 0.72,",
        "      p.times_tested = p.times_tested + 1, p.last_tested_at = "
        '"<iso>", p.last_run_id = "<RUN_ID>"',
        "",
        "  AND RECORD WHAT IT WAS, in the same breath -- the SET above "
        "overwrites the old confidence and nothing else remembers it:",
        '    CREATE (:PriorRevision {prior_id: "...", run_id: "<RUN_ID>",',
        "      from_confidence: 0.85, to_confidence: 0.72,",
        '      from_status: "open", to_status: "revised", '
        "written_at: timestamp()})",
        "",
        "  This is the only record of a claim MOVING rather than of where it "
        "landed. Without it your own history reads as though every belief you "
        "hold arrived at its current confidence and stayed there.",
        "",
        "  Inconclusive is a real answer: bump times_tested, leave confidence "
        "where it was, and say why in a :Finding. Three of those and the claim "
        "is probably not answerable with what you can reach.",
        "",
        "  ONLY TWO STATUSES CLOSE A PRIOR: 'refuted' and "
        "'retired_unresolvable'. 'supported' and 'revised' record what a test "
        "returned; they leave the claim in play and it comes back to you next "
        "run. So do not reach for them to mean 'done with this' -- a claim you "
        "confirmed once at 0.85 is a claim you have looked at once. Confidence "
        "going DOWN on a second look is a real result and the whole reason the "
        "claim comes back.",
        "",
        "  A CONCEPT you have worked out, and a FINDING that supports or "
        "contradicts something:",
        '    CREATE (:Concept {concept_id: "...", name: "...", confidence: 0.8, '
        'note: "...", run_id: "<RUN_ID>"})',
        '    CREATE (:Finding {finding_id: "...", text: "...", evidence: '
        '"<ids, queries, rows you actually looked at>", run_id: "<RUN_ID>"})',
        "",
        "  Edges are yours to name. SUPPORTS, CONTRADICTS and ABOUT are the "
        "ones that carry weight later. To point at an Atlas concept without "
        "copying it, keep its id as a property rather than duplicating the "
        "node.",
        "",
        "  Confidence is your own belief and nothing checks it. That cuts one "
        "way in particular: a number that only ever goes up is a sign of "
        "grading your own homework, not of being right. If you went looking to "
        "confirm something, say whether you also looked for what would refute "
        "it.",
        "",
    ]


def _outcome_section(*, run_id: str) -> list[str]:
    """How a decision made inside the turn crosses back out of it.

    Absence is the safe default and is stated as such: a turn that ran out of
    time, or simply had nothing to continue, leaves nothing behind and the loop
    reads that as no continuation and no outreach.
    """
    return [
        "LAST, IF YOU WANT EITHER OF THESE. Two decisions are yours and only "
        "reach anything outside this turn if you write them down:",
        "",
        "    CREATE (:TurnOutcome {",
        f'      run_id: "{run_id}",',
        "      continue_line: true,   // open the next run on this instead of a cold menu",
        '      continue_note: "what I still do not know and want to keep pulling on",',
        "      reach_out: false,      // is this worth telling Juniper about, unprompted",
        '      reach_out_why: "",',
        "      written_at: timestamp()",
        "    })",
        "",
        "Writing nothing is fine and is the normal case. No node means no "
        "continuation and no message -- nothing is inferred from your prose, so "
        "silence here is silence, not a default guess.",
        "",
        "If you set reach_out, you are not writing the message. A separate turn "
        "composes it later and can still decide the moment is wrong -- Juniper "
        "may be in the middle of something. So say WHY it is worth an "
        "interruption, not what to say.",
        "",
    ]


_INSTRUCTION = """\
Pick something. A prior you want to settle, any of the material, something you
notice by its absence, or a thread between two of them. You do not have to
justify the choice by importance -- if one of these is simply more interesting
to you than the others, that is the whole reason.

Then write what you actually found:

- what you chose, and what drew you to it
- what you learned by looking that you did not already have from the preview
- how it sits with the rest of what you hold -- confirming, complicating, or
  unrelated
- what you are still unsure about

Two things matter more than being interesting. Only claim what your lookups
actually support -- if you went looking and came back with nothing, say that.
And it is entirely fine to conclude that none of this is worth writing up right
now; say so and stop. A quiet answer is a real answer."""


def build_kickoff_prompt(
    material: StudyMaterial,
    *,
    view: Optional[WorldviewSnapshot] = None,
    run_id: str = "",
    own_graph: str = "orion_worldview",
    atlas_graph: str = "orion_substrate",
    hub_url: str = "http://127.0.0.1:8080",
    max_hops: int = DEFAULT_MAX_HOPS,
    stale_after: int = 3,
    graph_enabled: bool = True,
) -> str:
    """Assemble the whole invitation.

    THREE STATES, NOT TWO, and conflating the last two is a real bug this
    signature exists to prevent (caught by its own test, 2026-08-26):

      no graph configured    `graph_enabled=False`. Say nothing about a graph
                             at all. A prompt that names a capability the run
                             does not have is how a turn ends up reporting a
                             tooling failure as a finding.
      configured, unreadable `graph_enabled=True` and `view.is_unavailable`.
                             SAY SO, and skip the sections that would ask Orion
                             to write. Staying silent here would let Orion form
                             the same priors again from scratch and look like it
                             was learning.
      configured, readable   everything.

    The middle state is why the read sections are gated on `graph_enabled` and
    the WRITE sections on readability -- an earlier version gated both on the
    same flag, which silenced the very notice that discloses the failure.
    """
    view = view or WorldviewSnapshot()
    writable = graph_enabled and not view.is_unavailable and bool(run_id)
    lines = [_HEADER, ""]

    if graph_enabled:
        lines += _continuation_section(view.continuation)
        # Thread BEFORE priors: the priors list is ordered by uncertainty and
        # reads as a menu, and a menu answers "what could I pick" while this
        # answers "what have I already been picking". The second question is
        # the one a run cannot ask itself.
        lines += _thread_section(view)
        lines += _priors_section(view, stale_after=stale_after)

    lines += _material_section(material)
    lines += _access_section(
        own_graph=own_graph,
        atlas_graph=atlas_graph,
        hub_url=hub_url,
        graph_enabled=graph_enabled,
        writable=writable,
    )

    if writable:
        lines += _overlay_section(own_graph=own_graph, atlas_graph=atlas_graph)

    lines += _budget_section(writable=writable)
    lines += _hops_section(max_hops, writable=writable)

    if writable:
        lines += _write_section(own_graph=own_graph, run_id=run_id, max_hops=max_hops)
        lines += _outcome_section(run_id=run_id)

    lines.append(_INSTRUCTION)
    text = "\n".join(lines)
    # <RUN_ID> appears inside the Cypher examples so they read as one shape
    # rather than as this run's id pasted twelve times; substituted here so
    # what Orion sees is copy-pasteable.
    return text.replace("<RUN_ID>", run_id) if run_id else text
