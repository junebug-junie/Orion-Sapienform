"""The curiosity journal entry -- Orion's own written result of a run.

Moved VERBATIM out of `services/orion-hub/scripts/curiosity_investigation.py`
on 2026-09-06 so `orion-durable-runs` can write the same entry when it owns
the run's tail (journal is a node of the durable graph); Hub imports these
names from here unchanged. Pure functions, no Hub dependency: `material` is
anything with the five attributes the entry reads from `StudyMaterial`
(`MaterialCounts` below builds one from counts alone, which is all the runner
has once the cards themselves stayed in Hub).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from orion.curiosity.study_material import StudyMaterial
from orion.curiosity.worldview import FindingConnectivity
from orion.journaler.schemas import JournalEntryWriteV1

INVESTIGATION_TAG = "curiosity_investigation"
OUTREACH_TAG = "curiosity_outreach"
_JOURNAL_SOURCE_KIND = "self_study"
_AUTHOR = "orion"

def format_footprint(footprint: dict[str, int]) -> str:
    """`{'Prior': 2, 'Hop': 5}` -> `"Hop 5, Prior 2"`. Empty string for {}."""
    return ", ".join(f"{label} {n}" for label, n in sorted(footprint.items()))


def format_evidence(
    evidence: Optional[FindingConnectivity], *, graph_configured: bool = True
) -> str:
    """A named function rather than an inline conditional so the cases that
    matter are testable. `None` renders "unreadable", NOT "0/0 joined".

    Those two would be indistinguishable in the log while meaning opposite
    things -- "the graph did not answer" versus "Orion wrote findings and
    joined none of them" -- and the second is the live reading this metric was
    built to catch, so collapsing them would blind the one instrument watching
    for it. Same rule `format_footprint`'s caller applies one field over.

    `graph_configured=False` is the THIRD state and it is not an outage.
    `HUB_CURIOSITY_GRAPH_ORION_PASSWORD` ships blank in `.env_example`, so a
    default install has no reader at all and `_read_turn_result` short-circuits
    to all-empty. Rendering that as "unreadable" would have an operator
    watching for FalkorDB to come back from an outage that was never happening
    -- which is this field's own conflation, one level out, on the deployment
    that is most likely to be someone's first.
    """
    if not graph_configured:
        return "no graph"
    return "unreadable" if evidence is None else evidence.summary()


def build_investigation_journal_entry(
    *,
    material: StudyMaterial,
    body_text: str,
    correlation_id: str,
    run_id: str,
    harness_step_count: Optional[int] = None,
    harness_grounding_status: Optional[str] = None,
    harness_elapsed_sec: Optional[float] = None,
    harness_fcc_elapsed_sec: Optional[float] = None,
    graph_footprint: Optional[dict[str, int]] = None,
    hop_notes: Optional[list[tuple[int, str]]] = None,
    created_at: Optional[datetime] = None,
) -> JournalEntryWriteV1:
    """Orion's own written result.

    The title is deliberately NOT derived from a subject, because code no
    longer knows the subject -- Orion chose it inside the turn and it lives in
    the prose. Deriving one here would mean re-inferring Orion's choice with a
    heuristic, which is the exact move this rewrite exists to delete.

    THE FOOTPRINT IS THE EVIDENCE, and it is reported whether or not it is
    flattering. `graph_footprint` counts what Orion actually created in its own
    graph during THIS run; a run that wrote nothing says so in plain words
    rather than letting fluent prose imply that structure was formed. Same
    contract as the harness step count next to it: if Orion says it worked
    something out, there is an inspectable artifact behind the claim.

    `None` means the footprint could not be read (no graph configured, or the
    graph did not answer) and prints NOTHING, which is different from `{}`
    meaning Orion genuinely wrote nothing and saying so.
    """
    stamp = created_at or datetime.now(timezone.utc)
    offered = ", ".join(
        f"{kind} {count}" for kind, count in sorted(material.approved_by_kind.items())
    )
    lines = [body_text.strip()]

    if hop_notes:
        lines += ["", "---", "", "The path, as it was recorded at each stop:", ""]
        lines += [f"{n}. {note}" for n, note in hop_notes]

    lines += [
        "",
        f"(Offered {len(material.crystallizations)} of "
        f"{material.approved_total} approved concepts [{offered}] and "
        f"{len(material.relations)} of {material.relation_total} relation "
        "judgements, all sampled at random.",
    ]
    if harness_step_count is not None:
        # The evidence that this was a lookup and not a recollection, kept in
        # the artifact so the claim stays checkable after the fact.
        lines[-1] += (
            f" Investigated over {harness_step_count} harness steps"
            + (f", grounding: {harness_grounding_status}" if harness_grounding_status else "")
            + (
                # WHOLE-TURN WALL TIME, HUB SIDE, AND IT IS NOT THE BUDGET THE
                # `fcc_timeout` LABEL REFERS TO. Three nested deadlines are in
                # play and only the innermost one ever kills a run that gets
                # this far (all three confirmed against the live containers,
                # 2026-09-01; raised again 2026-09-03 alongside the move to the
                # slower `agent` lane -- see HARNESS_FCC_TIMEOUT_SEC's own
                # comment in orion-harness-governor/.env_example):
                #
                #   HARNESS_FCC_TIMEOUT_SEC          2400s  governor process
                #   HUB_HARNESS_GOVERNOR_RPC_TIMEOUT 2960s  hub
                #   HUB_CURIOSITY_INVESTIGATION_...  3500s  hub, this clock
                #
                # `fcc_timeout` is emitted by the GOVERNOR at 2400s
                # (`orion/harness/fcc_motor.py`), which then yields its partial
                # draft as an ordinary final frame -- which is the only reason
                # a timed-out run has a journal at all. The 3500s budget
                # structurally cannot kill a journaled run: if it fires,
                # `_generate` returns no text and `_investigate` bails at
                # `empty_generation` before anything is written. So every entry
                # carrying this number came from a turn where 3500s was slack.
                #
                # It is therefore NOT the investigation's duration. It spans
                # all four legs -- stance (<=400s), governor queue, the FCC
                # turn, and the finalize chain (<=485s) -- so up to ~885s of it
                # is provably not investigation, and the legs are not measured
                # separately anywhere. Named in the text rather than left to
                # position, because `in 3499s` sitting after "harness steps"
                # reads as the harness leg and is not.
                #
                # What it is good for: a `grounded` run's distance from the
                # 2400s FCC ceiling is real headroom, and until now the number
                # survived only for runs that FAILED to journal (logged in the
                # debug dict at `curiosity_investigation_no_text`) and was lost
                # for every run that succeeded. Read it as an upper bound on
                # the FCC leg, never as the leg itself.
                f", whole turn {harness_elapsed_sec:.0f}s "
                "(stance + harness + finalize)"
                if harness_elapsed_sec is not None
                else ""
            )
            + (
                # THE LEG THE TIMEOUT ACTUALLY GOVERNS. `whole turn` above
                # bounds it; this is it. Reported second and only when known,
                # so the difference between the two IS the stance+finalize
                # overhead and nobody has to infer it. A `grounded` run's
                # distance from HARNESS_FCC_TIMEOUT_SEC (2400s) is the real
                # headroom figure -- the thing that decides whether the budget
                # is genuinely too small or the turn simply never converged.
                f", of which harness {harness_fcc_elapsed_sec:.0f}s"
                if harness_fcc_elapsed_sec is not None
                else ""
            )
        )
    if graph_footprint is not None:
        # `{}` and `None` are DIFFERENT here, and the distinction lands in the
        # one artifact Juniper actually reads: `{}` is "Orion wrote nothing",
        # `None` is "the graph could not answer", and printing the former for
        # the latter would put a false claim about Orion's own work in its
        # journal. `read_run_footprint` keeps them apart for this reason.
        lines[-1] += (
            f" Wrote to its own graph: {format_footprint(graph_footprint)}"
            if graph_footprint
            else " Wrote nothing to its own graph this run"
        )
    lines[-1] += ".)"
    return JournalEntryWriteV1(
        created_at=stamp,
        author=_AUTHOR,
        mode="manual",
        title="Curiosity",
        body="\n".join(lines),
        source_kind=_JOURNAL_SOURCE_KIND,
        # Namespaced away from the four self-study analysis sources, whose own
        # cooldown matches on a `<source>:` prefix. Keyed on the run rather
        # than on a subject, since there is no code-known subject any more.
        source_ref=f"curiosity:{run_id}",
        correlation_id=correlation_id,
    )


class MaterialCounts:
    """A `StudyMaterial`-shaped object built from counts alone. The journal
    entry reads exactly these five attributes (`approved_by_kind`,
    `approved_total`, `relation_total`, and the lengths of `crystallizations`
    and `relations`); the runner carries the counts in
    `CuriosityMaterialBriefV1`, never the cards."""

    def __init__(
        self,
        *,
        approved_total: int,
        approved_by_kind: dict[str, int],
        crystallization_count: int,
        relation_total: int,
        relation_count: int,
    ) -> None:
        self.approved_total = int(approved_total)
        self.approved_by_kind = dict(approved_by_kind or {})
        self.crystallizations = range(max(0, int(crystallization_count)))
        self.relation_total = int(relation_total)
        self.relations = range(max(0, int(relation_count)))

