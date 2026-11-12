"""Consult outcomes as evidence on Orion's beliefs: what a curiosity run is asked to write, and
how Hub checks afterwards that it did. Pure: no I/O.

Design: docs/superpowers/specs/2026-11-11-consult-claude-before-juniper-design.md, section 5,
and "Patch 1d" in the same file.

Orion alone writes ``orion_worldview`` (the PR #2199 boundary); Hub only reads it. So a consult
outcome reaches a belief in three steps:

1. Offer (curiosity kickoff): the outcome, the beliefs it is most likely about (named by its
   source), and ONE fixed finding id to write: ``consult:<consult_id>:answer`` for an answered
   consult, ``consult:<consult_id>:decision`` for Juniper's answer to a brief.
2. Orion writes ``MERGE (f:Finding {finding_id: <that id>})`` and joins it to a ``:Prior``
   (one it holds, or a new one). MERGE, so a re-offer cannot write a second copy.
3. Read-back (turn end): one fixed read of every consult finding joined to a prior. Found: Hub
   stamps the consult row with the finding and the priors. Not found: the outcome is offered to
   a later run, at most ``MAX_BELIEF_OFFERS`` times in all.

Why two finding ids, not one: Juniper's answer says what she wants, not what is true. Only a
later measured outcome (Patch 2, its own finding) is evidence about the claim itself. Keeping
them as separate nodes keeps Orion from learning "what Juniper approves" as "what is true".
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

from orion.curiosity.worldview import LABEL_FINDING, LABEL_PRIOR, LIVE_PRIORS_LIMIT, Prior

FINDING_PREFIX = "consult:"
# How many curiosity runs are shown one outcome before Hub stops asking. A run can be cut short
# or pick other work; three tries is enough to say "Orion chose not to record it".
MAX_BELIEF_OFFERS = 3
# At most this many source beliefs are named per consult: a menu, not the whole graph.
MAX_SOURCE_PRIORS = 5

# Fixed reads (no id is ever spliced into Cypher); filtered in Python.
PRIOR_RUNS_CYPHER = (
    f"MATCH (p:{LABEL_PRIOR}) RETURN p.prior_id AS prior_id, p.run_id AS run_id, "
    f"p.last_run_id AS last_run_id LIMIT {LIVE_PRIORS_LIMIT}"
)
CONSULT_EVIDENCE_CYPHER = (
    f"MATCH (f:{LABEL_FINDING})-[e]-(p:{LABEL_PRIOR}) WHERE f.finding_id STARTS WITH '{FINDING_PREFIX}' "
    f"RETURN f.finding_id AS finding_id, f.run_id AS run_id, type(e) AS edge, p.prior_id AS prior_id "
    f"LIMIT {LIVE_PRIORS_LIMIT}"
)


def finding_id_for(consult_id: str, ending: Optional[str]) -> str:
    """The one finding id a run writes for this consult's outcome."""
    kind = "answer" if ending == "answered" else "decision"
    return f"{FINDING_PREFIX}{consult_id}:{kind}"


def source_prior_ids(row: Mapping[str, Any], prior_runs: Iterable[Mapping[str, Any]]) -> list[str]:
    """The beliefs a consult is most likely about, named by its source. A stuck-prior consult names
    its prior. A curiosity-run consult names the priors that run formed or last tested. A
    self-monitor consult names none (Orion makes or picks one)."""
    ref = str(row.get("source_ref") or "")
    kind = str(row.get("source_kind") or "")
    if kind == "stuck_prior" and ref.startswith("prior:"):
        return [ref[len("prior:"):]]
    if kind != "curiosity_run" or not ref.startswith("curiosity_run:"):
        return []
    run_id = ref[len("curiosity_run:"):]
    out: list[str] = []
    for r in prior_runs:
        pid = str(r.get("prior_id") or "").strip()
        if pid and pid not in out and run_id in (str(r.get("run_id") or ""), str(r.get("last_run_id") or "")):
            out.append(pid)
    return out[:MAX_SOURCE_PRIORS]


def recorded_evidence(
    consult_rows: Sequence[Mapping[str, Any]], evidence_rows: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[str, list[str]]]:
    """consult_id -> (finding_id, prior_ids joined to it), for each consult whose expected finding
    exists in the graph joined to at least one prior. A consult with no such finding is absent."""
    wanted = {finding_id_for(str(r["consult_id"]), r.get("ending")): str(r["consult_id"]) for r in consult_rows}
    out: dict[str, tuple[str, list[str]]] = {}
    for e in evidence_rows:
        fid = str(e.get("finding_id") or "")
        pid = str(e.get("prior_id") or "").strip()
        cid = wanted.get(fid)
        if cid is None or not pid:
            continue
        _, priors = out.setdefault(cid, (fid, []))
        if pid not in priors:
            priors.append(pid)
    return out


def _quote(text: Any, limit: int = 300) -> str:
    return " ".join(str(text or "").split())[:limit]


def _said(row: Mapping[str, Any]) -> str:
    decision = str(row.get("decision") or "")
    if decision == "no_because":
        return f"no, because: {_quote(row.get('decision_reason'))}"
    return decision or "(no answer recorded)"


def consult_outcomes_section(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Kickoff lines: each consult outcome, the beliefs it is likely about (``row['belief_priors']``:
    list of Prior, or None when the graph could not be read), and the exact writes. Empty when
    nothing was claimed."""
    if not rows:
        return []
    lines = [
        "## Your consults with Claude",
        "These consults ended and you have not yet recorded them on a belief. For each, write the "
        "finding below with its exact finding_id, and join it to the :Prior it bears on (one listed, "
        "another you hold, or a new one). Afterwards Hub reads back whether that finding exists and "
        "which prior it is joined to. One you leave out is offered to a later run, "
        f"{MAX_BELIEF_OFFERS} times at most.",
        "",
    ]
    for n, row in enumerate(rows, start=1):
        cid = str(row.get("consult_id"))
        ending = str(row.get("ending") or "")
        fid = finding_id_for(cid, ending)
        question = _quote(row.get("question"), 240)
        lines.append(f"  {n}. Consult {cid} (question: {question})")
        if ending == "answered":
            lines += [
                f"     Answered by you and Claude: {_quote(row.get('answer'), 600)}",
                "     This is evidence about the claim. Test the prior it bears on with the testing "
                "MATCH ... SET under WRITING TO YOUR OWN GRAPH (times_tested + 1, a :PriorRevision "
                "if the confidence moves), and join the finding with SUPPORTS or CONTRADICTS.",
            ]
        else:
            lines += [
                f"     It went to Juniper as a brief ({ending}). She said {_said(row)}.",
                "     Her answer is evidence about what she wants done, NOT proof the claim is true. "
                "Do not move the prior's confidence on it. Join the finding with ABOUT. Only the "
                "measured outcome after a change ships can move the confidence.",
            ]
        priors = row.get("belief_priors")
        if priors is None:
            lines.append("     The beliefs it came from could not be read this time; pick or make one.")
        elif not priors:
            lines.append("     No belief is named by its source; pick one you hold or make one.")
        else:
            lines.append("     Beliefs its source formed or tested:")
            lines += [f"    - {p.preview()}" for p in priors if isinstance(p, Prior)]
        edge = "SUPPORTS" if ending == "answered" else "ABOUT"
        lines += [
            "     Write (MERGE, so writing it twice is harmless):",
            f'       MERGE (f:Finding {{finding_id: "{fid}"}}) ON CREATE SET f.text = "<what it says, '
            f'in one sentence>", f.evidence = "orion_consult {cid}", f.run_id = "<RUN_ID>"',
            f'       MATCH (f:Finding {{finding_id: "{fid}"}}), (p:Prior {{prior_id: "..."}}) '
            f'MERGE (f)-[e:{edge}]->(p) ON CREATE SET e.run_id = "<RUN_ID>"'
            + ("  (CONTRADICTS instead when it goes against the claim)" if edge == "SUPPORTS" else ""),
            "",
        ]
    return lines
