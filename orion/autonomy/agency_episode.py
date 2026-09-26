"""Read-only reconstruction of existing evidence, not a new cognition producer.

This audit deliberately cannot declare a learned or causally closed episode:
the current sources lack a precommit receipt and a later decision receipt.
Only allowlisted metadata leaves the reconstruction; prose stays at its source.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any


def stamp(value: Any) -> str | None:
    """Normalize real source timestamps; never substitute the audit clock."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)) or str(value).isdigit():
            dt = datetime.fromtimestamp(float(value) / 1000, timezone.utc)
        else:
            dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return None
        return dt.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def _ref(source: str, key: Any) -> str:
    return f"{source}:{key}"


def _link(status: str, refs: list[str], note: str) -> dict:
    return {"status": status, "refs": sorted(set(refs)), "note": note}


def _rows(bundle: dict, source: str) -> list[dict]:
    data = bundle.get(source, {})
    return data.get("rows", []) if data.get("status") == "ok" else []


def _available(bundle: dict, *sources: str) -> bool:
    return all(bundle.get(source, {}).get("status") == "ok" for source in sources)


def _presence(bundle: dict, source: str, refs: list[str], note: str) -> dict:
    return _link("observed" if refs else ("missing" if _available(bundle, source) else "unverified"), refs, note)


def _unique(rows: list[dict], key: str) -> tuple[list[dict], list[str]]:
    """Identical replays collapse; conflicting identities are never chosen arbitrarily."""
    by_id: dict[str, dict] = {}
    conflicts: set[str] = set()
    for row in rows:
        rid = str(row.get(key) or "")
        if not rid:
            conflicts.add("<missing identity>")
            continue
        if rid in by_id and json.dumps(by_id[rid], sort_keys=True, default=str) != json.dumps(row, sort_keys=True, default=str):
            conflicts.add(rid)
        by_id[rid] = row
    return [by_id[k] for k in sorted(by_id) if k not in conflicts], sorted(conflicts)


def reconstruct(bundle: dict) -> dict:
    """Rebuild from a bounded snapshot. No network, inference, writes or clock reads."""
    sources: dict[str, list[dict]] = {}
    conflicts: dict[str, list[str]] = {}
    keys = {"asks": "help_id", "graph_briefs": "brief_id", "sql_briefs": "brief_id",
            "results": "result_id", "outcomes": "id", "dispatch_frames": "frame_id",
            "proposal_frames": "frame_id", "feedback_frames": "frame_id",
            "result_sample": "result_id", "outcome_sample": "dispatch_id"}
    # Older captures predate these sources: retain their original report shape,
    # without interpreting missing capture fields as empty live queries.
    for key, identity in (("ask_commits", "help_id"), ("brief_decisions", "receipt_id")):
        if key in bundle:
            keys[key] = identity
    for source, key in keys.items():
        sources[source], bad = _unique(_rows(bundle, source), key)
        if bad:
            conflicts[source] = bad

    # Contradictory evidence is unavailable for a negative claim. Preserve the
    # input, and do not turn rejected duplicate identities into "missing".
    bundle = {**bundle, **{s: {**bundle[s], "status": "conflict"} for s in conflicts}}

    episodes = []
    for ask in sources["asks"]:
        aid = ask["help_id"]
        briefs = [b for b in sources["graph_briefs"] if b.get("help_id") == aid]
        sql = [b for b in sources["sql_briefs"] if b.get("help_id") == aid]
        refs = [_ref("graph:PeerBrief", b["brief_id"]) for b in briefs]
        consumed = [b for b in briefs if b.get("consumed") is True or b.get("consumed") == "true"]
        issues = []
        ask_time = stamp(ask.get("written_at"))
        for b in briefs:
            when = stamp(b.get("written_at"))
            if ask_time and when and when < ask_time:
                issues.append("brief_precedes_ask:" + b["brief_id"])
            if b.get("run_id") != ask.get("run_id"):
                issues.append("brief_run_mismatch:" + b["brief_id"])
            if not b.get("answers_help_id") == aid:
                issues.append("answers_edge_missing:" + b["brief_id"])
        for b in sql:
            graph = next((g for g in briefs if g["brief_id"] == b["brief_id"]), None)
            if graph and any(graph.get(k) != b.get(k) for k in ("help_id", "run_id", "peer", "status")):
                issues.append("brief_store_disagreement:" + b["brief_id"])
        episodes.append({
            "episode_id": "ask:" + aid, "lane": "contractor_ask", "verdict": "UNVERIFIED",
            "source_times": {"ask_written_at": ask_time},
            "links": {
                "concern": _link("observed", [_ref("graph:HelpRequest", aid)], "Request exists; its prose is not exported."),
                "alternatives": _link("missing", [], "HelpRequest does not record the candidate set or rejected choices."),
                "expectation_precommitted": _link("unverified", [], "Success criteria are not a dated outcome forecast or durable pre-send receipt."),
                "intervention": _link("unverified", [], "Neither a request node nor a returned brief is a delivery receipt."),
                "response": _presence(bundle, "graph_briefs", refs, "A peer response is not independent verification of its claims or closure of the concern."),
                "response_persisted": _presence(bundle, "sql_briefs", [_ref("sql:curiosity_peer_brief", b["brief_id"]) for b in sql], "SQL persistence is a separate leg of the graph/bus dual write."),
                "consumption_marker": _presence(bundle, "graph_briefs", [_ref("graph:PeerBrief", b["brief_id"]) for b in consumed], "Consumed means offered in a kickoff prompt; it does not name an executed later decision."),
                "later_choice": _link("unverified", [], "No consuming run/decision receipt in the current peer contract."),
            },
            "responses": [{"brief_id": b["brief_id"], "status": b.get("status"), "peer": b.get("peer"), "written_at": stamp(b.get("written_at"))} for b in briefs],
            "issues": sorted(issues),
        })
        episode = episodes[-1]
        commits = [c for c in sources.get("ask_commits", []) if c.get("help_id") == aid and c.get("run_id") == ask.get("run_id")]
        if commits and stamp(commits[0].get("committed_at")):
            commit = commits[0]
            episode["links"]["expectation_precommitted"] = _link("observed", [_ref("graph:PeerAskCommit", aid)], "Immutable expectation acknowledged before the peer invocation; a commit alone does not prove delivery.")
            episode["links"]["alternatives"] = _link("observed", [_ref("graph:PeerAskCommit", aid)], "The immutable request snapshot records the considered alternatives and hire_peer choice; prose remains at its source.")
            episode["source_times"]["committed_at"] = stamp(commit["committed_at"])
            episode["source_times"]["deadline_at"] = stamp(commit.get("deadline_at"))
            response_at = stamp(commit.get("responded_at"))
            episode["source_times"]["responded_at"] = response_at
            deadline = stamp(commit.get("deadline_at"))
            now = stamp(bundle.get("captured_at"))
            episode["response_window"] = ("late_returned" if deadline and response_at > deadline else "returned") if response_at else ("elapsed_without_recorded_reply" if deadline and now and now > deadline else "awaiting")
        decisions = [d for d in sources.get("brief_decisions", []) if d.get("help_id") == aid]
        verified = [d for d in decisions if d.get("status") == "attributed_self_report"]
        if verified:
            episode["links"]["later_choice"] = _link("attributed_self_report", [_ref("graph:PeerBriefDecision", d["receipt_id"]) for d in verified], "Offered brief and completed run join Orion's recorded decision; used claims also join a real Hop. This is attributed self-report, not a causal effect estimate.")
        if decisions:
            episode["decisions"] = decisions

    dispatches = sorted({str(r["dispatch_id"]) for src in ("results", "outcomes") for r in sources[src] if r.get("dispatch_id")})
    for did in dispatches:
        results = [r for r in sources["results"] if r.get("dispatch_id") == did]
        outcomes = [o for o in sources["outcomes"] if o.get("dispatch_id") == did]
        frame_ids = {r.get("frame_id") for r in results} | {o.get("dispatch_frame_id") for o in outcomes}
        frames = [f for f in sources["dispatch_frames"] if f["frame_id"] in frame_ids]
        candidates = [c for f in frames for c in f.get("candidates", []) if c.get("dispatch_id") == did]
        issues = []
        if len(frame_ids - {None}) > 1:
            issues.append("dispatch_frame_disagreement")
        candidates, bad = _unique(candidates, "dispatch_id")
        if bad:
            issues.append("conflicting_dispatch_candidate")
        candidate = candidates[0] if len(candidates) == 1 else {}
        expected = candidate.get("expected_effect")
        visual = candidate.get("cortex_verb") == "skills.imagination.render_scene.v1"
        proposal_frames = [p for p in sources["proposal_frames"] if p["frame_id"] in {f.get("source_proposal_frame_id") for f in frames}]
        proposal_ids = sorted({p for f in proposal_frames for p in f.get("proposal_ids", [])})
        source_proposal = candidate.get("source_proposal_id")
        if source_proposal and source_proposal not in proposal_ids:
            issues.append("selected_proposal_not_in_loaded_alternatives")
        for outcome in outcomes:
            if expected and (outcome.get("signal_id") != expected.get("signal_id") or outcome.get("predicted_delta") != expected.get("predicted_delta")):
                issues.append("outcome_prediction_mismatch:" + str(outcome["id"]))
            feedback = next((f for f in sources["feedback_frames"] if f["frame_id"] == outcome.get("feedback_frame_id")), None)
            if feedback is None:
                issues.append("feedback_frame_not_loaded:" + str(outcome["id"]))
            elif feedback.get("source_execution_dispatch_frame_id") != outcome.get("dispatch_frame_id"):
                issues.append("feedback_dispatch_mismatch:" + str(outcome["id"]))
            sent, observed = stamp(candidate.get("dispatched_at")), stamp(outcome.get("observed_at"))
            if sent and observed and observed < sent:
                issues.append("outcome_precedes_dispatch:" + str(outcome["id"]))
        result_times = [stamp(r.get("created_at")) for r in results]
        result_times = [t for t in result_times if t]
        frame_times = [stamp(f.get("created_at")) for f in frames]
        frame_times = [t for t in frame_times if t]
        if expected and result_times and frame_times and min(frame_times) > min(result_times):
            issues.append("expectation_frame_inserted_after_result")
        refs = [_ref("sql:substrate_execution_dispatch_frames", f["frame_id"]) for f in frames]
        episodes.append({
            "episode_id": "motor:" + did, "lane": "motor", "verdict": "UNVERIFIED",
            "outcome_path": "visual" if visual else "field_or_undeclared",
            "source_times": {"dispatch_at": stamp(candidate.get("dispatched_at")), "frame_created_at": min(frame_times, default=None), "result_created_at": min(result_times, default=None)},
            "links": {
                "alternatives": _presence(bundle, "proposal_frames", [_ref("sql:substrate_proposal_frames", f["frame_id"]) for f in proposal_frames], "Persisted proposal set; not proof every alternative reached selection."),
                "selection": _link("observed" if candidate else "unverified", refs if candidate else [], "Dispatch candidate points to its proposal and decision."),
                "expectation_recorded": _link("observed" if expected else ("missing" if candidate else "unverified"), refs if expected else [], "Field prediction on saved dispatch frame. Current visual actions deliberately omit this field; absence alone is not a scoring outage."),
                "expectation_precommitted": _link("unverified", [], "Current worker saves the frame after sending. generated_at is not a commit acknowledgment."),
                "intervention": _presence(bundle, "results", [_ref("sql:substrate_dispatch_results", r["result_id"]) for r in results], "Execution result only; status success does not prove an external outcome."),
                "field_scored_outcome": _presence(bundle, "outcomes", [_ref("sql:substrate_action_outcomes", o["id"]) for o in outcomes], "Field ledger only; current visual actions bypass this ledger. No causal attribution from an individual before/after delta."),
                "visual_outcome": _presence(bundle, "results", [_ref("sql:substrate_dispatch_results", r["result_id"]) for r in results if r.get("visual_outcome") is not None], "Reported visual outcome, possibly deferred or unknown. This audit does not independently inspect the artifact or its later perception."),
                "later_choice": _link("unverified", [], "Posteriors are read by dispatch/allocator, but these rows do not identify a later consuming decision."),
            },
            "proposal_ids": proposal_ids,
            "selected_proposal_id": source_proposal,
            "results": [{"result_id": r["result_id"], "status": r.get("status"), "visual_outcome": r.get("visual_outcome")} for r in results],
            "outcomes": [{"id": o["id"], "signal_id": o.get("signal_id"), "arm": o.get("arm"), "observed_at": stamp(o.get("observed_at")), "frame_dispatch_count": o.get("frame_dispatch_count")} for o in outcomes],
            "issues": sorted(set(issues)),
        })

    return {"report_version": "agency_episode_audit.v1", "captured_at": stamp(bundle.get("captured_at")),
            "scope": "Bounded samples, not a complete history; graph and SQL are not an atomic cross-store snapshot.",
            "sources": {s: {"status": "conflict" if s in conflicts else bundle.get(s, {}).get("status", "unavailable"), "row_count": len(sources[s])} for s in keys},
            "conflicts": conflicts, "episodes": sorted(episodes, key=lambda e: e["episode_id"])}
