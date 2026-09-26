"""Bounded, metadata-only SQL/graph reads for the agency episode audit."""
from __future__ import annotations

from datetime import datetime, timezone
import json


def collect(conn, graph, *, limit: int = 10) -> dict:
    """Accept an autocommit PostgreSQL connection and WorldviewReader.

    Refuse writable SQL sessions. Graph reader must expose GRAPH.RO_QUERY via
    WorldviewReader; the CLI constructs it directly. No production writes.
    A failed source remains unavailable, never disguised as an empty result.
    """
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    from psycopg2.extras import RealDictCursor

    with conn.cursor() as cursor:
        cursor.execute("SHOW default_transaction_read_only")
        if cursor.fetchone()[0] != "on" or not conn.autocommit:
            raise ValueError("audit requires read-only autocommit SQL connection")
    bundle = {"captured_at": datetime.now(timezone.utc).isoformat()}

    def sql(name, query, params):
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                bundle[name] = {"status": "ok", "rows": [dict(row) for row in cursor.fetchall()]}
        except Exception as exc:
            # Exception messages may contain credentials, SQL or private values.
            bundle[name] = {"status": "unavailable", "rows": [], "error_type": type(exc).__name__}

    def cypher(name, query):
        try:
            bundle[name] = {"status": "ok", "rows": graph.query(query)}
        except Exception as exc:
            bundle[name] = {"status": "unavailable", "rows": [], "error_type": type(exc).__name__}

    cypher("asks", "MATCH (h:HelpRequest) RETURN h.help_id AS help_id, h.run_id AS run_id, "
           "h.written_at AS written_at ORDER BY h.written_at DESC, h.help_id LIMIT " + str(limit))
    help_ids = [r["help_id"] for r in bundle["asks"]["rows"] if r.get("help_id")]
    if bundle["asks"]["status"] == "ok":
        from orion.curiosity.agency_episode import decision_query, validate_decision

        cypher("ask_commits", "MATCH (c:PeerAskCommit) WHERE c.help_id IN " + json.dumps(help_ids) + " "
               "RETURN c.help_id AS help_id,c.run_id AS run_id,c.committed_at AS committed_at, "
               "c.deadline_at AS deadline_at,c.responded_at AS responded_at ORDER BY c.help_id LIMIT 51")
        if len(bundle["ask_commits"]["rows"]) > limit:
            bundle["ask_commits"] = {"status": "truncated", "rows": []}
        cypher("brief_decisions", decision_query(help_ids=help_ids))
        if len(bundle["brief_decisions"]["rows"]) >= 33:
            bundle["brief_decisions"] = {"status": "truncated", "rows": []}
        else:
            bundle["brief_decisions"]["rows"] = [
                {**validate_decision(row), "receipt_id": str(row.get("run_id")) + ":" + str(row.get("brief_id"))}
                for row in bundle["brief_decisions"]["rows"]
            ]
        cypher("graph_briefs", "MATCH (b:PeerBrief) WHERE b.help_id IN " + json.dumps(help_ids) + " "
               "OPTIONAL MATCH (b)-[:ANSWERS]->(h:HelpRequest) "
               "RETURN b.brief_id AS brief_id, b.help_id AS help_id, b.run_id AS run_id, "
               "b.peer AS peer, b.status AS status, b.written_at AS written_at, "
               "b.consumed AS consumed, h.help_id AS answers_help_id ORDER BY b.brief_id LIMIT " + str(limit * 10 + 1))
        sql("sql_briefs", "SELECT brief_id,help_id,run_id,peer,status,written_at,created_at "
            "FROM curiosity_peer_brief WHERE help_id = ANY(%s) ORDER BY brief_id LIMIT %s", (help_ids, limit * 10 + 1))

    # Sample recent activity AND recent scored history. Looking only at scored
    # rows would hide a scoring outage while dispatch continues.
    sql("result_sample", "SELECT result_id,dispatch_id,frame_id,status,created_at FROM substrate_dispatch_results "
        "ORDER BY created_at DESC,result_id LIMIT %s", (limit,))
    sql("outcome_sample", "SELECT dispatch_id FROM substrate_action_outcomes ORDER BY observed_at DESC,id DESC LIMIT %s", (limit,))
    dispatch_ids = sorted({r["dispatch_id"] for key in ("result_sample", "outcome_sample") for r in bundle[key]["rows"] if r.get("dispatch_id")})
    sql("results", "SELECT result_id,dispatch_id,frame_id,status,created_at, "
        "result_json->>'visual_outcome' AS visual_outcome FROM substrate_dispatch_results "
        "WHERE dispatch_id = ANY(%s) ORDER BY result_id LIMIT %s", (dispatch_ids, limit * 10 + 1))
    sql("outcomes", "SELECT id,dispatch_id,dispatch_frame_id,feedback_frame_id,signal_id,predicted_delta,"
        "observed_at,arm,frame_dispatch_count FROM substrate_action_outcomes "
        "WHERE dispatch_id = ANY(%s) ORDER BY id LIMIT %s", (dispatch_ids, limit * 10 + 1))
    frame_ids = sorted({r["frame_id"] for r in bundle["results"]["rows"]} |
                       {r["dispatch_frame_id"] for r in bundle["outcomes"]["rows"]})
    sql("dispatch_frames", "SELECT frame_id,source_proposal_frame_id,created_at, "
        "dispatch_frame_json FROM substrate_execution_dispatch_frames WHERE frame_id = ANY(%s)", (frame_ids,))
    # Strip request envelopes, reasons, target contents etc. before an export.
    for row in bundle["dispatch_frames"]["rows"]:
        payload = row.pop("dispatch_frame_json")
        candidates = []
        for key in ("candidates", "blocked_candidates", "dispatched_candidates"):
            for c in payload.get(key, []):
                candidates.append({k: c.get(k) for k in ("dispatch_id", "source_proposal_id", "source_decision_id", "dispatched_at", "expected_effect", "cortex_verb")})
        row["candidates"] = candidates
    proposal_ids = [r["source_proposal_frame_id"] for r in bundle["dispatch_frames"]["rows"]]
    sql("proposal_frames", "SELECT frame_id,proposal_frame_json FROM substrate_proposal_frames WHERE frame_id = ANY(%s)", (proposal_ids,))
    for row in bundle["proposal_frames"]["rows"]:
        payload = row.pop("proposal_frame_json")
        row["proposal_ids"] = [c["proposal_id"] for c in payload.get("candidates", []) if c.get("proposal_id")]
    feedback_ids = sorted({r["feedback_frame_id"] for r in bundle["outcomes"]["rows"]})
    sql("feedback_frames", "SELECT frame_id,source_execution_dispatch_frame_id FROM substrate_feedback_frames WHERE frame_id = ANY(%s)", (feedback_ids,))
    # Refuse a capped relationship set rather than asserting false missing links.
    for source in ("graph_briefs", "sql_briefs", "outcomes", "results"):
        if len(bundle.get(source, {}).get("rows", [])) > limit * 10:
            bundle[source] = {"status": "truncated", "rows": []}
    return bundle
