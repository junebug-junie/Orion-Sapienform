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

    # The dispatch rung (and the proposal rung before it) retired 2026-11-07 (ladder retirement PR 6,
    # docs/superpowers/specs/2026-11-04-retire-proposal-policy-dispatch-rungs.md). Their tables are
    # no longer read: each source reads "retired", which the audit reports as unverified, never as
    # an empty (and so "missing") link. Motor episodes now come only from non-act outcome rows: the
    # cooling world act (orion-durable-runs, episode id cooling_act:<want_id>).
    for retired in ("result_sample", "results", "dispatch_frames", "proposal_frames"):
        bundle[retired] = {"status": "retired", "rows": [], "retired_on": "2026-11-07"}
    # Act-ledger rows (dispatch_kind 'act:<verb>', orion/grammar/act_ledger.py) are not motor episodes.
    sql("outcome_sample", "SELECT dispatch_id FROM substrate_action_outcomes WHERE dispatch_kind NOT LIKE 'act:%%' "
        "ORDER BY observed_at DESC,id DESC LIMIT %s", (limit,))
    dispatch_ids = sorted({r["dispatch_id"] for r in bundle["outcome_sample"]["rows"] if r.get("dispatch_id")})
    sql("outcomes", "SELECT id,dispatch_id,dispatch_frame_id,feedback_frame_id,signal_id,predicted_delta,"
        "observed_at,arm,frame_dispatch_count FROM substrate_action_outcomes "
        "WHERE dispatch_id = ANY(%s) AND dispatch_kind NOT LIKE 'act:%%' ORDER BY id LIMIT %s", (dispatch_ids, limit * 10 + 1))
    # Attend-to-act loop (2026-10-01): world-action episodes for the sampled dispatches, the broadcast
    # rows they bound to, and the loop verdicts. Metadata only. The ledger table only exists once a
    # world action has been decided; until then the source is "unavailable", never "empty".
    sql("world_episodes", "SELECT episode_id, template, arm, decided_at, open_loop_id, broadcast_log_id, node_id, "
        "settlement_state, expected_effect IS NOT NULL AS expected_effect_recorded, scored_at, loop_outcome_id, "
        "outcome->>'excluded_reason' AS excluded_reason, outcome->'overlap' AS overlap, "
        "((outcome->'overlap') ? 'overlap:reflex' AND COALESCE((outcome->>'posterior_updated')::boolean, false)) "
        "AS posterior_from_reflex_overlap FROM substrate_world_action_episodes WHERE episode_id = ANY(%s)", (dispatch_ids,))
    loop_ids = sorted({r["open_loop_id"] for r in bundle["world_episodes"]["rows"] if r.get("open_loop_id")})
    log_ids = sorted({r["broadcast_log_id"] for r in bundle["world_episodes"]["rows"] if r.get("broadcast_log_id")})
    sql("broadcast_rows", "SELECT log_id, generated_at, projection_json->>'selected_open_loop_id' AS selected_open_loop_id "
        "FROM substrate_attention_broadcast_log WHERE log_id = ANY(%s)", (log_ids,))
    sql("loop_outcomes", "SELECT outcome_id, loop_id, verdict, actor, created_at, "
        "features_at_close->>'episode_id' AS episode_id FROM attention_loop_outcome WHERE loop_id = ANY(%s) "
        "ORDER BY created_at DESC LIMIT %s", (loop_ids, limit * 10 + 1))
    # Refuse a capped relationship set rather than asserting false missing links.
    # loop_outcomes: newest first, and refused when capped -- one loop id (world:chat's fixed label,
    # 2026-11-12) gathers outcomes across conversations, so an oldest-first cap would drop the
    # newest episodes' 'acted' rows and report their links as missing.
    for source in ("graph_briefs", "sql_briefs", "outcomes", "loop_outcomes"):
        if len(bundle.get(source, {}).get("rows", [])) > limit * 10:
            bundle[source] = {"status": "truncated", "rows": []}
    return bundle
