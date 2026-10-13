BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL statement_timeout = '120s';
SET LOCAL TIME ZONE 'UTC';
SELECT row_to_json(r) FROM (
 SELECT 'turn' AS kind, created_at AT TIME ZONE 'UTC' AS at FROM chat_history_log
 WHERE NULLIF(btrim(prompt), '') IS NOT NULL AND COALESCE(client_meta->>'unsolicited', 'false') <> 'true' AND created_at < '2026-10-13T19:00:00+00:00'::timestamptz
 AND (created_at >= '2026-10-09T06:00:00+00:00'::timestamptz OR id IN (SELECT id FROM chat_history_log
      WHERE NULLIF(btrim(prompt), '') IS NOT NULL AND COALESCE(client_meta->>'unsolicited', 'false') <> 'true' AND created_at < '2026-10-09T06:00:00+00:00'::timestamptz ORDER BY created_at DESC LIMIT 1))
 ORDER BY created_at
) r;
SELECT row_to_json(r) FROM (
 SELECT 'heat' AS kind, timestamp::timestamptz AS at, (measurements->>'cabinet_temp_c')::float AS temp_c
 FROM orion_biometrics_summary WHERE node='athena'
 AND timestamp >= '2026-10-09 06:00:00' AND timestamp < '2026-10-13 19:00:00'
 AND measurements ? 'cabinet_temp_c' ORDER BY timestamp
) r;
SELECT row_to_json(r) FROM (
 SELECT 'gpu_state' AS kind, generated_at AS at, host, queue_depth FROM gpu_pool_state_history
 WHERE generated_at >= '2026-10-09T06:00:00+00:00'::timestamptz AND generated_at < '2026-10-13T19:00:00+00:00'::timestamptz ORDER BY host, generated_at
) r;
SELECT row_to_json(r) FROM (
 SELECT 'gpu_event' AS kind, generated_at AS at, event, reason, role FROM gpu_pool_events
 WHERE event IN ('actuate_refused', 'swap_failed', 'swapped')
 AND generated_at >= '2026-10-09T05:30:00+00:00'::timestamptz
 AND generated_at < '2026-10-13T19:00:00+00:00'::timestamptz ORDER BY generated_at
) r;

SELECT row_to_json(r) FROM (
 SELECT 'decision' AS kind, 'dream' AS reader, started_at AS at FROM dream_cycle
  WHERE trigger = 'pressure' AND started_at >= '2026-10-09T07:00:00+00:00'::timestamptz AND started_at < '2026-10-13T19:00:00+00:00'::timestamptz
 UNION ALL
 SELECT 'decision', CASE WHEN request->'brief'->>'line' = 'self_inquiry' THEN 'self_inquiry' ELSE 'curiosity' END,
        created_at FROM durable_admission_runs
  WHERE request->>'workflow' = 'curiosity.investigate'
    AND COALESCE(request->'brief'->>'line', 'investigate') IN ('investigate', 'self_inquiry')
    AND COALESCE(request->'brief'->>'urgent', '') = ''
    AND created_at >= '2026-10-08T07:00:00+00:00'::timestamptz AND created_at < '2026-10-13T19:00:00+00:00'::timestamptz
 UNION ALL
 SELECT 'decision', 'outreach', decided_at FROM endogenous_outreach_decisions
  WHERE outreach AND COALESCE(result_json->>'source', '') <> 'curiosity_outreach'
    AND decided_at >= '2026-10-09T07:00:00+00:00'::timestamptz AND decided_at < '2026-10-13T19:00:00+00:00'::timestamptz
 UNION ALL
 SELECT 'decision', 'visual_baseline', started_at FROM reverie_visual_attempt
  WHERE need_id IS NOT NULL AND started_at >= '2026-10-09T07:00:00+00:00'::timestamptz AND started_at < '2026-10-13T19:00:00+00:00'::timestamptz
 UNION ALL
 SELECT 'decision', 'reverie_chain', created_at FROM substrate_reverie_chain
  WHERE created_at >= '2026-10-09T07:00:00+00:00'::timestamptz AND created_at < '2026-10-13T19:00:00+00:00'::timestamptz
 UNION ALL
 SELECT 'decision', 'mutation', created_at FROM substrate_mutation_proposal
  WHERE created_at >= '2026-10-09T07:00:00+00:00'::timestamptz AND created_at < '2026-10-13T19:00:00+00:00'::timestamptz
) r ORDER BY at;
SELECT row_to_json(r) FROM (
 SELECT 'live_level' AS kind, occurred_at AS at, label AS level FROM temporal_self_event
  WHERE source_kind = 'arousal_transition' AND occurred_at < '2026-10-13T19:00:00+00:00'::timestamptz ORDER BY occurred_at
) r;
ROLLBACK;

