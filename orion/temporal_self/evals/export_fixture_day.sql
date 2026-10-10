-- Read-only fixture export for the Temporal Self arc-precision eval.
-- One JSON object per line, tagged "k" with the source kind. Text-free: ids, timestamps,
-- numbers, verdict words and system-written short labels only. No chat text, no model prose
-- (summary, mantra, claim, statement, narrative and descriptions cut from chat are excluded).
--
--   docker exec -i orion-athena-sql-db psql -XqAt -v ON_ERROR_STOP=1 -U postgres -d conjourney \
--     -v start="'2026-10-08T06:00:00Z'" -v end="'2026-10-10T06:00:00Z'" \
--     -v body_start="'2026-10-09T06:00:00Z'" \
--     < orion/temporal_self/evals/export_fixture_day.sql | gzip -n > orion/temporal_self/evals/fixtures/day.jsonl.gz
BEGIN TRANSACTION READ ONLY;

SELECT json_build_object('k','broadcast','log_id',log_id,'generated_at',generated_at,
  'ref',(SELECT l->'source_refs'->>0 FROM jsonb_array_elements(projection_json->'frame'->'open_loops') l
         WHERE l->>'id' = projection_json->>'selected_open_loop_id' LIMIT 1),
  'label',(SELECT l->>'description' FROM jsonb_array_elements(projection_json->'frame'->'open_loops') l
         WHERE l->>'id' = projection_json->>'selected_open_loop_id' LIMIT 1))
FROM substrate_attention_broadcast_log WHERE generated_at >= :start AND generated_at < :end;

SELECT json_build_object('k','chat_turn','id',id,'correlation_id',correlation_id,'session_id',session_id,
  'source',source,'created_at',created_at,'has_prompt',btrim(coalesce(prompt,'')) <> '',
  'unsolicited',coalesce((client_meta->>'unsolicited')::boolean, false))
FROM chat_history_log WHERE created_at >= (:start)::timestamptz AT TIME ZONE 'UTC' AND created_at < (:end)::timestamptz AT TIME ZONE 'UTC';

SELECT json_build_object('k','curiosity_run','run_id',d.run_id,'turn_started_at',d.turn_started_at,
  'decided_at',d.decided_at,'arm',d.arm,'offered',d.offered,'completed_at',o.completed_at,'turn_ok',o.turn_ok,
  'n_tested',o.n_tested,'n_moved',o.n_moved,'n_formed',o.n_formed)
FROM curiosity_offer_decisions d LEFT JOIN curiosity_run_outcomes o USING (run_id)
WHERE d.decided_at >= (:start)::timestamptz - interval '6 hours' AND d.decided_at < :end;

SELECT json_build_object('k','reverie_chain','chain_id',c.chain_id,'created_at',c.created_at,
  'theme_key',c.theme_key,'terminal_reason',c.terminal_reason,
  'thoughts',COALESCE((SELECT json_agg(json_build_object('thought_id',t.thought_id,'created_at',t.created_at,
      'correlation_id',t.correlation_id) ORDER BY t.created_at)
    FROM substrate_reverie_thought t WHERE t.thought_json->>'chain_id' = c.chain_id), '[]'::json))
FROM substrate_reverie_chain c WHERE c.created_at >= :start AND c.created_at < :end;

SELECT json_build_object('k','expectation_verdict','thought_id',thought_id,'correlation_id',correlation_id,
  'chain_id',thought_json->>'chain_id','created_at',created_at,'expectation_verdict',expectation_verdict,
  'expectation_scored_at',expectation_scored_at)
FROM substrate_reverie_thought WHERE expectation_scored_at >= :start AND expectation_scored_at < :end;

SELECT json_build_object('k','visual_run','chain_id',c.chain_id,'created_at',c.created_at,'theme_key',c.theme_key,
  'terminal_reason',c.terminal_reason,'attempt_started_at',a.started_at,'thermal_state',a.result_json->'detail'->>'state')
FROM reverie_visual_chain c LEFT JOIN LATERAL (SELECT started_at, result_json FROM reverie_visual_attempt a
  WHERE a.result_json->>'chain_id' = c.chain_id ORDER BY started_at DESC LIMIT 1) a ON true
WHERE c.created_at >= :start AND c.created_at < :end;

SELECT json_build_object('k','visual_deferral','attempt_id',attempt_id,'started_at',started_at,'outcome',outcome,
  'result_json',json_build_object('reason',result_json->>'reason','refused',result_json->'refused',
     'detail',json_build_object('state',result_json->'detail'->>'state')))
FROM reverie_visual_attempt WHERE started_at >= :start AND started_at < :end;

SELECT json_build_object('k','gpu_wait','event_id',event_id,'generated_at',generated_at,'event',event,
  'holder',holder,'priority',priority,'work_class',work_class,'waited_ms',waited_ms,'turn_correlation_id',turn_correlation_id)
FROM gpu_pool_events WHERE generated_at >= :start AND generated_at < :end
  AND priority = 'background' AND holder NOT LIKE 'http:%'
  AND (event = 'unavailable' OR (event = 'granted' AND waited_ms >= 500));

SELECT json_build_object('k','dream_cycle','cycle_id',cycle_id,'trigger',trigger,'status',status,
  'started_at',started_at,'ended_at',ended_at,'pressure',pressure,'replay_count',replay_count,
  'hypothesis_count',hypothesis_count,'cycle_json',json_build_object('pressure',json_build_object('since',cycle_json->'pressure'->>'since')))
FROM dream_cycle WHERE started_at >= :start AND started_at < :end;

SELECT json_build_object('k','dream_hypothesis','hypothesis_id',hypothesis_id,'cycle_id',cycle_id,'arm',arm,
  'created_at',created_at,'expires_at',expires_at,'offered_at',offered_at,'offered_run_id',offered_run_id)
FROM dream_hypothesis WHERE created_at >= :start AND created_at < :end;

SELECT json_build_object('k','action_outcome','id',id,'observed_at',observed_at,'created_at',created_at,
  'claim_upheld',claim_upheld,'dispatch_kind',dispatch_kind,'target_id',target_id,'prediction_error',prediction_error)
FROM substrate_action_outcomes WHERE created_at >= :start AND created_at < :end;

SELECT json_build_object('k','metacog_observation','id',m.id,'correlation_id',m.correlation_id,'severity',m.severity,
  'trigger_kind',m.trigger_kind,'timestamp',m.timestamp,'trigger_timestamp',t.timestamp)
FROM orion_metacog m LEFT JOIN LATERAL (SELECT timestamp FROM metacog_trigger t WHERE t.correlation_id = m.correlation_id
  ORDER BY timestamp LIMIT 1) t ON true
WHERE m.severity IN ('degraded','critical')
  AND m.timestamp::timestamptz >= :start AND m.timestamp::timestamptz < :end;

SELECT json_build_object('k','consolidation_window_close','memory_window_id',memory_window_id,'closed_at',closed_at,
  'turn_correlation_ids',turn_correlation_ids,'close_reason',close_reason,'source_platform',source_platform)
FROM memory_consolidation_windows WHERE closed_at >= :start AND closed_at < :end;

SELECT json_build_object('k','attention_row','entry_id',entry_id,'generated_at',generated_at,'process',process,
  'correlation_id',correlation_id,'attention_reason',attention_reason)
FROM substrate_attention_schema WHERE generated_at >= :start AND generated_at < :end;

SELECT json_build_object('k','attention_loop_raised','trace_id',trace_id,'loop_id',loop_id,'scope',scope,
  'correlation_id',correlation_id,'created_at',created_at,
  'chat_turn',EXISTS (SELECT 1 FROM chat_history_log c WHERE c.correlation_id = t.correlation_id))
FROM attention_salience_trace t WHERE scope = 'chat' AND created_at >= :start AND created_at < :end;

SELECT json_build_object('k','attention_loop_verdict','outcome_id',outcome_id,'loop_id',loop_id,'verdict',verdict,
  'actor',actor,'created_at',created_at)
FROM attention_loop_outcome WHERE created_at >= :start AND created_at < :end;

SELECT json_build_object('k','field_dominance_run','run_id',run_id,'target_id',target_id,'target_kind',target_kind,
  'started_at',started_at,'ended_at',ended_at,'tick_count',tick_count,'min_streak_at_run',min_streak_at_run,
  'left_censored',left_censored)
FROM field_dominance_run WHERE ended_at >= :start AND ended_at < :end;

SELECT json_build_object('k','vision_percept','event_id',event_id,'stream_id',stream_id,'event_type',event_type,
  'entities',entities,'created_at',created_at)
FROM vision_events WHERE created_at >= :start AND created_at < :end AND jsonb_array_length(entities::jsonb) > 0;

SELECT json_build_object('k','memory_episode','memory_id',memory_id,'episode_id',episode_id,'purpose',purpose,
  'occurred_at',occurred_at)
FROM episode_memory WHERE occurred_at >= :start AND occurred_at < :end;

-- Body rows for the eval day only (the metric gate's live sanity check).
SELECT json_build_object('k','body_cluster','observed_at',observed_at,'chassis_watts',chassis_watts)
FROM orion_biometrics_cluster WHERE observed_at >= :body_start AND observed_at < :end;

SELECT json_build_object('k','body_cabinet','timestamp',timestamp,'cabinet_temp_c',measurements->'cabinet_temp_c')
FROM orion_biometrics_summary WHERE node = 'athena' AND measurements ? 'cabinet_temp_c'
  -- TEXT column, ' ' separator ('2026-10-10 18:38:57.43073+00'): cast, never compare as text.
  AND timestamp::timestamptz >= :body_start AND timestamp::timestamptz < :end;

SELECT json_build_object('k','body_spike','spike_id',spike_id,'timestamp',timestamp)
FROM cabinet_ambient_spike WHERE timestamp >= :body_start AND timestamp < :end;

ROLLBACK;
