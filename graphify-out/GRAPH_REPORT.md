<<<<<<< Updated upstream
# Graph Report - Orion-Sapienform  (2026-07-29)
=======
# Graph Report - Orion-Sapienform  (2026-07-28)
>>>>>>> Stashed changes

## Corpus Check
- 0 files · ~0 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
<<<<<<< Updated upstream
- 28306 nodes · 81046 edges · 1395 communities (1047 shown, 348 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 11337 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `97b634c5`
=======
- 2489 nodes · 3241 edges · 293 communities (190 shown, 103 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 321 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `fd647649`
>>>>>>> Stashed changes
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Service: orion-vector-writer
- Channel "orion:collapse:triage" (kind=event, schema=CollapseMirrorEntryV2) producers=[orion-collapse-mirror] consumers=[orion-meta-tags, orion-vector-writer, orion-sql-writer, orion-actions]
- Context-exec Workbench
- Hub Gateway (Harness) README
- Channel "orion:stt:intake" (kind=request, schema=STTRequestPayload) producers=[orion-hub] consumers=[orion-whisper-tts]
- orion-llm-gateway / model routing
- services/orion-proposal-runtime/README.md — Layer 7 substrate service: converts SelfStateV1 (+ optional attention/field context) into possible actions (ProposalFrameV1), not automatic actions
- orion-cortex-orch service
- Memory Constellation (tri-layer SQL/RDF/vector)
- Channel "orion:feedback:frame" (kind=event, schema=FeedbackFrameV1) producers=[orion-feedback-runtime] consumers=[orion-spark-concept-induction]
- Channel "orion:exec:request:RecallService" (kind=request, schema=RecallQueryV1) producers=[orion-cortex-exec, orion-cortex-orch, orion-hub, orion-context-exec, orion-spark-concept-induction] consumers=[orion-recall]
- ctx:substrate-tier-telemetry-v1 (context pack metadata)
- Knowledge Forge Lint Report 2026-05-20 (empty)
- orion-self-state-runtime docker-compose.yml
- conjourney
- orion-self-state-runtime requirements.txt
<<<<<<< Updated upstream
- test_consolidation_expectations.py
- websocket_endpoint
- InMemoryReasoningRepository
- LLMMessage
- ThoughtEventV1
- GraphDBSubstrateStore
- .text
- FieldAttentionFrameV1
- CallSyneRoomMessageV1
- ChatStanceBrief
- ProposalEnvelopeV1
- PolicyDecisionFrameV1
- NodeCatalog
- EquilibriumService
- profile_repository.py
- models.py
- ConceptSettings
- executor.py
- main.py
- CortexChatRequest
- Collapse Mirror split invariant (Strict/Juniper vs Metacog/Orion; rationale: metacog mirrors must never hit Juniper's triage/enrichment pipeline by default)
- TopicFoundryBusPublisher
- app.js
- worker.py
- OrganClass
- SubstrateMoleculeV1
- EndogenousTriggerEvaluator
- MemoryTurnPersistedV1
- main.py
- SubstratePolicyProfileStore
- ExecutionDispatchFrameV1
- ContextExecRunV1
- finalize_pass.py
- FieldStateV1
- ReductionReceiptV1
- molecules.py
- BiometricsSubstrateWorker
- ProposalCandidateV1
- __init__.py
- association.py
- GrammarEventV1
- ServiceState
- GraphReviewQueue
- drives.py
- chain.py
- EmbeddingGenerateV1
- test_compaction_applier.py
- context_exec.py
- router.py
- self_study.py
- test_memory_crystallization.py
- draft_to_graph
- VisionWindowPayload
- Service: orion-sql-writer
- GrammarProvenanceV1
- main.py
- consumer_readiness.py
- SubstrateMutationStore
- llm_backend.py
- RouterState
- UnifiedRelationalBeliefSetV1
- TurnAppraisalBundleV1
- SelfStateV1
- memory_cards.py
- AttentionBroadcastProjectionV1
- view_model.py
- appraise_repair_pressure
- BaseSubstrateNodeV1
- GrammarAtomV1
- attention_frame.py
- test_policy_act.py
- goal_actions.py
- test_stance_react_pipeline.py
- JournalPageIndexService
- SocialRoomBridgeService
- agent_synthesis.py
- detector_worker.py
- reverie.py
- Service: orion-spark-introspector
- .patch
- DiscussionWindowResultV1
- RedisStreamWorkQueue
- DriveEngine
- persist_context_exec_run
- recall_v2.py
- SystemHealthV1
- registry.py
- substrate_lattice_routes.py
- Service: orion-hub
- SemanticSynthesisV1
- EndogenousRuntimeAdoptionService
- recall_utils.py
- showToast
- Orion Bus Channels Registry (channels.yaml)
- WindowStore
- worker.py
- SparqlHttpClient
- curiosity.py
- digest.py
- suggest_with_escalation
- ConceptProfile
- worker.py
- VisionArtifactPayload
- test_substrate_lattice_routes.py
- test_signal_tension.py
- crystallization_routes.py
- supervisor.py
- EpisodicFederator
- OpenLoopV1
- DecisionRouter
- context_mode_hooks_smoke.py
- pcr_chat_memory.py
- main.py
- TransportBusProjectionV1
- models.py
- repo_tools.py
- Service: orion-cortex-exec
- fusion.py
- resolve
- test_route_substrate_reducer.py
- test_curiosity_reuse.py
- test_reducer_lane_adapters.py
- main.py
- VisionRunner
- EmbodimentIntentV1
- mind_runtime.py
- test_execution_dispatch_runtime_worker.py
- main.py
- HarnessRunV1
- test_publish_paths.py
- test_turn_change_classify.py
- ConceptRelationDecision
- service_logs.py
- thought.py
- CalibrationAdoptionRequestV1
- attention_loops_store.py
- brain_frame_producer.py
- test_social_memory_service.py
- test_memory_crystallization_dynamics.py
- WorldPerceptionV1
- grammar.py
- test_harness_governor_client_liveness.py
- engine.py
- SparkStateSnapshotV1
- bus_worker.py
- social_room.py
- thought-process.js
- main.py
- situation.py
- test_openai_passthrough.py
- snapshot_from_window
- bus_listener.py
- anthropic_passthrough.py
- Context Engineering Pack (Substrate Trace Adoption)
- test_chat_relational_stance.py
- FakeQueue
- test_salience_combiner.py
- test_grammar_truth.py
- Context Pack: orion-substrate-telemetry (Cursor, markdown)
- GraphitiAdapter
- detect_resonance
- main.py
- types.py
- ResonanceHealthMonitor
- main.py
- train_v3_moc.py
- Service: orion-vision-host
- VisionSceneInterpretationV1
- _common.py
- builder.py
- BaseEnvelope
- consolidation_memory_gate
- worker.py
- suggest_validate.py
- HyperbolicGPT
- Consolidation Policy v1
- Shared compactor helpers README
- test_reasoning_emit.py
- FolderFrameSource
- fcc_claude_bridge.py
- Phase 13b SQL-Writer Durability
- MemoryCompactionDeltaV1
- EmbodimentWorker
- SpontaneousThoughtV1
- check_inner_state_registry.py
- FieldDigesterWorker
- ConceptWorker
- SuggestDraftV1
- EvidenceUnitV1
- build_harness_reasoning_trace
- WorldPulseSourceV1
- client.py
- pg_conn
- _run_training
- github_repo_context.py
- rdf_adapter.py
- self_study_harness.py
- _WPBase
- test_substrate_mutation_scheduler_runtime.py
- map_curiosity_ctx_to_substrate
- AgentConfig
- test_graphiti_core_backend.py
- test_recall_strategy_readiness.py
- RowBlock
- PackManager
- service.py
- aitown_client.py
- test_metacog_phase_contract.py
- HarnessStepRelay
- MemoryCrystallizationV1
- SocialScenarioReplayHarness
- orchestrator.py
- test_fcc_motor_mcp.py
- mutation_control_surface.py
- AnswerContract
- StreamMessage
- MindRunRequestV1
- PsuService
- bootstrap_orion_agent.py
- Service: orion-harness-governor
- get_profile
- EndogenousEvaluationRequestV1
- VisionFramePointerPayload
- extract_repair_evidence
- CrystallizationEvidenceRefV1
- memory_consolidation_draft_routes.py
- test_queue_service_chassis.py
- run_answer_depth_live_proof.py
- HealthMonitor
- submitExplicitChatText
- LocalProfileStore
- active_packet.py
- draft_sanitize.py
- ArticleRecordV1
- test_drive_state_divergence_audit.py
- test_drive_history_reflection_synthesis.py
- graphiti_core.py
- AutonomyVerificationHarness
- test_introspection_fixture.py
- persist_turn_referent
- schemas.py
- CompressionRegionV1
- schemas.py
- fit_phi_encoder.py
- drive_history_reflection_synthesis.py
- CouncilService
- appendMessage
- mind_enrichment.py
- main.py
- WorldPulseStreamConsumer
- map_recall_bundle_to_substrate
- test_phi_encoder_fit_script.py
- validate_for_escalation
- SceneBeliefTracker
- Vision Services Documentation
- SocialGifUsageStateV1
- settings.py
- SignalsInspectCache
- substrate-atlas.js
- EwmaBand
- tts.py
- main.py
- Endogenous Action v1 Motor Nerve Spec
- WorldPulseRunResultV1
- Service: orion-actions
- McpPreflightError
- extract_suggest_draft_dict_from_cortex_payload
- derive_retrieval_intent
- orion-landing-pad Docker Compose
- test_measure_autonomy_gate.py
- endogenous_curiosity_candidates
- test_rlm_eval_fixtures.py
- proposal_review_client.py
- probe_structured_output.py
- test_interpretation_v2.py
- StepExecutionResult
- memory_graph_suggest_timeout.py
- check_single_consumer_channels.py
- bound_capability_exec.py
- syncDebugModalScrollLock
- memory.js
- service.py
- Service: orion-landing-pad
- build_delivery_grounding_context
- ChannelCatalogEnforcer
- TestLLMBackendHelpers
- test_consolidation_tensorize.py
- measure_autonomy_gate.py
- trace_unified_turn.py
- utils.py
- mind_provenance.js
- boundary.py
- test_proposal_review_api.py
- test_router_identity_boundary.py
- hub_memory_graph_suggest_text
- SemanticPlanner
- CrystallizationGovernanceV1
- ActiveCognitiveFrontierV1
- ContextExecRunner
- orion-hub service (README) — browser gateway into the mesh
- substrate_effect_pipeline.py
- substrateReviewFetch
- _make_engine_sequence
- __init__.py
- query.py
- should_rewrite_for_instructional
- __init__.py
- EpisodicConsolidationEvaluator
- SalienceState
- ensure_delivery_pack_in_packs
- pipeline.py
- test_skill_verbs.py
- fcc_env_catalog.py
- main.py
- test_voluntary_attention_wiring.py
- grammar_integration_helpers.py
- main.py
- __init__.py
- memory_routes.py
- rdf_retention.py
- is_caption_prompt_echo
- MindRunResultV1
- test_rem_compaction.py
- main.py
- CollapseMirrorEntryV2
- test_orion_proposal_cli.py
- orion-agent-council service (multi-agent deliberation stub)
- test_notify_attention_ack.py
- resolve_user_workflow_invocation
- build_crystallization_from_window
- InnerStateFeaturesV1
- HealthMonitor
- memory-graph-draft-form.js
- test_worker_speech.py
- test_reverie_observability_section.py
- main.py
- test_llm_uncertainty.py
- capability_policy.py
- test_proposal_runtime_worker.py
- test_bc_mode_understand.py
- substrate_execution_dispatch_routes.py
- chat_stance.py
- Service: orion-spark-introspector
- ChatHistorySparkMetaPatchV1
- test_boundary.py
- spark_narrative.py
- settings.py
- sql_timeline.py
- orion-vision-council service
- context_exec_permissions_for_llm_profile
- VectorHostEmbeddingProvider
- preview_dataset
- Orion Cortex Exec Service
- test_turn_effect.py
- model_moc.py
- BiometricsCollector
- build_route_arbitration_grammar_events
- refreshForgeTab
- test_recall_canary_battle_harness.py
- intention.py
- StateJournaler
- action_outcomes.py
- extract_cortex_payload_text
- classify.py
- WorkflowDispatchRequestV1
- sql_fetch.py
- test_single_consumer_channels_gate.py
- sync_file
- test_async_notify_producers.py
- bus_observer.py
- build_substrate_attention_frame
- renderNotifications
- test_substrate_review_runtime_hub_debug.py
- fetch_graph_compression_fragments
- build_substrate_grammar_truth
- test_reverie_chain.py
- train.py
- substrate_consolidation_routes.py
- resolve_destination
- Idea 2: Hub turn-hop WebSocket relay
- parse_json_object
- test_phi_corpus_diag_script.py
- SchedulerCursorStore
- refreshMindRunsForCorrelation
- test_proposal_review_hub.py
- test_anthropic_passthrough.py
- MindRunBudget
- synthesis.py
- enrichment.py
- test_check_service_env_compose_parity.py
- CLAUDE.md — Orion Subagent Development Contract: repo-wide rules for how AI coding agents work in Orion-Sapienform, identical content to AGENTS.md (symlinked)
- NISUPSClient
- __init__.py
- Proposal Review API
- test_chat_prompt_context_guardrails.py
- ModelManager
- AutonomyStateV2
- TensionRateLimiter
- social.py
- memory_graph_suggest.py
- test_mind_light_snapshot.py
- Settings
- substrate_observability_routes.py
- self-brain.js
- substrate-lattice.js
- apply_structured_output_to_payload
- snippet_dedupe.py
=======
- Collapse Mirror split invariant (Strict/Juniper vs Metacog/Orion; rationale: metacog mirrors must never hit Juniper's triage/enrichment pipeline by default)
- Service: orion-sql-writer
- Service: orion-spark-introspector
- Service: orion-hub
- Orion Bus Channels Registry (channels.yaml)
- Service: orion-cortex-exec
- Context Engineering Pack (Substrate Trace Adoption)
- Context Pack: orion-substrate-telemetry (Cursor, markdown)
- Service: orion-vision-host
- Consolidation Policy v1
- Shared compactor helpers README
- Phase 13b SQL-Writer Durability
- Service: orion-harness-governor
- Vision Services Documentation
- Endogenous Action v1 Motor Nerve Spec
- Service: orion-actions
- orion-landing-pad Docker Compose
- Service: orion-landing-pad
- orion-hub service (README) — browser gateway into the mesh
- orion-agent-council service (multi-agent deliberation stub)
- Service: orion-spark-introspector
- orion-vision-council service
- Orion Cortex Exec Service
- CLAUDE.md — Orion Subagent Development Contract: repo-wide rules for how AI coding agents work in Orion-Sapienform, identical content to AGENTS.md (symlinked)
- Proposal Review API
>>>>>>> Stashed changes
- orion-substrate-runtime service (biometrics closed loop, grammar reducers, Layers 1-5)
- config/mesh_remediation_roster.yaml (auto-remediation roster)
<<<<<<< Updated upstream
- map_drive_state_to_intent
- bus-core (Redis broker container)
- test_attention_loops_reader.py
- compaction.py
- grammar_atlas_routes.py
- memory-graph-draft-ui.js
- rem_compaction.py
- AGENTS.md — Orion Subagent Development Contract: repo-wide rules for how AI coding agents work in Orion-Sapienform, aimed at inspectable, testable, parallel-safe agent behavior
- Metrics Swamp Arsonist Review
- Recall Epistemic Honesty + Observability Spec
- properties
- build_perception
- memory_graph_routes.py
- Settings
- identity.py
- _project_reverie_glimpse
- conversation_front.py
- build_readout
- NotificationCache
- test_lane_routes.py
- SparkContractMetrics
- receipt_pruner.py
- FakeConn
- cognition_trace_cache.py
- AutonomyStateV2 evidence pipeline (chat, env-gated)
- test_agent_chain_guards.py
- test_proposal.py
- BiometricsAdapter
- HyperbolicGPTConfig
- foundry.py
- test_check_concept_relation_digest_liveness.py
- test_deviation_gate.py
- build_autonomy_slice
- resolve_llm_lane_for_step
- mcp_stdio_proxy.py
- run_recall_canary_battle.py
- workflow-schedule-ui.js
- workflow-ui.js
- test_fcc_claude_bridge_run.py
- test_memory_graph_suggest_coalesce_ui.py
- Stance Assembly / ChatStanceBrief
- projection_context.py
- required
- parse_json_object
- test_ouroboros_invariants.py
- WorkflowScheduleStore
- grammar_truth_gate.py
- _skip_journal_pageindex_for_automated_trigger
- _project_recent_dispatch_actions
- OrchConceptProfileSettings
- renderScheduleInventory
- social-inspection.js
- test_thought_candidate.py
- HealthMonitor
- test_stt_engine.py
- Concept Induction (Spark)
- Endogenous Drive Origination Design
- rdf_sync.py
- properties
- is_active
- graph_view.py
- assert_hub_context_exec_routing
- orion_fresh_main_smoke.sh
- _should_prepare_brain_reply_context
- _worker
- proposal-review-ui.js
- service-logs-ui.js
- substrate-effect-ui.js
- test_memory_graph_from_chat_live.py
- test_profile_forwarding.py
- _FakeSession
- test_check_daily_schedule_collisions.py
- reflect.v1 recall profile
- train_moc.py
- test_chatgpt_qlora_pipeline.py
- test_journal_pageindex_mvp.py
- memory-crystallization-ui.js
- test_grammar_atlas_api.py
- test_substrate_observability_api.py
- appraisal.py
- test_capability_policy.py
- test_autonomy_goal_actions.py
- Integrated Memory Cognition Loop Design
- Docker readiness (§8): run docker compose builds/deploys through scripts/safe_docker_build.sh instead of calling docker compose directly; it refuses to run from the shared/primary checkout and applies the --env-file/-f pattern automatically; raw docker compose examples retained only as reference for one-off logs/ps commands
- render_aitown_tab_blocks
=======
- bus-core (Redis broker container)
- AGENTS.md — Orion Subagent Development Contract: repo-wide rules for how AI coding agents work in Orion-Sapienform, aimed at inspectable, testable, parallel-safe agent behavior
- Metrics Swamp Arsonist Review
- Recall Epistemic Honesty + Observability Spec
- cognition_trace_cache.py
- AutonomyStateV2 evidence pipeline (chat, env-gated)
- Stance Assembly / ChatStanceBrief
- Concept Induction (Spark)
- Endogenous Drive Origination Design
- reflect.v1 recall profile
- Integrated Memory Cognition Loop Design
- Docker readiness (§8): run docker compose builds/deploys through scripts/safe_docker_build.sh instead of calling docker compose directly; it refuses to run from the shared/primary checkout and applies the --env-file/-f pattern automatically; raw docker compose examples retained only as reference for one-off logs/ps commands
>>>>>>> Stashed changes
- orion-llamacpp-host service (profile-driven llama.cpp GGUF wrapper, Atlas topology)
- config/proposals/proposal_policy.v1.yaml — Layer 7 proposal policy: limits, priority/risk thresholds, dimension weights, and named proposal_templates that turn substrate state into ProposalFrameV1 candidates
- Unified Cognitive Substrate Phase 6 (Frontier Expansion / Typed Graph-Delta Generation)
- Orion Titanium Contracts
- Orion Platform Contract
- MetacogTriggerV1
- Vision Grounded Pipeline Design
- Orion Unified Turn (canonical spec)
- Felt-State Arc Roadmap Spec
<<<<<<< Updated upstream
- VerbRegistry
- test_query.py
- ChatResponseFeedbackV1
- test_drive_pressure_probe.py
- _clean_raw_llm_content
- test_self_study_graphdb.py
- mind_routes.py
- test_hub_grammar_emit.py
- test_mind_routes.py
- profiles.py
- AlertPayload
- Orion Signals Roster v1 (mesh service tiers)
- Phase 4 Cluster-Weighting Research
- test_worker_prediction_error_node.py
- test_cortex_gateway_error_reply.py
- _ev
=======
- Orion Signals Roster v1 (mesh service tiers)
>>>>>>> Stashed changes
- services/orion-memory-consolidation/README.md — subscribes to orion:memory:turn:persisted, classifies each chat turn via LLM gateway quick-lane logprobs, patches chat_history_log.spark_meta, tracks consolidation windows, and on boundary closure runs a deterministic consolidation gate (default) or legacy graph suggest
- services/orion-execution-dispatch-runtime/README.md — Layer 9 of the Orion cognition substrate: converts PolicyDecisionFrameV1 + ProposalFrameV1 + SelfStateV1 into ExecutionDispatchFrameV1 envelopes, the motor-nerve service that can actually send real actions
- config/llm_profiles.yaml (LLM profile registry)
- MemoryCardV1
- Orion Heartbeat Research Charter
- SuggestDraftV1
- Runtime Trace Signal Nexus Design
- orion-signal-gateway (normalizes organ-bus events into OrionSignalV1)
<<<<<<< Updated upstream
- test_attention_ack.py
- proposal_review_routes.py
- generate_descriptions.py
- aitown-panel.js
- test_hub_ui_polish.py
- test_memory_consolidation_draft_routes.py
- main.py
- orion-spark-introspector: Spark metacognitive streaming service driving the phi/EKG chart
- test_roster.py
- FakeWorker
- order
- main.py
- Pipeline: Retina Dense
- World Pulse Sources Policy v1
- LLM Services and Agentic Flow
- up_all_services_batched.sh
- VectorStore
- check_service_env_compose_parity.py
- plan
- trace_hub_skill_runner_e2e.py
- test_generate_descriptions.py
- test_worker_social.py
- Settings
- agent-claude-trace.js
- test_fcc_claude_bridge_mcp.py
- test_hub_presence.py
- test_mind_provenance_normalizer.py
- test_turn_stop_command.py
- timeout_ms
- _body
- test_action_outcome_sql_shape.py
- test_worker_attention_broadcast_tick.py
- claim:orion:substrate-telemetry:0001 — orion-substrate-telemetry persists tier outcomes
=======
- orion-spark-introspector: Spark metacognitive streaming service driving the phi/EKG chart
- Pipeline: Retina Dense
- World Pulse Sources Policy v1
- LLM Services and Agentic Flow
>>>>>>> Stashed changes
- Reasoning Schema Phase 1
- Unified Cognitive Substrate Phase 1 (Shared Ontology + Canonical Contracts)
- Unified Cognitive Substrate Phase 11 (Narrow Runtime Review Execution)
- FCC-Cortex GWT Dispatch Design
- execution_trajectory reducer / ExecutionRunStateV1
- source:2026-05-20-knowledge-forge-v1-merge (metadata)
- Service: orion-notify
- Cognition Packs (Memory, Executive, Emergent)
<<<<<<< Updated upstream
- walkable_tiles
- test_fcc_motor_summarize.py
- recommend_actions_from_alerts
- test_signal_drive_consumer.py
- model.py
- refit_salience_weights.py
- _SingleConnPool
- verify_agent_repl_live.py
- compact_vision_scene_interpretation_json_schema
- agent-trace.js
- organ-signals-graph-ui.js
- _Store
- identity_snapshot.py
- context.py
- self_state_prediction.py
- _reload_settings
- readiness_payload
- models.py
- test_curiosity.py
- _facts_fixture
- .path
- test_vision_retina_settings.py
- Endogenous Drive Origination Design
=======
>>>>>>> Stashed changes
- orion-social-memory service
- Unified Cognitive Substrate Phase 13 (GraphDB-Backed Persistence)
- Brainstorming Session #1 - Appendix Ideas 3-10
- orion-mesh-guardian service
- Orion Relational Stance Design (v1)
- Concept Relation Resolution Design
- Channel "orion:kg:edge:ingest.v1" (kind=event, schema=KgEdgeIngestV1) producers=[orion-topic-foundry] consumers=[orion-rdf-writer, orion-graphdb]
<<<<<<< Updated upstream
- test_simulate_no_db_writes
- organ_layer
- test_draft_patch_does_not_write_files
- check_daily_schedule_collisions.py
- test_draft_patch_503_when_config_not_found
- _attempt_mind_handoff_chat_stance_shortcut
- endogenous_runtime.py
- test_gates_contract_not_quiet_when_contract_pressure_1
- correlation_chain_from_cognition_trace
- renderMemoryDebugModal
- substrate.js
- test_mind_enabled_contract.py
- test_projection_starvation.py
- test_gates_contract_unknown_when_m3_stale
- test_gates_attention_pass_when_capability_transport_present
- test_gates_attention_blocked_when_capability_transport_absent
- test_normalize_targets_strings_become_objects
- test_embodiment_c_hook.py
- test_normalize_targets_dicts_preserve_fields
- Channel "orion:dream:log" (kind=event, schema=DreamResultV1) producers=[orion-cortex-exec] consumers=[orion-sql-writer, orion-dream]
- test_coerce_str_list_handles_dict_channel_shapes
- STTEngine
- test_transport_latest_404_when_no_projection_row
=======
- Channel "orion:dream:log" (kind=event, schema=DreamResultV1) producers=[orion-cortex-exec] consumers=[orion-sql-writer, orion-dream]
>>>>>>> Stashed changes
- concept_induction_pass Workflow
- Unified Cognitive Substrate Phase 17 (Operator-Controlled Policy Adoption and Rollback)
- Biometrics Reference Adapter
- orion-memory-consolidation service
- Metacog Prompt Slim Context Design
- Stance React
<<<<<<< Updated upstream
- dataset.py
- Settings
- scan_cognition_library
- social_room_inspection_cache.py
- test_memory_graph_structured_output.py
- test_substrate_biometrics_debug_api.py
- test_substrate_execution_dispatch_debug_api.py
- test_substrate_field_debug_api.py
- test_http_contract.py
- orion-rag: retrieval-augmented generation orchestrator, enriches queries with vector-db context before delegating to LLM host
- ctx:substrate-tier-telemetry-v1 (context pack metadata)
- Knowledge Forge Lint Report 2026-05-20 (empty)
- orion-sql-db: PostgreSQL database + pgAdmin client
- test_phi_reward_sql_shape.py
- Landing Pad Metrics Explorer UI
- test_mind_http_client.py
- introspect.py
- orion-self-state-runtime docker-compose.yml
- main.py
- orion-self-state-runtime requirements.txt
- orion-social-memory service
=======
- orion-rag: retrieval-augmented generation orchestrator, enriches queries with vector-db context before delegating to LLM host
- orion-sql-db: PostgreSQL database + pgAdmin client
>>>>>>> Stashed changes
- GraphDB Semantic vs SQL Operational Ownership Split
- Unified Cognitive Substrate Phase 4 (Graph Dynamics and Pressure Propagation)
- Recall Service (orion-recall)
- AutonomyStateV2
- SubstrateGraphRecordV1
- DriveEngine + Concept-Induction Deactivation Design
- Inner-State Unification Design
- Vector Audit
- Orion Node Bootstrap README (Ubuntu 24.04)
<<<<<<< Updated upstream
- run_attention_bound_proposal_eval.py
- enum
- Orion Cognitive Dashboard UI (index.html + tissue_viz.js)
- chat.py
- Canonical phi: _phi_from_self_state() / _get_phi_stats
- smoke_all_notifications.sh
- test_health.py
- conftest.py
- run_dream
- record_turn
- test_hub_local_time_naive_utc.py
- test_memory_graph_bridge_ui.py
- test_mind_hub_tab.py
- test_presence_chat_injection.py
- _FakeBus
- test_self_brain_routes.py
- test_stop_chat_ui_smoke.py
- orion-notify: minimal notification host centralizing email delivery, attention requests, chat messages, recipient preferences, escalation
- test_rdf_chatturn_windowing.py
- test_phase21_wiring_verification.py
- orion-llm-gateway manual smoke tests: bus chat/exec_step envelopes, ollama vs vllm backend selection
- config/autonomy/capability_policy.v1.yaml — policy config gating which autonomy capabilities may auto-execute per cycle, by side-effect class, required goal status, required drive origins/signal kinds, and per-cycle budget
- Social GIF Expression Layer
- Social Scenario Replay Harness
=======
- Canonical phi: _phi_from_self_state() / _get_phi_stats
- orion-notify: minimal notification host centralizing email delivery, attention requests, chat messages, recipient preferences, escalation
- orion-social-memory: relational continuity synthesizer for social-room turns
- orion-llm-gateway manual smoke tests: bus chat/exec_step envelopes, ollama vs vllm backend selection
- config/autonomy/capability_policy.v1.yaml — policy config gating which autonomy capabilities may auto-execute per cycle, by side-effect class, required goal status, required drive origins/signal kinds, and per-cycle budget
- Social GIF Expression Layer
- Orion Landing Pad Guide
>>>>>>> Stashed changes
- live_state vs recovery_state
- Hub OTEL Traces + Metrics Observability Design
- CognitiveUnificationLayer
- Phi seed-v4 Feature Set Design
- Self-State & Mesh Substrate Redesign
- Channel "orion:evidence:index:upsert" (kind=event, schema=EvidenceUnitV1) producers=[orion-sql-writer, *] consumers=[orion-evidence-index, orion-sql-writer, *]
- Channel "orion:graph:compression:stale" (kind=event, schema=CompressionStalenessMarkV1) producers=[orion-rdf-writer, orion-graph-compression] consumers=[orion-graph-compression]
- orion-rdf-writer (bus → triples → RDF store service)
<<<<<<< Updated upstream
- smoke_memory_cognition_loop_e2e.sh
=======
>>>>>>> Stashed changes
- town_cards.yaml (cast source of truth)
- orion-dream Service
<<<<<<< Updated upstream
- fcc_model_mapping.py
- _Store
- test_substrate_attention_debug_api.py
- test_substrate_policy_debug_api.py
- test_substrate_proposal_debug_api.py
- test_turn_cancel.py
- _Conn
- test_registry_dag.py
- test_route_map_completeness.py
- whisper-tts (docker-compose service, GPU TTS/STT)
- test_tts_engine_settings.py
- test_execution_dispatch_bus_catalog.py
- Reasoning Promotion Phase 3
- Phase 5 Research Findings
=======
- whisper-tts (docker-compose service, GPU TTS/STT)
>>>>>>> Stashed changes
- Information-Dynamics Pillars
- OrionSignalV1
- memory.turn.persisted outbox
- Repair Pressure v2 + Pre-Turn Appraisal Rail Design
- inner_state_registry.py / InnerStateSignal
- AutonomyStateV2 Closed-Loop Wiring Design
- Topic Foundry (Windowing v2, Micro/Macro, Enrichment)
<<<<<<< Updated upstream
- verb.schema.json
- services
- enum
- Recall Memory
- Perceive: Retina Fast Pipeline (Embed, Detect, Caption)
- recall.py
- Spark organ (salience, change, concept formation)
- context_exec_beta_gate.sh
- context_exec_golden_probes.sh
- migrate_graphdb_to_fuseki.py
- smoke_actions_daily.sh
- smoke_vision_caption_provenance.sh
- test_scheduler_cursor_state_path.py
- test_conversation_proximity_patch.py
- test_town_chat_turns_patch.py
- orion-attention-runtime service
- get_settings
- filter_world_context_capsule
- up-with-tailscale.sh
- orion-equilibrium-service Docker Compose Service
- TestBaselineHygiene
- test_aitown_proxy.py
- test_self_observability_ui_panel.py
- test_substrate_feedback_debug_api.py
- test_world_pulse_proxy_routes.py
- verify_mind_llm_e2e.sh
- test_llm_uncertainty_telemetry.py
- test_scripts.py
- settings.py
- test_context_exec_proposal_storage_defaults.py
- test_recall_profiles_cards_knobs.py
- test_recent_turn_effect_alerts.py
=======
- Recall Memory
- Perceive: Retina Fast Pipeline (Embed, Detect, Caption)
- Spark organ (salience, change, concept formation)
- orion-attention-runtime service
- orion-equilibrium-service Docker Compose Service
>>>>>>> Stashed changes
- orion-memory-crystallizer: governed cognitive memory crystallization worker; proposes/validates MemoryCrystallizationV1, projects to Chroma/Graphiti/FalkorDB, never canonical without governor
- Agent Git Safety Mechanism Stack
- Workflow Schedule Production Hardening v1
- Substrate Trace Map Template
- Phi Snapshot
- Memory Graph Annotator (Hub) + Dual-Write GraphDB
- GoalProposalEngine v2 (dedupe + semantic goals)
- AutonomyStateV2 Evidence signal_tension Design
- Chat (Generalist)
- Daily Metacog v1
- Self Repo Inspect
- Orion Journaler Service Boundaries and Semantics
- orion-security-watcher (Guard: vision presence/alert debounce service)
- Bounded trigger loop in ConceptWorker.handle_envelope
<<<<<<< Updated upstream
- Phase 3B: Parity Evidence and Cutover-Readiness Model
- diagnose_cortex_bus_stack.py
- git-stash-table.sh
- smoke_active_verbs.sh
- smoke_presence_grounding.py
- verify-bound-capability-live.sh
- settings.py
- _exec_import_guard.py
- test_metacog_trigger_lineage.py
- _FakeRecallClient
- test_situation_prompt_integration.py
- _orch_import_guard.py
- TestBaselineStartupEmit
- conftest.py
- test_hub_direct_inspection_cache.py
- test_memory_crystallization_ui.py
- _ensure_hub_scripts_import_path
- test_memory_review_ui.py
- _FakeBus
- test_crystallization_repository_import.py
- _mind_import_guard.py
- test_sql_chat_windowing.py
- test_webhook_auth.py
- test_vision_persistence_lane.py
- conftest.py
- test_stance_prompt_renders_coloring.py
- test_agent_trace_js.py
- TestIsSafeSparqlIri
- test_memory_graph_core_pure.py
- _load_state_journaler
- test_vision_retina_no_detector.py
- hub_quick_playwright_live.py
=======
- orion-sql-writer: durable bus-to-Postgres persistence consumer
>>>>>>> Stashed changes
- orion-rdf-writer Canonical Writer
- Social Context Window Selection
- Social Thread Choreography
- Daily Delivery Burst After Restart Design
- Memory Graph Draft Viz + Bridge Turns Design
- chat_kids_story verb
- Substrate-Fed Motivation Design (v1)
- Chat History Compactor Design
- CortexOrchAdapter dispatch_failure signal
- Journal/Notification Flood Fix Design
<<<<<<< Updated upstream
- test_dream_trigger_contract.py
- Fact Extraction
- spec:knowledge-forge-ideation-review-v1 (YAML contract)
- SqlWriteRequest
- orion-self-experiments (typed self-experiment registry + context-exec dispatcher)
- cache_fineweb_edu.py
- dependencies
- check_fcc_context_parity.py
- context_exec_agent_route_probe.sh
- grammar_production_truth.sh
- run_answer_depth_proof_suite.py
- smoke_council_debug.sh
- smoke_graphiti_active_packet_search_e2e.sh
- smoke_graphiti_links_e2e.sh
- smoke_graphiti_search_e2e.sh
- smoke_metacog_phase_contract.py
- smoke_orion_bus_transport_full_stack.sh
- smoke_telemetry_normalization.py
- smoke_topic_foundry_bertopic.sh
- smoke_topic_foundry_remote.sh
- smoke_vision_persistence_live.sh
- Settings
- conftest.py
- conftest.py
- test_speech_settings_defaults.py
- entrypoint.sh
- settings.py
- settings.py
- conftest.py
- verify_agent_claude_stream_live.py
- verify_agent_repl_stream_live.py
- self_observability.js
- test_websocket_agent_claude_routing.py
- validate_llamacpp_upgrade.sh
- settings.py
- orion-policy-runtime: Layer 8 substrate service evaluating ProposalFrameV1 against SubstratePolicyV1, persists PolicyDecisionFrameV1 (policy is not execution)
- smoke.sh
- conftest.py
- test_settings.py
- claim:test:0001 (accepted claim fixture)
- test_channel_prefix_guardrail.py
- _NullAsyncCtx
- TestSparqlBuilders
- test_recall_alert_profile.py
- test_workflow_ui_js.py
=======
- Fact Extraction
- spec:knowledge-forge-ideation-review-v1 (YAML contract)
- orion-self-experiments (typed self-experiment registry + context-exec dispatcher)
- orion-policy-runtime: Layer 8 substrate service evaluating ProposalFrameV1 against SubstratePolicyV1, persists PolicyDecisionFrameV1 (policy is not execution)
- claim:test:0001 (accepted claim fixture)
>>>>>>> Stashed changes
- Substrate Atlas (grammar-atom Cytoscape.js graph)
- Channel triage heuristic report (audit_001/reports_postfix, remediated)
- Topology Node: prometheus
- Qwen3 Thinking-Off Policy
- resolve_subject_identity
- CallSyne Handoff Spec
- Hub Mind Tab v1 Completion
- SmolagentsCodeEngine REPL reasoning loop
- agent_repl context-exec mode
- Graphiti Rail Activation Design (A-B-C slices)
- ENABLE_TOOL_SEARCH env contract
- Reverie Narration Continuity Design
- requirements-dev.txt (repo-wide dev/test deps)
- Assess Runtime State
- Dream Cycle
- Finalize Response
<<<<<<< Updated upstream
- conftest.py
=======
>>>>>>> Stashed changes
- spec:substrate-tier-telemetry-v1
- Orion Recall Profiles Overview (Multi-Backend Ensemble Policy)
- graph.compressions.v1 recall profile (unified)
- Autonomous Event-Driven Concept Induction Trigger Loop note
<<<<<<< Updated upstream
- cache_fineweb_edu_sample.sh
- generate_moc_sample.sh
- generate_sample.sh
- smoke_test.sh
- smoke_test_moc.sh
- smoke_test_v2.sh
- train_fineweb_edu_text_12l_768d.sh
- train_moc_fineweb_edu_12l_768d.sh
- train_moc_tinystories_12l_768d.sh
- train_tinystories.sh
- train_v2_fineweb_edu_12l_768d.sh
- train_v2_tinystories.sh
- bootstrap_test_envs.sh
- check_activation_saturation.py
- collapse_mirror_live_path_truth.sh
- complete_fuseki_graphdb_migration.sh
- locate-bound-capability-live-path.sh
- migrate_fuseki_data_to_graphdb.sh
- run_all_audits.sh
- smoke_biometrics_grammar.sh
- smoke_chat_message.sh
- smoke_digest.sh
- smoke_execution_field_digestion.sh
- smoke_memory_crystallization_e2e.sh
- smoke_metacog_source_service.py
- smoke_topic_digest.sh
- smoke_topic_foundry_all.sh
- smoke_topic_foundry_dataset_update.sh
- smoke_topic_foundry_enrich.sh
- smoke_topic_foundry_facets.sh
- smoke_topic_foundry_introspect.sh
- smoke_topic_foundry_preview.sh
- smoke_topic_foundry_preview_conversation_bound.sh
- smoke_topic_foundry_train.sh
- smoke_topic_foundry_windowing.sh
- smoke_turn_effect_evidence.py
- test_service.sh
- smoke_llm_rail.sh
- settings.py
- settings.py
- models.py
- _sync_runner_settings_module
- resolve_service
- test_conftest_cross_service_isolation.py
=======
>>>>>>> Stashed changes
- orion-fcc Docker Compose Service
- orion-feedback-runtime Docker Compose Service
- orion-knowledge-forge Docker Compose
- orion-meta-tags: LLM-based enrichment of collapse events (entities, sentiment, tags) via orion:collapse:triage -> orion:tags:enriched
- orion-notify service (notification policy owner)
- orion-rdf-store: operator/deployment stack for Orion's primary RDF datastore (Apache Jena Fuseki); not a Python service, no app/settings.py/requirements.txt
<<<<<<< Updated upstream
- get_pg_connection
- test_cards_adapter_active_only.py
- test_sql_anchor_since_minutes.py
- down.sh
- up.sh
- test_chat_response_feedback_routing.py
- test_integration_postgres.py
- settings.py
- orion-world-pulse (docker-compose service, Firecrawl-backed curiosity fetch)
- claim:test:bad-ref (disputed claim fixture with dangling references)
- DriveAuditEvent
- TestJournalComposePrompt
- test_print_recent_turn_effects.py
- test_schema_registry_import_does_not_load_substrate
=======
- Golden phi + node attribution (trained encoder overrides phi_now coherence/energy/novelty)
- orion-world-pulse (docker-compose service, Firecrawl-backed curiosity fetch)
- claim:test:bad-ref (disputed claim fixture with dangling references)
>>>>>>> Stashed changes
- Rationale: weights are calibrated starting points, tunable via repair_pressure_v2_eval.py
- chat_template_kwargs Per-Request Reasoning
- Context Exec RLM Integration
- Sebo & Long 2025 - Moral Consideration for AI
- Frontier Buddy Fast Training Design
- Local /mnt/scripts Backup Strategy Design
- Hub Social Room Ops v1 Design
- GitHub Compactor Design
- Testing Contract (Global)
- Env Refresh Exclude Services List
- Direct Answer
- Auto Depth Select
- Concept Induction
- Introspect
- Skills — Docker PS Status
- Skills — GPU NVIDIA SMI Snapshot
- Summarize Context Verb
<<<<<<< Updated upstream
- __init__.py
- __init__.py
- __init__.py
=======
>>>>>>> Stashed changes
- Hub Source Delta Test (smoke-003)
- Laplace's Demon-lite loop (forecast/observe/delta/reflect/adjust)
- orion-execution-dispatch-runtime Docker Compose Service
- Substrate Atlas UI (orion-hub)
- orion-power-guard docker-compose (UPS/SNMP monitoring, on-battery grace, shutdown command, host SSH key mount)
<<<<<<< Updated upstream
- run_server.sh
- __init__.py
- smoke_otel_phase1.sh
- __init__.py
- __init__.py
=======
>>>>>>> Stashed changes
- orion-spark-concept-induction docker-compose.yml
- orion-vision-window (docker-compose service)
<<<<<<< Updated upstream
- __init__.py
=======
>>>>>>> Stashed changes
- /subagent-driven-development command (parallel sprint orchestration)
- Profile: Action Recognition
- Profile: Affect Signals
- Profile: Depth Estimation
- Profile: OCR Read
- Profile: Person Re-ID
- Profile: VLM VQA
- Grammar Production Observe/Deploy Verification
- Unified-turn Brain Shim Sunset Checklist
- orion-signal-gateway-tests CI workflow
- schedule-browser-smoke CI workflow (Playwright)
- Disk Onboarding Utility
- Risk Assessment
- Deep Context Chat
- Chat History Compactor Digest
- Context Exec: Belief Provenance
- Context Exec: Grammar Collision Audit
- Context Exec: Memory Contradiction Review
- Context Exec: Repo Impact Analysis
- Context Exec: Trace Autopsy
- Counterfactual Exploration
- Harness Finalize Reflect
- Housekeep Runtime
- Inspect Docker Container Status
- Inspect GPU Status (NVIDIA)
- List Recent Biometrics Readings
- Action Planning
- Self Critique
- Send Operator Chat Notification
- Show Biometrics Snapshot
- Show Landing Pad Metrics Snapshot
- Skills — Biometrics Raw Recent
- Skills — Biometrics Snapshot
- Skills — Chat discussion window (SQL)
- Skills — Landing Pad Last Events
- Skills — Landing Pad Metrics Snapshot
- Skills — Mesh Ops Round
- Skills — Mesh Refresh Service .env Files
- Skills — Mesh Bring Up All Docker Stacks
- Skills — Repo Recent PRs
- Skills — Notify Chat Message
- Skills — System Time Now
- Synthesize Patterns Verb
- Tag Enrich Verb
- Triage Verb
- Write Guide Verb
- Write Recommendation Verb
- Write Runbook Verb
- Write Tutorial Verb
- biographical.v1 Recall Profile (Cards-Only Autobiographical)
- brain.recall.v1 Recall Profile (Structured Memory, Vector Amputated)
- chat.belief.contradiction.v1 Recall Profile
- chat.belief.open_loop.v1 Recall Profile
- chat.belief.procedural.v1 Recall Profile
- chat.belief.relational.v1 Recall Profile
- chat.belief.semantic.v1 Recall Profile
- chat.story.kids.v1 Recall Profile
<<<<<<< Updated upstream
- orion-aitown-mcp
- orion-cognition
- orion-sapienform
=======
>>>>>>> Stashed changes
- orion-agent-council service
- Collapse Mirrors (causal-density markers)
- Prometheus node (development/utility)
- orion-actions service
- Orion mission (digital-mind / emergent intelligence experiment)
- orion-state-service
- Orion Platform Audit Scripts (scripts/platform/README.md)
- scripts/README.md (smoke test catalog)
- orion-llm-gateway Python dependencies (fastapi, uvicorn, redis[hiredis], pyyaml, numpy, scipy, loguru)
- orion-memory-consolidation Python dependencies (fastapi 0.111, asyncpg, psycopg2-binary, rdflib, chromadb, numpy)
- orion-proposal-runtime docker-compose (port 8119, PROPOSAL_POLICY_PATH, reverie propose/autoaction flags)
- orion-proposal-runtime dependencies (fastapi, sqlalchemy, psycopg2-binary, PyYAML)
- orion-state-service dependencies (fastapi/redis/asyncpg)

## God Nodes (most connected - your core abstractions)
<<<<<<< Updated upstream
1. `ServiceRef` - 1033 edges
2. `BaseEnvelope` - 980 edges
3. `OrionBusAsync` - 628 edges
4. `SchemaRegistration` - 555 edges
5. `Orion Bus Channels Registry (channels.yaml)` - 262 edges
6. `ContextExecRequestV1` - 217 edges
7. `GrammarEventV1` - 183 edges
8. `SubstrateMutationStore` - 181 edges
9. `CortexClientRequest` - 158 edges
10. `LLMMessage` - 156 edges
=======
1. `Orion Bus Channels Registry (channels.yaml)` - 262 edges
2. `Service: orion-hub` - 98 edges
3. `Service: orion-sql-writer` - 77 edges
4. `Service: orion-cortex-exec` - 54 edges
5. `Service: orion-social-memory` - 30 edges
6. `Service: orion-cortex-orch` - 26 edges
7. `Service: orion-rdf-writer` - 24 edges
8. `Service: orion-vision-host` - 21 edges
9. `Service: orion-world-pulse` - 20 edges
10. `Service: orion-spark-concept-induction` - 19 edges
>>>>>>> Stashed changes

## Surprising Connections (you probably didn't know these)
- `Agent Trace inspection modal (Hub UI, fail case screenshot)` --semantically_similar_to--> `Idea 4: Click-through payload cards`  [INFERRED] [semantically similar]
  .verify-run/hub_agent_trace_timeline.png → 2026-07-11-turn-visibility-design-spec.md
- `Agent Trace inspection modal (Hub UI, fail case screenshot)` --conceptually_related_to--> `agent-trace.js (plain-text step consumer)`  [AMBIGUOUS]
  .verify-run/hub_agent_trace_timeline.png → 2026-07-11-turn-visibility-design-spec.md
<<<<<<< Updated upstream
- `Autonomy Origination Measurement Gate (scripts/analysis/README.md)` --semantically_similar_to--> `Phase 3B: Parity Evidence and Cutover-Readiness Model`  [INFERRED] [semantically similar]
  scripts/analysis/README.md → orion/spark/concept_induction/PHASE3B_PARITY_EVIDENCE_READINESS.md
- `Turn Visibility Design Spec (2026-07-11)` --semantically_similar_to--> `/brainstorming command (sentience-development ideation)`  [INFERRED] [semantically similar]
  2026-07-11-turn-visibility-design-spec.md → .claude/commands/superpowers/brainstorming.md
- `Ethics & non-instrumental stance` --semantically_similar_to--> `Privacy and blocked material stay blocked: raw private traces, journals, mirrors, and internal memory artifacts must not leak through convenience surfaces; summaries/projections preserve privacy boundaries`  [INFERRED] [semantically similar]
  README.md → AGENTS.md

## Import Cycles
- 3-file cycle: `services/orion-cortex-exec/app/executor.py -> services/orion-cortex-exec/app/verb_adapters.py -> services/orion-cortex-exec/app/router.py -> services/orion-cortex-exec/app/executor.py`
- 4-file cycle: `services/orion-cortex-exec/app/executor.py -> services/orion-cortex-exec/app/verb_adapters.py -> services/orion-cortex-exec/app/router.py -> services/orion-cortex-exec/app/supervisor.py -> services/orion-cortex-exec/app/executor.py`
- 4-file cycle: `services/orion-cortex-exec/app/executor.py -> services/orion-cortex-exec/app/verb_adapters.py -> services/orion-cortex-exec/app/router.py -> services/orion-cortex-exec/app/pcr_chat_memory.py -> services/orion-cortex-exec/app/executor.py`
- 5-file cycle: `services/orion-cortex-exec/app/executor.py -> services/orion-cortex-exec/app/verb_adapters.py -> services/orion-cortex-exec/app/router.py -> services/orion-cortex-exec/app/supervisor.py -> services/orion-cortex-exec/app/pcr_chat_memory.py -> services/orion-cortex-exec/app/executor.py`
- 5-file cycle: `services/orion-cortex-exec/app/executor.py -> services/orion-cortex-exec/app/verb_adapters.py -> services/orion-cortex-exec/app/router.py -> services/orion-cortex-exec/app/grounding_capsule.py -> services/orion-cortex-exec/app/pcr_chat_memory.py -> services/orion-cortex-exec/app/executor.py`
=======
- `Turn Visibility Design Spec (2026-07-11)` --semantically_similar_to--> `/brainstorming command (sentience-development ideation)`  [INFERRED] [semantically similar]
  2026-07-11-turn-visibility-design-spec.md → .claude/commands/superpowers/brainstorming.md
- `config/llm_profiles.yaml (LLM profile registry)` --references--> `LLM Profiles and Model Routing (section)`  [INFERRED]
  config/llm_profiles.yaml → README.md
- `cortex-orch remediation entry` --references--> `orion-cortex-orch service`  [INFERRED]
  config/mesh_remediation_roster.yaml → README.md

## Import Cycles
- None detected.
>>>>>>> Stashed changes

## Hyperedges (group relationships)
- **Endogenous Runtime Phase Chain (6-16)** — docs_architecture_mentor_gateway_phase6_mentor_gateway, docs_architecture_endogenous_trigger_orchestration_phase7_workflow_orchestrator, docs_architecture_endogenous_runtime_adoption_phase8_adoption_service, docs_architecture_endogenous_runtime_fidelity_phase9_runtime_execution_records, docs_architecture_endogenous_runtime_durability_phase10_record_store [EXTRACTED 0.90]
- **Calibration Durability Loop (write/read/state/debug)** — docs_architecture_phase13b_sql_writer_endogenous_runtime_and_calibration_audit_orion_sql_writer, docs_architecture_phase14_sql_backed_read_path_endogenous_runtime_and_calibration_audit_sql_reader, docs_architecture_phase15_durable_calibration_profile_state_sql_sql_profile_store, docs_architecture_phase16_operator_debug_surface_endogenous_runtime_and_calibration_state_unified_inspection [EXTRACTED 0.85]
- **Concept Induction Workflow to Journal Synthesis Flow** — docs_architecture_chat_invoked_cognitive_workflows_concept_induction_pass, docs_architecture_concept_induction_details_modal_and_journal_synthesis_details_modal, docs_architecture_concept_induction_details_modal_and_journal_synthesis_synthesize_to_journal, docs_architecture_chat_invoked_cognitive_workflows_journal_write_boundary [EXTRACTED 0.85]
- **Reasoning Epistemic Schema-to-Summary Pipeline** — docs_architecture_reasoning_schema_phase1, docs_architecture_reasoning_promotion_phase3, docs_architecture_reasoning_summary_phase4 [EXTRACTED 0.90]
- **Social Room Live Turn Path** — docs_architecture_social_room_bridge_service, docs_architecture_social_room_bridge_hub_social_room, docs_architecture_social_relational_memory_orion_social_memory [EXTRACTED 0.85]
- **Social Shakedown Regression Loop** — docs_architecture_social_scenario_replay_harness, docs_architecture_social_shakedown_workflow, docs_architecture_social_state_inspection [INFERRED 0.75]
- **Frontier Expansion / Landing / Invocation Decomposition** — docs_architecture_unified_cognitive_substrate_phase6_frontier_expansion, docs_architecture_unified_cognitive_substrate_phase7_frontier_landing, docs_architecture_unified_cognitive_substrate_phase8_frontier_invocation [EXTRACTED 0.95]
- **Review Loop: Scheduling, Runtime Execution, Telemetry/Calibration** — docs_architecture_unified_cognitive_substrate_phase10_review_scheduling, docs_architecture_unified_cognitive_substrate_phase11_runtime_review_execution, docs_architecture_unified_cognitive_substrate_phase12_review_telemetry_and_calibration [EXTRACTED 0.95]
- **Policy Lifecycle Durability Evolution (In-Memory to JSON to SQL to Postgres)** — docs_architecture_unified_cognitive_substrate_phase17_policy_adoption_rollout, docs_architecture_unified_cognitive_substrate_phase18_durable_policy_store_and_cache_runtime_wiring, docs_architecture_unified_cognitive_substrate_phase19_sql_policy_persistence, docs_architecture_unified_cognitive_substrate_phase20c_postgres_comparison_parity [INFERRED 0.90]
- **Layer 4-11 Typed Frame Chain** — docs_context_engineering_04_layer_1_to_11_pipeline_fieldstatev1, docs_context_engineering_04_layer_1_to_11_pipeline_fieldattentionframev1, docs_context_engineering_04_layer_1_to_11_pipeline_selfstatev1, docs_context_engineering_04_layer_1_to_11_pipeline_proposalframev1, docs_context_engineering_04_layer_1_to_11_pipeline_policydecisionframev1, docs_context_engineering_04_layer_1_to_11_pipeline_executiondispatchframev1, docs_context_engineering_04_layer_1_to_11_pipeline_feedbackframev1, docs_context_engineering_04_layer_1_to_11_pipeline_consolidationframev1 [EXTRACTED 1.00]
- **Chat History Bus to Vector Memory Ingest Flow** — docs_chat_history_vector_memory_orion_hub, docs_chat_history_vector_memory_chat_history_log_channel, docs_chat_history_vector_memory_orion_vector_host, docs_chat_history_vector_memory_orion_vector_writer, docs_chat_history_vector_memory_vectorupsertv1 [EXTRACTED 1.00]
- **Cognition Trace Fan-Out to SQL/RDF/Vector/Spark** — docs_cognition_trace_contracts_cognitiontracepayload, docs_cognition_trace_contracts_orion_cortex_exec, docs_cognition_trace_contracts_spark_introspector, docs_cognition_trace_contracts_sparktelemetrypayload, docs_cognition_trace_contracts_orion_tissue [EXTRACTED 1.00]
- **Cortex Execution Spine (Orch-Exec-LLM)** — docs_llm_services_cortex_orch, docs_llm_services_cortex_exec, docs_llm_services_llm_gateway, docs_platform_contract_execution_spine [EXTRACTED 0.90]
- **Metacognition Trigger-to-Persist Flow** — docs_metacognition_logging_metacog_trigger_v1, docs_equilibrium_service, docs_llm_services_cortex_exec, docs_metacognition_logging_collapse_mirror_entry_v2 [EXTRACTED 0.85]
- **Proposal Review Lifecycle** — docs_context_exec_beta_runbook_proposal_envelope_v1, docs_context_exec_beta_runbook_proposal_ledger_record_v1, docs_proposal_review_api, docs_proposal_review_api_hub_pending_decisions [EXTRACTED 0.85]
- **Heartbeat Tensor-Network Substrate Program (charter + spec + measurement)** — docs_research_2026_05_01_orion_heartbeat_research_charter, docs_superpowers_specs_2026_05_01_orion_heartbeat_engineering_spec, docs_superpowers_specs_2026_05_01_orion_heartbeat_engineering_spec_mps_substrate_quimb, docs_research_2026_05_01_orion_heartbeat_research_charter_pre_registered_hypotheses, docs_research_2026_05_01_orion_heartbeat_research_charter_active_inference [EXTRACTED 1.00]
- **Organ Signal Gateway signal lineage (schema, registry, OTEL, biometrics adapter)** — docs_superpowers_specs_2026_05_01_organ_signal_gateway_design_orion_signal_v1, docs_superpowers_specs_2026_05_01_organ_signal_gateway_design_causal_dag_registry, docs_superpowers_specs_2026_05_01_organ_signal_gateway_design_otel_trace_propagation, docs_superpowers_specs_2026_05_01_organ_signal_gateway_design_biometrics_adapter, docs_superpowers_specs_2026_05_01_organ_signal_gateway_design_metric_lineage_stages [EXTRACTED 1.00]
- **Memory Cards + Graph Annotator recall/memory stack** — docs_superpowers_specs_2026_05_01_orion_memory_cards_v1_design_memory_card_v1, docs_superpowers_specs_2026_05_01_orion_memory_cards_v1_design_cards_backend, docs_recall_service, docs_superpowers_specs_2026_05_02_memory_graph_annotator_hub_design_synchronous_dual_write, docs_superpowers_specs_2026_05_02_memory_graph_annotator_hub_design_subschema_memory_graph [INFERRED 0.85]
- **Runtime causality signal nexus (trace to OrionSignalV1 join)** — docs_superpowers_specs_2026_05_20_runtime_trace_signal_nexus_design_cognition_trace_adapter, docs_superpowers_specs_2026_05_20_chat_stance_signal_adapter_contract_orion_signal_v1, docs_superpowers_specs_2026_05_20_runtime_trace_signal_nexus_design_cognition_trace_payload, docs_superpowers_specs_2026_05_03_hub_otel_traces_metrics_observability_design_otel_trace_id [INFERRED 0.75]
- **Per-turn memory + turn-change classify pipeline** — docs_superpowers_specs_2026_06_16_memory_consolidation_design_memory_turn_persisted, docs_superpowers_specs_2026_06_16_memory_consolidation_design_logprob_classify, docs_superpowers_specs_2026_06_22_turn_change_appraisal_v1_design_turn_change_appraisal, docs_superpowers_specs_2026_06_22_turn_change_classify_hardening_design_metacog_route [INFERRED 0.85]
- **Mesh health detect to remediate to email escalation** — docs_superpowers_specs_2026_06_18_mesh_bus_resilience_design_numsub_probe, docs_superpowers_specs_2026_06_18_mesh_bus_resilience_design_tiered_remediation, docs_superpowers_specs_2026_06_19_mesh_critical_email_design_escalation_loop [EXTRACTED 0.85]
- **Unified Orion turn cognitive loop (stance -> harness -> finalize -> learning)** — docs_superpowers_specs_2026_07_05_orion_unified_turn_design_thought_event, docs_superpowers_specs_2026_07_05_orion_unified_turn_design_three_beat_finalize, docs_superpowers_specs_2026_07_05_orion_unified_turn_design_substrate_finalize_appraisal, docs_superpowers_specs_2026_07_05_orion_unified_turn_design_post_turn_closure [EXTRACTED 1.00]
- **FCC agent-lane evolution (hub bridge -> GWT dispatch -> unified turn -> MCP)** — docs_superpowers_specs_2026_07_04_hub_agent_claude_design, docs_superpowers_specs_2026_07_05_fcc_cortex_gwt_dispatch_design, docs_superpowers_specs_2026_07_05_orion_unified_turn_design, docs_superpowers_specs_2026_07_06_fcc_claude_mcp_aitown_design [INFERRED 0.85]
- **Repair pressure appraisal rail (relational stance -> speech wiring -> pre-turn appraisal v2)** — docs_superpowers_specs_2026_06_30_orion_relational_stance_v2_design_compile_speech_contract, docs_superpowers_specs_2026_07_03_repair_pressure_speech_wiring_design_repair_pressure_contract, docs_superpowers_specs_2026_07_03_repair_pressure_v2_pre_turn_appraisal_design_pre_turn_appraisal [EXTRACTED 1.00]
- **Move-the-origin-of-wanting-inside 4-area arc** — docs_superpowers_specs_2026_07_07_endogenous_drive_origination_design, docs_superpowers_specs_2026_07_07_voluntary_attention_override_design, docs_superpowers_specs_2026_07_07_phi_intrinsic_reward_value_learning_design, docs_superpowers_specs_2026_07_07_internal_economy_scarcity_allocation_design [EXTRACTED 1.00]
- **φ truthful inner-state → encoder → intrinsic reward arc** — docs_superpowers_specs_2026_07_07_phi_inner_state_truthful_design, docs_superpowers_specs_2026_07_08_phi_encoder_plan2_design, docs_superpowers_specs_2026_07_08_phi_intrinsic_reward_value_learning_design, docs_superpowers_specs_2026_07_07_phi_intrinsic_reward_value_learning_design, inner_state_features_v1 [EXTRACTED 1.00]
- **Unified memory cognition program (encode/read/lifecycle/grounding)** — docs_superpowers_specs_2026_07_08_integrated_memory_cognition_loop_design, docs_superpowers_specs_2026_07_07_consolidation_crystallization_gate_design, docs_superpowers_specs_2026_07_07_purpose_conditioned_recall_design, docs_superpowers_specs_2026_07_07_memory_weight_reinforcement_decay_design, docs_superpowers_specs_2026_07_07_unified_turn_self_grounding_design [EXTRACTED 1.00]
- **Phi truthful-corpus three-spec program (reasoning adapter feeds seed-v4, hygiene independent)** — docs_superpowers_specs_2026_07_09_phi_truthful_corpus_overview, docs_superpowers_specs_2026_07_09_reasoning_telemetry_adapter_design, docs_superpowers_specs_2026_07_09_phi_seedv4_feature_set_design, docs_superpowers_specs_2026_07_09_phi_corpus_hygiene_design [EXTRACTED 1.00]
- **DriveEngine vs AutonomyStateV2 dual-drive-taxonomy unification thread** — docs_superpowers_specs_2026_07_11_drive_engine_concept_induction_deactivation_design, docs_superpowers_specs_2026_07_11_drive_taxonomy_conceptual_audit_design, docs_superpowers_specs_2026_07_11_autonomy_v2_closed_loop_wiring_design, docs_superpowers_specs_2026_07_12_inner_state_unification_design [INFERRED 0.85]
- **Grammar-lane reducer + unbounded-projection LRU cap discipline** — docs_superpowers_specs_2026_07_12_orch_route_grammar_lane_design_routearbitrationprojectionv1, docs_superpowers_specs_2026_07_09_phi_cognitive_motor_unification_design_execution_trajectory_reducer, docs_superpowers_specs_2026_07_09_phi_corpus_hygiene_design_projection_cap [INFERRED 0.75]
- **Endogenous Action Motor-Nerve Patch Series (P0-P5)** — docs_superpowers_specs_2026_07_13_endogenous_action_motor_nerve_spec_dispatch_status_honesty, docs_superpowers_specs_2026_07_13_endogenous_action_motor_nerve_spec_motor_nerve, docs_superpowers_specs_2026_07_14_autonomy_p3_p5_design_drive_satisfaction, docs_superpowers_specs_2026_07_14_autonomy_p3_p5_design_recall_capability, docs_superpowers_specs_2026_07_14_autonomy_p3_p5_design_attention_bound_proposals [EXTRACTED 1.00]
- **Memory Recall Dynamics + Epistemic Honesty Cluster** — docs_superpowers_specs_2026_07_13_memory_recall_reinforcement_decay_wiring_spec_crystallization_dynamics, docs_superpowers_specs_2026_07_13_recall_epistemic_honesty_and_observability_spec_confidence_assignment, docs_superpowers_specs_2026_07_13_recall_epistemic_honesty_and_observability_spec_contradiction_detection, docs_superpowers_specs_2026_07_13_recall_followups_loop_retirement_saturation_gate_spec_belief_revision_digest [EXTRACTED 1.00]
- **Felt-State Arc Corpus-to-Cluster Pipeline** — docs_superpowers_specs_2026_07_13_felt_state_arc_roadmap_spec_mood_arc_corpus, docs_superpowers_specs_2026_07_13_felt_state_arc_roadmap_spec_mood_arc_encoder, docs_superpowers_specs_2026_07_13_felt_state_arc_roadmap_spec_shuffle_gate, docs_superpowers_specs_2026_07_13_felt_state_arc_roadmap_spec_hdbscan_clusters [EXTRACTED 1.00]
- **Vision Perception Pipeline Flow** — docs_vision_services_vision_edge, docs_vision_services_vision_frame_router, docs_vision_services_vision_host, docs_vision_services_vision_window, docs_vision_services_vision_council, docs_vision_services_vision_scribe [EXTRACTED 1.00]
- **World Pulse Coverage Gap-Fill Mechanism** — docs_world_pulse_dev_coverage_contract, docs_world_pulse_dev_compute_coverage, docs_world_pulse_dev_curiosity_gap_fill, docs_world_pulse_dev_hardware_compute_gpu_section [EXTRACTED 0.85]
- **Social memory event family (sole producer orion-social-memory)** — orion_bus_channels_orion_social_participant_continuity, orion_bus_channels_orion_social_room_continuity, orion_bus_channels_orion_social_stance_snapshot, orion_bus_channels_orion_social_relational_update, orion_bus_channels_orion_social_open_thread, orion_bus_channels_orion_social_peer_style, orion_bus_channels_orion_social_room_ritual, orion_bus_channels_orion_social_commitment, orion_bus_channels_orion_social_commitment_resolution, orion_bus_channels_orion_social_claim, orion_bus_channels_orion_social_claim_revision, orion_bus_channels_orion_social_claim_stance, orion_bus_channels_orion_social_claim_attribution, orion_bus_channels_orion_social_claim_consensus, orion_bus_channels_orion_social_claim_divergence, orion_bus_channels_orion_social_bridge_summary, orion_bus_channels_orion_social_clarifying_question, orion_bus_channels_orion_social_deliberation_decision, orion_bus_channels_orion_social_turn_handoff, orion_bus_channels_orion_social_closure_signal, orion_bus_channels_orion_social_floor_decision, orion_bus_channels_svc_orion_social_memory [INFERRED 0.95]
- **World Pulse event family (sole producer orion-world-pulse)** — orion_bus_channels_orion_world_pulse_run_result, orion_bus_channels_orion_stream_world_pulse_run_result, orion_bus_channels_orion_world_pulse_digest_created, orion_bus_channels_orion_world_pulse_digest_item, orion_bus_channels_orion_world_pulse_article_emit, orion_bus_channels_orion_world_pulse_cluster_emit, orion_bus_channels_orion_world_pulse_digest_published, orion_bus_channels_orion_world_pulse_learning_emit, orion_bus_channels_orion_world_pulse_worth_reading, orion_bus_channels_orion_world_pulse_worth_watching, orion_bus_channels_orion_world_pulse_claim_emit, orion_bus_channels_orion_world_pulse_event_emit, orion_bus_channels_orion_world_pulse_entity_emit, orion_bus_channels_orion_world_pulse_situation_brief_upsert, orion_bus_channels_orion_world_pulse_situation_change_emit, orion_bus_channels_orion_world_pulse_graph_upsert, orion_bus_channels_orion_world_context_daily_capsule, orion_bus_channels_orion_world_pulse_publish_status, orion_bus_channels_orion_hub_messages_create, orion_bus_channels_svc_orion_world_pulse [INFERRED 0.95]
- **Hub-originated channel family (sole producer orion-hub)** — orion_bus_channels_orion_conversation_request, orion_bus_channels_orion_cortex_gateway_request, orion_bus_channels_orion_cortex_pre_turn_appraisal_request, orion_bus_channels_orion_exec_request_councilservice, orion_bus_channels_orion_spark_introspect_candidate_log, orion_bus_channels_orion_chat_history_log, orion_bus_channels_orion_chat_history_turn, orion_bus_channels_orion_chat_social_turn, orion_bus_channels_orion_chat_gpt_log, orion_bus_channels_orion_chat_gpt_turn, orion_bus_channels_orion_chat_gpt_message_log, orion_bus_channels_orion_memory_cards_active, orion_bus_channels_orion_tts_intake, orion_bus_channels_orion_stt_intake, orion_bus_channels_orion_thought_request, orion_bus_channels_orion_attention_loop_outcome, orion_bus_channels_orion_harness_run_request, orion_bus_channels_orion_harness_run_cancel, orion_bus_channels_svc_orion_hub [INFERRED 0.95]
- **Cortex-exec-originated channel family (sole producer orion-cortex-exec)** — orion_bus_channels_orion_verb_result, orion_bus_channels_orion_verb_result_wild, orion_bus_channels_orion_effect_wild, orion_bus_channels_orion_dream_log, orion_bus_channels_orion_cortex_pre_turn_appraisal_result_wild, orion_bus_channels_orion_collapse_enrich, orion_bus_channels_orion_collapse_scored, orion_bus_channels_orion_spark_introspect_candidate, orion_bus_channels_orion_cognition_reasoning_call, orion_bus_channels_orion_rdf_worker, orion_bus_channels_orion_exec_request_metatagsservice, orion_bus_channels_orion_exec_request_collapsemirrorservice, orion_bus_channels_orion_cognition_trace, orion_bus_channels_orion_autonomy_goal_planned, orion_bus_channels_orion_endogenous_runtime_record, orion_bus_channels_orion_endogenous_runtime_audit, orion_bus_channels_orion_calibration_profile_audit, orion_bus_channels_orion_substrate_tier_outcomes, orion_bus_channels_svc_orion_cortex_exec [INFERRED 0.95]
- **Spark concept-induction event family (sole producer orion-spark-concept-induction)** — orion_bus_channels_orion_spark_concepts_profile, orion_bus_channels_orion_spark_concepts_delta, orion_bus_channels_orion_memory_drives_state, orion_bus_channels_orion_memory_tension_event, orion_bus_channels_orion_memory_drives_audit, orion_bus_channels_orion_memory_identity_snapshot, orion_bus_channels_orion_memory_goals_proposed, orion_bus_channels_orion_debug_turn_dossier, orion_bus_channels_orion_stream_world_pulse_run_result_dlq, orion_bus_channels_svc_orion_spark_concept_induction [INFERRED 0.95]
- **Harness governor event/result family (sole producer orion-harness-governor)** — orion_bus_channels_orion_harness_run_result_wild, orion_bus_channels_orion_harness_run_artifact, orion_bus_channels_orion_harness_run_step, orion_bus_channels_orion_substrate_finalize_appraisal_request, orion_bus_channels_orion_harness_verdict_artifact, orion_bus_channels_orion_substrate_turn_outcome, orion_bus_channels_orion_substrate_post_turn_closure, orion_bus_channels_svc_orion_harness_governor [INFERRED 0.95]
- **Notification service event family (sole producer orion-notify)** — orion_bus_channels_orion_notify_in_app, orion_bus_channels_orion_notify_persistence_request, orion_bus_channels_orion_notify_config_recipient, orion_bus_channels_orion_notify_config_preference, orion_bus_channels_orion_notify_persistence_receipt, orion_bus_channels_svc_orion_notify [INFERRED 0.95]
- **ChatGPT import pipeline event family (sole producer chatgpt-import)** — orion_bus_channels_orion_chat_gpt_import_run, orion_bus_channels_orion_chat_gpt_conversation, orion_bus_channels_orion_chat_gpt_example, orion_bus_channels_svc_chatgpt_import [INFERRED 0.95]
- **Channels sharing the generic catch-all payload schema (GenericPayloadV1)** — orion_bus_channels_orion_context_exec_event, orion_bus_channels_orion_exec_request_councilservice, orion_bus_channels_orion_llm_reply_wild, orion_bus_channels_orion_agent_council_intake, orion_bus_channels_orion_agent_council_reply_wild, orion_bus_channels_orion_actions_trigger_daily_pulse_v1, orion_bus_channels_orion_actions_trigger_daily_metacog_v1, orion_bus_channels_orion_actions_audit, orion_bus_channels_orion_pad_signal, orion_bus_channels_orion_pad_stats, orion_bus_channels_orion_exec_request_collapsemirrorservice, orion_bus_channels_orion_spark_introspector_reply_wild, orion_bus_channels_orion_world_pulse_publish_status, orion_bus_channels_schema_genericpayloadv1 [INFERRED 0.85]
- **Memory crystallization lifecycle events sharing schema MemoryCrystallizationV1** — orion_bus_channels_orion_memory_crystallization_proposed, orion_bus_channels_orion_memory_crystallization_validated, orion_bus_channels_orion_memory_crystallization_approved, orion_bus_channels_orion_memory_crystallization_rejected, orion_bus_channels_orion_memory_crystallization_quarantined, orion_bus_channels_orion_memory_crystallization_project, orion_bus_channels_orion_memory_crystallization_reinforced, orion_bus_channels_orion_memory_crystallization_auto_activated, orion_bus_channels_schema_memorycrystallizationv1 [INFERRED 0.85]
- **Collapse mirror pipeline events sharing schema CollapseMirrorEntryV2** — orion_bus_channels_orion_collapse_intake, orion_bus_channels_orion_collapse_triage, orion_bus_channels_orion_collapse_sql_write, orion_bus_channels_orion_collapse_events, orion_bus_channels_orion_collapse_enrich, orion_bus_channels_orion_collapse_scored, orion_bus_channels_schema_collapsemirrorentryv2 [INFERRED 0.85]
- **Layer 7-9 autonomy action loop: proposal frame, execution dispatch frame, and drive-tension relief closing the loop from possible action to executed action to homeostatic effect** — services_orion_proposal_runtime_readme_proposalframev1, services_orion_execution_dispatch_runtime_readme_executiondispatchframev1, services_orion_spark_concept_induction_readme_satisfaction_tensions_relief [EXTRACTED 1.00]
- **Readonly capability gating: the policy config's readonly capability rules and their live consumer implementation in orion-spark-concept-induction's readonly recall/fetch dispatch** — config_autonomy_capability_policy_v1_recall_query_readonly, config_autonomy_capability_policy_v1_web_fetch_readonly, services_orion_spark_concept_induction_readme_recall_readonly_capabilities [EXTRACTED 1.00]
- **Mesh Node Topology Consistency (atlas/athena/circe/prometheus across configs)** — config_biometrics_node_catalog_athena, config_biometrics_node_catalog_atlas, config_field_orion_field_topology_v1_node_athena, config_field_orion_field_topology_v1_node_atlas, config_consolidation_consolidation_policy_v1_policy [INFERRED 0.80]
- **Transport Lane Pressure Sensing Pipeline** — config_substrate_lattice_grammar_producer_registry_v1_orion_bus, config_substrate_lattice_transport_lattice_policy_v1_policy, config_field_orion_field_topology_v1_capability_transport, config_attention_field_attention_policy_v1_policy [INFERRED 0.80]
- **Layered Dry-Run/Read-Only Safety Ceiling** — config_execution_dispatch_execution_dispatch_policy_v1_policy, config_substrate_lattice_action_ceiling_policy_v1_policy, config_substrate_lattice_gate_policy_v1_policy, config_policy_substrate_policy_v1_policy [INFERRED 0.80]
- **Proposed turn-hop event spine (schema + channel + relay + swimlane UI)** — 2026_07_11_turn_visibility_design_spec_turn_hop_v1, 2026_07_11_turn_visibility_design_spec_orion_turn_hop_channel, 2026_07_11_turn_visibility_design_spec_turn_hop_relay, 2026_07_11_turn_visibility_design_spec_swimlane_pipeline_view [INFERRED 0.85]
- **Core cognitive-loop organs (Cortex, Recall, Spark, Metacog, Stance)** — readme_cortex_orch, readme_recall, readme_spark, readme_metacognition, readme_stance_assembly_chatstancebrief [INFERRED 0.85]
- **Channel triage audit lineage across audit_001 (pre/postfix) and audit_002** — codex_reviews_audit_001_reports_channel_triage, codex_reviews_audit_001_reports_postfix_channel_triage, codex_reviews_audit_002_reports_channel_triage [INFERRED 0.85]
- **Cognition Pack + Verb System** — orion_cognition_readme_cognition_packs, orion_cognition_packs_executive_pack, orion_cognition_verbs_analyze_text [EXTRACTED 0.90]
- **Context-Exec Investigative Verb Family** — orion_cognition_verbs_context_exec_belief_provenance_context_exec_belief_provenance, orion_cognition_verbs_context_exec_grammar_collision_audit_context_exec_grammar_collision_audit, orion_cognition_verbs_context_exec_memory_contradiction_review_context_exec_memory_contradiction_review, orion_cognition_verbs_context_exec_repo_impact_analysis_context_exec_repo_impact_analysis, orion_cognition_verbs_context_exec_trace_autopsy_context_exec_trace_autopsy [INFERRED 0.85]
- **Capability-Selector-Backed Assessment Verbs** — orion_cognition_verbs_assess_mesh_presence_assess_mesh_presence, orion_cognition_verbs_assess_runtime_state_assess_runtime_state, orion_cognition_verbs_assess_storage_health_assess_storage_health [INFERRED 0.85]
- **Chat Lane Verb Family** — orion_cognition_verbs_chat_general_chat_general, orion_cognition_verbs_chat_quick_chat_quick, orion_cognition_verbs_chat_kids_story_chat_kids_story, orion_cognition_verbs_chat_deep_graph_chat_deep_graph, orion_cognition_verbs_chat_social_room_chat_social_room [INFERRED 0.75]
- **Dream Cycle Consolidation Family** — orion_cognition_verbs_dream_cycle_dream_cycle, orion_cognition_verbs_dream_preprocess_dream_preprocess, orion_cognition_verbs_dream_simple_dream_simple [EXTRACTED 1.00]
- **Recall-then-Draft Journaling Family** — orion_cognition_verbs_daily_metacog_v1_daily_metacog_v1, orion_cognition_verbs_daily_pulse_v1_daily_pulse_v1, orion_cognition_verbs_journal_compose_journal_compose [INFERRED 0.85]
- **Capability-Backed Runtime Inspection Family** — orion_cognition_verbs_housekeep_runtime_housekeep_runtime, orion_cognition_verbs_inspect_docker_container_status_inspect_docker_container_status, orion_cognition_verbs_inspect_gpu_status_inspect_gpu_status, orion_cognition_verbs_list_biometrics_recent_readings_list_biometrics_recent_readings [INFERRED 0.80]
- **Vision Perception Pipeline (Host retina_fast composing Embed/Detect/Caption, chaining into Window/Council/Scribe)** — orion_cognition_verbs_perceive_caption_frame_perceive_caption_frame, orion_cognition_verbs_perceive_detect_open_vocab_perceive_detect_open_vocab, orion_cognition_verbs_perceive_embed_image_perceive_embed_image, orion_cognition_verbs_perceive_retina_fast_perceive_retina_fast, orion_cognition_verbs_perceive_vision_events_perceive_vision_events, orion_cognition_verbs_perceive_vision_memory_perceive_vision_memory [EXTRACTED 1.00]
- **Self-Study Trust Lanes (authoritative -> induced -> reflective -> lane-aware retrieval)** — orion_cognition_verbs_self_repo_inspect_self_repo_inspect, orion_cognition_verbs_self_concept_induce_self_concept_induce, orion_cognition_verbs_self_concept_reflect_self_concept_reflect, orion_cognition_verbs_self_retrieve_self_retrieve [EXTRACTED 1.00]
- **Capability-Backed Selector Verbs (notification/snapshot-metric read verbs sharing identical execution_mode + schema shape)** — orion_cognition_verbs_send_operator_notification_send_operator_notification, orion_cognition_verbs_show_biometrics_snapshot_show_biometrics_snapshot, orion_cognition_verbs_show_landing_pad_metrics_show_landing_pad_metrics [EXTRACTED 1.00]
- **Mesh Operations Skill Family (skills.mesh.*)** — orion_cognition_verbs_skills_mesh_mesh_ops_round_v1_skills_mesh_mesh_ops_round_v1, orion_cognition_verbs_skills_mesh_refresh_service_envs_v1_skills_mesh_refresh_service_envs_v1, orion_cognition_verbs_skills_mesh_tailscale_mesh_status_v1_skills_mesh_tailscale_mesh_status_v1, orion_cognition_verbs_skills_mesh_up_all_services_v1_skills_mesh_up_all_services_v1 [EXTRACTED 1.00]
- **Read-Only Diagnostic Snapshot Skill Family (skills.*)** — orion_cognition_verbs_skills_docker_ps_status_v1_skills_docker_ps_status_v1, orion_cognition_verbs_skills_gpu_nvidia_smi_snapshot_v1_skills_gpu_nvidia_smi_snapshot_v1, orion_cognition_verbs_skills_storage_disk_health_snapshot_v1_skills_storage_disk_health_snapshot_v1, orion_cognition_verbs_skills_biometrics_raw_recent_v1_skills_biometrics_raw_recent_v1, orion_cognition_verbs_skills_biometrics_snapshot_v1_skills_biometrics_snapshot_v1, orion_cognition_verbs_skills_landing_pad_last_events_v1_skills_landing_pad_last_events_v1, orion_cognition_verbs_skills_landing_pad_metrics_snapshot_v1_skills_landing_pad_metrics_snapshot_v1, orion_cognition_verbs_skills_system_time_now_v1_skills_system_time_now_v1 [EXTRACTED 1.00]
- **Substrate Probe Verb Family (substrate.inspect/observe/summarize)** — orion_cognition_verbs_substrate_inspect_substrate_inspect, orion_cognition_verbs_substrate_observe_substrate_observe, orion_cognition_verbs_substrate_summarize_substrate_summarize [EXTRACTED 1.00]
- **Chat Belief Profile Family (Contradiction, Open Loop, Procedural, Relational, Semantic)** — orion_recall_profiles_chat_belief_contradiction_v1_profile, orion_recall_profiles_chat_belief_open_loop_v1_profile, orion_recall_profiles_chat_belief_procedural_v1_profile, orion_recall_profiles_chat_belief_relational_v1_profile, orion_recall_profiles_chat_belief_semantic_v1_profile [INFERRED 0.95]
- **Generative Writing Verbs Family (Guide, Recommendation, Runbook, Tutorial)** — orion_cognition_verbs_write_guide_write_guide, orion_cognition_verbs_write_recommendation_write_recommendation, orion_cognition_verbs_write_runbook_write_runbook, orion_cognition_verbs_write_tutorial_write_tutorial [INFERRED 0.95]
- **Lightweight Non-Vector Chat Recall Profiles** — orion_recall_profiles_chat_general_v1_profile, orion_recall_profiles_chat_continuity_v1_profile, orion_recall_profiles_chat_story_kids_v1_profile, orion_recall_profiles_assist_light_v1_profile [INFERRED 0.75]
- **Journal *.grounded.v1 recall profile family** — orion_recall_profiles_journal_daily_grounded_v1_profile, orion_recall_profiles_journal_daily_metacog_grounded_v1_profile, orion_recall_profiles_journal_notify_grounded_v1_profile, orion_recall_profiles_journal_world_pulse_grounded_v1_profile [EXTRACTED 1.00]
- **reflect.* recall profile family** — orion_recall_profiles_reflect_v1_profile, orion_recall_profiles_reflect_alerts_v1_profile, orion_recall_profiles_reflect_anchor_v1_profile, orion_recall_profiles_reflect_sql_only_v1_profile [INFERRED 0.85]
- **graph.compressions.* recall profile family (global/local/unified)** — orion_recall_profiles_graph_compressions_global_v1_profile, orion_recall_profiles_graph_compressions_local_v1_profile, orion_recall_profiles_graph_compressions_v1_profile [INFERRED 0.85]
- **Spark ConceptProfile Graph-Backend Rollout (Phase 0-4)** — orion_spark_concept_induction_profile_repository_seam, orion_spark_concept_induction_phase2_concept_profile_graph_read_model, orion_spark_concept_induction_phase3a_shadow_rollout_note, orion_spark_concept_induction_phase3b_parity_evidence_readiness, orion_spark_concept_induction_phase4_concept_profile_cutover_note [EXTRACTED 1.00]
- **Brainstorming Session #1 Appendix Ideas 3-10 (self-state/autonomy substrate proposals)** — reviews_pending_2026_06_20_brainstorming_session_1_appendix_idea3_action_outcome_feedback_loop, reviews_pending_2026_06_20_brainstorming_session_1_appendix_idea4_drive_pressures_bridge_self_state, reviews_pending_2026_06_20_brainstorming_session_1_appendix_idea5_substrate_surprise_signal, reviews_pending_2026_06_20_brainstorming_session_1_appendix_idea6_rolling_self_state_archive, reviews_pending_2026_06_20_brainstorming_session_1_appendix_idea7_drive_audit_loop, reviews_pending_2026_06_20_brainstorming_session_1_appendix_idea8_identity_snapshot, reviews_pending_2026_06_20_brainstorming_session_1_appendix_idea9_cross_reducer_coherence_checker, reviews_pending_2026_06_20_brainstorming_session_1_appendix_idea10_predictive_substrate [EXTRACTED 1.00]
- **Evidence-gated cutover/build readiness pattern (measure before cathedral)** — scripts_analysis_readme, orion_spark_concept_induction_phase3b_parity_evidence_readiness, orion_spark_concept_induction_phase4_concept_profile_cutover_note [INFERRED 0.75]
- **Bus observability/wiretap tooling group (bus, mirror, tap)** — services_orion_bus_readme_orion_bus, services_orion_bus_mirror_readme_orion_bus_mirror, services_orion_bus_tap_readme_orion_bus_tap [INFERRED 0.85]
- **orion-bus core Redis stack (bus-core, bus-exporter, bus-observer)** — services_orion_bus_docker_compose_bus_core, services_orion_bus_docker_compose_bus_exporter, services_orion_bus_docker_compose_bus_observer [EXTRACTED 1.00]
- **Field attention production pipeline (digester -> field state -> attention frames)** — services_orion_attention_runtime_readme_orion_attention_runtime, services_orion_attention_runtime_readme_fieldstatev1, services_orion_attention_runtime_readme_fieldattentionframev1, services_orion_attention_runtime_readme_orion_field_digester [EXTRACTED 1.00]
- **Cortex Request Routing Chain (Gateway to Context-Exec)** — services_orion_cortex_gateway_readme_service, services_orion_cortex_orch_readme_service, services_orion_cortex_exec_readme_service, services_orion_context_exec_readme_service [EXTRACTED 1.00]
- **Concept Profile Repository Seam Family** — services_orion_cortex_orch_concept_profile_config_adapter_note_rationale, orion_concept_profile_repository_concept, services_orion_cortex_orch_docker_compose_service, services_orion_cortex_exec_docker_compose_service [INFERRED 0.85]
- **Substrate Layer Aggregation Family (Consolidation, Exec, Orch)** — services_orion_consolidation_runtime_readme_service, services_orion_cortex_exec_readme_service, services_orion_cortex_orch_readme_service [INFERRED 0.65]
- **Bridge/Façade Services Connecting Cognition to External Mediums** — services_orion_dream_readme_orion_dream_service, services_orion_embodiment_readme_orion_embodiment_service, services_orion_fcc_readme_orion_fcc_service [INFERRED 0.65]
- **Health and System-State Monitoring Services** — services_orion_equilibrium_service_readme_orion_equilibrium_service, services_orion_field_digester_readme_orion_field_digester_service, services_orion_feedback_runtime_readme_orion_feedback_runtime_service [INFERRED 0.65]
- **Hub debug/observability UI panel family** — services_orion_hub_templates_index_dashboard, services_orion_hub_static_pressure_analytics_panel, services_orion_hub_static_substrate_lattice_panel, services_orion_hub_static_self_brain_panel [INFERRED 0.85]
- **Graph/knowledge projection service family (compression + temporal graph)** — services_orion_graph_compression_readme_service, services_orion_graphiti_adapter_readme_service, services_orion_graphiti_adapter_docker_compose_falkordb [INFERRED 0.75]
- **Harness governor semantic self-indexing stack (GitNexus + Context Mode)** — services_orion_harness_governor_readme_service, services_orion_harness_governor_readme_gitnexus, services_orion_harness_governor_readme_context_mode [EXTRACTED 1.00]
- **GPU LLM hosting service family (llama-cola, llamacpp, llamacpp-neural)** — services_orion_llama_cola_host_docker_compose_service, services_orion_llamacpp_host_docker_compose_service, services_orion_llamacpp_neural_host_docker_compose_service [INFERRED 0.85]
- **Orion debug/inspector UI surfaces (substrate, substrate atlas, landing pad explorer)** — services_orion_hub_templates_substrate_ui, services_orion_hub_templates_substrate_atlas_ui, services_orion_landing_pad_app_static_landing_pad_index_ui [INFERRED 0.75]
- **Atlas LLM serving pipeline (llamacpp Atlas workers -> gateway route table -> profile registry)** — services_orion_llamacpp_host_docker_compose_atlas_workers_service, services_orion_llm_gateway_readme_service, config_llm_profiles_yaml_registry [EXTRACTED 1.00]
- **Services wired to the shared LLMGatewayService intake channel** — services_orion_llm_gateway_tests_llm_gateway_smoketests, services_orion_memory_consolidation_docker_compose, services_orion_mind_docker_compose [EXTRACTED 1.00]
- **Notify-centered attention/escalation ecosystem** — services_orion_notify_readme, services_orion_notify_digest_readme, services_orion_mesh_guardian_readme [EXTRACTED 1.00]
<<<<<<< Updated upstream
=======
- **RDF Triple Write and Storage Subsystem (Writer + Fuseki Store)** — services_orion_rdf_writer_readme_service, services_orion_rdf_store_readme_service, services_orion_rdf_store_docker_compose_service_config [EXTRACTED 1.00]
>>>>>>> Stashed changes
- **Notification Policy and Digest Family** — services_orion_notify_app_policy_rules_orion_notify_service, services_orion_notify_app_policy_rules_notification_rules, services_orion_notify_digest_requirements_dependencies [INFERRED 0.75]
- **orion-signal-gateway OTel observability stack (collector + Tempo + Grafana)** — services_orion_signal_gateway_otel_collector_config_config, services_orion_signal_gateway_otel_grafana_datasources_config, services_orion_signal_gateway_otel_tempo_config [EXTRACTED 1.00]
- **Social Room Continuity Substrate** — services_orion_social_memory_service, services_orion_social_room_bridge_service, services_orion_social_memory_readme_chat_social_stored_channel [EXTRACTED 1.00]
- **Orion SQL Persistence Family** — services_orion_sql_db_service, services_orion_sql_writer_service, services_orion_state_journaler_service [INFERRED 0.85]
- **Substrate service family (organs contract layer, runtime reducer worker, telemetry)** — services_orion_substrate_organs_readme_service, services_orion_substrate_runtime_readme_service, services_orion_substrate_telemetry_docker_compose_service [INFERRED 0.65]
- **Vector pipeline family (vector-host embeds, vector-writer persists, vector-db stores)** — services_orion_vector_host_readme_service, services_orion_vector_writer_readme_service, services_orion_vector_db_readme_service [INFERRED 0.85]
- **Host Vision Pipeline (Retina -> Router -> Host -> Window -> Council -> Scribe)** — services_orion_vision_retina_readme_service, services_orion_vision_frame_router_readme_service, services_orion_vision_host_readme_service, services_orion_vision_window_readme_service, services_orion_vision_council_readme_service, services_orion_vision_scribe_readme_service [EXTRACTED 1.00]
- **orion-vision-edge Service Bundle (README, compose, deps, debug UI)** — services_orion_vision_edge_readme_service, services_orion_vision_edge_docker_compose_service, services_orion_vision_edge_requirements_deps, services_orion_vision_edge_app_static_index_ui [INFERRED 0.75]
- **Vision Event Persistence Pipeline (Council -> Scribe -> Vector Writer)** — services_orion_vision_council_readme_service, services_orion_vision_scribe_readme_service, services_orion_vector_writer_requirements_service [EXTRACTED 1.00]
- **Knowledge Forge claim-source-spec provenance chain** — tests_fixtures_knowledge_forge_claims_accepted_claim_test_0001_claim, tests_fixtures_knowledge_forge_raw_sources_source_test_fixture_source, tests_fixtures_knowledge_forge_specs_execution_ready_spec_test_compile_spec [EXTRACTED 1.00]
- **Knowledge Forge dangling-reference test family** — tests_fixtures_knowledge_forge_claims_disputed_claim_test_bad_ref_claim, tests_fixtures_knowledge_forge_claims_disputed_claim_test_bad_ref_missing_claim, tests_fixtures_knowledge_forge_claims_disputed_claim_test_bad_ref_missing_source [INFERRED 0.65]
- **orion-whisper-tts service definition family (README + compose + requirements)** — services_orion_whisper_tts_readme_doc, services_orion_whisper_tts_docker_compose_whisper_tts, services_orion_whisper_tts_requirements_dependencies [EXTRACTED 1.00]

<<<<<<< Updated upstream
## Communities (1395 total, 348 thin omitted)

### Community 0 - "Service: orion-vector-writer"
Cohesion: 0.01
Nodes (247): dispatch_autonomy_episode_journal(), _new_reply_channel(), Compose autonomy episode journal via cortex RPC and publish journal write., new_goal_task_id(), test_dispatch_autonomy_episode_journal_publishes_write(), _BusStub, test_process_recall_includes_sql(), test_recall_handler_returns_bundle() (+239 more)

### Community 1 - "Channel "orion:collapse:triage" (kind=event, schema=CollapseMirrorEntryV2) producers=[orion-collapse-mirror] consumers=[orion-meta-tags, orion-vector-writer, orion-sql-writer, orion-actions]"
Cohesion: 0.10
Nodes (13): OrionTissue, Baseline-relative novelty using a rolling z-score of cosine distance.          n, Hybrid coherence:           - If an embedding is provided (spark_vector or featu, Main update cycle:           1. Update expectation (learning)           2. Evolv, Advance the tissue by one or more local-update steps.          v0 local rule:, Lightweight rolling mean/std tracker for novelty + coherence stability., Convert a SurfaceEncoding into a stimulus and integrate it., Compute a low-dimensional "self state" φ from the tissue.          v2: (+5 more)

### Community 2 - "Context-exec Workbench"
Cohesion: 0.02
Nodes (202): build_feedback_category_options(), eval_row_to_v1_v2_compare(), infer_pressure_category_for_eval_row(), pressure_evidence_from_eval_suite_rows(), Helpers to map recall_eval suite rows into mutation pressure metadata (proposal-, Compact V1 vs V2 summary for MutationPressureEvidenceV1.metadata["v1_v2_compare", Build first-class pressure evidence rows from recall_eval-style dicts (manual in, _aitown_convex_internal_url() (+194 more)

### Community 3 - "Hub Gateway (Harness) README"
Cohesion: 0.03
Nodes (101): build_plan_for_verb(), load_prompt_template(), load_verb_yaml(), Honor explicit `services: []` on a plan step; do not treat it as 'inherit defaul, _resolve_step_services(), Unified async message iterator. Yields dicts with fields similar to redis-py's l, Serialized result for a verb execution., VerbResultV1 (+93 more)

### Community 4 - "Channel "orion:stt:intake" (kind=request, schema=STTRequestPayload) producers=[orion-hub] consumers=[orion-whisper-tts]"
Cohesion: 0.02
Nodes (104): DecodeResult, OrionCodec, Bulletproof encode/decode layer.      Goals:     - Never leak raw JSON dicts int, _attempt_from_envelope(), _consumer_suffix(), _expires_at(), _not_before_ms(), QueueRabbit (+96 more)

### Community 5 - "orion-llm-gateway / model routing"
Cohesion: 0.04
Nodes (97): AutonomyAdapter, _extract_state(), _extract_summary_dict(), _pressure_dimensions(), Autonomy summary/state → autonomy_state OrionSignalV1 (Milestone B4)., OrionSignalAdapter, Abstract base class for all Orion signal adapters., Base class for all organ adapters. (+89 more)

### Community 6 - "services/orion-proposal-runtime/README.md — Layer 7 substrate service: converts SelfStateV1 (+ optional attention/field context) into possible actions (ProposalFrameV1), not automatic actions"
Cohesion: 0.03
Nodes (142): _concept(), ConceptNodeV1, EntityNodeV1, EventNodeV1, EvidenceNodeV1, OntologyBranchNodeV1, Unified cognitive substrate contracts (Phase 1 foundation)., Batch envelope for canonical substrate writes/reads. (+134 more)

### Community 7 - "orion-cortex-orch service"
Cohesion: 0.04
Nodes (65): _build_request(), main(), _parse_args(), _service_ref(), _tap(), Hub Gateway (Harness) README, AutoRouteDecisionV1, AutoRouteRecallDecisionV1 (+57 more)

### Community 8 - "Memory Constellation (tri-layer SQL/RDF/vector)"
Cohesion: 0.03
Nodes (83): ContextExecPermissionV1, TraceAutopsyReportV1, AlexZhangRLMEngine, _engine_selection_grep_path(), _engine_selection_risk(), _extract_claim_from_text(), _extract_corr_id_from_text(), _is_engine_selection_query() (+75 more)

### Community 9 - "Channel "orion:feedback:frame" (kind=event, schema=FeedbackFrameV1) producers=[orion-feedback-runtime] consumers=[orion-spark-concept-induction]"
Cohesion: 0.05
Nodes (67): JournalDispatchPolicy, Declarative journal notification dispatch policy, keyed off `trigger_kind`.  Con, Fail-closed lookup: an unregistered trigger_kind sends nothing., resolve_policy(), JournalEntryDraftV1, JournalEntryIndexV1, JournalTriggerV1, test_episode_journal_carries_spawned_correlation_id() (+59 more)

### Community 10 - "Channel "orion:exec:request:RecallService" (kind=request, schema=RecallQueryV1) producers=[orion-cortex-exec, orion-cortex-orch, orion-hub, orion-context-exec, orion-spark-concept-induction] consumers=[orion-recall]"
Cohesion: 0.06
Nodes (108): MutationDecisionV1, MutationPatchV1, MutationPressureV1, MutationProposalV1, MutationTrialV1, Typed mutation contracts for Substrate adaptation V2.1., GraphReviewTelemetryRecordV1, PatchApplier (+100 more)

### Community 11 - "ctx:substrate-tier-telemetry-v1 (context pack metadata)"
Cohesion: 0.09
Nodes (67): build_introspection_context(), execute_chat_workflow(), DummyBus, _load_sql_writer_worker(), Load orion-sql-writer's worker under its own ``app`` package.      The worker im, Production path: digest arrives only as final_text JSON, no metadata dict., A transient DB error on card persist must not discard the digest., _req() (+59 more)

### Community 12 - "Knowledge Forge Lint Report 2026-05-20 (empty)"
Cohesion: 0.08
Nodes (125): SocialArtifactConfirmationV1, SocialArtifactProposalV1, SocialArtifactRevisionV1, SocialCalibrationSignalV1, SocialPeerCalibrationV1, SocialTrustBoundaryV1, SocialClaimAttributionV1, SocialClaimRevisionV1 (+117 more)

### Community 13 - "orion-self-state-runtime docker-compose.yml"
Cohesion: 0.03
Nodes (97): create_entry_from_v2(), ErrorInfo, BaseChassis, ChassisConfig, Hunter, Rabbit, Restart subscriber loops if they exit without an explicit stop signal., RPC / synchronous pattern.     Listens on a single request channel and replies t (+89 more)
=======
## Communities (293 total, 103 thin omitted)

### Community 0 - "Service: orion-vector-writer"
Cohesion: 0.11
Nodes (27): Channel "orion:chat:history:spark_meta:patch" (kind=event, schema=ChatHistorySparkMetaPatchV1) producers=[orion-memory-consolidation] consumers=[orion-sql-writer], Channel "orion:memory:cards:active" (kind=event, schema=MemoryCardV1) producers=[orion-hub] consumers=[orion-memory-crystallizer], Channel "orion:memory:crystallization:approved" (kind=event, schema=MemoryCrystallizationV1) producers=[orion-memory-crystallizer, orion-hub] consumers=[orion-memory-crystallizer, orion-hub, orion-vector-writer], Channel "orion:memory:crystallization:auto_activated" (kind=event, schema=MemoryCrystallizationV1) producers=[orion-memory-consolidation, orion-memory-crystallizer, orion-hub] consumers=[orion-memory-crystallizer, orion-hub, orion-vector-writer], Channel "orion:memory:crystallization:project" (kind=event, schema=MemoryCrystallizationV1) producers=[orion-memory-crystallizer, orion-hub] consumers=[orion-vector-writer], Channel "orion:memory:crystallization:proposed" (kind=event, schema=MemoryCrystallizationV1) producers=[orion-memory-crystallizer, orion-hub] consumers=[orion-memory-crystallizer, orion-hub], Channel "orion:memory:crystallization:quarantined" (kind=event, schema=MemoryCrystallizationV1) producers=[orion-memory-crystallizer, orion-hub] consumers=[orion-memory-crystallizer, orion-hub], Channel "orion:memory:crystallization:reinforced" (kind=event, schema=MemoryCrystallizationV1) producers=[orion-memory-consolidation, orion-memory-crystallizer, orion-hub] consumers=[orion-memory-crystallizer, orion-hub] (+19 more)

### Community 1 - "Channel "orion:collapse:triage" (kind=event, schema=CollapseMirrorEntryV2) producers=[orion-collapse-mirror] consumers=[orion-meta-tags, orion-vector-writer, orion-sql-writer, orion-actions]"
Cohesion: 0.10
Nodes (21): Channel "orion:chat:gpt:conversation" (kind=event, schema=ChatGptConversationV1) producers=[chatgpt-import] consumers=[orion-sql-writer, orion-vector-host], Channel "orion:chat:gpt:example" (kind=event, schema=ChatGptDerivedExampleV1) producers=[chatgpt-import] consumers=[orion-sql-writer, orion-vector-host], Channel "orion:chat:gpt:import:run" (kind=event, schema=ChatGptImportRunV1) producers=[chatgpt-import] consumers=[orion-sql-writer], Channel "orion:chat:gpt:log" (kind=event, schema=ChatGptMessageV1) producers=[orion-hub] consumers=[orion-sql-writer, orion-vector-host], Channel "orion:chat:gpt:message:log" (kind=event, schema=ChatGptMessageV1) producers=[orion-hub] consumers=[orion-sql-writer, orion-vector-host], Channel "orion:chat:gpt:turn" (kind=event, schema=ChatGptLogTurnV1) producers=[orion-hub] consumers=[orion-sql-writer, orion-vector-host], Channel "orion:chat:history:log" (kind=event, schema=ChatHistoryMessageV1) producers=[orion-hub] consumers=[orion-sql-writer, orion-vector-writer, orion-vector-host, orion-spark-concept-induction], Channel "orion:chat:history:turn" (kind=event, schema=ChatHistoryTurnV1) producers=[orion-hub] consumers=[orion-sql-writer, orion-vector-writer, orion-vector-host, orion-spark-concept-induction] (+13 more)

### Community 2 - "Context-exec Workbench"
Cohesion: 0.17
Nodes (12): Turn Visibility Design Spec (2026-07-11), orion:turn:hop bus channel (proposed), Idea 8: Self-observability consumer (deferred), Idea 3: Swimlane pipeline view (not a node graph), Idea 5: Timestamp scrubber (rewind/replay/fast-forward), scripts/trace_unified_turn.py (regex hop detector), Idea 2: Hub turn-hop WebSocket relay, TurnHopV1 schema (proposed) (+4 more)

### Community 4 - "Channel "orion:stt:intake" (kind=request, schema=STTRequestPayload) producers=[orion-hub] consumers=[orion-whisper-tts]"
Cohesion: 0.28
Nodes (9): Context-exec Beta Runbook, AgentChain Replacement, AgentChainService (legacy), belief_provenance mode, Context-exec Workbench, Cortex Sovereignty, repo_impact_analysis mode, RLM Engine (fake/alexzhang) (+1 more)

### Community 5 - "orion-llm-gateway / model routing"
Cohesion: 0.28
Nodes (9): Channel "orion:cortex:gateway:request" (kind=request, schema=CortexChatRequest) producers=[orion-hub] consumers=[orion-cortex-gateway], Channel "orion:cortex:gateway:result:*" (kind=result, schema=CortexChatResult) producers=[orion-cortex-gateway] consumers=[orion-hub], Channel "orion:cortex:request" (kind=request, schema=CortexClientRequest) producers=[orion-hub, orion-cortex-gateway, orion-actions, orion-memory-consolidation] consumers=[orion-cortex-orch], Channel "orion:cortex:result" (kind=result, schema=CortexClientResult) producers=[orion-cortex-orch] consumers=[orion-hub, orion-cortex-gateway], Channel "orion:cortex:result*" (kind=result, schema=CortexClientResult) producers=[orion-cortex-orch] consumers=[orion-hub, orion-cortex-gateway], Schema: CortexChatRequest, Schema: CortexChatResult, Schema: CortexClientResult (+1 more)

### Community 6 - "services/orion-proposal-runtime/README.md — Layer 7 substrate service: converts SelfStateV1 (+ optional attention/field context) into possible actions (ProposalFrameV1), not automatic actions"
Cohesion: 0.33
Nodes (6): llm-gateway remediation entry, Atlas node (primary LLM/GPU workhorse), Circe node (high-density GPU expansion server), orion-llm-gateway / model routing, LLM Profiles and Model Routing (section), Reasoning Trace Philosophy

### Community 7 - "orion-cortex-orch service"
Cohesion: 0.33
Nodes (6): template inspect_attended_target (P5) — target_binding self_state.dominant_attention_targets[0] with a fallback literal to capability:orchestration if the binding fails to resolve; shipped live at base_priority 0.34 per explicit user go-ahead, not dark-shipped at 0.0, since it's a read-only inspect under the same policy gate/risk class as the other already-live inspect templates; carries a falsifiable 7-day kill criterion (distinct target_id count must be >= 3 or the binding isn't doing anything the literal templates don't already do), checked by run_attention_bound_proposal_eval.py, Prerequisites: substrate_policy_decision_frames populated by orion-policy-runtime (port 8120); substrate_proposal_frames and substrate_self_state from Layers 7-6; two manual SQL migrations applied for ExecutionDispatchFrameV1 and substrate_dispatch_results_v1, Attention-bound proposals (P5): ProposalTemplateV1.target_binding lets a template point at a live field on inbound context; the only binding implemented is ATTENTION_FIRST_TARGET_BINDING = self_state.dominant_attention_targets[0] (orion/proposals/builder.py _resolve_binding_target), which never raises and falls through to the template's static target on an empty list or unsupported target_kind; ProposalCandidateV1 gains binding_resolved_from for traceability, Idempotency: one proposal frame per source_self_state_id; re-running the worker for the same self-state snapshot is a no-op; policy/template changes do not regenerate until a new self-state row exists (v1 semantics), ProposalFrameV1 — data flow substrate_self_state + substrate_attention_frames + substrate_field_state -> orion-proposal-runtime -> ProposalFrameV1 -> substrate_proposal_frames; execution_intent on candidates is descriptive only, no policy approval or cortex-exec calls happen here, services/orion-proposal-runtime/README.md — Layer 7 substrate service: converts SelfStateV1 (+ optional attention/field context) into possible actions (ProposalFrameV1), not automatic actions

### Community 8 - "Memory Constellation (tri-layer SQL/RDF/vector)"
Cohesion: 0.33
Nodes (6): Cognitive EKG dashboard + Hub chat panel (screenshot), Bus and platform law, orion-cortex-orch service, Athena node (core services/Hub/orchestration), Orion Hub (main interface), Service spine / cognitive loop (experience->...->changed future stance)

### Community 9 - "Channel "orion:feedback:frame" (kind=event, schema=FeedbackFrameV1) producers=[orion-feedback-runtime] consumers=[orion-spark-concept-induction]"
Cohesion: 0.33
Nodes (6): Memory Constellation (tri-layer SQL/RDF/vector), orion-rdf-writer service, orion-social-memory service, orion-social-room-bridge service, orion-sql-writer service, orion-vector-writer / vector-host / vector-db services

### Community 10 - "Channel "orion:exec:request:RecallService" (kind=request, schema=RecallQueryV1) producers=[orion-cortex-exec, orion-cortex-orch, orion-hub, orion-context-exec, orion-spark-concept-induction] consumers=[orion-recall]"
Cohesion: 0.67
Nodes (3): Channel "orion:feedback:frame" (kind=event, schema=FeedbackFrameV1) producers=[orion-feedback-runtime] consumers=[orion-spark-concept-induction], Schema: FeedbackFrameV1, Service: orion-feedback-runtime
>>>>>>> Stashed changes

### Community 14 - "conjourney"
Cohesion: 0.01
Nodes (179): "public"."action_outcomes", "public"."attention_loop_outcome", "public"."attention_salience_trace", "public"."autonomy_state_v2", "public"."bus_fallback_log", "public"."calibration_profile_audit", "public"."calibration_profiles", "public"."chat_gpt_conversation" (+171 more)

<<<<<<< Updated upstream
### Community 15 - "orion-self-state-runtime requirements.txt"
Cohesion: 0.04
Nodes (67): AutonomyLookupV1, AutonomyRepository, AutonomyRepositoryStatus, _bounded_reason(), build_autonomy_repository(), _classify_query_error(), _dominant_drive_from_evidence(), _drives_facet_ok() (+59 more)

### Community 16 - "test_consolidation_expectations.py"
Cohesion: 0.07
Nodes (66): build_consolidation_frame(), _attention_target_pressure(), _detect_attention_saturated_execution(), _detect_blocked_review_loop(), _detect_dry_run_feedback_loop(), _detect_loaded_but_reliable(), _detect_read_only_policy_loop(), _detect_stable_after_dry_run() (+58 more)

### Community 17 - "websocket_endpoint"
Cohesion: 0.02
Nodes (168): ChatResponseFeedbackEnvelope, api_chat(), api_chat_response_feedback(), api_chat_turn_cancel(), api_debug_cortex_bus_stack(), api_session(), api_workflow_schedule_action(), api_workflow_schedule_history() (+160 more)

### Community 18 - "InMemoryReasoningRepository"
Cohesion: 0.17
Nodes (27): MentorContextSliceV1, MentorGatewayResultV1, MentorProposalItemV1, MentorRequestV1, MentorResponseV1, Mentor gateway contracts for bounded external critique loop (Phase 6)., ReasoningWriteContextV1, MentorProposalV1 (+19 more)

### Community 19 - "LLMMessage"
Cohesion: 0.04
Nodes (158): ChatRequestPayload, ChatResultPayload, LLMMessage, Request payload for LLM chat., Standard response from LLM Gateway.     Strictly normalizes 'text' (Gateway) vs, Vacuum Validator:         1. Ensures 'content' is populated from 'text'., RecallReplyV1, BaseVerb (+150 more)

### Community 20 - "ThoughtEventV1"
Cohesion: 0.07
Nodes (108): Layer attribution evals (spec §11.2) — structural replay with mocked inputs., High surprise appraisal blocks quick lane; low surprise allows deterministic 5b., Misaligned reflection must change voice finalize output vs aligned., Turn N closure with surprise_unresolved exposes reducer-facing strain signal., test_5a_affects_5b_verdict(), test_5b_affects_5c_text(), test_turn_n_error_shifts_turn_n_plus_one_strain(), A 'How are you?' unified turn carries Orion self-context into both passes. (+100 more)

### Community 21 - "GraphDBSubstrateStore"
Cohesion: 0.05
Nodes (13): GraphDBSubstrateStore, SPARQL STRING_LITERAL_LONG2 — safe for JSON payloads with quotes and newlines., GraphDB-backed substrate store with bounded cache fallback., GraphDB accepts SPARQL UPDATE only on the RDF4J statements endpoint, not the rep, SubstrateQueryExecutionMetaV1, SubstrateQueryExecutionV1, SubstrateQueryPlanStepV1, SubstrateQueryPlanV1 (+5 more)

### Community 22 - ".text"
Cohesion: 0.03
Nodes (31): EpisodeSummaryV1, AttentionRuntimeStore, FieldDigesterStore, CompressionStore, _coerce(), _engine(), frames_range(), frames_tail() (+23 more)

### Community 23 - "FieldAttentionFrameV1"
Cohesion: 0.09
Nodes (51): build_attention_frame(), stable_frame_id(), AttentionLimitsV1, AttentionThresholdsV1, AttentionWeightsV1, FieldAttentionPolicyV1, load_attention_policy(), ObservationModesV1 (+43 more)

### Community 24 - "CallSyneRoomMessageV1"
Cohesion: 0.06
Nodes (53): SocialOpenThreadV1, SocialTurnPolicyDecisionV1, CallSyneRoomMessageV1, ExternalRoomMessageV1, ExternalRoomParticipantV1, ExternalRoomPostResultV1, ExternalRoomTurnSkippedV1, Thin transport contract for inbound CallSyne-style room traffic. (+45 more)

### Community 25 - "ChatStanceBrief"
Cohesion: 0.05
Nodes (58): _as_optional_str(), build_journal_entry_index_payload(), _list_of_str(), Build a denormalized journal retrieval payload.      This stays journal-specific, ChatStanceBrief, Bounded internal stance brief used by chat_general speech pass., _brief_response_hazards(), enforce_chat_stance_quality() (+50 more)

### Community 26 - "ProposalEnvelopeV1"
Cohesion: 0.03
Nodes (161): assert_context_exec_proposal_safe(), build_memory_correction_proposal_envelope(), build_patch_proposal_envelope(), MemoryCorrectionProposalV1, PatchProposalV1, ProposalEnvelopeV1, Shared review wrapper for context-exec proposal artifacts., Context-exec may only emit draft/pending_review envelopes with mutation disallow (+153 more)

### Community 27 - "PolicyDecisionFrameV1"
Cohesion: 0.05
Nodes (51): PolicyDecisionFrameV1, PolicyDecisionV1, apply_compaction_delta(), CompactionApplyReceiptV1, CompactionMemoryStore, policy_approves_execution(), Phase G — compaction applier (memory mutation, hard-gated).  THIS IS THE ONE RUN, True only if this delta's proposal was approved *for execution*.      Reverie pr (+43 more)

### Community 28 - "NodeCatalog"
Cohesion: 0.05
Nodes (102): NodeCatalog, NodeProfile, test_node_biometrics_projection_defaults(), test_organ_emission_roundtrip(), test_projection_update_roundtrip(), test_reduction_receipt_requires_schema_version(), test_state_delta_roundtrip(), ActiveNodePressureProjectionV1 (+94 more)

### Community 29 - "EquilibriumService"
Cohesion: 0.04
Nodes (52): Normalized signal contract for Spark., SparkSignalV1, EquilibriumServiceState, EquilibriumSnapshotV1, Versioned heartbeat contract for Titanium bus., Current service state used by the Equilibrium snapshot publisher., Aggregate view of system equilibrium and distress., SystemHealthV1 (+44 more)

### Community 30 - "profile_repository.py"
Cohesion: 0.05
Nodes (70): ConceptCluster, ConceptEvidenceRef, ConceptItem, ConceptProfile, make_concept_id(), Canonical schemas for Concept Induction artifacts., Versioned snapshot of induced concepts., Deterministic-ish concept id helper.      Uses a simple stable hash of the norma (+62 more)

### Community 31 - "models.py"
Cohesion: 0.06
Nodes (67): AttentionItemV1, AutonomyActiveGoalV1, AutonomyGoalHeadlineV1, AutonomyStateV1, AutonomyStateV2, AutonomySummaryV1, CandidateImpulseV1, DriveCompetitionSummaryV1 (+59 more)

### Community 32 - "ConceptSettings"
Cohesion: 0.08
Nodes (22): ClusterResult, ConceptClusterer, _cosine(), _jaccard(), EmbeddingClient, EmbeddingResponse, HTTP client for the vector-host embedding endpoint., spaCy-first concept candidate extractor with graceful fallback. (+14 more)

### Community 33 - "executor.py"
Cohesion: 0.03
Nodes (124): build_compact_skill_catalog(), _bounded_memory_digest(), _load_daily_metacog_template(), test_daily_metacog_prompt_rejects_oversize_without_truncation(), test_daily_metacog_rendered_prompt_stays_bounded(), test_prompt_render_ctx_preserves_journal_lane_bundle_by_default(), test_prompt_render_ctx_strips_debug_recall_bundle_only_when_opted_in(), explain_alerts() (+116 more)

### Community 34 - "main.py"
Cohesion: 0.04
Nodes (90): EmailTransport, _split_mime(), AgentTraceSummaryV1, ChatAttentionRequest, ChatAttentionState, ChatMessageNotification, ChatMessageState, DeliveryAttempt (+82 more)

### Community 35 - "CortexChatRequest"
Cohesion: 0.16
Nodes (91): GoalActionError, GoalActionResult, CognitiveProposalDraftV1, CognitiveProposalReviewV1, CognitiveStanceNoteV1, MutationPressureEvidenceV1, MutationSignalV1, RecallCanaryJudgmentRecordV1 (+83 more)

### Community 36 - "Collapse Mirror split invariant (Strict/Juniper vs Metacog/Orion; rationale: metacog mirrors must never hit Juniper's triage/enrichment pipeline by default)"
Cohesion: 0.03
Nodes (94): Collapse Mirror split invariant (Strict/Juniper vs Metacog/Orion; rationale: metacog mirrors must never hit Juniper's triage/enrichment pipeline by default), Metacog/Spark Surgical Patch Tracker, Oríon identity + response-policy profile, actions.respond_to_juniper_collapse_mirror.v1 verb, attach_llm_uncertainty_to_collapse_payload(), _canonical_phi_hint(), _coerce_change_type_payload(), CollapseMirrorConstraints (+86 more)

### Community 37 - "TopicFoundryBusPublisher"
Cohesion: 0.20
Nodes (15): KgEdgeIngestItemV1, KgEdgeIngestV1, TopicFoundryDriftAlertV1, TopicFoundryEnrichCompleteV1, TopicFoundryRunCompleteV1, get_bus_publisher(), _safe_run(), TopicFoundryBusPublisher (+7 more)

### Community 38 - "app.js"
Cohesion: 0.02
Nodes (92): API_BASE_URL, appendExecutionStepsPanel(), appendSocialInspectionStateList(), applyMindPrefsToControls(), applyPreferenceRows(), audioContext, audioQueue, buildMemoryGraphSuggestUserContent() (+84 more)

### Community 39 - "worker.py"
Cohesion: 0.03
Nodes (126): Telemetry for recall decisions., RecallDecisionV1, fetch_card_fragments(), fetch_card_fragments_guarded(), _neighbor_confidence_ok(), Score memory cards for fusion via embedding cosine similarity (source=cards)., HTTP API request model (backwards compatibility)., RecallCompareRequestBody (+118 more)

### Community 40 - "OrganClass"
Cohesion: 0.03
Nodes (56): normalize_adapter_result(), test_normalize_list(), test_normalize_single_signal(), Causal parent miss notes (spec §7.B)., _sig(), test_no_note_when_parents_present(), test_note_when_parent_missing(), Causal provenance helpers for OrionSignalV1 (shared by gateway and tests). (+48 more)

### Community 41 - "SubstrateMoleculeV1"
Cohesion: 0.05
Nodes (67): compute_daily_rollup(), _contradiction_clusters(), _gradient_stats(), _health_score(), Per-day rollup computation + JSON persistence., Write a daily rollup to ``runs_dir/YYYY-MM-DD.json`` and return the path., Cluster molecules with contradiction>0 by their atom signature., Compute a DailyMetricsV1 for ``day`` using harness events and store state. (+59 more)

### Community 42 - "EndogenousTriggerEvaluator"
Cohesion: 0.11
Nodes (32): EndogenousHistoryEntryV1, EndogenousTriggerDebugV1, EndogenousTriggerRequestV1, EndogenousTriggerSignalV1, EndogenousWorkflowActionV1, EndogenousWorkflowExecutionResultV1, Endogenous trigger orchestration contracts (Phase 7)., ReasoningSummaryV1 (+24 more)

### Community 43 - "MemoryTurnPersistedV1"
Cohesion: 0.14
Nodes (23): _rpc_request(), MemoryTurnPersistedV1, Config, Settings, _load(), test_classify_turn_first_turn_baseline_none(), test_classify_turn_invalid_route_falls_back_to_metacog(), test_classify_turn_llm_failure_preserves_baseline_context() (+15 more)

### Community 44 - "main.py"
Cohesion: 0.04
Nodes (84): VisionArtifactOutputs, VisionArtifactPayload, VisionCaption, VisionEmbedding, VisionObject, VisionScribeAckPayload, VisionScribeRequestPayload, VisionScribeResultPayload (+76 more)

### Community 45 - "SubstratePolicyProfileStore"
Cohesion: 0.07
Nodes (37): Operator-controlled substrate policy profile adoption contracts (Phase 17)., SubstratePolicyAdoptionResultV1, SubstratePolicyAuditEventV1, SubstratePolicyComparisonV1, SubstratePolicyInspectionV1, SubstratePolicyProfileV1, SubstratePolicyResolutionV1, SubstratePolicyRollbackRequestV1 (+29 more)

### Community 46 - "ExecutionDispatchFrameV1"
Cohesion: 0.06
Nodes (70): _aggregate_outcome_status(), build_feedback_frame(), _candidate_outcome_kind(), _cortex_status_to_outcome(), _observation(), _policy_decision_outcome(), _score_for_outcome_kind(), stable_feedback_frame_id() (+62 more)

### Community 47 - "ContextExecRunV1"
Cohesion: 0.04
Nodes (84): ContextExecOperatorSummaryV1, ContextExecRunV1, ContextExecSafetySummaryV1, ContextExecVerbStepV1, Operator-facing summary for Hub Agent mode responses., AgentSynthesisResult, build_operator_summary(), Skip LLM RPC when gateway bus consumer is not ready. (+76 more)

### Community 48 - "finalize_pass.py"
Cohesion: 0.05
Nodes (63): bootstrap_answer_contract_on_request(), build_answer_contract_draft_for_hub(), enrich_answer_contract_after_routing(), heuristic_answer_contract(), investigation_state_for_contract(), merge_draft(), _norm_user_text(), output_modes_for_answer_contract_style() (+55 more)

### Community 49 - "FieldStateV1"
Cohesion: 0.05
Nodes (82): FieldEdgeV1, FieldStateV1, apply_decay(), apply_diffusion(), _clamp01(), Recompute every diffused capability channel fresh from this tick's node/     cap, apply_perturbations(), apply_suppression() (+74 more)

### Community 50 - "ReductionReceiptV1"
Cohesion: 0.05
Nodes (91): ReductionReceiptV1, StateDeltaV1, emission_touches_node(), _normalize_node_id(), Node-scoped filtering for substrate receipts and organ emissions., receipt_touches_node(), state_deltas_for_node(), derive_episode_id() (+83 more)

### Community 51 - "molecules.py"
Cohesion: 0.04
Nodes (73): emit_contradiction(), emit_pressure(), Thin substrate-emit helpers for the autonomy/pressure organ.  These do not modif, A pressure molecule is a constraint+gradient pair.      ``magnitude`` is folded, Emit a contradiction molecule that points at two other molecule ids., build_turn_change_signal(), ConceptAtomV1, Atoms — reusable semantic invariants.  An atom is *not* a domain noun (memory, d (+65 more)

### Community 52 - "BiometricsSubstrateWorker"
Cohesion: 0.03
Nodes (79): execution_prediction_error(), _mean(), 0-1 surprise score: how much did execution pressure hints change this batch?, 0-1 surprise score: how much did transport bus health change this batch?, transport_prediction_error(), clear_tail_seeds_for_tests(), has_cold_start_tail_seed(), has_recent_tail_seed() (+71 more)

### Community 53 - "ProposalCandidateV1"
Cohesion: 0.06
Nodes (71): CortexRouteTemplateV1, build_policy_decision_frame(), build_unevaluable_policy_decision_frame(), A proposal whose source self-state could not be loaded (missing, or a     row sa, stable_policy_frame_id(), evaluate_proposal_candidate(), _finish(), _policy_gate_for_decision() (+63 more)

### Community 54 - "__init__.py"
Cohesion: 0.07
Nodes (56): GoalNodeV1, HypothesisNodeV1, FrontierDeltaItemV1, FrontierExpansionRequestV1, FrontierExpansionResponseV1, FrontierGraphDeltaBundleV1, FrontierSourceProvenanceV1, Frontier expansion contracts for typed substrate graph-delta generation (Phase 6 (+48 more)

### Community 55 - "association.py"
Cohesion: 0.05
Nodes (59): _broadcast_enabled(), _broadcast_is_stale(), build_hub_association_bundle(), _default_reader(), _parse_broadcast(), Ensure the current Hub turn is always a coalition member for fail-closed evidenc, Orion capability: felt-state context for the stance turn.      Supplies Thought, _read_association_data() (+51 more)

### Community 56 - "GrammarEventV1"
Cohesion: 0.04
Nodes (96): apply_grammar_event(), apply_grammar_trace_batch(), _atom_row(), _bulk_insert_derived(), _bulk_insert_events(), _compaction_row(), _created_at(), _edge_row() (+88 more)

### Community 57 - "ServiceState"
Cohesion: 0.06
Nodes (65): AttentionPublisher, ProbeResult, run_probe(), build_compose_build_command(), build_compose_command(), build_compose_up_command(), execute_remediation(), RemediationResult (+57 more)

### Community 58 - "GraphReviewQueue"
Cohesion: 0.06
Nodes (71): Canonical shared schemas for Orion core services., GraphConsolidationDecisionV1, GraphConsolidationResultV1, GraphReviewCycleRecordV1, GraphStateDeltaDigestV1, Bounded reflective graph consolidation contracts (Phase 9)., GraphReviewCycleBudgetV1, GraphReviewCyclePolicyV1 (+63 more)

### Community 59 - "drives.py"
Cohesion: 0.05
Nodes (92): ConceptProfileDelta, Delta between two concept profile revisions., ArtifactEventRef, ArtifactEvidence, ArtifactProvenance, DriveAuditV1, DriveStateV1, GoalProposalV1 (+84 more)

### Community 60 - "chain.py"
Cohesion: 0.06
Nodes (45): One chain occurrence: a theme fired at a time., ThemeEvent, CompactionRequestV1, Readout of one train of thought — successive climbs of the ladder.      Continui, A typed *ask* from the awake reverie (reasoning) to the offline dream     (stora, ReverieChainV1, build_compaction_request(), DbRefractoryStore (+37 more)

### Community 61 - "EmbeddingGenerateV1"
Cohesion: 0.03
Nodes (97): _embed_bus(), ChatGptLogTurnV1, ChatGptMessageV1, ChatGPT imported message payload for isolated SQL + vector fanout., Turn-level ChatGPT import row (prompt + response) for `chat_gpt_log`., ChatHistoryMessageEnvelope, ChatHistoryMessageV1, ChatHistoryTurnEnvelope (+89 more)

### Community 62 - "test_compaction_applier.py"
Cohesion: 0.04
Nodes (77): CompactionMetricsV1, ConsolidateEntryV1, DownscaleEntryV1, MemoryCompactionDeltaV1, PruneEntryV1, Memory-compaction delta — the dream's *proposed* housekeeping (Phase F).  A `Mem, True when there is nothing to compact tonight (an honest zero, not a         fak, A proposed gist card that would *supersede* a batch of episodes.      The card t (+69 more)

### Community 63 - "context_exec.py"
Cohesion: 0.06
Nodes (81): BusConsumerReadinessResult, EvidenceBundle, InvestigationSectionV2, SourceResult, Schema tests for investigation_v2 mode and profile permissions., test_context_exec_permissions_for_llm_profile_agent_read_repo(), test_context_exec_request_accepts_investigation_v2_mode(), test_evidence_bundle_and_source_result_schema() (+73 more)

### Community 64 - "router.py"
Cohesion: 0.05
Nodes (75): _count(), main(), _forward_llm_uncertainty_metadata(), prepare_brain_reply_context(), prepare_chat_quick_reply_context(), Copy gateway meta.llm_uncertainty into execution ctx metadata for Hub spark_meta, Hub quick lane: identity YAML only — no stance/autonomy graph (must stay fast; G, Canonical preparation hook for brain-lane reply context.     Ensures identity an (+67 more)

### Community 65 - "self_study.py"
Cohesion: 0.06
Nodes (100): JournalEntryWriteV1, Backward compatibility: a payload built before this patch (no trigger_kind key, test_trigger_kind_absent_when_no_trigger_kind_on_older_producer_payload(), Direct request to write raw triples or triggers to the RDF writer.     Used for, RdfWriteRequest, SelfConceptEvidenceRefV1, SelfConceptInduceResultV1, SelfConceptReflectResultV1 (+92 more)

### Community 66 - "test_memory_crystallization.py"
Cohesion: 0.06
Nodes (49): fetch_similar_candidates(), Vector-similarity candidate retrieval across ALL active crystallizations, not sc, approve(), can_activate(), GovernorError, quarantine(), Mark crystallization superseded; preserves artifact., A crystallization can only become active via governor path. (+41 more)

### Community 67 - "draft_to_graph"
Cohesion: 0.05
Nodes (82): GraphStoreClient, SparqlUpdateClient, approve_memory_graph_draft(), ApproveOutcome, preview_validate_only(), Validate RDF + SHACL without persistence; return preview card shells., validate → RDF graph store (Fuseki graph-store HTTP + SPARQL update, or legacy G, _sparql_compensate_batch() (+74 more)

### Community 68 - "VisionWindowPayload"
Cohesion: 0.13
Nodes (38): VisionWindowPayload, _attach_raw_model_output(), build_interpretation_prompt(), _clamp_float(), _coerce_event_candidate_item(), _coerce_legacy_events_field(), _coerce_llm_text(), _coerce_salient_observation_item() (+30 more)

### Community 69 - "Service: orion-sql-writer"
Cohesion: 0.04
Nodes (93): Channel "orion:chat:gpt:conversation" (kind=event, schema=ChatGptConversationV1) producers=[chatgpt-import] consumers=[orion-sql-writer, orion-vector-host], Channel "orion:chat:gpt:example" (kind=event, schema=ChatGptDerivedExampleV1) producers=[chatgpt-import] consumers=[orion-sql-writer, orion-vector-host], Channel "orion:chat:gpt:import:run" (kind=event, schema=ChatGptImportRunV1) producers=[chatgpt-import] consumers=[orion-sql-writer], Channel "orion:chat:gpt:log" (kind=event, schema=ChatGptMessageV1) producers=[orion-hub] consumers=[orion-sql-writer, orion-vector-host], Channel "orion:chat:gpt:message:log" (kind=event, schema=ChatGptMessageV1) producers=[orion-hub] consumers=[orion-sql-writer, orion-vector-host], Channel "orion:chat:gpt:turn" (kind=event, schema=ChatGptLogTurnV1) producers=[orion-hub] consumers=[orion-sql-writer, orion-vector-host], Channel "orion:chat:history:log" (kind=event, schema=ChatHistoryMessageV1) producers=[orion-hub] consumers=[orion-sql-writer, orion-vector-writer, orion-vector-host, orion-spark-concept-induction], Channel "orion:chat:history:turn" (kind=event, schema=ChatHistoryTurnV1) producers=[orion-hub] consumers=[orion-sql-writer, orion-vector-writer, orion-vector-host, orion-spark-concept-induction] (+85 more)

### Community 70 - "GrammarProvenanceV1"
Cohesion: 0.08
Nodes (68): ExecutionRunStateV1, ExecutionTrajectoryProjectionV1, GrammarProvenanceV1, _boolish(), compute_pressure_hints(), extract_execution_state_from_events(), _parse_summary_kv(), _utc_now() (+60 more)

### Community 71 - "main.py"
Cohesion: 0.06
Nodes (70): _default_verbs_dir(), _family_for_skill(), load_skill_manifest(), _risk_for_skill(), SkillManifestEntry, Typed self-experiment registry schemas., Accept legacy skill_id payloads or typed experiment fields., SelfExperimentCreateRequestV1 (+62 more)

### Community 72 - "consumer_readiness.py"
Cohesion: 0.10
Nodes (29): bus_consumer_readiness_v1(), check_bus_consumer_readiness(), check_heartbeat_fresh(), _decode_redis_val(), _heartbeat_matches_service(), _parse_last_seen_ts(), Map internal readiness result to telemetry schema (single http_alive source)., redis_pubsub_numsub() (+21 more)

### Community 73 - "SubstrateMutationStore"
Cohesion: 0.04
Nodes (50): CognitiveDraftRecommendationV1, MutationAdoptionV1, MutationQueueItemV1, MutationRollbackV1, SubstrateMutationStore, _utc_now(), test_blocked_apply_attribution_persists_with_reason_and_context(), test_pending_review_compat_no_sql_enum_migration() (+42 more)

### Community 74 - "llm_backend.py"
Cohesion: 0.08
Nodes (41): _build_ollama_payload(), _common_http_client(), _debug_len(), _debug_snippet(), _debug_think_capture(), _execute_llamacpp_native_completion(), _execute_ollama_chat(), _execute_openai_chat() (+33 more)

### Community 75 - "RouterState"
Cohesion: 0.05
Nodes (57): VisionFramePointerPayload, VisionTaskResultPayload, FrameDispatcher, stream_id_from_host_result(), FrameRouterService, healthz(), lifespan(), _log_task_failure() (+49 more)

### Community 76 - "UnifiedRelationalBeliefSetV1"
Cohesion: 0.04
Nodes (102): build_cognitive_projection_for_context(), build_cognitive_projection_for_mind_with_diagnostics(), build_projection_unification_registry(), _env_float(), get_projection_unification_layer(), _publish_tier_outcomes_if_needed(), Shared CognitiveUnificationLayer → CognitiveProjection builder.  Phase-3 seam: m, Return the process-level CognitiveUnificationLayer used by projection builders. (+94 more)

### Community 77 - "TurnAppraisalBundleV1"
Cohesion: 0.05
Nodes (85): _run_pre_turn_appraisal(), _WebSocketLike, build_orion_turn_request(), Thin Orion-mode turn dict — not the Brain chat request builder., PreTurnAppraisalOptionsV1, PreTurnAppraisalRequestV1, TurnAppraisalBundleV1, TurnAppraisalParadigmSliceV1 (+77 more)

### Community 78 - "SelfStateV1"
Cohesion: 0.05
Nodes (82): apply_causal_density_to_entry(), _coerce_self_state(), CollapseMirrorStore, _condition_severity_rank(), enrich_entry(), _get_store(), _label_for_score(), _phi_evidence_score() (+74 more)

### Community 79 - "memory_cards.py"
Cohesion: 0.05
Nodes (76): EvidenceItemV1, MemoryCardCreateV1, MemoryCardEdgeCreateV1, MemoryCardEdgeV1, MemoryCardHistoryEntryV1, MemoryCardPatchV1, MemoryCardV1, Payload for creating a card (Hub POST). (+68 more)

### Community 80 - "AttentionBroadcastProjectionV1"
Cohesion: 0.05
Nodes (59): AttentionFrameV1, attention_broadcast_enabled(), broadcast_projection_from_frame(), build_substrate_attention_frame(), _current_history(), _node_salience(), Continuous global broadcast — rung 3 of the self-modeling loop.  The workspace c, Map graph nodes into workspace signals; tolerant, never raises per-node. (+51 more)

### Community 81 - "view_model.py"
Cohesion: 0.04
Nodes (97): emit_claim(), emit_observation(), Thin substrate-emit helpers for the mind/chat organ.  This module deliberately d, Build an ``observation`` molecule from a chat turn.      The molecule binds two, Build a ``claim`` molecule. Supports/contradicts are other molecule ids., Repair pressure evidence schema (bus/registry layer).  Lives under orion.schemas, RepairEvidenceV1, Substrate-derived appraisers. See docs/plans/substrate/2026-05-23-repair-pressur (+89 more)

### Community 82 - "appraise_repair_pressure"
Cohesion: 0.06
Nodes (90): Compact input summary for Orch vs Exec projection parity comparison., summarize_projection_inputs(), Map CortexClientRequest.recall → recall_cfg dict for ``build_recall_query_v1``., recall_cfg_from_recall_directive(), CortexClientRequest, Public request contract for Cortex-Orch., PlanExecutionRequest, The Payload Contract. (+82 more)

### Community 83 - "BaseSubstrateNodeV1"
Cohesion: 0.12
Nodes (51): BaseSubstrateNodeV1, SubstrateEdgeV1, _component_count(), extract_graph_features(), build_graph_views(), build_graph_views_from_store(), _build_view(), ConceptGraphViewV1 (+43 more)

### Community 84 - "GrammarAtomV1"
Cohesion: 0.06
Nodes (49): test_atom_roundtrip(), test_grammar_event_requires_provenance(), test_invalid_atom_type_rejected(), build_harness_grammar_events(), build_harness_grammar_finalize_events(), compute_harness_reasoning_present(), compute_harness_thinking_source(), _event() (+41 more)

### Community 85 - "attention_frame.py"
Cohesion: 0.07
Nodes (52): AttentionSignalV1, CuriosityCandidateActionV1, CuriositySuppressionV1, compact(), stable_id(), unique(), AutonomySignalDetector, AttentionSignalDetector (+44 more)

### Community 86 - "test_policy_act.py"
Cohesion: 0.08
Nodes (71): ActionOutcomeRefV1, CapabilityDecisionV1, FetchedArticleRefV1, SubstrateActResultV1, SubstrateEpisodeIntentV1, build_episode_narrative_seed(), build_readonly_fetch_query(), curiosity_strength_from_signals() (+63 more)

### Community 87 - "goal_actions.py"
Cohesion: 0.23
Nodes (25): ContradictionFindingV1, EntityLifecycleEvaluationRequestV1, EntityLifecycleEvaluationResultV1, PromotionEvaluationItemV1, PromotionEvaluationRequestV1, Deterministic promotion and lifecycle policy contracts (Phase 3)., PromotionDecisionV1, _ensure_tz() (+17 more)

### Community 88 - "test_stance_react_pipeline.py"
Cohesion: 0.07
Nodes (59): HubAssociationBundleV1, align_evidence_refs_to_coalition(), coalition_ids_from_association(), attended_node_ids + open_loop ids + always the current Hub turn anchor., Snap LLM evidence_refs to coalition-backed ids; default to hub turn anchor., evaluate_trust_rupture_fixture(), _load_fixtures(), Return failure messages for a single fixture; empty when expectations pass. (+51 more)

### Community 89 - "JournalPageIndexService"
Cohesion: 0.07
Nodes (33): JournalRepository, chat_episodes_query(), chat_episodes_status(), journals_query(), journals_status(), rebuild_chat_episodes(), rebuild_journals(), BuildResponse (+25 more)

### Community 90 - "SocialRoomBridgeService"
Cohesion: 0.10
Nodes (63): ExternalRoomPostRequestV1, _callsyne_bridge_post_body(), CallSyneClient, HubClient, Shape for POST /api/bridge/messages using top-level snake_case fields., SocialMemoryClient, SocialRoomBridgeService, _FailingCallSyneClient (+55 more)

### Community 91 - "agent_synthesis.py"
Cohesion: 0.05
Nodes (60): ContextExecRequestV1, Schema validation for context-exec llm_profile., test_context_exec_request_accepts_valid_llm_profiles(), test_context_exec_request_normalizes_llm_profile_case(), test_context_exec_request_omitted_llm_profile_allowed(), test_context_exec_request_rejects_invalid_llm_profile(), _build_synthesis_prompt(), _default_title() (+52 more)

### Community 92 - "detector_worker.py"
Cohesion: 0.05
Nodes (42): VisionEdgeActivityPayload, VisionEdgeError, VisionEdgeHealth, ActivityRateLimiter, build_activity_payload(), labels_from_detections(), publish_activity_if_allowed(), CameraSource (+34 more)

### Community 93 - "reverie.py"
Cohesion: 0.04
Nodes (100): Dependency-free stable identifiers (hashlib only).  Use from thin services (orio, Return ``{prefix}_{sha256(preimage)[:24]}`` from ordered semantic parts., stable_hash_id(), Eval: reverie semantic lift quality bar — referent, voice, grounding., test_bad_meta_fixture_fails_infra_vocab(), test_good_fixture_passes_semantic_gates(), _database_url(), default_referent_loader() (+92 more)

### Community 94 - "Service: orion-spark-introspector"
Cohesion: 0.06
Nodes (45): Channel "orion:autonomy:action:outcome" (kind=event, schema=ActionOutcomeEmitV1) producers=[orion-spark-concept-induction, orion-execution-dispatch-runtime] consumers=[orion-sql-writer, *], Channel "orion:cortex:exec:request" (kind=request, schema=CortexExecRequestPayload) producers=[orion-cortex-orch, orion-thought] consumers=[orion-cortex-exec], Channel "orion:cortex:exec:request:background" (kind=request, schema=CortexExecRequestPayload) producers=[orion-cortex-orch, orion-actions, orion-harness-governor, orion-execution-dispatch-runtime] consumers=[orion-cortex-exec], Channel "orion:cortex:exec:request:chat" (kind=request, schema=CortexExecRequestPayload) producers=[orion-cortex-orch] consumers=[orion-cortex-exec], Channel "orion:cortex:exec:request:spark" (kind=request, schema=CortexExecRequestPayload) producers=[orion-cortex-orch] consumers=[orion-cortex-exec], Channel "orion:cortex:gateway:request" (kind=request, schema=CortexChatRequest) producers=[orion-hub] consumers=[orion-cortex-gateway], Channel "orion:cortex:gateway:result:*" (kind=result, schema=CortexChatResult) producers=[orion-cortex-gateway] consumers=[orion-hub], Channel "orion:cortex:request" (kind=request, schema=CortexClientRequest) producers=[orion-hub, orion-cortex-gateway, orion-actions, orion-memory-consolidation] consumers=[orion-cortex-orch] (+37 more)

### Community 95 - ".patch"
Cohesion: 0.10
Nodes (57): test_ready_200_when_consumer_ready(), test_ready_503_when_intake_bus_not_connected(), _perception_in_convo(), Observability seam: a healthy loop (all other logs exception-only) must still, The 'void' regression, corrected: the worker must NOT send `finishSendingMessage, test_empty_reply_is_not_injected(), test_heartbeat_logs_once_then_throttles(), test_injectable_reply_is_injected() (+49 more)

### Community 96 - "DiscussionWindowResultV1"
Cohesion: 0.07
Nodes (52): assert_chat_compactor_digest_within_budget(), build_quiet_day_chat_digest(), parse_chat_history_compactor_digest_json(), stable_chat_compactor_journal_entry_id(), trim_chat_history_compactor_input(), test_assert_chat_compactor_digest_within_budget(), test_build_quiet_day_chat_digest(), test_chat_history_compactor_digest_v1_rejects_empty_card_summary() (+44 more)

### Community 97 - "RedisStreamWorkQueue"
Cohesion: 0.05
Nodes (64): build_expectations_from_motifs(), detect_motifs(), FeedbackFrameV1, OutcomeObservationV1, health(), get_settings(), Settings, FeedbackRuntimeStore (+56 more)

### Community 98 - "DriveEngine"
Cohesion: 0.08
Nodes (36): ActionOutcomeEmitV1, Bus payload carrying an action outcome for durable persistence via sql-writer., ConceptInductionTrigger, WorkerLivenessState, drive_state_from_values(), DriveEngine, DriveMathConfig, Clamp to [-1, 1] -- same shape as _clamp01 but preserves sign, for         drive (+28 more)

### Community 99 - "persist_context_exec_run"
Cohesion: 0.08
Nodes (35): configured_storage_paths(), ensure_storage_dirs(), _path_status(), persist_context_exec_run(), Best-effort conversion to a JSON-serializable structure.      Never raises; un-s, Persist an immutable forensic bundle for a completed context-exec run.      Retu, _redact(), run_dir() (+27 more)

### Community 100 - "recall_v2.py"
Cohesion: 0.06
Nodes (48): Minimal SPARQL Protocol HTTP clients (Fuseki + generic SPARQL endpoints)., Host + path + query + fragment only (strips userinfo from URL)., Resolve Basic Auth for substrate SPARQL HTTP (query + update).      Precedence (, SPARQL 1.1 Protocol over HTTP: separate query and update endpoints, optional Bas, redact_http_url_for_log(), resolve_substrate_sparql_http_basic_auth(), SparqlHttpClient, SparqlQueryClient (+40 more)

### Community 101 - "SystemHealthV1"
Cohesion: 0.08
Nodes (39): VisionEventBundleItem, VisionEventPayload, build_intake_envelope(), build_rdf_write_request(), build_smoke_bundle_item(), build_vision_event_payload(), coerce_sql_row(), _collect_envelopes() (+31 more)

### Community 102 - "registry.py"
Cohesion: 0.04
Nodes (91): Normalized recall response. Exec exposes debug counts, not raw fragments, by def, RecallResultPayload, EdgeRefV1, EvidenceRefV1, SubjectRefV1, SubstrateActivationV1, FrontierContextRefsV1, FrontierGraphRegionRefV1 (+83 more)

### Community 103 - "substrate_lattice_routes.py"
Cohesion: 0.06
Nodes (50): api_service_logs_services(), build_compose_logs_command(), collect_service_inventory(), discover_loggable_services(), _docker_diagnostics(), _repo_root(), RepoRootResolution, resolve_repo_root() (+42 more)

### Community 104 - "Service: orion-hub"
Cohesion: 0.05
Nodes (64): Channel "orion:attention:loop_outcome" (kind=event, schema=AttentionLoopOutcomeV1) producers=[orion-hub] consumers=[none], Channel "orion:attention:salience:trace" (kind=telemetry, schema=AttentionSalienceTraceV1) producers=[orion-thought] consumers=[none], Channel "orion:chat:history:spark_meta:patch" (kind=event, schema=ChatHistorySparkMetaPatchV1) producers=[orion-memory-consolidation] consumers=[orion-sql-writer], Channel "orion:chat:response:feedback" (kind=event, schema=ChatResponseFeedbackV1) producers=[orion-hub, *] consumers=[orion-sql-writer], Channel "orion:conversation:request" (kind=request, schema=ChatRequestPayload) producers=[orion-hub] consumers=[orion-cortex-orch], Channel "orion:conversation:result" (kind=result, schema=ChatResultPayload) producers=[orion-cortex-orch] consumers=[orion-hub], Channel "orion:council:reply*" (kind=result, schema=ChatResultPayload) producers=[orion-llm-gateway] consumers=[orion-vision-council], Channel "orion:dream:compaction-request" (kind=event, schema=CompactionRequestV1) producers=[orion-thought] consumers=[none] (+56 more)

### Community 105 - "SemanticSynthesisV1"
Cohesion: 0.09
Nodes (57): ActiveCognitiveFrontierV1, ActiveFrontierDiagnosticsV1, AppraisalFeatureVectorV1, DeferredFrontierMatterV1, MindEvidenceItemV1, MindEvidencePackV1, Mind semantic synthesis, appraisal, and stance handoff contracts., SemanticClaimV1 (+49 more)

### Community 106 - "EndogenousRuntimeAdoptionService"
Cohesion: 0.09
Nodes (34): apply_calibration_adoption(), compare_endogenous_runtime_profile_outcomes(), EndogenousRuntimeAdoptionService, inspect_calibration_profile_audit(), inspect_calibration_profile_audit_with_source(), inspect_calibration_profile_state_with_source(), inspect_calibration_profiles(), inspect_endogenous_operator_debug_surface() (+26 more)

### Community 107 - "recall_utils.py"
Cohesion: 0.06
Nodes (60): _format_message_history_for_chat_prompt(), Compact transcript for chat_general / chat_quick Jinja (message_history).     Sk, apply_fast_chat_recall_profile_clamp(), apply_hub_chat_lane_recall_clamp(), _clean_profile(), delivery_safe_recall_decision(), _is_concrete_ops_query(), _normalize_bool() (+52 more)

### Community 108 - "showToast"
Cohesion: 0.06
Nodes (68): applyCapabilityDefaults(), applySegmentsClientFilters(), bindTopicStudioPersistence(), copyText(), executePreview(), exportEventsCsv(), exportKgCsv(), exportSegmentsCsv() (+60 more)

### Community 109 - "Orion Bus Channels Registry (channels.yaml)"
Cohesion: 0.04
Nodes (71): Channel "orion:bridge:social:participant" (kind=event, schema=ExternalRoomParticipantV1) producers=[orion-social-room-bridge] consumers=[orion-sql-writer, *], Channel "orion:bridge:social:room:delivery" (kind=event, schema=ExternalRoomPostResultV1) producers=[orion-social-room-bridge] consumers=[orion-sql-writer, *], Channel "orion:bridge:social:room:intake" (kind=event, schema=ExternalRoomMessageV1) producers=[orion-social-room-bridge] consumers=[orion-sql-writer, *], Channel "orion:bridge:social:room:skipped" (kind=event, schema=ExternalRoomTurnSkippedV1) producers=[orion-social-room-bridge] consumers=[orion-sql-writer, *], Channel "orion:core:events" (kind=event, schema=CoreEventV1) producers=[*] consumers=[*], Channel "orion:rdf:error" (kind=event, schema=SystemErrorV1) producers=[orion-rdf-writer] consumers=[*], Channel "orion:social:bridge-summary" (kind=event, schema=SocialBridgeSummaryV1) producers=[orion-social-memory] consumers=[*, orion-social-room-bridge, orion-hub], Channel "orion:social:claim" (kind=event, schema=SocialClaimV1) producers=[orion-social-memory] consumers=[*, orion-social-room-bridge, orion-hub] (+63 more)

### Community 110 - "WindowStore"
Cohesion: 0.08
Nodes (19): _prior_turns_for(), Re-classify turns that degraded or never received turn_change_appraisal., retry_degraded_classifies(), run_classify_retry_loop(), _spark_meta_dict(), retry_failed_windows(), run_retry_loop(), WindowStore (+11 more)

### Community 111 - "worker.py"
Cohesion: 0.11
Nodes (62): ContradictionNodeV1, NodeRefV1, FrontierInvocationDecisionV1, FrontierInvocationPlanV1, FrontierInvocationRunResultV1, GraphConsolidationRequestV1, build_metacog_perception_brief(), MetacogPerceptionBriefV1 (+54 more)

### Community 112 - "SparqlHttpClient"
Cohesion: 0.07
Nodes (58): Autonomy graph IRIs (no heavy imports — safe for orion-actions / hub)., archive_subject_goals(), archive_subjects(), archive_subjects_drain(), _binding_value(), build_archive_candidates(), build_archive_status_update(), _build_client() (+50 more)

### Community 113 - "curiosity.py"
Cohesion: 0.10
Nodes (39): publish_harness_lifecycle_grammar(), short_error_kind(), publish_harness_step_grammar(), _compile_finalize_overlay(), map_repair_pressure_contract(), Map ingress repair_pressure_contract metadata to harness overlay fields.      Us, build_harness_prompt(), _default_harness_node_name() (+31 more)

### Community 114 - "digest.py"
Cohesion: 0.05
Nodes (67): get_db(), init_models(), generate_biometrics_model(), Returns a SQLAlchemy model with the table name defined in .env → TABLE_NAME, DigestRunDB, NotificationAttemptDB, NotificationRequestDB, build_digest_content() (+59 more)

### Community 115 - "suggest_with_escalation"
Cohesion: 0.06
Nodes (57): _first_json_object(), hub_memory_graph_suggest_text(), _openai_choice_message_text(), Extract memory_graph_suggest model text from CortexChatResult (final_text + step, Recover visible assistant text from gateway ``raw`` OpenAI completion shape., Prefer final_text; fall back to llm_memory_graph_suggest step LLMGatewayService, _step_text_candidates(), extract_gateway_structured_diagnostics() (+49 more)

### Community 116 - "ConceptProfile"
Cohesion: 0.11
Nodes (20): DriveNodeV1, Deterministic bounded dynamics operating on the materialized substrate graph., SubstrateDynamicsEngine, SubstrateGraphMaterializer, InMemorySubstrateGraphStore, Bounded persistent in-memory materialized substrate graph state., _edge(), _healthy_store() (+12 more)

### Community 117 - "worker.py"
Cohesion: 0.06
Nodes (62): cancel_active_grammar_persist(), Cancel the in-flight Postgres query for the active grammar persist on a shard., _apply_spark_meta_patch(), _build_collapse_stored_payload(), _build_social_turn_stored_payload(), _chat_history_llm_uncertainty_scalars(), _chat_history_thought_for_merge(), _coalesce_chat_history_turn_fields() (+54 more)

### Community 118 - "VisionArtifactPayload"
Cohesion: 0.06
Nodes (21): allocate_workspace(), _build_manifest(), _materialize_repo(), Conservative copy from canonical repo into workspace/repo. Fail-open., Allocate a per-run workspace under workspaces/{run_id}/.      Idempotent for the, workspace_dir(), Tests for agent_repl workbench tools: repo navigation, workspace artifacts, patc, test_agent_repl_runtime_debug_includes_max_steps() (+13 more)

### Community 119 - "test_substrate_lattice_routes.py"
Cohesion: 0.12
Nodes (16): _sample_proof_chain_for_gates(), test_gates_action_ceiling_reflects_dispatch_mode(), test_gates_contract_pass_when_below_threshold(), test_gates_contract_watch_when_high(), test_gates_evidence_blocked_when_no_receipts(), test_gates_freshness_blocked_when_stale(), test_gates_freshness_pass(), test_gates_lineage_blocked_when_no_trace_id() (+8 more)

### Community 120 - "test_signal_tension.py"
Cohesion: 0.05
Nodes (66): config/autonomy/signal_drive_map.yaml, _Baseline, DeviationGate, Deviation gate: turn a stream of per-dimension observations into impulses that f, Adaptive per-dimension deviation detector.      Args:         alpha: EWMA weight, Return the deviation impulse (>=0) for this observation, then fold it         in, End-to-end eval for homeostatic drives (spec/plan Task 8).  Replays a synthetic, run() (+58 more)

### Community 121 - "crystallization_routes.py"
Cohesion: 0.11
Nodes (62): emit_crystallization_lifecycle(), detect_contradictions(), detect_duplicates(), DetectionResult, _jaccard(), merge_detection(), _normalize_text(), _token_set() (+54 more)

### Community 122 - "supervisor.py"
Cohesion: 0.04
Nodes (89): BoundCapabilityExecutionRequestV1, AgentOpinion, AuditVerdict, BlinkJudgement, BlinkScores, ContextBlock, CouncilResult, DeliberationRequest (+81 more)

### Community 123 - "EpisodicFederator"
Cohesion: 0.08
Nodes (20): health(), get_settings(), Settings, ExecutionDispatchRuntimeStore, _frame(), _incompatible_dispatch_frame_payload(), test_count_dispatches_today_returns_row_count(), test_count_dispatches_today_zero_when_no_row() (+12 more)

### Community 124 - "OpenLoopV1"
Cohesion: 0.07
Nodes (56): OpenLoopV1, Recorded when top-down goal bias makes a lower-bottom-up loop win — an     inspe, VoluntaryOverrideV1, _apply_voluntary_attention(), Layer top-down goal bias onto the bottom-up frame (spec Step 2).      Default-of, _candidates(), _override_rate(), Eval: voluntary attention override dynamics (spec Step 2).  Replays synthetic ca (+48 more)

### Community 125 - "DecisionRouter"
Cohesion: 0.09
Nodes (42): filter_allowed(), load_verb_catalog(), rank_verbs_for_query(), serialize_shortlist(), _tokenize(), VerbInfo, AutoDepthDecisionV1, DecisionRouter (+34 more)

### Community 126 - "context_mode_hooks_smoke.py"
Cohesion: 0.08
Nodes (57): cleanup_mcp_config(), Unit tests for the pure helpers in scripts/context_mode_hooks_smoke.py.  These n, _stream_lines(), test_check_result_json_line_round_trips(), test_dir_diff_detects_added_file(), test_dir_diff_detects_modified_file(), test_exit_code_zero_only_without_fail(), test_extract_final_text_falls_back_to_assistant_text() (+49 more)

### Community 127 - "pcr_chat_memory.py"
Cohesion: 0.05
Nodes (59): PcrChatMemoryV1, Purpose-conditioned recall (PCR) schema types., Cortex ctx shape for PCR chat memory surfaces., _ctx_user_text_for_skill_hints(), _journal_pageindex_compose_context(), _journal_pageindex_query(), _last_user_message(), RecallService bus RPC. If ``rpc_timeout_sec`` is omitted, wait is ``min(STEP_TIM (+51 more)

### Community 128 - "main.py"
Cohesion: 0.04
Nodes (62): Clock, Periodic ticker/loop with safe cancellation., BiometricsContext, RPC request payload for the state read-model service., RPC reply payload from the state read-model service., StateGetLatestRequest, StateLatestReply, BiometricsClusterV1 (+54 more)

### Community 129 - "TransportBusProjectionV1"
Cohesion: 0.17
Nodes (23): TransportBusProjectionV1, TransportBusStateV1, _boolish(), compute_transport_pressures(), extract_transport_bus_state_from_events(), parse_bus_transport_trace_id(), _parse_summary_kv(), _utc_now() (+15 more)

### Community 130 - "models.py"
Cohesion: 0.05
Nodes (49): DatasetCreateRequest, DatasetCreateResponse, DatasetListResponse, DriftListResponse, DriftRecord, DriftRunRequest, DriftRunResponse, EventListResponse (+41 more)

### Community 131 - "repo_tools.py"
Cohesion: 0.05
Nodes (68): OrganRuntime, Read-only organ broker for a single context-exec run., record_repo(), _diff_path_from_header(), _is_allowed(), _is_denied(), _normalize_rel_path(), patch_validate() (+60 more)

### Community 132 - "Service: orion-cortex-exec"
Cohesion: 0.06
Nodes (45): Channel "orion:autonomy:goal:planned" (kind=event, schema=AutonomyGoalPlannedV1) producers=[orion-cortex-exec] consumers=[orion-cortex-exec, *], Channel "orion:calibration:profile:audit" (kind=event, schema=CalibrationProfileAuditV1) producers=[orion-cortex-exec] consumers=[orion-sql-writer], Channel "orion:cognition:reasoning_call" (kind=telemetry, schema=ReasoningCallV1) producers=[orion-cortex-exec] consumers=[orion-thought], Channel "orion:collapse:enrich" (kind=event, schema=CollapseMirrorEntryV2) producers=[orion-cortex-exec] consumers=[orion-collapse-mirror], Channel "orion:collapse:events" (kind=event, schema=CollapseMirrorEntryV2) producers=[orion-collapse-mirror, orion-cortex-exec] consumers=[orion-timeline, orion-athena-spark-introspector], Channel "orion:collapse:intake" (kind=event, schema=CollapseMirrorEntryV2) producers=[orion-cortex-exec, orion-collapse-mirror] consumers=[orion-collapse-mirror], Channel "orion:collapse:scored" (kind=event, schema=CollapseMirrorEntryV2) producers=[orion-cortex-exec] consumers=[orion-collapse-mirror], Channel "orion:collapse:sql-write" (kind=event, schema=CollapseMirrorEntryV2) producers=[orion-collapse-mirror] consumers=[orion-sql-writer] (+37 more)

### Community 133 - "fusion.py"
Cohesion: 0.04
Nodes (75): test_fuse_dedupe_and_limit(), test_self_factual_filters_exclude_induced_and_reflective_candidates(), _backend_weights(), _belief_source_rank(), _candidate_allowed(), _cards_rail_enabled(), _denial_patterns(), _exact_match_boost() (+67 more)

### Community 134 - "resolve"
Cohesion: 0.07
Nodes (37): resolve(), test_grounding_capsule_registered(), test_all_proposal_control_plane_schemas_registered(), test_proposal_execution_receipt_registered_schema_round_trip(), test_proposal_ledger_schemas_resolve(), test_schemas_registered(), Regression: the delta must be resolvable by the runtime `_REGISTRY`     (resolve, test_delta_kind_resolves_via_runtime_registry() (+29 more)

### Community 135 - "test_route_substrate_reducer.py"
Cohesion: 0.10
Nodes (44): RouteArbitrationProjectionV1, RouteArbitrationRunStateV1, _noop_receipt(), Deterministic substrate receipt and state-delta identifiers., _sorted_join(), stable_delta_id(), stable_receipt_id(), _boolish() (+36 more)

### Community 136 - "test_curiosity_reuse.py"
Cohesion: 0.11
Nodes (35): CapabilityEvaluationContext, _decision(), _env_bool(), _env_float(), evaluate_capability(), _find_rule(), _goal_status_level(), _layer_a_episode_journal_enabled() (+27 more)

### Community 137 - "test_reducer_lane_adapters.py"
Cohesion: 0.11
Nodes (32): _clamp(), _coerce(), _make_prov(), map_biometrics_ctx_to_substrate(), Biometrics-pressure adapter — binds the substrate's biometric "felt state".  Map, Map ``ctx['active_node_pressure_projection']`` → biometric belief nodes., _clamp(), _coerce() (+24 more)

### Community 138 - "main.py"
Cohesion: 0.07
Nodes (42): asterisk_cmd(), bootstrap_asterisk_and_cisco(), Ensure Asterisk dirs exist, write core configs and SEP<MAC>.cnf.xml (only if mis, Start in.tftpd (tftpd-hpa) serving /tftpboot., Minimal rtp.conf so RTP has sane defaults., Start Asterisk in the foreground (we keep the Popen object)., Run an Asterisk CLI command and capture output., Write content only if file doesn't exist or is empty. (+34 more)

### Community 139 - "VisionRunner"
Cohesion: 0.06
Nodes (36): is_caption_prompt_echo(), _normalize(), True when caption text is the VLM prompt echoed back, not scene description., test_is_caption_prompt_echo_matches_current_prompt(), test_is_caption_prompt_echo_matches_legacy_phrase(), test_is_caption_prompt_echo_rejects_blip_suffix_noise(), test_is_caption_prompt_echo_rejects_real_caption(), sanitize_caption() (+28 more)

### Community 140 - "EmbodimentIntentV1"
Cohesion: 0.09
Nodes (45): ArbiterDecision, ArbiterState, decide(), Pure arbitration. Mutates only ``state.deliberate_hold_until`` on accept., DriveMapThresholds, map_drive_state_to_intent(), build_intent(), Single builder so the non-empty ``reason`` contract is enforced everywhere. (+37 more)

### Community 141 - "mind_runtime.py"
Cohesion: 0.03
Nodes (105): _fresh_prefetch_diagnostics(), log_mind_projection_prebuild_ctx_summary(), prefetch_recall_bundle_for_projection(), Recall bundle prefetch for Mind preflight (Orch, before Exec)., Run recall bus RPC; return (ctx_merge, diagnostics).      On failure/timeout ret, Structured pre-build ctx summary (after recall prefetch, before projection)., active_turn_ids_for_recall(), build_recall_query_v1() (+97 more)

### Community 142 - "test_execution_dispatch_runtime_worker.py"
Cohesion: 0.22
Nodes (28): _candidate(), _FakeClient, _frame_with_candidates(), _make_worker(), _patch_bus_and_client(), _policy_frame(), _proposal(), Stand-in for ExecutionDispatchCortexClient -- returns canned results     or rais (+20 more)

### Community 143 - "main.py"
Cohesion: 0.06
Nodes (43): _cast_embedding_to_vecf32(), _embed_query(), ensure_graphiti_indices(), _ensure_target_entity_stub(), _extract_crystallization_ids(), _falkor_driver(), _filter_intimate_crystallization_ids(), _get_search_stack() (+35 more)

### Community 144 - "HarnessRunV1"
Cohesion: 0.10
Nodes (48): apply_context_overflow_hint(), is_context_overflow_text(), execute_unified_turn(), _finalize_phase_error(), _harness_error_frame(), _partial_draft_from_run(), _publish_unified_turn_chat_grammar(), _publish_unified_turn_chat_history() (+40 more)

### Community 145 - "test_publish_paths.py"
Cohesion: 0.06
Nodes (66): metabolize_substrate_signals(), _gpu_gap_result(), test_metabolism_skips_covered_sections(), test_metabolism_sparse_gpu_section_raises_predictive(), build_world_pulse_prompt_seed(), Deterministic compact seed for journal.compose (no LLM inside world-pulse pipeli, DailyWorldPulseItemV1, DailyWorldPulseSectionsV1 (+58 more)

### Community 146 - "test_turn_change_classify.py"
Cohesion: 0.17
Nodes (21): appraisal_confidence(), binary_margin(), build_change_only_prompt(), _clip_pair(), enum_scores_from_top_logprobs(), novel_margin_below_threshold(), parse_novel_shift_lines(), Lift novelty when shift mass is strong but the NOVEL line says NO. (+13 more)

### Community 147 - "ConceptRelationDecision"
Cohesion: 0.09
Nodes (23): _build_relation_prompt(), ConceptRelationDecision, maybe_resolve_concept_relation(), merge_new_evidence(), Bounded structured-output LLM call. NEVER raises -- degrades to     ConceptRelat, Returns a (crystallization_id, row, outcome) tuple ONLY when it took a decisive, Local to this seam -- not a bus-published event, no registry entry needed     (s, Moved here from intake_pipeline.py so both the same-window Jaccard reinforce (+15 more)

### Community 148 - "service_logs.py"
Cohesion: 0.13
Nodes (33): Append-only social_room turn persistence payload., Post-commit stored event emitted by sql-writer., SocialGroundingStateV1, SocialRedactionScoreV1, SocialRoomTurnStoredV1, SocialRoomTurnV1, artifact_dialogue_records(), _artifact_scope_matches() (+25 more)

### Community 149 - "thought.py"
Cohesion: 0.07
Nodes (61): StanceReactRequestV1, build_stance_react_failure_thought(), Valid defer ThoughtEventV1 for RPC reply channels that require ThoughtEventV1., slim_association_for_prompt(), slim_repair_bundle_for_prompt(), build_stance_react_context(), build_stance_react_plan_request(), _coalition_projection() (+53 more)

### Community 150 - "CalibrationAdoptionRequestV1"
Cohesion: 0.10
Nodes (27): CalibrationAdoptionRequestV1, CalibrationAdoptionResultV1, CalibrationProfileAuditV1, CalibrationProfileResolutionV1, CalibrationProfileV1, CalibrationRollbackRequestV1, CalibrationRollbackResultV1, CalibrationRolloutScopeV1 (+19 more)

### Community 151 - "attention_loops_store.py"
Cohesion: 0.08
Nodes (30): _cards_enabled(), _close(), CloseRequest, dismiss_loop(), list_loops(), publish_loop_outcome(), Operator Pending Attention API — cognitive-loop rows + Resolve/Dismiss.  Flag-ga, Best-effort publish of the label event; swallow any bus failure. (+22 more)

### Community 152 - "brain_frame_producer.py"
Cohesion: 0.08
Nodes (46): BrainEdgeSampleV1, BrainNodeSampleV1, BrainRegionV1, BrainSpotlightV1, The spine of the contract: a continuous, trackable per-region signal., Best-effort decoration. NO continuity guarantee across frames., SubstrateBrainFrameV1, assemble_brain_frame() (+38 more)

### Community 153 - "test_social_memory_service.py"
Cohesion: 0.10
Nodes (54): _FakeBus, _payload(), _service_and_session(), test_accepted_artifact_confirmation_expands_active_continuity(), test_accepted_confirmation_without_clear_scope_stays_non_active(), test_active_commitment_is_selected_over_old_ritual_hint(), test_addressed_peer_context_is_preferred_over_generic_room_context(), test_ambiguous_divergence_prefers_one_clarifying_question() (+46 more)

### Community 154 - "test_memory_crystallization_dynamics.py"
Cohesion: 0.09
Nodes (34): decay_activation(), Half-life activation decay — stdlib only (safe for lightweight service imports)., _aware(), _clamp(), decay(), decayed_activation(), Dynamic memory weight for crystallizations: encode weakly, strengthen on reinfor, Activation decayed to `now` on its half-life. Pure read — does not mutate. (+26 more)

### Community 155 - "WorldPerceptionV1"
Cohesion: 0.08
Nodes (46): build_speech_prompt(), _interlocutor_name(), is_injectable(), latest_partner_line(), _participants(), Pure helpers for the cortex-generated town speech bridge.  No I/O. The worker ow, Anti empty-shell guard: only non-empty, non-whitespace replies are injectable., True only when ``own_player_id`` is a participant of an active conversation. (+38 more)

### Community 156 - "grammar.py"
Cohesion: 0.11
Nodes (42): ChatSessionProjectionV1, ChatTurnStateV1, compute_chat_pressure_hints(), extract_chat_turn_state(), _parse_trace_id(), Parse hub.chat:{node_id}:{turn_id} → (node_id, turn_id)., _utc_now(), process_chat_grammar_events() (+34 more)

### Community 157 - "test_harness_governor_client_liveness.py"
Cohesion: 0.10
Nodes (24): _get_message_within(), HarnessGovernorClient, pubsub.get_message() performs exactly one read per call: if that single read, _FakeBus, _FakePubSub, _FakeWorkerBus, HarnessGovernorClient.run should extend its RPC wait while the governor is still, Mimics the REAL redis.asyncio PubSub.get_message() one-read-per-call semantics: (+16 more)

### Community 158 - "engine.py"
Cohesion: 0.11
Nodes (47): Orion Mind shared contracts (types only; runtime lives in services/orion-mind)., MindStanceHandoffV1, test_hash_snapshot_inputs_stable(), test_mind_run_result_roundtrip(), test_universe_snapshot_facets(), test_validate_merged_stance_brief_accepts_minimal(), MindControlDecisionV1, MindHypothesisV1 (+39 more)

### Community 159 - "SparkStateSnapshotV1"
Cohesion: 0.08
Nodes (34): ReasoningSparkStateSnapshotV1, Canonical spark source seam used by reasoning adapters (Phase 5)., SparkSourceSnapshotV1, Normalization utilities for Orion payloads., _as_mapping(), _coerce_datetime(), _coerce_telemetry_timestamp(), normalize_spark() (+26 more)

### Community 160 - "bus_worker.py"
Cohesion: 0.13
Nodes (30): _overall_action_pressure(), _overall_risk(), _artifact_id(), _canonical_pressures_for_spread(), clamp01(), _delta_from_turn_effect(), derive_pressure_competition_tensions(), extract_tensions() (+22 more)

### Community 161 - "social_room.py"
Cohesion: 0.06
Nodes (61): SocialSkillRequestV1, SocialSkillResultV1, SocialSkillSelectionV1, SocialStyleAdaptationSnapshotV1, api_debug_skill_runner_deterministic(), Resolve prompt to catalogue verb (no cortex). Use to verify deterministic-lane e, build_cortex_chat_request(), _build_recall_payload() (+53 more)

### Community 162 - "thought-process.js"
Cohesion: 0.10
Nodes (52): asList(), asObject(), attach(), buildExecutionStepsPanel(), cleanText(), collectBase(), compactProjectionItems(), comparisonColumn() (+44 more)

### Community 163 - "main.py"
Cohesion: 0.23
Nodes (4): _decode_envelope(), Bounded Redis recovery index for vision-window (design §4.2, §4.3). Stores JSON, Return (latest_cursor, earliest_cursor) from last_n list if parseable., RecoveryStore

### Community 164 - "situation.py"
Cohesion: 0.12
Nodes (42): AgendaContextV1, EnvironmentContextV1, LabContextV1, RequestorContextV1, SituationAffordanceV1, SituationDiagnosticsV1, SituationPromptFragmentV1, SurfaceContextV1 (+34 more)

### Community 166 - "snapshot_from_window"
Cohesion: 0.14
Nodes (28): EvidenceSnapshot, EvidenceTransitionDecision, EvidenceTransitionTracker, _hard_labels_from_evidence(), _labels_for_gate(), _labels_summary(), Deterministic pre-LLM gate: interpret only on host evidence transitions., snapshot_from_window() (+20 more)

### Community 167 - "bus_listener.py"
Cohesion: 0.15
Nodes (30): HarnessDraftMoleculeV1, CoalitionSnapshotV1, Typed subset of AttentionBroadcastProjectionV1 at draft emit time., appraise_draft_molecule(), _clamp01(), _collect_learning_refs(), _compute_alignment_hints(), _compute_open_loop_pressure() (+22 more)

### Community 168 - "anthropic_passthrough.py"
Cohesion: 0.12
Nodes (34): _available_route_keys(), _backend_supports_anthropic_messages(), build_models_list_payload(), _extract_correlation_id(), _forwardable_request_headers(), _forwardable_response_headers(), handle_messages_get(), handle_messages_head() (+26 more)
=======
### Community 36 - "Collapse Mirror split invariant (Strict/Juniper vs Metacog/Orion; rationale: metacog mirrors must never hit Juniper's triage/enrichment pipeline by default)"
Cohesion: 0.50
Nodes (4): Collapse Mirror split invariant (Strict/Juniper vs Metacog/Orion; rationale: metacog mirrors must never hit Juniper's triage/enrichment pipeline by default), Metacog/Spark Surgical Patch Tracker, Oríon identity + response-policy profile, actions.respond_to_juniper_collapse_mirror.v1 verb

### Community 69 - "Service: orion-sql-writer"
Cohesion: 0.05
Nodes (68): Channel "orion:autonomy:action:outcome" (kind=event, schema=ActionOutcomeEmitV1) producers=[orion-spark-concept-induction, orion-execution-dispatch-runtime] consumers=[orion-sql-writer, *], Channel "orion:chat:social:stored" (kind=event, schema=SocialRoomTurnStoredV1) producers=[orion-sql-writer] consumers=[orion-vector-host, orion-rdf-writer, orion-meta-tags, orion-spark-concept-induction, *], Channel "orion:chat:social:turn" (kind=event, schema=SocialRoomTurnV1) producers=[orion-hub] consumers=[orion-sql-writer], Channel "orion:debug:turn:dossier" (kind=event, schema=TurnDossierV1) producers=[orion-spark-concept-induction] consumers=[orion-rdf-writer, orion-sql-writer, *], Channel "orion:effect:*" (kind=event, schema=VerbEffectV1) producers=[orion-cortex-exec] consumers=[orion-rdf-writer, orion-sql-writer, orion-vector-writer], Channel "orion:hub:messages:create" (kind=event, schema=HubWorldPulseMessageV1) producers=[orion-world-pulse] consumers=[orion-hub, orion-sql-writer, *], Channel "orion:memory:drives:audit" (kind=event, schema=DriveAuditV1) producers=[orion-spark-concept-induction] consumers=[orion-rdf-writer, orion-sql-writer, *], Channel "orion:memory:drives:state" (kind=event, schema=DriveStateV1) producers=[orion-spark-concept-induction] consumers=[orion-rdf-writer, orion-sql-writer, *] (+60 more)

### Community 94 - "Service: orion-spark-introspector"
Cohesion: 0.06
Nodes (48): Channel "orion:autonomy:goal:planned" (kind=event, schema=AutonomyGoalPlannedV1) producers=[orion-cortex-exec] consumers=[orion-cortex-exec, *], Channel "orion:calibration:profile:audit" (kind=event, schema=CalibrationProfileAuditV1) producers=[orion-cortex-exec] consumers=[orion-sql-writer], Channel "orion:cortex:exec:request" (kind=request, schema=CortexExecRequestPayload) producers=[orion-cortex-orch, orion-thought] consumers=[orion-cortex-exec], Channel "orion:cortex:exec:request:background" (kind=request, schema=CortexExecRequestPayload) producers=[orion-cortex-orch, orion-actions, orion-harness-governor, orion-execution-dispatch-runtime] consumers=[orion-cortex-exec], Channel "orion:cortex:exec:request:chat" (kind=request, schema=CortexExecRequestPayload) producers=[orion-cortex-orch] consumers=[orion-cortex-exec], Channel "orion:cortex:exec:request:spark" (kind=request, schema=CortexExecRequestPayload) producers=[orion-cortex-orch] consumers=[orion-cortex-exec], Channel "orion:endogenous:runtime:audit" (kind=event, schema=EndogenousRuntimeAuditV1) producers=[orion-cortex-exec] consumers=[orion-sql-writer], Channel "orion:endogenous:runtime:record" (kind=event, schema=EndogenousRuntimeExecutionRecordV1) producers=[orion-cortex-exec] consumers=[orion-sql-writer] (+40 more)

### Community 104 - "Service: orion-hub"
Cohesion: 0.05
Nodes (58): Channel "orion:agent-council:intake" (kind=request, schema=GenericPayloadV1) producers=[orion-hub, orion-cortex-exec] consumers=[orion-agent-council], Channel "orion:agent-council:reply*" (kind=result, schema=GenericPayloadV1) producers=[orion-agent-council] consumers=[orion-hub, orion-cortex-exec], Channel "orion:attention:loop_outcome" (kind=event, schema=AttentionLoopOutcomeV1) producers=[orion-hub] consumers=[none], Channel "orion:attention:salience:trace" (kind=telemetry, schema=AttentionSalienceTraceV1) producers=[orion-thought] consumers=[none], Channel "orion:chat:response:feedback" (kind=event, schema=ChatResponseFeedbackV1) producers=[orion-hub, *] consumers=[orion-sql-writer], Channel "orion:cognition:reasoning_call" (kind=telemetry, schema=ReasoningCallV1) producers=[orion-cortex-exec] consumers=[orion-thought], Channel "orion:conversation:request" (kind=request, schema=ChatRequestPayload) producers=[orion-hub] consumers=[orion-cortex-orch], Channel "orion:conversation:result" (kind=result, schema=ChatResultPayload) producers=[orion-cortex-orch] consumers=[orion-hub] (+50 more)

### Community 109 - "Orion Bus Channels Registry (channels.yaml)"
Cohesion: 0.04
Nodes (78): Channel "orion:bridge:social:participant" (kind=event, schema=ExternalRoomParticipantV1) producers=[orion-social-room-bridge] consumers=[orion-sql-writer, *], Channel "orion:bridge:social:room:delivery" (kind=event, schema=ExternalRoomPostResultV1) producers=[orion-social-room-bridge] consumers=[orion-sql-writer, *], Channel "orion:bridge:social:room:intake" (kind=event, schema=ExternalRoomMessageV1) producers=[orion-social-room-bridge] consumers=[orion-sql-writer, *], Channel "orion:bridge:social:room:skipped" (kind=event, schema=ExternalRoomTurnSkippedV1) producers=[orion-social-room-bridge] consumers=[orion-sql-writer, *], Channel "orion:collapse:enrich" (kind=event, schema=CollapseMirrorEntryV2) producers=[orion-cortex-exec] consumers=[orion-collapse-mirror], Channel "orion:collapse:events" (kind=event, schema=CollapseMirrorEntryV2) producers=[orion-collapse-mirror, orion-cortex-exec] consumers=[orion-timeline, orion-athena-spark-introspector], Channel "orion:collapse:intake" (kind=event, schema=CollapseMirrorEntryV2) producers=[orion-cortex-exec, orion-collapse-mirror] consumers=[orion-collapse-mirror], Channel "orion:collapse:scored" (kind=event, schema=CollapseMirrorEntryV2) producers=[orion-cortex-exec] consumers=[orion-collapse-mirror] (+70 more)

### Community 132 - "Service: orion-cortex-exec"
Cohesion: 0.29
Nodes (8): Channel "orion:exec:request:MetaTagsService" (kind=request, schema=MetaTagsRequestV1) producers=[orion-cortex-exec] consumers=[orion-meta-tags], Channel "orion:exec:result:MetaTagsService" (kind=result, schema=MetaTagsResultV1) producers=[orion-meta-tags] consumers=[orion-cortex-exec], Channel "orion:tags:chat:enriched" (kind=event, schema=MetaTagsPayload) producers=[orion-meta-tags] consumers=[orion-rdf-writer], Channel "orion:tags:enriched" (kind=event, schema=MetaTagsPayload) producers=[orion-meta-tags] consumers=[orion-rdf-writer, orion-sql-writer], Schema: MetaTagsPayload, Schema: MetaTagsRequestV1, Schema: MetaTagsResultV1, Service: orion-meta-tags
>>>>>>> Stashed changes

### Community 169 - "Context Engineering Pack (Substrate Trace Adoption)"
Cohesion: 0.06
Nodes (50): Bounded Review-Cycle Philosophy, Consolidation Outcome Semantics, Phase 9 Reflective Graph Consolidation, Zone-Aware Conservatism, AutonomyStateDeltaV1, AutonomyStateV1, AutonomyStateV2, chat_evidence_to_tension (+42 more)

<<<<<<< Updated upstream
### Community 170 - "test_chat_relational_stance.py"
Cohesion: 0.12
Nodes (29): _compile_repair_speech_overlay(), compile_speech_contract(), _inject_prior_stance_to_inputs(), Deterministic regime-specific contract injected near TASK in chat_general.j2., Copy prior brief summary into stance inputs and expose it as a TOP-LEVEL ctx, _instrumental_brief(), _relational_brief(), test_chat_stance_brief_new_fields_default_none() (+21 more)

### Community 171 - "FakeQueue"
Cohesion: 0.07
Nodes (11): _FakeBus, _request(), test_publish_reflection_gracefully_skips_without_bus(), test_publish_self_concepts_skips_unchanged_repeated_run(), test_publish_self_study_artifacts_gracefully_skips_without_bus(), test_repeat_publish_skips_unchanged_graph_but_keeps_journal_append_intent(), test_self_concept_induce_verb_publishes_induced_graph_only(), test_self_concept_reflect_verb_writes_reflective_graph_and_journal() (+3 more)

### Community 172 - "test_salience_combiner.py"
Cohesion: 0.10
Nodes (42): Evidence-derived feature vector scored by the salience combiner.      Replaces t, SalienceFeaturesV1, bounded(), compute_features(), compute_salience(), default_combiner(), _habituation(), LinearSalienceCombiner (+34 more)

### Community 173 - "test_grammar_truth.py"
Cohesion: 0.07
Nodes (37): apply_grammar_events_retention(), build_grammar_truth_snapshot(), _fallback_counts(), _grammar_index_valid(), GrammarRetentionState, _latest_events_by_source(), Runtime snapshot and bounded retention for grammar production observe mode., Count grammar.event.v1 fallbacks using typed created_at_ts when available. (+29 more)

### Community 174 - "Context Pack: orion-substrate-telemetry (Cursor, markdown)"
Cohesion: 0.08
Nodes (22): Bus payloads for substrate observability (tier merge outcomes)., Emitted on ``orion:substrate:tier_outcomes`` when cold-path materialization runs, SubstrateTierOutcomesPayloadV1, _correlation_uuid(), publish_substrate_tier_outcomes_sync(), Synchronous Redis publish for substrate tier telemetry (cortex-exec chat_stance, Publish tier outcome telemetry when cold-path fan-out ran. Best-effort; never ra, _reset_redis() (+14 more)

### Community 175 - "GraphitiAdapter"
Cohesion: 0.15
Nodes (15): emit_active_packet_retrieved(), emit_vector_upsert(), _source(), _embed_http(), publish_crystallization_to_chroma(), Build upsert, optionally embed, publish to vector bus. Returns updated crystalli, Governed cognitive memory crystallization layer., build_chroma_upsert() (+7 more)

### Community 176 - "detect_resonance"
Cohesion: 0.09
Nodes (39): action_usefulness_rate(), pressure_discharge_rate(), Phase H — efficacy metrics for the reverie/dream weave.  Pure, deterministic red, Fraction of chains where post-chain pressure fell below pre-chain pressure., Fraction of reverie-originated action outcomes judged useful (FeedbackFrameV1)., Before/after recall metrics around a compaction., Deterministic before/after recall reduction. Negative deltas are wins:     lower, recall_delta() (+31 more)

### Community 177 - "main.py"
Cohesion: 0.09
Nodes (34): AgentChainRequest, AgentChainResult, Primary entry point for the Agent Chain service., Response from Agent Chain., agent_chain_request_to_context_exec(), context_exec_run_to_agent_chain_result(), _parse_answer_contract(), agent_lane_health_block() (+26 more)

### Community 178 - "types.py"
Cohesion: 0.07
Nodes (40): collect_fragments(), Fan-out collector.      This is the only place that knows which storage backends, End-to-end recall pipeline:      1. Collect candidate fragments from SQL and RDF, run_recall_pipeline(), _dedupe(), _mix_kinds(), postprocess_fragments(), Very light kind mixing: prefer a blend of collapse + chat + assoc     instead of (+32 more)

### Community 179 - "ResonanceHealthMonitor"
Cohesion: 0.09
Nodes (44): _check(), check_resonance_worsening(), HealthCheck, _is_worsening(), Phase H+ — resonance health monitor.  The resonance tripwire (`orion.reverie.res, Edge-triggered per-theme resonance monitor.      Tracks every theme currently be, Reconstruct tracked themes (and their known-unhealthy state) from         orion-, Call once per completed chain, after `_maybe_emit_resonance_alert`         persi (+36 more)

### Community 180 - "main.py"
Cohesion: 0.11
Nodes (22): CompressionRegionV1, CompressionStalenessMarkV1, GraphCompressionRegionMaterializedV1, A cached semantic compression of a graph region, written to orion:compressions., Bus payload marking a graph region stale when source triples are written., Bus event emitted after each compression artifact is written to Fuseki., CompressionWriter, Emit a passive materialization event for any region kind. (+14 more)

### Community 181 - "train_v3_moc.py"
Cohesion: 0.13
Nodes (31): iter_batches(), load_fineweb_edu_tokens(), load_gpt2_tokenizer(), _load_streaming_text_tokens(), load_text_file_tokens(), load_tinystories_tokens(), Load a streaming FineWeb-Edu token slice.      Default uses HuggingFaceFW/finewe, Streams token blocks (x, y) with y = x shifted by 1. (+23 more)

### Community 182 - "Service: orion-vision-host"
Cohesion: 0.06
Nodes (48): Channel "orion:exec:request:VisionCouncilService" (kind=request, schema=VisionCouncilRequestPayload) producers=[orion-vision-window] consumers=[orion-vision-council], Channel "orion:exec:request:VisionHostService" (kind=request, schema=VisionTaskRequestPayload) producers=[orion-cortex-exec, orion-hub, orion-vision-host, orion-vision-frame-router] consumers=[orion-vision-host], Channel "orion:exec:request:VisionScribeService" (kind=request, schema=VisionScribeRequestPayload) producers=[orion-vision-council] consumers=[orion-vision-scribe], Channel "orion:exec:request:VisionWindowService" (kind=request, schema=VisionWindowRequestPayload) producers=[orion-vision-window] consumers=[orion-vision-council], Channel "orion:exec:result:VisionCouncilService" (kind=result, schema=VisionCouncilResultPayload) producers=[orion-vision-council] consumers=[orion-vision-window], Channel "orion:exec:result:VisionScribeService" (kind=result, schema=VisionScribeResultPayload) producers=[orion-vision-scribe] consumers=[orion-vision-council], Channel "orion:exec:result:VisionWindowService" (kind=result, schema=VisionWindowResultPayload) producers=[orion-vision-window] consumers=[orion-vision-council], Channel "orion:grammar:event" (kind=event, schema=GrammarEventV1) producers=[orion-vision-retina, orion-hub, orion-vision-edge, orion-vision-window, orion-biometrics, orion-cortex-exec, orion-bus, orion-harness-governor, orion-cortex-orch] consumers=[orion-sql-writer] (+40 more)

### Community 183 - "VisionSceneInterpretationV1"
Cohesion: 0.17
Nodes (25): VisionEventCandidateV1, VisionSceneInterpretationV1, _activity_claim_has_caption_slop(), build_person_presence_fallback(), enforce_evidence_grounding(), ensure_grounded_person_presence(), _events_mention_person(), _filter_person_entity_names() (+17 more)

### Community 184 - "_common.py"
Cohesion: 0.16
Nodes (32): main(), extract_from_python(), extract_from_yaml(), _extract_str(), main(), merge_inv(), triage_unknown(), compose_has_environment_block() (+24 more)

### Community 185 - "builder.py"
Cohesion: 0.16
Nodes (30): build_execution_dispatch_frame(), build_unevaluable_execution_dispatch_frame(), _candidate_status_for_mode(), _is_hard_blocked(), _proposal_by_id(), A policy frame whose proposal or self-state could not be loaded still     needs, _resolve_dispatch_mode(), stable_dispatch_id() (+22 more)

### Community 186 - "BaseEnvelope"
Cohesion: 0.11
Nodes (32): build_conversation_envelope(), build_envelopes_for_turn(), build_example_envelope(), build_import_run_envelope(), build_import_run_id(), build_message_envelope(), ChatMessage, ChatTurn (+24 more)

### Community 187 - "consolidation_memory_gate"
Cohesion: 0.16
Nodes (22): _appraisal(), consolidation_memory_gate(), _significance(), _atom_semantic_role(), fetch_grammar_evidence_for_window(), _parse_event_json(), is_low_info_social(), _normalize_whitespace() (+14 more)

### Community 188 - "worker.py"
Cohesion: 0.07
Nodes (45): insert_pending_draft(), compact_schema_for_llamacpp(), compact_suggest_draft_json_schema(), JSON Schema contracts for memory-graph suggest (llama.cpp-friendly compact form), Full Pydantic JSON schema (may include $defs / $ref)., Strip Pydantic noise; keep inline object shapes only., Stage-1 schema: top-level keys + array item shapes without $ref/oneOf., suggest_draft_json_schema() (+37 more)

### Community 189 - "suggest_validate.py"
Cohesion: 0.10
Nodes (44): collect_topical_spine_warnings(), _draft_utterance_corpus(), _entity_surface_forms_lower(), _entity_tokens(), extract_selected_role_evidence(), _has_assistant_role_entity(), _has_role_entity(), _has_topical_entity() (+36 more)

### Community 190 - "HyperbolicGPT"
Cohesion: 0.33
Nodes (5): main(), parse_args(), HyperbolicGPT, main(), _pick_device()
=======
### Community 174 - "Context Pack: orion-substrate-telemetry (Cursor, markdown)"
Cohesion: 1.00
Nodes (3): claim:orion:substrate-telemetry:0001 — orion-substrate-telemetry persists tier outcomes, claim:orion:substrate-telemetry:0002 — orion-cortex-orch optionally merges telemetry facet, source:2026-05-14-substrate-tier-telemetry-design-ref (metadata)

### Community 182 - "Service: orion-vision-host"
Cohesion: 0.07
Nodes (43): Channel "orion:exec:request:VisionCouncilService" (kind=request, schema=VisionCouncilRequestPayload) producers=[orion-vision-window] consumers=[orion-vision-council], Channel "orion:exec:request:VisionHostService" (kind=request, schema=VisionTaskRequestPayload) producers=[orion-cortex-exec, orion-hub, orion-vision-host, orion-vision-frame-router] consumers=[orion-vision-host], Channel "orion:exec:request:VisionScribeService" (kind=request, schema=VisionScribeRequestPayload) producers=[orion-vision-council] consumers=[orion-vision-scribe], Channel "orion:exec:request:VisionWindowService" (kind=request, schema=VisionWindowRequestPayload) producers=[orion-vision-window] consumers=[orion-vision-council], Channel "orion:exec:result:VisionCouncilService" (kind=result, schema=VisionCouncilResultPayload) producers=[orion-vision-council] consumers=[orion-vision-window], Channel "orion:exec:result:VisionScribeService" (kind=result, schema=VisionScribeResultPayload) producers=[orion-vision-scribe] consumers=[orion-vision-council], Channel "orion:exec:result:VisionWindowService" (kind=result, schema=VisionWindowResultPayload) producers=[orion-vision-window] consumers=[orion-vision-council], Channel "orion:vision:artifacts" (kind=event, schema=VisionArtifactPayload) producers=[orion-vision-edge, orion-vision-host] consumers=[orion-security-watcher, orion-vision-window, orion-vision-council] (+35 more)
>>>>>>> Stashed changes

### Community 191 - "Consolidation Policy v1"
Cohesion: 0.05
Nodes (46): Field Attention Policy v1, Signal Kind: biometrics_state, Signal Kind: chat_reasoning_quality, Signal Kind: chat_social_hazard, Signal Kind: failure_event, Signal Kind: mesh_health, Signal to Drive Map v1, Signal Kind: spark_signal (+38 more)

<<<<<<< Updated upstream
### Community 192 - "Shared compactor helpers README"
Cohesion: 0.09
Nodes (15): GpuInfo, GpuInspector, Pick GPU with highest free VRAM among candidates, honoring reserve/hard_floor., Raised when the bounded async queue cannot accept another pending task., Submit a unit of work to the scheduler; returns handler result., Multi-GPU scheduler with:     - global inflight cap     - per-GPU inflight cap, Pending items waiting on the worker (best-effort backlog indicator)., True if VRAM policy allows scheduling onto at least one configured CUDA device. (+7 more)

### Community 193 - "test_reasoning_emit.py"
Cohesion: 0.06
Nodes (63): Reasoning telemetry — per-call cognition metadata + windowed activity.  `Reasoni, One LLM call's reasoning metadata. No trace text — privacy-preserving., Rolling-window aggregate of ReasoningCallV1, read by φ (spark-introspector)., ReasoningActivityV1, ReasoningCallV1, build_reasoning_call(), _coerce_correlation_uuid(), _coerce_str() (+55 more)

### Community 194 - "FolderFrameSource"
Cohesion: 0.08
Nodes (18): create_frame_source(), FolderFrameSource, FrameReadResult, FrameSource, MockFrameSource, Test/dev: random sample from folder like legacy mock., RtspFrameSource, _VideoCaptureSource (+10 more)

### Community 195 - "fcc_claude_bridge.py"
Cohesion: 0.10
Nodes (38): annotate_harness_step(), build_context_pressure_step(), chars_per_token_estimate(), context_fill_pct(), context_overflow_operator_hint(), context_pressure_threshold_chars(), context_pressure_threshold_pct(), context_risk_level() (+30 more)

=======
>>>>>>> Stashed changes
### Community 196 - "Phase 13b SQL-Writer Durability"
Cohesion: 0.05
Nodes (45): Calibration Adoption Typed Contracts, InMemoryCalibrationProfileStore Manual Adoption Seam, Phase 12 Calibration Profile Adoption, Advisory-Only Recommendations Posture, EndogenousCalibrationEngine, EndogenousOfflineEvaluator, Phase 11 Offline Evaluation and Calibration, EndogenousRuntimeAdoptionService (+37 more)

<<<<<<< Updated upstream
### Community 197 - "MemoryCompactionDeltaV1"
Cohesion: 0.13
Nodes (20): MindLLMFailOpenRecord, Fail-open metadata preserved when LLM synthesis falls back to deterministic Mind, MindPhaseTelemetry, phase_telemetry_machine_keys(), Per-phase LLM telemetry for Mind run inspectability., maybe_publish_llm_surface_instability_trigger(), _publish_metacog_trigger_async(), Advisory metacog trigger when LLM language surface appears unstable. (+12 more)

### Community 198 - "EmbodimentWorker"
Cohesion: 0.06
Nodes (27): publish_with_reconnect(), Publish once; on transport failure reconnect the command client and retry., _FlakyBus, orion.core.bus.resilience must not crash-on-import when loguru is missing., test_publish_with_reconnect_retries_after_transport_error(), test_resilience_importable_without_loguru(), EmbodimentWorker, Re-read AI Town ids from the mounted FCC env each perception tick. (+19 more)

### Community 199 - "SpontaneousThoughtV1"
Cohesion: 0.07
Nodes (36): Return a copy with the hollow flag + reason stamped from the guard.          `ex, Unprompted narration of the current winning coalition.      Grounding is the *sa, The set of coalition ids that legitimately anchor this thought., True if this thought is empty-shell cognition (§0A).          Rejects three fail, SpontaneousThoughtV1, Latest fresh non-hollow spontaneous thought, or None (Phase B).          Degrade, _is_hollow(), _is_meta_mechanism_narration() (+28 more)

### Community 200 - "check_inner_state_registry.py"
Cohesion: 0.16
Nodes (8): _channel_schema_ids(), _covered_schema_names(), main(), _matches_keyword(), new_duplicate_heuristic_check(), Returns a list of failure messages; empty means all entries are healthy., _registry_py_schema_names(), rot_check()

### Community 201 - "FieldDigesterWorker"
Cohesion: 0.08
Nodes (19): check_field_coherence(), Return 0-1 incoherence score for one node vector., Return per-node incoherence scores (0-1). Empty if no suspicion found., _rule_suspicion(), FieldChannelCorpusRowV1, Field-channel corpus row schema -- Item 1 v2 of docs/superpowers/specs/ 2026-07-, One per-tick training-data row of raw `FieldStateV1` channel     pressures, stra, Shared contract for JSONL corpus-sink rotation naming and file resolution.  Sing (+11 more)

### Community 202 - "ConceptWorker"
Cohesion: 0.05
Nodes (31): ConceptWorker, Manages windowed intake and triggers induction., Handler for durable world-pulse run-result stream messages.          Routes thro, Pub/sub intake channels, minus any that are consumed via a durable stream., Classify a channel as a homeostatic source: 'signal' | 'failure', or         Non, Measurement-only (Phase 4, 2026-07-12): log `DriveEngine`'s pressure         vec, Thin drive-only rail for homeostatic sources: mint deviation tensions,         u, _service_ref() (+23 more)

### Community 203 - "SuggestDraftV1"
Cohesion: 0.17
Nodes (30): DispositionDraft, MemoryGraphSubschemaV1, Appendix D projection embedded under card.subschema.memory_graph., SituationDraft, _anchor_class_for_entity(), build_memory_graph_subschema(), _card_types_for(), _collect_rdf_edges() (+22 more)

### Community 204 - "EvidenceUnitV1"
Cohesion: 0.10
Nodes (36): EvidenceAdapter, _as_dt(), CollapseMirrorEvidenceAdapter, JournalEvidenceAdapter, _is_table_line(), _link_siblings(), MarkdownSpecEvidenceAdapter, _strip_lines() (+28 more)

### Community 205 - "build_harness_reasoning_trace"
Cohesion: 0.08
Nodes (36): auto_approve_from_env(), claude_permission_argv(), extend_mcp_argv(), mcp_disallowed_tool_patterns(), Shared ``claude -p`` argv helpers for FCC harness bridges., Block Bash fallbacks that fail in headless Hub (gh not installed)., Auto-approve tool permissions for non-interactive FCC turns., Whether to auto-approve when env is unset: root containers yes, host non-root ye (+28 more)

### Community 206 - "WorldPulseSourceV1"
Cohesion: 0.06
Nodes (58): SourceRegistryV1, SourceTrustAssessmentV1, WorldPulseAllowedUsesV1, WorldPulseSourceV1, fetch_rss_articles(), _parse_date(), bounded_urls(), extract_links() (+50 more)

### Community 207 - "client.py"
Cohesion: 0.09
Nodes (34): _admin_key(), AitownClientError, _base_url(), convex_mutation(), convex_query(), convex_request(), _default_player_id(), fetch_version() (+26 more)

### Community 208 - "pg_conn"
Cohesion: 0.07
Nodes (52): ModelCreateRequest, SegmentListPage, SegmentListResponse, SegmentRawResponse, SegmentRecord, create_model_endpoint(), compare_runs(), get_segment() (+44 more)

### Community 209 - "_run_training"
Cohesion: 0.11
Nodes (29): DatasetSpec, ModelSpec, RunRecord, RunSpecSnapshot, RunTrainRequest, RunTrainResponse, train_run_endpoint(), fetch_dataset_rows() (+21 more)

### Community 210 - "github_repo_context.py"
Cohesion: 0.14
Nodes (24): append_github_mcp_harness_brief(), default_harness_workspace(), github_mcp_brief_lines(), github_mcp_repo_brief_line(), harness_mcp_enabled(), parse_github_remote_url(), Resolve GitHub owner/repo for FCC MCP operator briefs., Append GitHub MCP operator lines when harness MCP is enabled. (+16 more)

### Community 211 - "rdf_adapter.py"
Cohesion: 0.12
Nodes (24): detect_drive_tensions(), DriveTensionV1, One detected tension between two drives at a single tick.      `drive_a` is the, Detect pairwise inverse-coactivation tensions between drives.      Definition: a, _pairs(), Spec for cross-drive tension detection (`drive_tension.py`).  Tension definition, pressures[drive_b] == deactivate_threshold exactly -> NOT suppressed,     no ten, A hair below the threshold -> suppressed, tension fires. (+16 more)

### Community 212 - "self_study_harness.py"
Cohesion: 0.33
Nodes (8): _scenario_map(), test_conceptual_retrieval_exposes_authoritative_and_induced_only(), test_consumer_modes_preserve_trust_boundaries(), test_degraded_consumer_backend_is_reported_cleanly(), test_factual_retrieval_stays_authoritative_only(), test_harness_runs_golden_path_scenarios(), test_reflective_retrieval_exposes_all_tiers_without_upcasting(), test_soak_mode_reports_stable_repeated_runs()

### Community 213 - "_WPBase"
Cohesion: 0.09
Nodes (69): ArticleClusterV1, ArticleRecordV1, ClaimRecordV1, EntityRecordV1, EventRecordV1, SituationChangeV1, SituationEvidenceV1, SituationObservationV1 (+61 more)

### Community 214 - "test_substrate_mutation_scheduler_runtime.py"
Cohesion: 0.11
Nodes (7): scheduler_fixture(), _self_revision_self_state(), test_scheduler_operator_gated_class_never_auto_applies(), test_scheduler_self_revision_disabled_by_default_signal_never_enters_cycle(), test_scheduler_self_revision_requires_cognitive_lane_double_gate(), test_scheduler_self_revision_respects_routing_proposals_kill_lever(), test_scheduler_self_revision_signals_flow_into_cognitive_proposal_when_double_gated()

### Community 215 - "map_curiosity_ctx_to_substrate"
Cohesion: 0.08
Nodes (41): _coerce(), map_curiosity_ctx_to_substrate(), Map ``ctx['curiosity_signals']`` → one ``curiosity:unresolved_gaps`` node., Coerce raw input to list of FrontierInvocationSignalV1.      Accepts:     - list, _make_signal(), Tests for curiosity_ctx adapter., emitted node has salience exactly 0.4 (verify it, don't relax)., coerce dict version of FrontierInvocationSignalV1. (+33 more)

### Community 216 - "AgentConfig"
Cohesion: 0.06
Nodes (49): PromptContext, PromptFactory, Turns (AgentConfig + PromptContext) into a messages[] list.      This is where φ, Factory / registry that knows which agents live in which 'universe'.      - Load, UniverseRegistry, CouncilDecision, CouncilPolicy, Encapsulates:       - how we interpret auditor verdict + blink scores       - ho (+41 more)

### Community 217 - "test_graphiti_core_backend.py"
Cohesion: 0.07
Nodes (30): mock_pool(), test_ingest_episode_returns_ids(), _edge_data_calls(), _entity_data_calls(), _FakeEntityEdge, _FakeEntityNode, _patch_graphiti_node_edge_modules(), Stands in for graphiti_core.nodes.EntityNode in this dev venv (graphiti-core isn (+22 more)

### Community 218 - "test_recall_strategy_readiness.py"
Cohesion: 0.17
Nodes (24): Recall V2 / recall-strategy promotion readiness (advisory only; no live apply)., RecallStrategyReadinessV1, _collect_compare_rows_from_pressure(), compare_rows_from_telemetry_records(), compute_recall_strategy_readiness(), default_eval_corpus_total_cases(), _f(), Deterministic recall strategy / Recall V2 shadow promotion readiness (advisory). (+16 more)

### Community 219 - "RowBlock"
Cohesion: 0.16
Nodes (29): WindowingSpec, _build_prompt(), _cache_key(), _call_llm(), _hash_text(), judge_boundaries(), _write_artifact(), _cosine_similarity() (+21 more)

### Community 220 - "PackManager"
Cohesion: 0.14
Nodes (8): CognitionPack, PackManager, Represents a pack as defined in packs/*.yaml., Given one or more pack names, return consolidated list of unique verbs., Manages cognitive packs.      Responsibilities:     - Load packs from packs/*.ya, Validate that all verbs in the pack exist in verbs/*.yaml.         Returns a dic, test_pack_validation_memory_pack(), test_packs_load()

### Community 221 - "service.py"
Cohesion: 0.13
Nodes (15): query_chroma_collection(), Query Chroma HTTP API for semantic hits (Postgres remains canonical)., _apply_recall_boost(), _embed_query(), Multi-rail retrieval: Postgres crystallizations + cards + Chroma + Graphiti., Reinforce every crystallization that actually made it into the render budget., retrieve_active_packet(), Acceptance check 1: head-to-head recall competition.          Two crystallizatio (+7 more)

### Community 222 - "aitown_client.py"
Cohesion: 0.09
Nodes (37): accept_invite(), _admin_key(), AitownClientError, _base_url(), convex_mutation(), convex_query(), convex_request(), _game_descriptions() (+29 more)

### Community 223 - "test_metacog_phase_contract.py"
Cohesion: 0.11
Nodes (36): MetacogCausalDensityV1, MetacogConstraintsV1, MetacogDraftTextPatchV1, MetacogDraftWhatChangedV1, MetacogEnrichScorePatchV1, MetacogNumericSistersV1, _draft_ctx(), _load_executor_module() (+28 more)

### Community 224 - "HarnessStepRelay"
Cohesion: 0.09
Nodes (20): Forward harness governor FCC steps to Hub WS as claude_step frames., relay_harness_run_steps(), test_relay_harness_run_steps_forwards_matching_correlation(), HarnessRunStepV1, HarnessStepRelay, Drop liveness bookkeeping for a finished correlation_id., True if a harness step for this correlation_id was observed within `within_sec`., Opportunistically evict liveness entries older than the TTL. Cheap: only scans (+12 more)

### Community 225 - "MemoryCrystallizationV1"
Cohesion: 0.14
Nodes (29): Auto-encode at a fraction of salience — weak initial footprint., seed_weak_dynamics(), auto_activate(), GovernorPathRequired, _utc_now(), FormationPolicy, _has_identity_scope(), resolve_formation_policy() (+21 more)

### Community 226 - "SocialScenarioReplayHarness"
Cohesion: 0.08
Nodes (28): SocialScenarioEvaluationResultV1, SocialScenarioExpectationV1, SocialScenarioFixtureV1, SocialScenarioSeedStateV1, SocialScenarioTurnFixtureV1, SocialShakedownFixV1, SocialShakedownIssueV1, _clear_app_modules() (+20 more)

### Community 227 - "orchestrator.py"
Cohesion: 0.09
Nodes (51): _clean(), extract_anchors(), _clean_block(), extract_blocks(), _extract_code_or_command(), _extract_logs(), _reasoning_summary(), _candidate_claims() (+43 more)

### Community 228 - "test_fcc_motor_mcp.py"
Cohesion: 0.07
Nodes (13): _BlockingAfterLinesStream, _fake_fcc_env(), _FakeProc, _FakeStream, Near the whole-turn deadline, fcc_timeout fires, not fcc_stream_stalled., A message that never completes must not hang for the whole turn budget.      Rep, test_run_fcc_turn_adds_mcp_config_when_enabled(), test_run_fcc_turn_fails_fast_on_stalled_stream() (+5 more)

### Community 229 - "mutation_control_surface.py"
Cohesion: 0.19
Nodes (9): control_surface_store(), get_chat_reflective_lane_threshold(), inspect_chat_reflective_lane_threshold(), _resolve_postgres_url(), _resolve_sqlite_path(), RuntimeControlSurfaceStore, set_chat_reflective_lane_threshold(), _utc_now() (+1 more)

### Community 230 - "AnswerContract"
Cohesion: 0.07
Nodes (62): harness_motor_instruction(), is_relational_motor_stance(), Deterministic FCC operator briefs for harness motor turns., _stance_slice(), compile_harness_prefix(), _format_autonomy_slice(), _format_grounding_self_block(), _format_stance_slice() (+54 more)

### Community 231 - "StreamMessage"
Cohesion: 0.18
Nodes (19): apply_repair_pressure_contract(), assemble_repair_contract_delta(), _evidence_kinds_from_dimensions(), Behavior consumer: repair_pressure signal → response contract mode.  This is int, Return a new contract dict adjusted by the repair_pressure signal.      Spec §11, v2: level gates mode; active kind scores union rules., Behavior contract tests for repair pressure — spec §14.4 and §11.1., _signal() (+11 more)

### Community 232 - "MindRunRequestV1"
Cohesion: 0.20
Nodes (25): test_mind_run_request_roundtrip(), MindRunPolicyV1, MindRunRequestV1, run_mind(), FakeMindLLMClient, Deterministic JSON responses for unit tests., set_llm_client_override(), mind_run() (+17 more)

### Community 233 - "PsuService"
Cohesion: 0.10
Nodes (21): check_token(), create_app(), create_service(), get_service(), heartbeat_loop(), psu_cycle(), psu_off(), psu_on() (+13 more)

### Community 234 - "bootstrap_orion_agent.py"
Cohesion: 0.07
Nodes (44): default_fetch_backend(), expand_env_path(), load_fcc_env(), resolve_fetch_backend(), resolve_firecrawl_api_key(), firecrawl_search_backend(), Tier B readonly search via Firecrawl REST API., test_default_fetch_backend_raises() (+36 more)

### Community 235 - "Service: orion-harness-governor"
Cohesion: 0.06
Nodes (40): Channel "orion:actions:trigger:journal.v1" (kind=event, schema=JournalTriggerV1) producers=[orion-actions, orion-embodiment, *] consumers=[orion-actions], Channel "orion:embodiment:intent" (kind=event, schema=EmbodimentIntentV1) producers=[orion-substrate-runtime, orion-harness-governor, orion-cortex-exec] consumers=[orion-embodiment], Channel "orion:embodiment:outcome" (kind=event, schema=EmbodimentOutcomeV1) producers=[orion-embodiment] consumers=[orion-hub], Channel "orion:embodiment:perception" (kind=event, schema=WorldPerceptionV1) producers=[orion-embodiment] consumers=[orion-substrate-runtime, orion-self-state-runtime, orion-cortex-exec], Channel "orion:exec:result:*" (kind=result, schema=CortexExecResultPayload) producers=[orion-cortex-exec, *] consumers=[orion-cortex-orch, orion-harness-governor, orion-thought, orion-hub, *], Channel "orion:grammar:accepted-pressure" (kind=event, schema=GrammarEventV1) producers=[orion-substrate-runtime] consumers=[none], Channel "orion:harness:run:artifact" (kind=event, schema=HarnessRunV1) producers=[orion-harness-governor] consumers=[*], Channel "orion:harness:run:cancel" (kind=event, schema=HarnessRunCancelV1) producers=[orion-hub] consumers=[orion-harness-governor] (+32 more)

### Community 236 - "get_profile"
Cohesion: 0.08
Nodes (23): Hub legacy label ``recall.v1`` must resolve to structured brain recall, not chat, test_dream_v1_profile_loads(), test_profiles_load_default(), test_recall_v1_hub_alias_maps_to_brain_recall(), _canonical_profile_name(), _find_profiles_dir(), get_profile(), load_profiles() (+15 more)

### Community 237 - "EndogenousEvaluationRequestV1"
Cohesion: 0.14
Nodes (13): HarnessRunCancelV1, Fire-and-forget cancel for an in-flight FCC motor turn (Hub disconnect / abort)., run_bus_worker(), apply_harness_run_cancel(), handle_cancel_bus_message(), Validate cancel payload and kill the matching FCC subprocess if live., Subscribe to harness run cancel events and kill matching FCC motors., run_cancel_worker() (+5 more)

### Community 238 - "VisionFramePointerPayload"
Cohesion: 0.27
Nodes (13): configure_parity_evidence_store(), ConsumerParityEvidence, get_parity_evidence_snapshot(), ParityEvidenceStore, ParityReadinessThresholds, record_parity_evidence(), reset_parity_evidence_store(), setup_function() (+5 more)

### Community 239 - "extract_repair_evidence"
Cohesion: 0.18
Nodes (24): _clamp01(), extract_repair_evidence(), _extract_text(), _new_evidence_id(), _Phrase, _phrase_hit(), Deterministic phrase-match detector for repair evidence.  No LLM. No embeddings., Return (matched, span_text_or_None). (+16 more)

### Community 240 - "CrystallizationEvidenceRefV1"
Cohesion: 0.19
Nodes (13): infer_confidence(), Deterministic confidence tier from evidence + recurrence. No LLM, no I/O.      R, CrystallizationEvidenceRefV1, Best-effort: grammar events may live in substrate tables or bus-only., resolve_crystallization_sources(), resolve_evidence_ref(), resolve_grammar_event_ref(), resolve_memory_card_ref() (+5 more)

### Community 241 - "memory_consolidation_draft_routes.py"
Cohesion: 0.16
Nodes (6): EndogenousRuntimeSqlReader, SqlReadResult, _Conn, _Engine, test_calibration_reader_profile_history_query(), test_runtime_reader_filters_and_limit_are_bounded()

### Community 242 - "test_queue_service_chassis.py"
Cohesion: 0.21
Nodes (5): LlamaMergeMLP, LlamaRMSNorm, LlamaRotaryEmbedding, LlamaRMSNorm is equivalent to T5LayerNorm, dynamic RoPE layers should recompute `inv_freq` in the following situations:

### Community 243 - "run_answer_depth_live_proof.py"
Cohesion: 0.12
Nodes (28): _amain(), _extract(), LiveScenario, main(), _now_iso(), _ordered_hops(), _parse_args(), ProbeCollector (+20 more)

### Community 244 - "HealthMonitor"
Cohesion: 0.16
Nodes (23): _check(), HealthCheck, HealthMonitor, Edge-triggered health monitor: fires an orion-notify attention request only, run_checks(), health(), get_settings(), Settings (+15 more)

### Community 245 - "submitExplicitChatText"
Cohesion: 0.08
Nodes (41): agentAnswerHeadline(), applyAgentClaudePayloadFields(), applyHubModeSelection(), applyOrionUnifiedPayloadFields(), audienceChipLabel(), beginWsReadyWait(), confirmDownRouteOrProceed(), drawVisualizer() (+33 more)

### Community 246 - "LocalProfileStore"
Cohesion: 0.16
Nodes (6): LocalProfileStore, Minimal JSON-backed store for latest profiles., True if an autonomy episode has already been composed for this world-pulse run., test_episode_run_processed_ignores_empty_run_id(), test_episode_run_processed_persists_across_instances(), test_episode_run_processed_roundtrip()

### Community 247 - "active_packet.py"
Cohesion: 0.12
Nodes (24): build_active_packet(), _entry(), _task_boost(), eligible_for_recall(), Return True when an active crystallization is strong enough for recall injection, CrystallizationDynamicsV1, _allowed_buckets(), _enabled() (+16 more)

### Community 248 - "draft_sanitize.py"
Cohesion: 0.21
Nodes (12): OperationalSemanticHarnessTests, Regression: direct lane RPC must reuse the same plan as the verb path (router +, Regression: lane/mind/output_mode arbitration facts must reach the caller on, _settings_routing_off(), _settings_routing_on(), _spark_req(), test_call_verb_runtime_direct_exec_surfaces_route_metadata(), test_call_verb_runtime_uses_direct_exec_for_spark_when_routing_on() (+4 more)

### Community 249 - "ArticleRecordV1"
Cohesion: 0.24
Nodes (13): extract_final_text(), parse_structured_observation(), Parse a substrate.* verb's structured JSON output.      Expected shape: {"observ, Pull the plan's final_text out of a decoded CortexExecResultPayload.      Mirror, test_extract_final_text_from_result_final_text(), test_extract_final_text_from_step_result_fallback(), test_extract_final_text_missing_returns_empty(), test_parse_structured_observation_empty_text() (+5 more)

### Community 250 - "test_drive_state_divergence_audit.py"
Cohesion: 0.10
Nodes (29): _coerce_float(), compare_drives(), load_autonomy_state(), load_drive_state_v1(), main(), Load the current autonomy_state_v2 for `subject`.      `load_autonomy_state_v2`, Best-effort numeric coercion for values read from the unvalidated     LocalProfi, Per-drive pressure divergence + activation-flag agreement.      `drive_state.v1` (+21 more)

### Community 252 - "test_drive_history_reflection_synthesis.py"
Cohesion: 0.14
Nodes (18): _binding(), FakeConn, FakeSparqlClient, _grounded_llm_content(), _make_bus(), fetch_drive_history_events is list-only (no detail join) -- see its     docstrin, Minimal in-process double for the asyncpg.Connection calls     repository.py::in, _sufficient_history_bindings() (+10 more)

### Community 253 - "graphiti_core.py"
Cohesion: 0.29
Nodes (13): _availability_summary(), build_biometrics_node_grammar_events(), _dt(), _event(), _hash_id(), _safe_ts(), _fixtures(), test_athena_capability_surface_mentions_graphdb_not_heavy_llm() (+5 more)

### Community 254 - "AutonomyVerificationHarness"
Cohesion: 0.18
Nodes (14): __getattr__(), AutonomyVerificationHarness, CheckResult, GraphDBClient, load_scenarios(), ScenarioFixture, ScenarioRunResult, write_report() (+6 more)

### Community 255 - "test_introspection_fixture.py"
Cohesion: 0.11
Nodes (31): extract_tool_metrics(), _iter_message_blocks(), load_fixture(), Deterministic scoring for the unified-turn introspection eval.  Experiment artif, Return (passed, failed) assertion id lists for one answer., Fold fcc_motor step frames (raw stream-json events) into navigation metrics., score_answer(), ToolMetrics (+23 more)

### Community 256 - "persist_turn_referent"
Cohesion: 0.08
Nodes (36): HarnessPostTurnClosureV1, _coalition_ref(), persist_turn_referent(), Best-effort writer for substrate_turn_referent (reverie semantic lift v1)., _make_worker(), _sample_closure(), test_handle_post_turn_closure_bus_message_decodes_envelope(), test_handle_post_turn_closure_message_invokes_handler() (+28 more)

### Community 257 - "schemas.py"
Cohesion: 0.33
Nodes (5): PromptContext, PromptFactory, Shared context for building prompts for any agent.     Keeps council logic DRY,, Responsible for turning (AgentConfig + PromptContext) into LLM message lists., Single entrypoint: build a messages[] list for an LLM from agent + context.

### Community 258 - "CompressionRegionV1"
Cohesion: 0.05
Nodes (39): build_graph_from_triples(), leiden_cluster(), build_region(), Deterministic region ID from scope + kind + sorted node URIs., stable_region_id(), EpisodicFederator, CompressionWorker, Construct a RegionSummarizer when LLM summarization is viable, else None. (+31 more)

### Community 259 - "schemas.py"
Cohesion: 0.20
Nodes (18): main(), _parse_args(), evaluate_training_run(), _load_jsonl(), _now(), ensure_runtime_packages(), missing_runtime_packages(), AdapterArtifactManifest (+10 more)

### Community 260 - "fit_phi_encoder.py"
Cohesion: 0.13
Nodes (39): InnerStateFeaturesV1, MoodArcCorpusRowV1, MoodArcEncoderManifestV1, Mood-arc corpus row schema -- Item 1 of docs/superpowers/specs/2026-07-13-felt-s, One per-tick training-data row for the not-yet-built windowed felt-     state au, Item 2's windowed felt-state-trajectory encoder manifest -- dark     artifact, d, AttributionV1, CorpusStatsV1 (+31 more)

### Community 261 - "drive_history_reflection_synthesis.py"
Cohesion: 0.10
Nodes (36): build_drive_audit_detail_sparql(), build_drive_audit_event_list_sparql(), build_fact_sheet(), _build_narrative_prompt(), _build_reflection_crystallization(), _call_llm_narrative(), DriveHistoryAggregationV1, _escape_sparql() (+28 more)

### Community 262 - "CouncilService"
Cohesion: 0.18
Nodes (12): VisionCouncilRequestPayload, VisionCouncilResultPayload, stream_key_from_window(), InterpretationParseOutcome, CouncilService, lifespan(), test_finalize_drops_youtube_activity_without_hard_person(), test_finalize_host_fallback_on_parse_failure() (+4 more)

### Community 263 - "appendMessage"
Cohesion: 0.06
Nodes (48): appendMessage(), autonomyAvailabilityRowsForDisplay(), backfillLatestUserTurnIdForGraph(), buildAgentTraceOverviewNode(), buildAgentTraceRawPayloadsNode(), buildAgentTraceTimelineNode(), buildAgentTraceToolGroupsNode(), buildChatStanceSection() (+40 more)

### Community 264 - "mind_enrichment.py"
Cohesion: 0.14
Nodes (18): Shared Mind contract constants (no runtime / IO)., MindRunArtifactV1, Bus + Postgres artifact for a completed Mind run (producer: orch, consumer: sql-, _clip(), _clip_str_or_none(), _envelope_correlation_id(), publish_mind_run_artifact_for_thought(), orion-thought → orion-mind advisory enrichment (unified turn coloring).  The uni (+10 more)

### Community 265 - "main.py"
Cohesion: 0.05
Nodes (61): CausalityLink, Envelope, Create a child envelope that extends the causality chain with this message as a, Typed envelope: payload is a Pydantic model.      This is the preferred new path, A single step in a causality chain. Used for Conjourney lineage., CortexExecRequestPayload, CortexExecResultPayload, Typed payload for cortex.exec.request. (+53 more)

### Community 266 - "WorldPulseStreamConsumer"
Cohesion: 0.16
Nodes (9): configure_tracing(), _DropSpanExporter, OpenTelemetry tracer setup for the signal gateway (spec §5 gateway instrumentati, Swallows finished spans (no backend); span context IDs are still real., Install a TracerProvider. If ``otlp_endpoint`` is set, export spans there; else, lifespan(), orion-signal-gateway FastAPI entrypoint., Return most recent OrionSignalV1 per organ_id from the in-memory window. (+1 more)

### Community 267 - "map_recall_bundle_to_substrate"
Cohesion: 0.36
Nodes (13): maybe_quick_lane_verdict(), quick_lane_block_reason(), Return a block reason when quick lane is disallowed; None when eligible., _eligible_inputs(), test_quick_lane_allowed_when_all_criteria_pass(), test_quick_lane_blocked_on_alignment_hints(), test_quick_lane_blocked_on_boundary_register(), test_quick_lane_blocked_on_high_surprise() (+5 more)

### Community 268 - "test_phi_encoder_fit_script.py"
Cohesion: 0.14
Nodes (28): All files backing one corpus path, oldest first: any rotated     siblings (match, resolve_rotated_corpus_files(), input_features_for_version(), _load_jsonl(), _variance_gate(), _feature(), _inner_row(), Without feature_names, behavior stays the plain fraction-based gate --     backw (+20 more)

### Community 269 - "validate_for_escalation"
Cohesion: 0.37
Nodes (13): resolve_execution_lane(), _req(), test_explicit_lane_spark_wins_over_verb_chat(), test_fallback_background_auto_mode(), test_mode_agent_background(), test_mode_brain_chat_lane_without_matching_verb(), test_resolve_chat_general(), test_resolve_chat_kids_story() (+5 more)

### Community 270 - "SceneBeliefTracker"
Cohesion: 0.12
Nodes (13): BeliefObserveResult, Per-stream scene label habituation (observed → believed tier)., Enter votes: empty ring slots inherit last non-empty labels (flicker fix)., Exit votes: count only labels present in each raw observation., SceneBeliefRegistry, SceneBeliefTracker, test_belief_ignores_single_empty_observation(), test_belief_requires_enter_votes() (+5 more)
=======
### Community 235 - "Service: orion-harness-governor"
Cohesion: 0.06
Nodes (39): Channel "orion:actions:trigger:journal.v1" (kind=event, schema=JournalTriggerV1) producers=[orion-actions, orion-embodiment, *] consumers=[orion-actions], Channel "orion:embodiment:intent" (kind=event, schema=EmbodimentIntentV1) producers=[orion-substrate-runtime, orion-harness-governor, orion-cortex-exec] consumers=[orion-embodiment], Channel "orion:embodiment:outcome" (kind=event, schema=EmbodimentOutcomeV1) producers=[orion-embodiment] consumers=[orion-hub], Channel "orion:embodiment:perception" (kind=event, schema=WorldPerceptionV1) producers=[orion-embodiment] consumers=[orion-substrate-runtime, orion-self-state-runtime, orion-cortex-exec], Channel "orion:grammar:accepted-pressure" (kind=event, schema=GrammarEventV1) producers=[orion-substrate-runtime] consumers=[none], Channel "orion:grammar:event" (kind=event, schema=GrammarEventV1) producers=[orion-vision-retina, orion-hub, orion-vision-edge, orion-vision-window, orion-biometrics, orion-cortex-exec, orion-bus, orion-harness-governor, orion-cortex-orch] consumers=[orion-sql-writer], Channel "orion:harness:run:artifact" (kind=event, schema=HarnessRunV1) producers=[orion-harness-governor] consumers=[*], Channel "orion:harness:run:cancel" (kind=event, schema=HarnessRunCancelV1) producers=[orion-hub] consumers=[orion-harness-governor] (+31 more)
>>>>>>> Stashed changes

### Community 271 - "Vision Services Documentation"
Cohesion: 0.08
Nodes (33): Vision Services Documentation, BaseEnvelope, Cortex-Exec Vision Verb Enablement, Frame Router Dispatch Tiers (baseline/triggered), GroundingDINO, Titanium Contract Stack, VisionArtifactPayload, Vision Council (orion-vision-council) (+25 more)

<<<<<<< Updated upstream
### Community 272 - "SocialGifUsageStateV1"
Cohesion: 0.16
Nodes (23): SocialGifUsageStateV1, _eligible_summary(), _FakeBus, _FakeCallSyneClient, _FakeHubClient, _FakeSocialMemoryClient, _payload(), _policy_and_message() (+15 more)

### Community 273 - "settings.py"
Cohesion: 0.24
Nodes (11): attach_repair_pressure_contract(), _contract_changed(), Wire substrate repair contract into Hub → Cortex chat metadata., Mutate req.metadata in place when repair pressure changed behavior., WS builds chat_req before pipeline; attach must run after pipeline., _snapshot(), test_attach_skips_when_disabled(), test_attach_skips_when_mode_unchanged() (+3 more)

### Community 274 - "SignalsInspectCache"
Cohesion: 0.07
Nodes (31): is_stub_signal(), Shared stub-signal detection for Hub inspect cache and adapter tests (spec §5.9), True when the signal is a placeholder stub emission, not real organ truth., test_is_stub_signal_detects_placeholder_dimensions(), test_is_stub_signal_false_for_real_equilibrium(), api_observability_grafana_tempo_trace(), api_signals_trace(), Rolling trace cache by ``otel_trace_id`` (bounded by ``TRACE_CACHE_*`` settings) (+23 more)

### Community 275 - "substrate-atlas.js"
Cohesion: 0.15
Nodes (30): activateAtlasPanel(), apiFetch(), applyGraphFilters(), destroyCy(), escapeHtml(), fetchTraces(), fitAtlasGraph(), formatTs() (+22 more)

### Community 276 - "EwmaBand"
Cohesion: 0.14
Nodes (5): clamp11(), EwmaBand, Tests for normalization utilities., TestClamp, TestEwmaBand

### Community 277 - "tts.py"
Cohesion: 0.12
Nodes (23): Config, Settings, CoquiBackend, _ensure_torch_load_compat(), _is_xtts_model(), Facade: select backend by TTS_BACKEND., PyTorch 2.6+ defaults weights_only=True; Coqui XTTS checkpoints need False., _resolve_speaker_wav_path() (+15 more)

### Community 278 - "main.py"
Cohesion: 0.05
Nodes (52): set_active_goal(), clear_cursor_resets_for_tests(), cursor_reset_snapshot(), CursorResetRecord, last_reset_skipped_history(), parse_timestamp_at(), Operator cursor reset auth, validation, and audit trail., record_cursor_reset() (+44 more)

=======
>>>>>>> Stashed changes
### Community 279 - "Endogenous Action v1 Motor Nerve Spec"
Cohesion: 0.09
Nodes (32): Endogenous Action v1 Motor Nerve Spec, P0 Dispatch Status Honesty (prepared_for_dispatch), P2 Experience Loop (act becomes experience), Endogenous-Origination Measurement Gate (verdict a/b), Motor Nerve (Layer 9 Dispatch Send), substrate.inspect/summarize/observe Verbs, Cognition Theater Tripwire, Execution Dispatch Motor Nerve P1 Design (+24 more)

<<<<<<< Updated upstream
### Community 280 - "WorldPulseRunResultV1"
Cohesion: 0.11
Nodes (39): outcome_from_followup(), Return the followup whose section matches the first gap-section label the     re, Rebuild an ActionOutcomeRefV1 from a world-pulse curiosity followup so the     r, select_reusable_followup(), iter_gap_section_labels(), Yield normalized section labels from `section:` focal refs across     world_cove, _followup(), _gap_signal() (+31 more)

### Community 281 - "Service: orion-actions"
Cohesion: 0.10
Nodes (24): Channel "orion:actions:audit" (kind=event, schema=GenericPayloadV1) producers=[orion-actions] consumers=[none], Channel "orion:actions:manage:result:*" (kind=result, schema=WorkflowScheduleManageResponseV1) producers=[orion-actions] consumers=[orion-cortex-orch], Channel "orion:actions:manage:workflow.v1" (kind=request, schema=WorkflowScheduleManageRequestV1) producers=[orion-cortex-orch] consumers=[orion-actions], Channel "orion:actions:trigger:daily_metacog.v1" (kind=event, schema=GenericPayloadV1) producers=[orion-actions] consumers=[orion-actions], Channel "orion:actions:trigger:daily_pulse.v1" (kind=event, schema=GenericPayloadV1) producers=[orion-actions, orion-cortex-exec] consumers=[orion-actions], Channel "orion:actions:trigger:workflow.v1" (kind=event, schema=WorkflowDispatchRequestV1) producers=[orion-cortex-orch, orion-actions] consumers=[orion-actions], Channel "orion:agent-council:intake" (kind=request, schema=GenericPayloadV1) producers=[orion-hub, orion-cortex-exec] consumers=[orion-agent-council], Channel "orion:agent-council:reply*" (kind=result, schema=GenericPayloadV1) producers=[orion-agent-council] consumers=[orion-hub, orion-cortex-exec] (+16 more)

### Community 282 - "McpPreflightError"
Cohesion: 0.18
Nodes (22): mcp_allowed_tool_patterns(), Per-server allow patterns for Claude Code 2.1+ MCP pre-approval.      Use ``mcp_, _deep_replace(), McpPreflightError, _probe_convex_auth(), _probe_convex_version(), Render ephemeral MCP config for fcc-claude turns., Context Mode needs an absolute, writable storage root (Docker volume). (+14 more)

### Community 283 - "extract_suggest_draft_dict_from_cortex_payload"
Cohesion: 0.14
Nodes (26): extract_suggest_draft_dict_from_cortex_payload(), extract_suggest_text_from_cortex_payload(), _openai_choice_message_text(), Extract memory_graph_suggest draft JSON from CortexClientResult-shaped payloads., Parse a SuggestDraftV1 dict from a CortexClientResult-shaped payload., Return best-effort model text containing a suggest draft from a cortex payload., _service_block_text_candidates(), _sorted_steps() (+18 more)

### Community 284 - "derive_retrieval_intent"
Cohesion: 0.14
Nodes (27): Deterministic Phase 0 gate: skip recall on low-info social turns., recall_skip_gate(), RecallSkipGateResult, _coerce_novelty(), derive_retrieval_intent(), _has_contradiction_seed(), _has_entity_query(), _has_planning_like_priority() (+19 more)

### Community 286 - "test_measure_autonomy_gate.py"
Cohesion: 0.09
Nodes (9): Deterministic unit tests for the pure layer of measure_autonomy_gate.  No DB, no, _self_state(), test_percentile_and_median(), test_verdict_a_busy_zero_variance_passes_when_silent_moves(), test_verdict_a_flat_but_measured_is_no_go_not_unmeasurable(), test_verdict_a_flat_silent_is_no_go(), test_verdict_a_moving_silent_is_go(), test_verdict_b_frequent_coactivation_is_go() (+1 more)

### Community 287 - "endogenous_curiosity_candidates"
Cohesion: 0.12
Nodes (34): MetabolismResultV1, _gap_strength(), _gaps_from_rollups(), _load_recommended_sections(), _signal_from_gap(), _tension_from_gap(), test_metabolism_result_v1_accepts_tensions_and_curiosity(), FrontierInvocationSignalV1 (+26 more)

### Community 288 - "test_rlm_eval_fixtures.py"
Cohesion: 0.08
Nodes (50): AnswerGroundingStatus, Finding, FindingsBundle, User-visible grounding summary (§9.2)., RenderedAnswer, Cognition-facing contracts (evidence-first answering, etc.)., test_answer_contract_json_roundtrip(), test_findings_bundle_roundtrip() (+42 more)

### Community 289 - "proposal_review_client.py"
Cohesion: 0.14
Nodes (20): _assert_get_path(), _assert_post_path(), _base_url(), fetch_health(), get_eligibility(), _get_json(), get_proposal(), list_proposals() (+12 more)

### Community 290 - "probe_structured_output.py"
Cohesion: 0.12
Nodes (23): build_response_format_for_method(), _default_base_url(), _default_model(), _evaluate_content(), has_forbidden_content(), main(), parse_json_content(), probe_method() (+15 more)

### Community 291 - "test_interpretation_v2.py"
Cohesion: 0.18
Nodes (31): project_interpretation_to_events(), _parse(), Council V2 scene interpretation parsing and projection tests (no live LLM/Redis/, Live failure mode: LLM put event_candidates fields under salient_observations., test_build_interpretation_prompt_caps_artifact_bloat(), test_debug_parse_mode_available_from_outcome(), test_empty_event_candidates_projection_returns_empty_payload(), test_event_candidates_populated_from_misplaced_salient_observations() (+23 more)

### Community 292 - "StepExecutionResult"
Cohesion: 0.06
Nodes (78): _agent_chain_delegate_status(), _agent_chain_delegate_summary(), _agent_chain_failure_detail(), _agent_chain_failure_signals(), _agent_delegate_payload(), _agent_delegate_service_key(), _bound_capability_payload(), _bound_effect_kind() (+70 more)

### Community 294 - "check_single_consumer_channels.py"
Cohesion: 0.29
Nodes (9): evaluate_counts(), fetch_live_counts(), load_single_consumer_channels(), main(), parse_numsub_output(), Pure decision core: (violations, warnings, status_by_channel).      count > 1  -, Query live subscriber counts via redis-cli. Raises RuntimeError on infra failure, All catalog channel names marked single_consumer: true, glob names skipped. (+1 more)

### Community 295 - "bound_capability_exec.py"
Cohesion: 0.17
Nodes (19): ActionSkillManifestEntry, ActionsSkillRegistry, _family_for_skill(), Normalized orion-actions skill manifest derived from skills.* verb YAMLs., _risk_for_skill(), _bound_capability_service_payload(), execute_bound_capability(), _last_user_message() (+11 more)

### Community 296 - "syncDebugModalScrollLock"
Cohesion: 0.12
Nodes (31): closeAgentTraceModal(), closeAutonomyConstitutionModal(), closeAutonomyDebugModal(), closeChatInputExpandModal(), closeChatStanceDebugModal(), closeCognitiveReviewModal(), closeMemoryDebugModal(), closeRecallCanaryModal() (+23 more)

### Community 297 - "memory.js"
Cohesion: 0.16
Nodes (27): activateSubview(), apiFetch(), escapeHtml(), formatHistoryWhen(), formatMemoryApiError(), graphSetOut(), loadAll(), loadConsolidationDrafts() (+19 more)

### Community 298 - "service.py"
Cohesion: 0.17
Nodes (15): build_social_gif_proxy_context(), _FakeBus, _FakeCallSyneClient, _FakeHubClient, _FakeSocialMemoryClient, _payload(), _service(), _settings() (+7 more)

### Community 299 - "Service: orion-landing-pad"
Cohesion: 0.08
Nodes (30): Channel "orion:biometrics:cluster" (kind=telemetry, schema=BiometricsClusterV1) producers=[orion-biometrics-hub] consumers=[orion-state-service, orion-sql-writer], Channel "orion:biometrics:induction" (kind=telemetry, schema=BiometricsInductionV1) producers=[orion-biometrics, orion-biometrics-hub] consumers=[orion-state-service, orion-sql-writer], Channel "orion:biometrics:sample" (kind=telemetry, schema=BiometricsSampleV1) producers=[orion-biometrics] consumers=[orion-sql-writer, orion-landing-pad], Channel "orion:biometrics:summary" (kind=telemetry, schema=BiometricsSummaryV1) producers=[orion-biometrics, orion-biometrics-hub] consumers=[orion-state-service, orion-sql-writer, orion-landing-pad], Channel "orion:exec:result:PadRpc:*" (kind=result, schema=PadRpcResponseV1) producers=[orion-landing-pad] consumers=[orion-cortex-exec, *], Channel "orion:exec:result:StateService:*" (kind=result, schema=StateLatestReply) producers=[orion-state-service] consumers=[orion-cortex-exec, *], Channel "orion:pad:event" (kind=event, schema=PadEventV1) producers=[orion-landing-pad] consumers=[*], Channel "orion:pad:frame" (kind=event, schema=StateFrameV1) producers=[orion-landing-pad] consumers=[*] (+22 more)

### Community 300 - "build_delivery_grounding_context"
Cohesion: 0.07
Nodes (45): build_answer_grounding_context(), _compose_grounding_context(), delivery_grounding_mode(), extract_trace_preferred_output(), Answer/runtime grounding strings split from delivery-oriented helpers., Structured grounding for answer/render verbs (repo + runtime; Discord only when, build_delivery_grounding_context(), Shared helpers for grounding delivery-oriented answers (re-exports answer ground (+37 more)

### Community 301 - "ChannelCatalogEnforcer"
Cohesion: 0.14
Nodes (16): load_channel_catalog(), ChannelCatalogEnforcer, _assert_channel_wiring(), main(), _run_importer_dry_run(), _sample_export_file(), _catalog_names(), test_channel_enforcer_accepts_cortex_result_reply_channel() (+8 more)

### Community 302 - "TestLLMBackendHelpers"
Cohesion: 0.12
Nodes (9): _extract_reasoning_from_openai_response(), _extract_text_from_ollama_response(), _extract_text_from_openai_response(), _extract_vector_from_openai_response(), Generic extractor for OpenAI-compatible responses (vLLM & llama.cpp)., Extract structured reasoning emitted by provider/OpenAI-compatible payloads only, Best-effort extraction of an embedding/state vector from OpenAI-compatible     r, _split_think_blocks() (+1 more)

### Community 303 - "test_consolidation_tensorize.py"
Cohesion: 0.22
Nodes (18): _attention_frame(), _dim(), _dispatch_frame(), _feedback_frame(), _motif(), _policy_frame(), _proposal_frame(), _self_state() (+10 more)

### Community 304 - "measure_autonomy_gate.py"
Cohesion: 0.05
Nodes (68): _abs_trajectory_for_row(), _as_utc(), bucket_class_for(), bucket_index(), BucketActivity, build_arg_parser(), build_bucket_activity(), build_drive_coactivation_histogram_sparql() (+60 more)

### Community 305 - "trace_unified_turn.py"
Cohesion: 0.18
Nodes (25): build_turn_trace(), classify_log_line(), cmd_dump(), cmd_latest(), cmd_live(), _docker_logs(), extract_correlation_id(), _fetch_grammar_summaries() (+17 more)

### Community 306 - "utils.py"
Cohesion: 0.26
Nodes (11): main(), _main_async(), monitor_ups(), _publish_event(), _run_shutdown(), setup_logging(), _source(), PowerEvent (+3 more)

### Community 307 - "mind_provenance.js"
Cohesion: 0.22
Nodes (29): asBool(), asList(), asObject(), calloutSeverityClass(), chip(), escapeHtml(), findPhaseTelemetry(), hasSourceTagLeakage() (+21 more)

### Community 309 - "boundary.py"
Cohesion: 0.38
Nodes (8): binary_score_from_top_logprobs(), build_classify_prompt(), parse_classify_lines(), build_turn_change_prompt(), test_binary_score_from_top_logprobs_yes_wins(), test_build_classify_prompt_four_lines_no_phi(), test_build_classify_prompt_includes_phase(), test_parse_classify_lines()

### Community 310 - "test_proposal_review_api.py"
Cohesion: 0.12
Nodes (24): app_client(), _load_app(), Tests for proposal review API on orion-context-exec., _seed_store(), test_context_exec_app_mounts_proposal_review_routes_when_enabled(), test_eligibility_endpoint(), test_get_proposal_detail(), test_health_proposal_review_block_store_configured() (+16 more)

### Community 311 - "test_router_identity_boundary.py"
Cohesion: 0.15
Nodes (16): Drop leading Juniper/Oríon label recital from speech on ordinary turns., Remove identity-kernel prose from llm_chat_general prompt context on ordinary tu, strip_identity_recital_leadin(), suppress_chat_general_speech_identity_priming(), _apply_chat_general_identity_boundary_guard(), test_strip_identity_recital_leadin_skips_relational_turn(), _step(), test_chat_general_identity_boundary_repairs_user_role_inversion() (+8 more)

### Community 312 - "hub_memory_graph_suggest_text"
Cohesion: 0.26
Nodes (13): _graph_node(), _make_worker(), Unit tests for rung-5 endogenous curiosity tick wiring in substrate-runtime., When no seeds qualify, tick still writes an empty candidate set for observabilit, Evaluator signals are persisted endogenous-first, capped at 8., test_endogenous_curiosity_disabled_is_noop(), test_endogenous_curiosity_fails_open_on_evaluator_error(), test_endogenous_curiosity_noop_persist_failure_does_not_break_tick() (+5 more)

### Community 313 - "SemanticPlanner"
Cohesion: 0.12
Nodes (15): create_default_planner(), Convenience helper to construct a SemanticPlanner using the standard     verbs/, ExecutionPlan, ExecutionStep, Semantic Planner      This version:     - Reads verb definitions directly from v, Simple generic safety rule evaluation.          Expects safety_rules in YAML lik, Turn the verb_def['plan'] list into ordered ExecutionSteps., Lightweight representation of Orion's current mode/context.     You can extend t (+7 more)

### Community 314 - "CrystallizationGovernanceV1"
Cohesion: 0.09
Nodes (37): ConsolidationGateResult, attach_grammar_to_crystallization(), envelope_from_grammar_event(), Build memory grammar envelope from existing GrammarEventV1-shaped dict., build_crystallization_from_episode(), _episode_scope(), _episode_summary(), Build a proposed episode crystallization from an autonomy journal entry. (+29 more)

### Community 315 - "ActiveCognitiveFrontierV1"
Cohesion: 0.17
Nodes (22): SelectedFrontierMatterV1, Project the mode-agnostic self/attention subset of a Mind run.      Returns None, select_mind_coloring(), _frontier(), Eval: Mind coloring is mode-agnostic and never leaks task-control into the unifi, _result(), test_no_task_control_leaks_across_all_modes(), test_relational_turn_carries_self_and_curiosity_signal() (+14 more)

### Community 316 - "ContextExecRunner"
Cohesion: 0.14
Nodes (26): _artifact_evidence_count(), artifact_repo_evidence_count(), _blocked_summary(), count_evidence(), evaluate_investigation_outcome(), explicit_fake_run_requested(), grounding_required(), _grounding_sources_attempted() (+18 more)
=======
### Community 281 - "Service: orion-actions"
Cohesion: 0.12
Nodes (19): Channel "orion:actions:audit" (kind=event, schema=GenericPayloadV1) producers=[orion-actions] consumers=[none], Channel "orion:actions:manage:result:*" (kind=result, schema=WorkflowScheduleManageResponseV1) producers=[orion-actions] consumers=[orion-cortex-orch], Channel "orion:actions:manage:workflow.v1" (kind=request, schema=WorkflowScheduleManageRequestV1) producers=[orion-cortex-orch] consumers=[orion-actions], Channel "orion:actions:trigger:daily_metacog.v1" (kind=event, schema=GenericPayloadV1) producers=[orion-actions] consumers=[orion-actions], Channel "orion:actions:trigger:daily_pulse.v1" (kind=event, schema=GenericPayloadV1) producers=[orion-actions, orion-cortex-exec] consumers=[orion-actions], Channel "orion:actions:trigger:workflow.v1" (kind=event, schema=WorkflowDispatchRequestV1) producers=[orion-cortex-orch, orion-actions] consumers=[orion-actions], Channel "orion:collapse:stored" (kind=event, schema=CollapseMirrorStoredV1) producers=[orion-sql-writer] consumers=[orion-actions, *], Channel "orion:context_exec:event" (kind=event, schema=GenericPayloadV1) producers=[orion-context-exec] consumers=[*] (+11 more)

### Community 285 - "orion-landing-pad Docker Compose"
Cohesion: 0.50
Nodes (4): Landing Pad Metrics Explorer UI, orion-landing-pad Docker Compose, Orion Landing Pad service (sensory gateway + working memory), orion-landing-pad Python dependencies

### Community 299 - "Service: orion-landing-pad"
Cohesion: 0.07
Nodes (32): Channel "orion:biometrics:cluster" (kind=telemetry, schema=BiometricsClusterV1) producers=[orion-biometrics-hub] consumers=[orion-state-service, orion-sql-writer], Channel "orion:biometrics:induction" (kind=telemetry, schema=BiometricsInductionV1) producers=[orion-biometrics, orion-biometrics-hub] consumers=[orion-state-service, orion-sql-writer], Channel "orion:biometrics:sample" (kind=telemetry, schema=BiometricsSampleV1) producers=[orion-biometrics] consumers=[orion-sql-writer, orion-landing-pad], Channel "orion:biometrics:summary" (kind=telemetry, schema=BiometricsSummaryV1) producers=[orion-biometrics, orion-biometrics-hub] consumers=[orion-state-service, orion-sql-writer, orion-landing-pad], Channel "orion:cognition:trace" (kind=event, schema=CognitionTracePayload) producers=[orion-cortex-exec] consumers=[orion-spark-introspector, orion-landing-pad, orion-bus-tap], Channel "orion:exec:result:StateService:*" (kind=result, schema=StateLatestReply) producers=[orion-state-service] consumers=[orion-cortex-exec, *], Channel "orion:pad:event" (kind=event, schema=PadEventV1) producers=[orion-landing-pad] consumers=[*], Channel "orion:pad:frame" (kind=event, schema=StateFrameV1) producers=[orion-landing-pad] consumers=[*] (+24 more)
>>>>>>> Stashed changes

### Community 317 - "orion-hub service (README) — browser gateway into the mesh"
Cohesion: 0.10
Nodes (27): orion-gpu-cluster-power service (compose), orion-gpu-cluster-power requirements.txt, compression_policy.v1.yaml (scopes, clustering, summarization policy), orion-graph-compression service (compose), GraphCompressionRegionMaterializedV1 (bus event schema), MutationPressureEvidenceV1 (bus event schema), orion-graph-compression service (README), orion-graph-compression requirements.txt (networkx/leidenalg/igraph) (+19 more)

<<<<<<< Updated upstream
### Community 318 - "substrate_effect_pipeline.py"
Cohesion: 0.12
Nodes (19): In-memory LRU cache of per-turn substrate effect snapshots., SubstrateEffectCache, SubstrateEffectSnapshot, _push_observation(), Orchestrate the repair_pressure appraisal pipeline for one chat turn.  Failure m, Run the appraiser end-to-end. Stash a snapshot in `cache`. Return summary., run_substrate_effect_pipeline(), _summary_dict() (+11 more)

### Community 319 - "substrateReviewFetch"
Cohesion: 0.10
Nodes (29): createRecallCanaryReviewArtifact(), hydrateRecallCanaryProfileSelect(), recordRecallCanaryJudgment(), refreshAutonomyConstitutionModal(), refreshAutonomyReadinessPanel(), refreshCognitiveReviewModal(), refreshCognitiveReviewPanelInto(), refreshRecallCanaryModal() (+21 more)

### Community 320 - "_make_engine_sequence"
Cohesion: 0.11
Nodes (19): _make_engine_sequence(), Returns a fake engine where each successive execute() call returns the next resu, M3 status is 'fresh' when projection updated_at is recent., M3 status is 'stale' when projection updated_at is old., L11 status is 'missing' when consolidation frames table has no rows., capability_transport_bucket is set when target is in dominant_targets., capability_transport_bucket is 'capability_targets' when only in that bucket., capability_transport_bucket is 'suppressed_targets' when only suppressed. (+11 more)

### Community 321 - "__init__.py"
Cohesion: 0.12
Nodes (11): BiometricsInput, ChatHistoryInput, DreamFragmentMeta, DreamInput, DreamMetrics, Legacy row-shape models. The canonical bus payload is DreamResultV1 from orion.s, EnrichmentInput, MirrorInput (+3 more)

### Community 322 - "query.py"
Cohesion: 0.21
Nodes (19): _atom_to_dict(), _bfs_atom_ids(), _compaction_to_dict(), _count_atoms(), _count_edges(), _distinct_dimensions(), _distinct_layers(), _edge_to_dict() (+11 more)

### Community 323 - "should_rewrite_for_instructional"
Cohesion: 0.22
Nodes (6): _drive_state(), _FakeStore, _minimal_goal(), test_goal_proposal_v1_accepts_new_optional_fields(), test_signature_change_sets_supersedes_artifact_for_same_drive_origin(), test_signature_stable_when_trace_changes()

### Community 324 - "__init__.py"
Cohesion: 0.07
Nodes (51): _goal_to_reasoning_claim(), ClaimV1, ConceptV1, ContradictionV1, I/O contracts for reasoning artifact materialization (Phase 2)., ReasoningWriteRequestV1, ReasoningWriteResultV1, Canonical epistemic reasoning schemas (Phase 1). (+43 more)

### Community 325 - "EpisodicConsolidationEvaluator"
Cohesion: 0.22
Nodes (5): _CapabilityBridgeBus, CapabilityHop, _Codec, _HarnessBus, test_bound_capability_full_runtime_path_emits_non_empty_result_without_timeout()

### Community 326 - "SalienceState"
Cohesion: 0.15
Nodes (24): evaluate_salience(), Pure salience gate for town embodiment episodes.  Decides whether a world event, Cross-event memory for the salience gate.      ``seen_players`` dedupes first-en, Decide whether ``event`` is worth journaling.      Salient cases:       - ``conv, SalienceEvaluation, SalienceState, _utcnow(), test_bare_proximity_not_salient() (+16 more)

### Community 327 - "ensure_delivery_pack_in_packs"
Cohesion: 0.18
Nodes (13): ensure_delivery_pack_in_packs(), Runtime pack merging for agent-chain and orch.  Ensures delivery_pack is present, Return a new pack list with delivery_pack appended when appropriate.      If out, Pass 2: merged packs include delivery verbs (YAML-level, no jinja)., test_implementation_guide_packs_include_write_guide_and_finalize(), _verb_services(), Pass 2: runtime pack merge proves delivery_pack for instructional / code asks., test_code_scaffold_merges_delivery_pack() (+5 more)

### Community 328 - "pipeline.py"
Cohesion: 0.27
Nodes (9): autonomy_subject_fanout_from_runtime_ctx(), Runtime policy for Graph autonomy multi-subject SPARQL fan-out (bounded vs full), Return ``bounded`` only for the default Hub quick lane; deep lanes use ``full``., test_fanout_bounded_chat_kids_story(), test_fanout_bounded_plain_chat_quick(), test_fanout_full_agent_mode_even_if_verb_chat_quick(), test_fanout_full_chat_general(), test_fanout_full_chat_quick_with_hub_full_stance() (+1 more)

### Community 329 - "test_skill_verbs.py"
Cohesion: 0.11
Nodes (17): _docker_prune_runner_factory(), _plan_request(), Live host Docker: preview only (natural-language dry-run cleanup phrase)., Live host Docker: execute intent hits policy gate — must not call docker rm., test_biometrics_snapshot_maps_mock_http(), test_docker_prune_dry_run_behavior(), test_docker_prune_execute_policy_gate(), test_docker_prune_execute_runs_rm_when_policy_allows() (+9 more)

### Community 330 - "fcc_env_catalog.py"
Cohesion: 0.15
Nodes (19): _convex_base_from_settings(), fetch_aitown_status(), AI Town status probe for Hub API., aitown_status(), api_fcc_model_labels(), _maybe_render_mcp_config(), run_turn_from_settings(), catalog_from_settings() (+11 more)

### Community 332 - "main.py"
Cohesion: 0.36
Nodes (7): _extract_cortex_result(), main(), MatrixRow, _one_row(), ProbeCollector, _service_ref(), _summarize()

### Community 333 - "test_voluntary_attention_wiring.py"
Cohesion: 0.31
Nodes (8): _operator_brief_for_workspace(), prepare_agent_claude_input(), Prepare Hub agent-claude turn input. v2 adds slash-command dispatch., TurnRequest, test_operator_brief_prepended_to_prompt(), test_prepare_agent_claude_input_includes_github_brief_when_env_set(), test_v1_empty_becomes_empty_string(), test_http_agent_claude_collects_events()

### Community 334 - "grammar_integration_helpers.py"
Cohesion: 0.08
Nodes (35): empty_transport_projection(), apply_sql_file(), assert_grammar_event_indexes_valid(), bus_transport_trace_batch(), _clear_app_namespace(), delete_trace(), ensure_grammar_schema(), grammar_session_factory() (+27 more)

### Community 335 - "main.py"
Cohesion: 0.05
Nodes (24): AgentStepRelay, substrate_page(), EmbodimentOutcomeCache, build_hub_ui_asset_version(), _discover_git_sha(), HubStaticFiles, _memory_store_banner(), PresenceContextStore (+16 more)

### Community 336 - "__init__.py"
Cohesion: 0.11
Nodes (23): Contract smoke tests for dream modernization schemas., test_dream_internal_trigger_v1(), test_dream_result_v1_defaults_and_audit(), test_dream_trigger_payload_minimal(), DreamFragmentV1, DreamInternalTriggerV1, DreamMetricsV1, DreamResultV1 (+15 more)

### Community 337 - "memory_routes.py"
Cohesion: 0.12
Nodes (40): MemoryCardStatusChangeV1, get_consolidation_draft(), list_consolidation_drafts(), _row_to_dict(), update_consolidation_draft_status(), get_memory_consolidation_draft(), list_memory_consolidation_drafts(), _need_session() (+32 more)

### Community 338 - "rdf_retention.py"
Cohesion: 0.20
Nodes (16): build_artifact_cap_select(), build_artifact_child_delete(), build_subject_age_delete(), cutoff_literal(), GraphRetentionPolicy, parse_retention_policies(), PruneGraphResult, SPARQL retention/pruning for Fuseki / SPARQL RDF stores. (+8 more)

### Community 339 - "is_caption_prompt_echo"
Cohesion: 0.27
Nodes (6): client(), _crys(), _FixedDatetime, _now(), No regression: existing consumers reading pre-existing fields still work., test_list_endpoint_preserves_existing_response_shape()

### Community 340 - "MindRunResultV1"
Cohesion: 0.18
Nodes (31): MindHandoffBriefV1, MindRunResultV1, build_synthetic_mind_http_failure_result(), merge_mind_brief_into_plan_metadata(), _mind_result_is_deterministic_contract_only(), _mind_result_quality(), _client_request(), _orch_prep() (+23 more)

### Community 341 - "test_rem_compaction.py"
Cohesion: 0.18
Nodes (3): IntentionModel, _prepare_4d_causal_attention_mask_with_cache_position(), Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_lengt

### Community 342 - "main.py"
Cohesion: 0.31
Nodes (6): mirror_to_buffer(), _norm_payload(), _now_ts(), _pubsub_loop(), _stream_loop(), Settings

### Community 343 - "CollapseMirrorEntryV2"
Cohesion: 0.47
Nodes (9): _make_worker(), Unit tests for the Orion embodiment perception->substrate ingest consumer.  Veri, _set_decode(), test_flag_off_writes_nothing(), test_flag_on_writes_real_proximity_node(), test_invalid_payload_fails_open(), test_no_store_fails_open(), test_store_write_exception_fails_open() (+1 more)

### Community 344 - "test_orion_proposal_cli.py"
Cohesion: 0.15
Nodes (20): Tests for operator proposal CLI., _run_cli(), _seed_and_approve(), test_dry_run_execute_creates_receipt_without_mutation(), test_dry_run_execute_does_not_change_memory(), test_dry_run_execute_does_not_change_repo(), test_dry_run_execute_does_not_mark_executed_success(), test_dry_run_execute_rejects_pending_review() (+12 more)

=======
>>>>>>> Stashed changes
### Community 346 - "orion-agent-council service (multi-agent deliberation stub)"
Cohesion: 0.09
Nodes (26): datasets (PyPI dependency), fastapi (PyPI dependency), httpx (PyPI dependency), loguru (PyPI dependency), numpy (PyPI dependency), orjson (PyPI dependency), pydantic (PyPI dependency), PyYAML (PyPI dependency) (+18 more)

<<<<<<< Updated upstream
### Community 347 - "test_notify_attention_ack.py"
Cohesion: 0.31
Nodes (5): _env(), FakeQueue, test_enqueue_failure_is_swallowed(), test_enqueue_run_result_stream_selects_run_result_envelope(), test_enqueue_skipped_when_no_run_result_envelope()

### Community 348 - "resolve_user_workflow_invocation"
Cohesion: 0.08
Nodes (43): _normalize(), _parse_time(), resolve_workflow_schedule_management(), WorkflowScheduleManagementIntent, get_workflow_definition(), _journal_discussion_window_command_intent(), list_workflows(), _normalize_text() (+35 more)

### Community 349 - "build_crystallization_from_window"
Cohesion: 0.42
Nodes (7): ExecutionDispatchCortexClient, Thin RPC client sending prepared_for_dispatch envelopes to cortex-exec.      Mir, _bus_returning(), test_dispatch_custom_timeout_overrides_default(), test_dispatch_raises_on_non_dict_payload(), test_dispatch_raises_on_rpc_not_ok(), test_dispatch_sends_real_verb_plan_and_returns_payload()

### Community 350 - "InnerStateFeaturesV1"
Cohesion: 0.15
Nodes (25): InnerFeatureV1, InnerStateFeaturesV1 — Orion's honest, decontaminated inner-state vector.  Felt+, is_corpus_row_healthy(), Pure gate predicate for the phi (InnerStateFeaturesV1) training corpus.  Garbage, Pure predicate: should this InnerStateFeaturesV1 row enter the phi corpus?, main(), _parse_args(), _all_none_cognitive_features() (+17 more)

### Community 351 - "HealthMonitor"
Cohesion: 0.09
Nodes (25): _check(), HealthCheck, HealthMonitor, Edge-triggered health monitor: fires an orion-notify attention request only, run_checks(), health(), get_settings(), Settings (+17 more)

### Community 352 - "memory-graph-draft-form.js"
Cohesion: 0.11
Nodes (13): attachFormEditor(), dateInput(), debounce(), defaultCardProjectionDefaults(), el(), entityMultiSelect(), entityOptionLabel(), entitySelect() (+5 more)

### Community 353 - "test_worker_speech.py"
Cohesion: 0.42
Nodes (8): _coerce(), map_attention_broadcast_ctx_to_substrate(), Map ``ctx['attention_broadcast']`` → one ``attending:current_focus`` node., _payload(), test_accepts_json_string_payload(), test_maps_broadcast_to_single_attending_node(), test_returns_none_on_missing_or_garbage_ctx(), test_returns_none_when_nothing_attended()

### Community 354 - "test_reverie_observability_section.py"
Cohesion: 0.12
Nodes (16): _alert_row(), _delta_row(), _FakeConn, _FakeEngine, _FakeResult, _queue_row(), _row(), test_compaction_delta_section_none_when_empty() (+8 more)

### Community 355 - "main.py"
Cohesion: 0.10
Nodes (32): build_llama_server_cmd_and_env(), _ensure_draft_file(), _ensure_hf_gguf_file(), _ensure_mmproj_file(), _ensure_model_file(), _get_llama_server_build(), _get_supported_llama_server_flags(), main() (+24 more)

### Community 356 - "test_llm_uncertainty.py"
Cohesion: 0.16
Nodes (23): _count_unstable_spans(), _entropy_proxy(), extract_llm_uncertainty_from_native_completion(), extract_llm_uncertainty_from_openai_response(), native_completion_probs_to_logprob_content(), Summary-only language-surface stability metrics from OpenAI logprobs., Normalize llama.cpp /completion prob shapes into OpenAI logprobs.content entries, summarize_logprob_content() (+15 more)

### Community 357 - "capability_policy.py"
Cohesion: 0.33
Nodes (4): CoreEventCache, get_core_event_cache(), _is_turn_effect_alert(), _normalize_turn_effect_alert()

### Community 358 - "test_proposal_runtime_worker.py"
Cohesion: 0.14
Nodes (9): health(), get_settings(), Settings, ProposalRuntimeWorker, _existing_frame(), _field(), _self_state(), test_worker_skips_when_field_missing() (+1 more)

### Community 359 - "test_bc_mode_understand.py"
Cohesion: 0.33
Nodes (8): _FakeBatch, _FakeTokenizer, test_bc_mode_is_deterministic_across_repeated_calls(), test_bc_mode_returns_three_tensors_with_valid_distribution(), test_understand_endpoint_pools_distribution_over_tokens(), test_understand_endpoint_rejects_empty_text(), _tiny_config(), _tiny_model()

### Community 360 - "substrate_execution_dispatch_routes.py"
Cohesion: 0.43
Nodes (6): _dispatch_status_summary(), _engine(), execution_dispatch_latest(), _load_latest_dispatch_frame(), Read-only debug API for substrate execution dispatch frames., Operator-visible breakdown of what actually happened in this frame --     P3's "

### Community 361 - "chat_stance.py"
Cohesion: 0.04
Nodes (89): autonomy_graph_backend_raw(), autonomy_graph_reads_explicitly_enabled(), AutonomyGraphReadPlan, _env_float(), is_quick_autonomy_graph_lane(), log_autonomy_graph_backend_decision(), _parse_subjects_csv(), _parse_subqueries_csv() (+81 more)

### Community 362 - "Service: orion-spark-introspector"
Cohesion: 0.15
Nodes (14): Channel "orion:cognition:trace" (kind=event, schema=CognitionTracePayload) producers=[orion-cortex-exec] consumers=[orion-spark-introspector, orion-landing-pad, orion-bus-tap], Channel "orion:self:inner_features" (kind=telemetry, schema=InnerStateFeaturesV1) producers=[orion-spark-introspector] consumers=[orion-hub, orion-sql-writer], Channel "orion:spark:introspect:candidate" (kind=event, schema=SparkCandidateV1) producers=[orion-cortex-exec] consumers=[orion-spark-introspector], Channel "orion:spark:introspect:candidate:log" (kind=event, schema=SparkCandidateV1) producers=[orion-hub] consumers=[orion-spark-introspector], Channel "orion:spark:introspector:reply:*" (kind=result, schema=GenericPayloadV1) producers=[orion-cortex-orch] consumers=[orion-spark-introspector], Channel "orion:spark:signal" (kind=telemetry, schema=SparkSignalV1) producers=[*] consumers=[orion-spark-introspector, orion-state-service], Channel "orion:spark:telemetry" (kind=telemetry, schema=SparkTelemetryPayload) producers=[orion-spark-introspector] consumers=[orion-sql-writer], Schema: CognitionTracePayload (+6 more)

### Community 363 - "ChatHistorySparkMetaPatchV1"
Cohesion: 0.16
Nodes (8): ChatHistorySparkMetaPatchV1, MemoryConsolidationWindowV1, _ExistingRow, _FakeQuery, _FakeSession, test_spark_meta_patch_merges_existing_row(), test_spark_meta_patch_missing_row(), test_consolidation_status_accepts_skipped()

### Community 364 - "test_boundary.py"
Cohesion: 0.15
Nodes (20): _label_line_reached(), _normalize_binary_tops(), _normalize_shift_tops(), _normalize_token(), Map BPE-split shift tokens (e.g. ' TOP') onto SHIFT_KINDS for softmax., _resolve_shift_kind(), scores_from_llm_result(), should_close_window() (+12 more)

### Community 365 - "spark_narrative.py"
Cohesion: 0.18
Nodes (22): _arousal_band(), _as_float(), _center_valence(), _clarity_band(), _format_value(), _overload_band(), Compact, structured bins suitable for embedding in mirror telemetry hints.     (, Compact hint naming which real hardware node is most salient this tick     (2026 (+14 more)

### Community 366 - "settings.py"
Cohesion: 0.36
Nodes (5): GPUConfig, LLMProfile, LLMProfileRegistry, Config, Settings

### Community 367 - "sql_timeline.py"
Cohesion: 0.11
Nodes (13): _filter_excluded_rows(), _is_current_turn_echo(), _memory_filter_clause(), _normalize_text(), _parse_row(), _pick_id_col(), _pick_session_col(), _pick_spark_meta_col() (+5 more)
=======
### Community 362 - "Service: orion-spark-introspector"
Cohesion: 0.10
Nodes (21): Channel "orion:equilibrium:metacog:trigger" (kind=event, schema=MetacogTriggerV1) producers=[orion-equilibrium-service] consumers=[orion-cortex-orch, orion-sql-writer], Channel "orion:equilibrium:snapshot" (kind=event, schema=EquilibriumSnapshotV1) producers=[orion-equilibrium-service] consumers=[orion-cortex-orch], Channel "orion:metacognition:tick" (kind=telemetry, schema=MetacognitionTickV1) producers=[orion-equilibrium-service] consumers=[orion-sql-writer], Channel "orion:self:inner_features" (kind=telemetry, schema=InnerStateFeaturesV1) producers=[orion-spark-introspector] consumers=[orion-hub, orion-sql-writer], Channel "orion:self:phi_reward" (kind=telemetry, schema=PhiIntrinsicRewardV1) producers=[orion-spark-introspector] consumers=[orion-substrate-runtime, orion-sql-writer, orion-hub], Channel "orion:spark:introspect:candidate" (kind=event, schema=SparkCandidateV1) producers=[orion-cortex-exec] consumers=[orion-spark-introspector], Channel "orion:spark:introspect:candidate:log" (kind=event, schema=SparkCandidateV1) producers=[orion-hub] consumers=[orion-spark-introspector], Channel "orion:spark:state:snapshot" (kind=telemetry, schema=SparkStateSnapshotV1) producers=[orion-spark-introspector, orion-equilibrium-service] consumers=[orion-state-service, orion-sql-writer] (+13 more)
>>>>>>> Stashed changes

### Community 368 - "orion-vision-council service"
Cohesion: 0.09
Nodes (25): chromadb dependency (vector store backend), orion-vector-writer service, orion-vision-council docker-compose config, evidence_grounding.py choke point, evidence_transition.py choke point, orion-vision-council service, orion-vision-council dependencies (fastapi, redis, pydantic), orion-vision-edge live MJPEG/SSE debug UI (+17 more)

<<<<<<< Updated upstream
### Community 369 - "context_exec_permissions_for_llm_profile"
Cohesion: 0.05
Nodes (71): context_exec_permissions_for_llm_profile(), Map Hub compute lane / LLM profile to a context-exec permission envelope., _call_name(), detect_semantic_tools_from_code(), _detect_via_ast(), _detect_via_regex(), Deterministic semantic tool detection for agent_repl code_action telemetry., Return registered workbench tool names found in *code*, first appearance order. (+63 more)

### Community 370 - "VectorHostEmbeddingProvider"
Cohesion: 0.14
Nodes (11): CapabilitiesResponse, capabilities(), VectorHostEmbeddingProvider, build_topic_engine(), _build_vectorizer(), _parse_topic_list(), _parse_zeroshot_list(), _require_topic_engine_runtime() (+3 more)

### Community 371 - "preview_dataset"
Cohesion: 0.25
Nodes (13): DatasetPreviewDoc, DatasetPreviewRequest, DatasetPreviewResponse, preview_dataset_endpoint(), apply_overrides(), build_conversations(), Conversation, OverrideRecord (+5 more)

=======
>>>>>>> Stashed changes
### Community 372 - "Orion Cortex Exec Service"
Cohesion: 0.11
Nodes (24): concept_induction_pass Workflow, Concept Profile Repository Seam, orion-chat-memory Docker Compose Service, orion-chat-memory Python Dependencies, orion-collapse-mirror Docker Compose Service, Orion Collapse Mirror Service, orion-collapse-mirror Python Dependencies, orion-consolidation-runtime Docker Compose Service (+16 more)

<<<<<<< Updated upstream
### Community 373 - "test_turn_effect.py"
Cohesion: 0.12
Nodes (32): _coerce_float(), compute_deltas_from_turn_effect(), _delta_block(), evaluate_turn_effect_alert(), _evidence_block(), should_emit_turn_effect_alert(), turn_effect_from_appraisal(), turn_effect_from_spark_meta() (+24 more)

### Community 374 - "model_moc.py"
Cohesion: 0.13
Nodes (20): main(), parse_args(), generate(), main(), parse_args(), entropy_floor_loss(), fisher_trace_proxy(), ManifoldBatchMetrics (+12 more)

### Community 375 - "BiometricsCollector"
Cohesion: 0.15
Nodes (8): filter_temps(), BiometricsCollector, collect_biometrics(), _CpuTimes, _DiskStats, _NetStats, collect_gpu_stats(), Collects the latest GPU stats by:      1. Running a shell script that writes a f

### Community 376 - "build_route_arbitration_grammar_events"
Cohesion: 0.31
Nodes (8): apply_rotary_pos_emb(), multinomial_num_samples_1(), Rotates half the hidden dims of the input., Applies Rotary Position Embedding to the query and key tensors.      Args:, # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generat, rotate_half(), sample(), sample_top_p()

### Community 377 - "refreshForgeTab"
Cohesion: 0.12
Nodes (24): forgeBadgeClass(), forgeHideSourceIngestError(), forgeRenderBadge(), forgeRenderClaimsList(), forgeRenderCompileResult(), forgeRenderCompileSpecCheckboxes(), forgeRenderReviewsList(), forgeRenderSearchResults() (+16 more)

### Community 378 - "test_recall_canary_battle_harness.py"
Cohesion: 0.28
Nodes (7): _FakeClient, _load_module(), _status_payload(), test_battle_runner_never_calls_judgment_or_review_or_execute_once(), test_battle_runner_rejects_invalid_profile_before_posting(), test_battle_runner_uses_default_profile_when_omitted(), test_operator_token_loader_prefers_explicit_then_env()

### Community 379 - "intention.py"
Cohesion: 0.18
Nodes (3): IntentionModel_v1a, IntentionModel_v1p, Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer

### Community 380 - "StateJournaler"
Cohesion: 0.16
Nodes (7): _fetch_rollups(), get_rollups(), StateJournaler, _utcnow(), Config, get_settings(), Settings

### Community 381 - "action_outcomes.py"
Cohesion: 0.12
Nodes (31): append_action_outcome(), _db_url(), _get_engine(), load_action_outcomes(), _load_from_sql(), _load_raw(), Read the most recent outcomes for a subject from the shared SQL store.      Retu, Exclusive lock for read-modify-write on the outcome store (best-effort). (+23 more)

### Community 382 - "extract_cortex_payload_text"
Cohesion: 0.16
Nodes (20): cortex_exec_failure_detail(), extract_cortex_payload_text(), _openai_choice_message_text(), Extract model text / JSON-bearing strings from Cortex PlanExecutionResult-shaped, Summarize why a cortex exec payload has no usable model text., Return best-effort model text from a cortex exec payload (may be JSON-ish prose), _service_block_text_candidates(), _sorted_steps() (+12 more)

### Community 383 - "classify.py"
Cohesion: 0.35
Nodes (12): build_turn_change_appraisal(), _build_window_transcript(), _classify_scores(), classify_turn(), _clip(), _degraded_patch(), _llm_classify(), _prior_turn_baseline() (+4 more)

### Community 384 - "WorkflowDispatchRequestV1"
Cohesion: 0.42
Nodes (8): _projection(), Unit tests for the self-observability store writers (curiosity + dwell)., _signal(), _store_with_conn(), test_save_coalition_dwell_inactive_when_zero_ticks(), test_save_coalition_dwell_row_shape_and_prune(), test_save_curiosity_candidates_empty_persists_heartbeat(), test_save_curiosity_candidates_inserts_json_array_and_prunes()

### Community 386 - "test_single_consumer_channels_gate.py"
Cohesion: 0.06
Nodes (21): find_orphaned_registry_entries(), find_unregistered_trigger_kinds(), _load(), main(), trigger_kinds the journaler actually understands but the dispatch registry     d, Registry rows for trigger_kinds no longer recognized by the journaler at all, Regression/live check: the actual orion.journaler.worker._TRIGGER_TO_MODE and, test_main_fails_when_a_trigger_kind_is_unregistered() (+13 more)

### Community 387 - "sync_file"
Cohesion: 0.16
Nodes (19): example_value_is_host_placeholder(), main(), parse_kv(), Skip syncing template placeholders that must stay host-specific in local .env., Result of syncing one service's .env from its .env_example.      updated: keys a, should_sync_key(), sync_file(), SyncResult (+11 more)

### Community 388 - "test_async_notify_producers.py"
Cohesion: 0.04
Nodes (75): DailyMetacogV1, DailyPulseV1, extract_message_sections(), new_reply_channel(), build_daily_window(), _cfg(), _clamp_daily_metacog_payload(), _daily_metacog_dedupe_key() (+67 more)

### Community 389 - "bus_observer.py"
Cohesion: 0.18
Nodes (12): build_rollup_from_redis_snapshot(), _fetch_redis_snapshot(), load_channel_catalog_names(), ObserverRollup, _resolve_catalog_path(), run_bus_observer_loop(), run_observer_tick(), _sample_window_id() (+4 more)

### Community 390 - "build_substrate_attention_frame"
Cohesion: 0.43
Nodes (7): _lane(), _reader(), test_attention_broadcast_lane_registered(), test_episode_lane_registered_with_extended_max_age(), test_hydrate_injects_episode_older_than_global_gate(), test_hydrate_rejects_episode_older_than_lane_max_age(), test_hydrate_rejects_stale_attention_broadcast()

### Community 391 - "renderNotifications"
Cohesion: 0.13
Nodes (23): addNotification(), focusChatInput(), formatHubLocalTime(), handleAttentionAck(), handleChatMessageReceipt(), isAttentionNotification(), isChatMessageNotification(), isNotificationTrayItem() (+15 more)

### Community 392 - "test_substrate_review_runtime_hub_debug.py"
Cohesion: 0.08
Nodes (4): test_execute_once_endpoint_is_single_cycle_operator_only_and_followup_default_off(), test_execute_once_followup_endpoint_keeps_followup_explicit(), test_mutation_lineage_readonly_endpoints(), test_self_experiments_trigger_daily_publishes_bus_event()

### Community 393 - "fetch_graph_compression_fragments"
Cohesion: 0.14
Nodes (19): _extract_keywords(), fetch_graph_compression_fragments(), _fetch_summary_from_fuseki(), _get_engine(), Query Postgres artifact index, rank by salience + keyword relevance,     fetch s, Simple salience + keyword hit scoring., _score_artifact(), _clear_engine_cache() (+11 more)

### Community 394 - "build_substrate_grammar_truth"
Cohesion: 0.39
Nodes (7): _lane(), Tests for the latest_reverie_thought lane in felt-state reader., _reader(), test_hydrate_absent_row_leaves_ctx_key_unset(), test_hydrate_injects_fresh_reverie_thought(), test_hydrate_rejects_stale_reverie_thought(), test_reverie_lane_registered()

### Community 395 - "test_reverie_chain.py"
Cohesion: 0.14
Nodes (14): _broadcast(), Phase C — reverie chain. Deterministic control is fully testable with injected f, _step_always(), test_chain_never_raises_on_step_error(), test_chain_none_when_no_coalition(), test_chain_step_none_terminates_without_raise(), test_chain_suppressed_by_refractory(), test_chain_suppresses_theme_after_completion() (+6 more)

### Community 396 - "train.py"
Cohesion: 0.26
Nodes (11): Rank-aware partition for IterableDataset training shards., shard_token_ids(), ddp_print(), ddp_setup(), eval_loss(), main(), parse_args(), save_checkpoint() (+3 more)

### Community 397 - "substrate_consolidation_routes.py"
Cohesion: 0.05
Nodes (59): _engine_for(), load_expectations_for_motif(), _load_latest_consolidation_frame(), load_latest_tensor_slices(), load_recent_motifs(), load_schema_candidates(), _parse_json(), Read-only Postgres accessors for consolidation substrate artifacts. (+51 more)

### Community 398 - "resolve_destination"
Cohesion: 0.26
Nodes (18): _match(), _nearest(), _others(), _pos(), resolve_destination(), _resolve_target(), ResolveResult, _intent() (+10 more)

### Community 399 - "Idea 2: Hub turn-hop WebSocket relay"
Cohesion: 0.17
Nodes (12): Turn Visibility Design Spec (2026-07-11), orion:turn:hop bus channel (proposed), Idea 8: Self-observability consumer (deferred), Idea 3: Swimlane pipeline view (not a node graph), Idea 5: Timestamp scrubber (rewind/replay/fast-forward), scripts/trace_unified_turn.py (regex hop detector), Idea 2: Hub turn-hop WebSocket relay, TurnHopV1 schema (proposed) (+4 more)

### Community 400 - "parse_json_object"
Cohesion: 0.43
Nodes (7): _Env, test_extractor_disabled_noop(), test_stage1_dedupes_via_fingerprint(), test_stage1_extracts_and_inserts(), test_stage1_noop_when_pool_unavailable(), test_stage2_raises(), _turn_env()

### Community 401 - "test_phi_corpus_diag_script.py"
Cohesion: 0.22
Nodes (19): main(), _parse_args(), Pure: load + gate-check a corpus, return the full report dict. Never raises., run_diag(), features_version(), input_features(), _feature(), _live_corpus() (+11 more)

### Community 402 - "SchedulerCursorStore"
Cohesion: 0.32
Nodes (4): _install_stdlib_logging_bridge(), _InterceptHandler, lifespan(), Route stdlib logging records into loguru.      The worker (`orion.spark.concept_

### Community 403 - "refreshMindRunsForCorrelation"
Cohesion: 0.11
Nodes (27): applyHashToTab(), enableMindModalFocusTrap(), ensureOrganSignalsGraph(), escapeHtml(), fetchMindRunDetail(), formatMindRunsApiError(), formatMindTs(), getMindModalTabbables() (+19 more)

### Community 404 - "test_proposal_review_hub.py"
Cohesion: 0.17
Nodes (18): _cleanup_hub_path_pollution(), hub_client(), Hub proposal review client, routes, and review actions., Prevent hub reload from shadowing orion-context-exec `app` in later tests., _reload_hub_modules(), test_hub_pending_decisions_shows_denver_memory_correction(), test_hub_proposal_review_client_get_allowlist_only(), test_hub_proposal_review_client_post_allowlist_only() (+10 more)

### Community 406 - "MindRunBudget"
Cohesion: 0.14
Nodes (13): MindRunBudget, Wall-clock budget for Cortex-governed Mind LLM phases., Tracks remaining wall time for a single MindRun and caps per-phase timeouts., _mind_prep(), Wall-clock budget enforcement (loop_budget_exceeded)., Simulate snapshot phase taking longer than policy allows., Regression for fix/mind-enrichment-wall-budget: a 12s wall cannot fit even a, The corrected 180s wall lets each of the 3 sequential phases use its full     co (+5 more)

### Community 407 - "synthesis.py"
Cohesion: 0.18
Nodes (21): Settings, _claims_list_empty(), _coerce_array_claim_item(), _coerce_evidence_refs(), _coerce_semantic_claim_item(), coerce_semantic_llm_root(), _float_field(), _has_current_claim_shape() (+13 more)

### Community 408 - "enrichment.py"
Cohesion: 0.17
Nodes (21): EnrichmentSpec, _build_run_record_for_update(), _elapsed_secs(), enqueue_enrichment(), _enrich_segment(), _extract_evidence(), _finalize_enrichment(), _generate_edges() (+13 more)

### Community 409 - "test_check_service_env_compose_parity.py"
Cohesion: 0.12
Nodes (14): _make_service(), Real syntax from services/orion-hub/docker-compose.yml -- the plain string-list, A sidecar service's own environment: keys must never leak into the checked     s, Regression test: the actual orion-recall docker-compose.yml, checked against the, docker-compose also supports `environment:` as a mapping (KEY: value) instead, test_compose_env_file_directive_detected_extended_mapping_form(), test_compose_env_keys_matches_mapping_form(), test_main_json_output_shape() (+6 more)

=======
>>>>>>> Stashed changes
### Community 410 - "CLAUDE.md — Orion Subagent Development Contract: repo-wide rules for how AI coding agents work in Orion-Sapienform, identical content to AGENTS.md (symlinked)"
Cohesion: 0.13
Nodes (15): Clean git and worktree rules (§2): start with git status/branch check, classify dirt before mixing, prefer a separate worktree for parallel subagent work — stated as prose policy, not an enforced mechanism, Completion status (§19): every task ends in exactly one of DONE, DONE_WITH_CONCERNS, BLOCKED, or NEEDS_CONTEXT — never 'partially done', Env/config/settings contract (§7): env parity is non-negotiable — any add/remove/rename of an env key must update .env_example, local .env, settings.py, docker-compose.yml, requirements.txt, and README in the same changeset, via scripts/sync_local_env_from_example.py, Event substrate first: grow from events -> schema -> trace -> reducer -> projection -> eval -> UI, not from a hand-authored mind-palace ontology with no runtime proof, Follow-through is part of the feature: a patch isn't done until the branch is clean, tests/evals pass, review is fixed, env is synced, Docker checks run, restart commands are listed, the branch is pushed, and the PR report exists, No keyword cathedrals: labels/enums/taxonomies/ontology nodes without a schema contract, producer, consumer, reducer, UI surface, metric, test, eval, or live smoke attached in the same patch are banned as junk that names the world without changing runtime behavior, PR description template (§18): required Markdown shape covering summary, outcome moved, architecture touched, files changed, schema/bus/API changes, env/config changes, tests/evals/docker checks run, review findings fixed, restart required, risks, Prime directive: do not build cathedrals — prefer thin seams, small patches, explicit contracts, fast tests, and visible evidence; design mode gets a design artifact, implementation mode gets a finished branch with tests/evals/docs/review/PR report (+7 more)

<<<<<<< Updated upstream
### Community 411 - "NISUPSClient"
Cohesion: 0.18
Nodes (8): PowerStatus, Parsed snapshot of UPS state from SNMP., NISUPSClient, Reads APC UPS status from a local or remote apcupsd NIS server (TCP 3551)., Connects to apcupsd and requests the 'status' dump., Async wrapper to fetch and parse status.          (We keep it async to match the, SNMPUPSClient, _to_int()

### Community 412 - "__init__.py"
Cohesion: 0.16
Nodes (6): FaceDetector, build_detectors(), MotionDetector, PresenceDetector, Run detection on BGR frame.         Returns list of (x, y, w, h, label, score)., YoloDetector

### Community 413 - "Proposal Review API"
Cohesion: 0.14
Nodes (21): Context-exec Beta Runbook, AgentChain Replacement, AgentChainService (legacy), belief_provenance mode, Context-exec Workbench, Cortex Sovereignty, memory_correction_proposal mode, MemoryCorrectionProposalV1 (+13 more)

### Community 414 - "test_chat_prompt_context_guardrails.py"
Cohesion: 0.32
Nodes (4): _broadcast(), Phase D — episode + motif grounding (read-only). Refs are capped, degrade to emp, test_tick_attaches_grounding_when_flag_on(), test_tick_no_grounding_when_flag_off()

### Community 415 - "ModelManager"
Cohesion: 0.38
Nodes (4): Settings, test_settings_reads_legacy_skip_enabled_alias(), test_settings_reads_legacy_skip_max_sec_alias(), test_settings_refresh_ttl_default_zero()

### Community 416 - "AutonomyStateV2"
Cohesion: 0.17
Nodes (22): _db_url(), _get_engine(), load_autonomy_state_v2(), Postgres-backed persistence for the latest AutonomyStateV2 per subject.  Closes, Load the most recently persisted AutonomyStateV2 for a subject.      Returns Non, Upsert the latest AutonomyStateV2 for a subject.      No-ops when no DSN is conf, save_autonomy_state_v2(), _make_db() (+14 more)

### Community 417 - "TensionRateLimiter"
Cohesion: 0.19
Nodes (14): Bound the tension stream: per-source cap, dedup, storm safety (spec §5).  Even w, Source identity: kind + the set of drives it pushes. Two tensions with the     s, Return the subset of ``candidates`` allowed through, updating windows., _signature(), TensionRateLimiter, Task 5: rate-limit / dedup / bounded state., A relief (negative-weight) tension must be rate-limited by which     drives it t, _t() (+6 more)

### Community 418 - "social.py"
Cohesion: 0.19
Nodes (20): _append_unique(), build_hub_direct_inspection_sections(), build_social_inspection_snapshot(), _candidate_bucket(), _compact_text(), _make_section(), Materialize hub-local inspection sections when bridge turn-policy surfaces are a, _safe_text() (+12 more)

### Community 419 - "memory_graph_suggest.py"
Cohesion: 0.60
Nodes (4): resolve_graphiti_adapter_url(), test_resolve_empty_when_both_unset(), test_resolve_falls_back_to_graphiti_url(), test_resolve_prefers_adapter_url_over_graphiti_url()

### Community 420 - "test_mind_light_snapshot.py"
Cohesion: 0.47
Nodes (5): api_llm_routes(), _base_url(), fetch_routes(), Hub client for LLM gateway GET /routes catalog., _timeout()

### Community 421 - "Settings"
Cohesion: 0.12
Nodes (10): _trigger_world_pulse_run(), Config, get_settings(), Settings, test_daily_goal_archive_enabled_by_default(), test_daily_goal_archive_run_on_startup_default(), test_blank_workflow_int_env_values_fall_back_to_defaults(), test_world_pulse_journal_disabled_by_default() (+2 more)

### Community 422 - "substrate_observability_routes.py"
Cohesion: 0.25
Nodes (19): _attention_broadcast_section(), _compaction_delta_section(), _compaction_queue_section(), _curiosity_section(), _engine(), _hub_presence_section(), _iso(), _latest_row() (+11 more)

### Community 423 - "self-brain.js"
Cohesion: 0.24
Nodes (20): buildDimRail(), DIMENSIONS, drawBrain(), drawEkg(), drawSpotlight(), _get(), goLive(), init() (+12 more)

### Community 424 - "substrate-lattice.js"
Cohesion: 0.21
Nodes (20): _asList(), _clearError(), _esc(), _fmt(), _gateColor(), _get(), _loadAll(), _post() (+12 more)

### Community 425 - "apply_structured_output_to_payload"
Cohesion: 0.16
Nodes (10): apply_structured_output_to_payload(), build_response_format(), Build llama.cpp / OpenAI-compatible response_format payloads from a named method, Mutate opts in place for structured output + thinking policy.     Returns (opts,, Pick method: options.structured_output_method → env → none., Return response_format dict for the given method, or None for none/unknown., resolve_structured_output_method(), response_format_shape_label() (+2 more)
=======
### Community 413 - "Proposal Review API"
Cohesion: 0.24
Nodes (12): memory_correction_proposal mode, MemoryCorrectionProposalV1, orion_proposal_cli.py, patch_proposal mode, PatchProposalV1, ProposalEnvelopeV1, Proposal Ledger, ProposalLedgerRecordV1 (+4 more)
>>>>>>> Stashed changes

### Community 427 - "orion-substrate-runtime service (biometrics closed loop, grammar reducers, Layers 1-5)"
Cohesion: 0.10
Nodes (21): biometrics_pressure organ contract (organ.contract.v1), orion-substrate-organs service (event-native organ contracts), orion-substrate-runtime compose config, brain-frame live stimulus-response smoke test (acceptance section 6), orion-substrate-runtime service (biometrics closed loop, grammar reducers, Layers 1-5), orion-substrate-runtime dependencies (fastapi/sqlalchemy/psycopg2/redis), orion-substrate-telemetry compose config, orion-substrate-telemetry dependencies (fastapi/asyncpg/redis) (+13 more)

<<<<<<< Updated upstream
### Community 428 - "_compute_current_distribution"
Cohesion: 0.32
Nodes (12): _compute_current_distribution(), drift_daemon_loop(), _is_drift_alert(), _js_divergence(), _load_baseline_distribution(), _load_clusterer(), _normalize_counts(), _require_hdbscan_predict() (+4 more)

### Community 429 - "config/mesh_remediation_roster.yaml (auto-remediation roster)"
Cohesion: 0.12
Nodes (19): cortex-exec remediation entry, cortex-gateway remediation entry, cortex-orch remediation entry, equilibrium-service remediation entry (auto_remediate: false), landing-pad remediation entry, config/mesh_remediation_roster.yaml (auto-remediation roster), notify remediation entry (auto_remediate: false), recall remediation entry (+11 more)

### Community 430 - "map_drive_state_to_intent"
Cohesion: 0.60
Nodes (5): build_state_response(), get_security_state(), get_security_state_root(), set_security_state(), set_security_state_root()
=======
### Community 429 - "config/mesh_remediation_roster.yaml (auto-remediation roster)"
Cohesion: 0.16
Nodes (14): cortex-exec remediation entry, cortex-gateway remediation entry, cortex-orch remediation entry, equilibrium-service remediation entry (auto_remediate: false), landing-pad remediation entry, config/mesh_remediation_roster.yaml (auto-remediation roster), notify remediation entry (auto_remediate: false), recall remediation entry (+6 more)
>>>>>>> Stashed changes

### Community 431 - "bus-core (Redis broker container)"
Cohesion: 0.14
Nodes (20): Orion (AI Town persona card), orion-biometrics service, fork_rpc_client (orion.core.bus.rpc_fork), GrammarEventV1 (bus substrate trace schema), bus-core (Redis broker container), bus-exporter (Prometheus redis_exporter), bus-observer (grammar trace sidecar), bus_transport_reducer (deferred Layer 3 reducer) (+12 more)

<<<<<<< Updated upstream
### Community 432 - "test_attention_loops_reader.py"
Cohesion: 0.15
Nodes (8): _Conn, _Engine, _Result, test_latest_salience_for_theme_dict_features(), test_latest_salience_for_theme_no_row(), test_latest_salience_for_theme_string_features(), test_load_pending_loops_falls_back_to_theme_key(), test_load_pending_loops_filters_and_builds()

### Community 433 - "compaction.py"
Cohesion: 0.47
Nodes (3): callsyne_room_message(), _normalize_signature(), _verify_webhook_hmac_signature()

### Community 434 - "grammar_atlas_routes.py"
Cohesion: 0.27
Nodes (17): atom_neighborhood_api(), atom_provenance_api(), atom_temporal_path_api(), _ensure_sql_writer_on_path(), get_trace_api(), get_trace_graph_api(), _grammar_query(), list_traces_api() (+9 more)

### Community 435 - "memory-graph-draft-ui.js"
Cohesion: 0.17
Nodes (14): attach(), coalesceChatSuggestDraft(), coalesceMemoryGraphSuggestEnvelope(), contentLooksLikeGatewayFailureBlurb(), debounce(), draftToCyElements(), emptySuggestDraft(), emptyValidSuggestDraft() (+6 more)

### Community 436 - "rem_compaction.py"
Cohesion: 0.60
Nodes (5): _base_source(), _journal_payload(), test_journal_entry_write_uses_nested_payload_for_validation_and_write(), test_non_regression_other_route_keeps_payload_shape(), test_regression_full_envelope_shape_fails_journal_schema_but_nested_payload_passes()

=======
>>>>>>> Stashed changes
### Community 437 - "AGENTS.md — Orion Subagent Development Contract: repo-wide rules for how AI coding agents work in Orion-Sapienform, aimed at inspectable, testable, parallel-safe agent behavior"
Cohesion: 0.11
Nodes (19): Clean git and worktree rules (§2): start with git status/branch check, classify dirt before mixing, prefer a separate worktree for parallel subagent work — stated as prose policy, not an enforced mechanism, Completion status (§19): every task ends in exactly one of DONE, DONE_WITH_CONCERNS, BLOCKED, or NEEDS_CONTEXT — never 'partially done', Deterministic gates over repeated yelling: if Juniper has to repeat a rule twice, turn it into a script/test/check/hook/make target instead of a louder prompt, Env/config/settings contract (§7): env parity is non-negotiable — any add/remove/rename of an env key must update .env_example, local .env, settings.py, docker-compose.yml, requirements.txt, and README in the same changeset, via scripts/sync_local_env_from_example.py, Event substrate first: grow from events -> schema -> trace -> reducer -> projection -> eval -> UI, not from a hand-authored mind-palace ontology with no runtime proof, Follow-through is part of the feature: a patch isn't done until the branch is clean, tests/evals pass, review is fixed, env is synced, Docker checks run, restart commands are listed, the branch is pushed, and the PR report exists, No empty-shell cognition: do not ship cognition-shaped output with no substance — empty semantic projections, placeholder memory cards, fallback text as generated cognition, raw_len=0 treated as success, stale reducer cursors are all invalid success states requiring inspectable evidence, No keyword cathedrals: labels/enums/taxonomies/ontology nodes without a schema contract, producer, consumer, reducer, UI surface, metric, test, eval, or live smoke attached in the same patch are banned as junk that names the world without changing runtime behavior (+11 more)

### Community 438 - "Metrics Swamp Arsonist Review"
<<<<<<< Updated upstream
Cohesion: 0.22
Nodes (14): Metrics Swamp Arsonist Review, AutonomyStateV2, DriveEngine, Endogenous Naming Collision, endogenous_origination.py (NO-GO), endogenous_runtime.py, phi encoder, SelfStateV1 (+6 more)
=======
Cohesion: 0.09
Nodes (33): Metrics Swamp Arsonist Review, AutonomyStateV2, DriveEngine, Endogenous Naming Collision, endogenous_origination.py (NO-GO), endogenous_runtime.py, Field-topology Edge Weights, phi encoder (+25 more)
>>>>>>> Stashed changes

### Community 439 - "Recall Epistemic Honesty + Observability Spec"
Cohesion: 0.15
Nodes (19): Graphiti-Core Backend Activation Spec, Memory Crystallization (belief), graphiti_core Backend (hybrid vector+graph search), RELATES_TO Edge Schema (FalkorDB), Search Stack Driver/Client Reuse Cache, FalkorDB Vectorf32 Cast Bug/Fix, Memory Recall Reinforcement + Decay Wiring Spec, CrystallizationDynamicsV1 (activation/decay) (+11 more)

<<<<<<< Updated upstream
### Community 440 - "properties"
Cohesion: 0.11
Nodes (19): description, type, type, description, type, description, type, description (+11 more)

### Community 441 - "build_perception"
Cohesion: 0.20
Nodes (16): _active_conversation(), build_perception(), _facing_partner(), True iff Orion's facing vector aligns with the direction to the partner.      Co, Shape the conversation Orion is a member of (invited/walkingOver/participating)., test_build_perception_computes_distances_and_nearby(), test_build_perception_exposes_own_facing_and_pathfinding(), test_build_perception_no_conversation_when_orion_not_member() (+8 more)

### Community 442 - "memory_graph_routes.py"
Cohesion: 0.08
Nodes (38): call_with_fuseki_retry(), fuseki_http_error_body(), fuseki_http_retry_attempts(), fuseki_http_retry_base_delay_sec(), is_fuseki_lock_exhaustion(), is_fuseki_retryable_http_error(), Fuseki/TDB transient HTTP failure detection and client-side retry., Retry Fuseki graph-store / SPARQL HTTP calls on lock exhaustion and gateway erro (+30 more)

### Community 443 - "Settings"
Cohesion: 0.23
Nodes (7): EmailWorldPulseRenderV1, Settings, test_curiosity_defaults_off(), test_curiosity_env_override(), _FakeNotifyClient, test_email_publish_is_blocked_when_disabled(), test_world_pulse_defaults_are_conservative()

### Community 444 - "identity.py"
Cohesion: 0.53
Nodes (5): _channel_index(), Bus catalog + schema registry alignment for unified Orion turn., test_harness_governor_consumes_cortex_exec_results(), test_harness_governor_produces_cortex_exec_requests(), test_unified_turn_channels_exist_with_registry_schema_ids()

### Community 446 - "_project_reverie_glimpse"
Cohesion: 0.20
Nodes (18): _project_reverie_glimpse(), Projection helper: surface the latest fresh, non-hollow reverie thought.      Re, _fresh_payload(), Guard the call-site contract: chat_reverie_glimpse, when set, is exactly     the, is_hollow() rejects absent_coalition even if the stored `hollow` bool     says F, is_hollow() rejects evidence outside the coalition's grounding ids, even     if, A payload that no longer validates as SpontaneousThoughtV1 at all     (e.g. sche, test_ctx_key_wiring_only_sets_no_other_fields() (+10 more)

### Community 447 - "conversation_front.py"
Cohesion: 0.18
Nodes (15): _build_memory_digest_from_fragments(), build_personality_summary(), ChatTurnPayload, ChatTurnResult, conversation_front_worker(), handle_chat_turn(), Turn raw recall fragments into a compact bullet-style digest.     This is intern, Core Conversation Front handler.      - Takes a high-level ChatTurnPayload from (+7 more)

### Community 448 - "build_readout"
Cohesion: 0.23
Nodes (10): wake_today(), fetch_dream_row_from_sql(), Load latest dream row from Postgres for wake readout (canonical path)., build_readout(), _default_readout(), _dream_path_for_date(), _latest_dream_path(), Prefer the latest row from the `dreams` SQL table (durable path).     Fall back (+2 more)

### Community 449 - "NotificationCache"
Cohesion: 0.50
Nodes (4): _heuristic_gate_score(), Bounded lightweight score used by tests and fallback heuristics., test_heuristic_gate_score_bounds(), test_heuristic_gate_score_prefers_longer_text()

### Community 450 - "test_lane_routes.py"
Cohesion: 0.20
Nodes (16): _first_route_key(), LlmLaneRouteDecision, _match_served_by(), _norm_lane(), Side-effect free: picks a route-table key for the LLM gateway HTTP hop.      Cha, resolve_llm_lane_route(), _resolve(), test_agent_prefers_agent_then_background() (+8 more)

### Community 451 - "SparkContractMetrics"
Cohesion: 0.15
Nodes (5): SparkContractMetrics, _legacy_action(), FakeLogger, TestSparkContractMetrics, TestSqlWriterLegacyMode

### Community 452 - "receipt_pruner.py"
Cohesion: 0.23
Nodes (13): disk_usage_pct(), get_cached_pressure_state(), log_receipt_pressure(), maybe_run_emergency_prune(), measure_pressure_state(), refresh_pressure_cache(), run_emergency_prune(), run_safe_prune() (+5 more)

### Community 453 - "FakeConn"
Cohesion: 0.16
Nodes (9): FakeConn, _NullAsyncCtx, Minimal in-process double for the handful of asyncpg.Connection calls this     s, _row(), _seeded_rows(), test_digest_no_pending_rows_reports_cleanly(), test_digest_reflection_crystallization_shape(), test_digest_report_counts_and_reflection_creation() (+1 more)

### Community 454 - "cognition_trace_cache.py"
Cohesion: 0.18
Nodes (11): agent-trace.js (plain-text step consumer), api_routes.py (GET /api/cognition/trace/{correlation_id}), Idea 4: Click-through payload cards, cognition_trace_cache.py, CognitionTracePayload schema, HarnessRunStepV1 schema, HarnessStepRelay (harness_step_relay.py), orion:cognition:trace bus channel (+3 more)

### Community 455 - "AutonomyStateV2 evidence pipeline (chat, env-gated)"
Cohesion: 0.17
Nodes (15): get_route_targets(), _load_route_targets(), _resolve_route(), RouteTarget, health(), routes_catalog(), build_routes_response(), _entry_to_dict() (+7 more)
=======
### Community 454 - "cognition_trace_cache.py"
Cohesion: 0.18
Nodes (11): agent-trace.js (plain-text step consumer), api_routes.py (GET /api/cognition/trace/{correlation_id}), Idea 4: Click-through payload cards, cognition_trace_cache.py, CognitionTracePayload schema, HarnessRunStepV1 schema, HarnessStepRelay (harness_step_relay.py), orion:cognition:trace bus channel (+3 more)
>>>>>>> Stashed changes

### Community 455 - "AutonomyStateV2 evidence pipeline (chat, env-gated)"
Cohesion: 0.67
Nodes (3): config/autonomy/signal_drive_map.yaml, AutonomyStateV2 evidence pipeline (chat, env-gated), Orion Autonomy README

<<<<<<< Updated upstream
### Community 457 - "test_proposal.py"
Cohesion: 0.24
Nodes (15): _clamp01(), Phase B — spontaneous thought → governed proposal candidate.  Maps a non-hollow, Convert a thought into a review-gated proposal candidate, or None.      Returns, spontaneous_thought_to_candidate(), _coalition(), Phase B — spontaneous thought → governed proposal candidate.  The security-criti, test_absent_coalition_targets_self_state_without_raising(), test_autoaction_posture_recorded_but_gate_unchanged() (+7 more)

### Community 458 - "BiometricsAdapter"
Cohesion: 0.12
Nodes (16): BiometricsAdapter, EquilibriumAdapter, adapter(), make_induction_payload(), norm_ctx(), Biometrics adapter contract (spec §7.A) — lives under ``orion/signals/adapters/t, test_adapt_leaves_otel_for_gateway(), TestAdapt (+8 more)

### Community 459 - "HyperbolicGPTConfig"
Cohesion: 0.10
Nodes (10): HyperbolicGPTConfig, Block, MLP, BlockMoC, HyperbolicCausalSelfAttentionMoC, _jittered_raw_values(), MLP, Create inverse-softplus raw parameters with optional multiplicative jitter. (+2 more)

### Community 460 - "foundry.py"
Cohesion: 0.24
Nodes (16): _load_from_postgres(), _answer_styles(), build_semantic_foundry(), _depth_expectation(), _developmental_fit(), _extract_seed_concepts(), _extract_unknown_concepts(), _failure_modes() (+8 more)

### Community 461 - "test_check_concept_relation_digest_liveness.py"
Cohesion: 0.15
Nodes (5): main(), _query_backlog(), FakeConn, test_fresh_backlog_within_threshold_is_healthy(), test_no_backlog_is_healthy()

### Community 462 - "test_deviation_gate.py"
Cohesion: 0.22
Nodes (13): _gate(), Task 2: deviation gate fires on change, not presence., A steady stream (the scene_state flood) mints ~0 after warm-up., homeostasis 0.82 steady, then a real drop to 0.55 → sized impulse     (worse='do, A rise when only a fall is 'worse' mints nothing., A perfectly constant series (var→0) then a small step does not explode., test_cold_start_mints_nothing(), test_confidence_scales_impulse() (+5 more)

### Community 463 - "build_autonomy_slice"
Cohesion: 0.24
Nodes (16): AutonomyStateDeltaV1, _bounded_unique_tensions(), build_autonomy_slice(), _pressure_trend(), Cheap before/after pressure comparison from ctx['chat_autonomy_movement_debug']., Assemble the compact slice from the V2 reducer output already in ctx.      Retur, _delta(), _state_v2() (+8 more)

### Community 464 - "resolve_llm_lane_for_step"
Cohesion: 0.26
Nodes (15): _coerce_allow_chat_fallback(), Return True/False if caller set allow_chat_fallback on options or ctx; else None, Decide LLM lane metadata for gateway Phase 3 routing (orthogonal to route keys l, resolve_llm_lane_for_step(), Mirror the finalize_reflect context (top-level llm_lane) as cortex-exec merges i, _settings(), test_chat_general_chat_lane(), test_chat_lane_allow_chat_fallback_can_be_false() (+7 more)

### Community 466 - "run_recall_canary_battle.py"
Cohesion: 0.27
Nodes (10): ApiClient, BattleSummary, build_parser(), _fmt_row(), load_battle_fixture(), _load_operator_token(), main(), print_case_table() (+2 more)

### Community 468 - "workflow-schedule-ui.js"
Cohesion: 0.19
Nodes (13): cadenceSummary(), healthChipClass(), normalizeAnalytics(), normalizeEventItem(), normalizeHistoryItem(), normalizeRecentOutcomes(), normalizeSchedule(), oneOf() (+5 more)

### Community 469 - "workflow-ui.js"
Cohesion: 0.21
Nodes (16): buildConceptInductionSections(), buildWorkflowDetailRows(), canRunAgain(), extractWorkflow(), getWorkflowBadgeLabel(), getWorkflowStatusLabel(), normalizeConceptInductionDetails(), normalizeStatus() (+8 more)

### Community 470 - "test_fcc_claude_bridge_run.py"
Cohesion: 0.13
Nodes (3): _FakeProc, _FakeStream, test_cancel_turn_sigterms_active()

### Community 471 - "test_memory_graph_suggest_coalesce_ui.py"
Cohesion: 0.24
Nodes (12): _assert_valid_suggest_draft_shape(), _node_coalesce(), _parse_draft_text(), Regression: coalescer must not assign data.text directly to draftText without va, test_coalesce_accepts_valid_empty_suggest_draft(), test_coalesce_accepts_valid_nonempty_suggest_draft(), test_coalesce_failed_api_returns_empty_draft(), test_coalesce_never_assigns_prose_pattern_in_helpers() (+4 more)

### Community 475 - "Stance Assembly / ChatStanceBrief"
Cohesion: 0.12
Nodes (17): Proposal mode required before invasive cognition changes: memory, identity, self-modeling, autonomy, private recall, social continuity, or cognition-loop changes need a proposal naming capability change, data touched, privacy boundary, proof trace, dangerous failure mode, and rollback before implementation, baseline dispatch policy (retina_fast, every_n_frames 10, no caption/embeddings), porch_eye camera config (open-vocab detect: person/package/vehicle/animal), triggered dispatch policy (person trigger, TTL 8s, caption+embeddings on), config/vision_frame_router.yaml (baseline vs triggered dispatch), Autonomy readiness / mutation pipeline, Dreams / Dream Weaver (symbolic residue processing), Journal layer (autobiographical compression) (+9 more)

### Community 477 - "projection_context.py"
Cohesion: 0.12
Nodes (24): build_identity_context(), load_identity_file(), resolve_identity_path(), enrich_projection_context(), identity_kernel_with_fallbacks(), inject_identity_context_for_projection(), _orion_state_from_ctx(), Shared projection-context enrichment for Orch Mind preflight and Exec parity. (+16 more)

### Community 478 - "required"
Cohesion: 0.15
Nodes (17): required, required, can_interrupt_others, category, description, interruptible, max_recursion_depth, name (+9 more)

### Community 480 - "parse_json_object"
Cohesion: 0.28
Nodes (13): _json_to_dict(), parse_json_object(), repair_json(), _strip_outer_quotes(), _try_candidate(), RFC 8259 has no \\'; models often emit it for possessives inside JSON strings., test_parse_json_object_double_encoded_string(), test_parse_json_object_escaped_object_text() (+5 more)

### Community 481 - "test_ouroboros_invariants.py"
Cohesion: 0.31
Nodes (8): Cross-cutting ouroboros invariants for the reverie/dream/compaction weave.  The, A producer of a weave channel must never also be listed as a consumer of     tha, The memory-touching / dispatch-adjacent channels stay dead-ended: no     service, test_channel_schema_ids_match_the_weave_contract(), test_dangerous_channels_have_no_live_consumer(), test_every_weave_kind_is_registered_and_resolvable(), test_no_process_reads_its_own_output_kind(), _weave_channel_entries()

### Community 482 - "WorkflowScheduleStore"
Cohesion: 0.06
Nodes (57): derive_workflow_execution_policy(), _has_explicit_schedule_intent(), _hour_from_token(), next_run_for_recurring_schedule(), _next_weekday_run(), _normalize_prompt(), _one_shot_tonight(), _parse_notify_on() (+49 more)

### Community 483 - "grammar_truth_gate.py"
Cohesion: 0.21
Nodes (14): _classify_degraded_reason(), format_degraded_reason_groups(), format_mode_summary(), format_reducer_health_summary(), _missing_fields(), Validate /grammar/truth payloads for the production smoke gate., validate_truth_payload(), Tests for grammar production truth smoke gate helpers. (+6 more)

### Community 484 - "_skip_journal_pageindex_for_automated_trigger"
Cohesion: 0.31
Nodes (8): Skip journal PageIndex for compose paths that would self-loop on journal_entry_i, Backward-compatible name; prefer _skip_journal_pageindex_for_automated_trigger., _scheduler_daily_journal_ctx(), _skip_journal_pageindex_for_automated_trigger(), test_manual_collapse_not_skipped(), test_skips_metacog_digest(), test_skips_notify_summary(), test_skips_scheduler_daily()

### Community 485 - "_project_recent_dispatch_actions"
Cohesion: 0.28
Nodes (16): _project_recent_dispatch_actions(), Projection helper: surface the most recent autonomous dispatch-action     outcom, _outcome(), _patched(), A schema-drifted item (e.g. a future writer that doesn't set kind/summary)     m, success=None (ActionOutcomeRefV1's documented "genuinely unknown" state)     mus, test_caps_at_three_newest_first_and_exact_keys(), test_empty_ctx_still_queries_and_returns_empty() (+8 more)

### Community 486 - "OrchConceptProfileSettings"
Cohesion: 0.21
Nodes (6): build_orch_concept_profile_settings(), Config, get_orch_concept_profile_settings(), OrchConceptProfileSettings, Concept-profile repository settings used by Orch runtime.      This adapter inte, Return concept-profile repository config for Orch runtime.      Environment is t

### Community 487 - "renderScheduleInventory"
Cohesion: 0.19
Nodes (17): buildWorkflowModalSummaryCard(), fetchScheduleInventory(), filteredSchedules(), formatOverdue(), loadScheduleInventory(), normalizeSchedule(), openScheduleDetails(), openScheduleEdit() (+9 more)

### Community 488 - "social-inspection.js"
Cohesion: 0.24
Nodes (15): buildOperatorSummary(), buildSurfaceModel(), cleanList(), countStateItems(), formatCountLabel(), getSection(), normalizeSection(), normalizeSnapshot() (+7 more)

### Community 492 - "test_thought_candidate.py"
Cohesion: 0.17
Nodes (5): _FakeQuery, _FakeRow, _FakeSession, test_chat_history_thought_for_merge_preserves_existing_non_empty_thought(), test_chat_history_thought_for_merge_writes_insert_and_update_when_empty_existing()

### Community 495 - "HealthMonitor"
Cohesion: 0.29
Nodes (7): _annotate_reason(), _check(), _format_local(), HealthCheck, HealthMonitor, Edge-triggered health monitor: fires an orion-notify attention request only, run_checks()

### Community 496 - "test_stt_engine.py"
Cohesion: 0.24
Nodes (12): _load_stt_module(), _mock_canonicalize(), stt_engine(), test_canonicalize_wav_produces_16k_mono_s16(), test_measure_wav_levels_silent_vs_tone(), test_peak_threshold_invalid_env_falls_back(), test_peak_threshold_reads_env(), test_transcribe_client_override_when_server_peak_zero() (+4 more)
=======
### Community 475 - "Stance Assembly / ChatStanceBrief"
Cohesion: 0.20
Nodes (11): Proposal mode required before invasive cognition changes: memory, identity, self-modeling, autonomy, private recall, social continuity, or cognition-loop changes need a proposal naming capability change, data touched, privacy boundary, proof trace, dangerous failure mode, and rollback before implementation, baseline dispatch policy (retina_fast, every_n_frames 10, no caption/embeddings), porch_eye camera config (open-vocab detect: person/package/vehicle/animal), triggered dispatch policy (person trigger, TTL 8s, caption+embeddings on), config/vision_frame_router.yaml (baseline vs triggered dispatch), Autonomy readiness / mutation pipeline, Dreams / Dream Weaver (symbolic residue processing), Journal layer (autobiographical compression) (+3 more)
>>>>>>> Stashed changes

### Community 497 - "Concept Induction (Spark)"
Cohesion: 0.13
Nodes (16): orion:chat:history:log channel, chat.history.message.v1 kind, ChatHistoryMessageEnvelope, skills.chat.discussion_window.v1, Chat History to Vector Memory Flow, orion-hub (chat history publisher), orion-vector-host, orion-vector-writer (+8 more)

### Community 498 - "Endogenous Drive Origination Design"
<<<<<<< Updated upstream
Cohesion: 0.19
Nodes (16): Phi Inner State Truthful Design, Reconstruction-Error φ Objective Function, Signal Veracity Audit (input hygiene), Phi Intrinsic Reward + Value Learning (Step 3b), Intrinsic Reward r = Δφ, TopDownBiasCombiner / VoluntaryOverrideV1, Phi Encoder Plan 2 Design, mlp_shallow_v1 Encoder (+8 more)

### Community 499 - "rdf_sync.py"
Cohesion: 0.17
Nodes (12): main(), generate_turtle_for_all(), _load_verbs(), Render a Python value as a Turtle-safe literal.      We only use this for *strin, Yield all verb definitions as dicts from verbs/*.yaml., Return Turtle triples for a single verb and its steps., Generate Turtle for all verbs in base_dir/verbs.      base_dir is typically the, Generate Turtle for all verbs and write it to ontology/orion_cognition_generated (+4 more)

### Community 500 - "properties"
Cohesion: 0.13
Nodes (16): description, type, properties, type, items, applies_when_state, requires_gpu, requires_memory (+8 more)

### Community 502 - "is_active"
Cohesion: 0.21
Nodes (13): build_verb_list(), _discover_verbs(), is_active(), is_runtime_entry_verb(), list_all_verbs(), _load_manifest(), _normalize_and_validate_verb(), api_verbs() (+5 more)

### Community 503 - "graph_view.py"
Cohesion: 0.20
Nodes (11): Canonical layer and dimension enums for Substrate Atlas (spec §5.4–5.5)., atom_row_to_node(), build_dimension_groups(), build_dimension_summary(), build_layer_groups(), build_layer_summary(), layer_index(), layer_y_position() (+3 more)

### Community 504 - "assert_hub_context_exec_routing"
Cohesion: 0.22
Nodes (13): assert_context_exec_engine_identity(), assert_context_exec_safety_posture(), assert_hub_context_exec_routing(), _blob(), _has_context_exec_signal(), Assertions for Hub golden probes — context-exec routing must be proven, not assu, Assert context-exec runtime_debug identifies the RLM engine., Assert read-only safety flags on context-exec runtime_debug. (+5 more)

### Community 505 - "orion_fresh_main_smoke.sh"
Cohesion: 0.22
Nodes (13): banner(), classify_failure(), FAIL_LOGS, FAIL_NAMES, record_fail(), record_pass(), record_skip(), require_repo_root() (+5 more)

### Community 507 - "_should_prepare_brain_reply_context"
Cohesion: 0.43
Nodes (7): _should_prepare_brain_reply_context(), test_chat_kids_story_skips_heavy_brain_reply_context_prep(), test_chat_quick_skips_heavy_brain_reply_context_prep(), test_introspect_spark_skips_heavy_brain_reply_context_prep(), test_memory_graph_suggest_skips_heavy_brain_reply_context_prep(), test_non_runtime_brain_step_keeps_context_prep_enabled(), test_runtime_skill_skips_chat_stance_autonomy_context_prep()

### Community 509 - "_worker"
Cohesion: 0.20
Nodes (10): health(), get_settings(), Settings, test_emit_perception_once_fail_open_on_list_players_error(), test_emit_perception_once_fail_open_on_malformed_row(), test_emit_perception_once_fail_open_on_publish_error(), test_emit_perception_once_none_when_orion_absent(), test_emit_perception_once_publishes_built_perception() (+2 more)

### Community 510 - "proposal-review-ui.js"
Cohesion: 0.29
Nodes (15): apiFetch(), escapeHtml(), evidenceSummary(), init(), loadPendingDecisions(), loadProposalDetail(), renderDetail(), renderListRow() (+7 more)

### Community 511 - "service-logs-ui.js"
Cohesion: 0.27
Nodes (14): appendLog(), bindTerminalScroll(), connectSocket(), createTerminalState(), diagnosticsHint(), ensureServiceTerminal(), flushTerminal(), loadInventory() (+6 more)

### Community 512 - "substrate-effect-ui.js"
Cohesion: 0.33
Nodes (15): el(), initSubstrateEffectTab(), loadRecentEffects(), openSubstrateEffectModal(), renderBehaviorDelta(), renderCausalChain(), renderEvidenceCards(), renderMoleculeSummaries() (+7 more)

### Community 513 - "test_memory_graph_from_chat_live.py"
Cohesion: 0.22
Nodes (11): Live Memory graph from chat cases for Playwright e2e., _artifact_dir(), _entity_labels(), hub_available(), _hub_reachable(), _is_evidence_only(), _is_strict_suggest_draft(), _looks_like_prose_outside_json() (+3 more)

### Community 514 - "test_profile_forwarding.py"
Cohesion: 0.18
Nodes (11): _find_flag_value(), config/llm_profiles qwen3-8b-q4km-v100-16gb-balanced: non-thinking template kwar, Atlas metacog: Q5_K_M, n_parallel=1 -> full 16k ctx per slot, reasoning off., Synthetic profile: only chat_template_kwargs={'enable_thinking': False} → policy, test_draft_fields_emit_speculative_decoding_flags(), test_gemma4_31b_multimodal_profile_forwards_mmproj_and_image_flags(), test_qwen3_64k_profile_forwards_validated_flags(), test_qwen3_64k_profile_skips_unsupported_reasoning_flag() (+3 more)

### Community 515 - "_FakeSession"
Cohesion: 0.17
Nodes (7): _FakeQuery, _FakeSession, test_chat_history_log_scalar_columns_from_meta_llm_uncertainty(), test_chat_history_spark_meta_merges_llm_uncertainty_from_meta(), test_chat_history_spark_meta_merges_llm_uncertainty_from_spark_meta(), _write_chat_history(), _write_chat_history_row()

### Community 516 - "test_check_daily_schedule_collisions.py"
Cohesion: 0.18
Nodes (8): True positive against the actual repo config: Daily Journal reuses Daily     Pul, test_load_cadences_includes_synthetic_daily_journal_entry(), test_main_fail_on_collision_exits_one(), test_main_fail_on_collision_exits_zero_when_below_threshold(), test_main_json_output_contains_collisions(), test_main_report_only_exits_zero_even_with_collision(), test_real_env_example_has_known_daily_pulse_journal_collision(), _write_env_example()
=======
Cohesion: 0.07
Nodes (42): Autonomy Arc Measurement Gate (Step 0), Endogenous Drive Origination Design, Origination Dynamics Under Leaky Math, EndogenousOriginationTicker, Homeostatic Drives Real Tensions Design, Deviation Gate (EWMA baseline, anti-flood), Internal Economy Scarcity Allocation Design, AllocationDecisionV1 / MetabolicBudgetV1 (+34 more)
>>>>>>> Stashed changes

### Community 518 - "reflect.v1 recall profile"
Cohesion: 0.13
Nodes (15): collapse_mirror.v1 recall profile, deep.graph.v1 recall profile, dream.v1 recall profile, graphtri.v1 recall profile, journal.daily.grounded.v1 recall profile, journal.daily.metacog.grounded.v1 recall profile, journal.notify.grounded.v1 recall profile, journal.world_pulse.grounded.v1 recall profile (+7 more)

<<<<<<< Updated upstream
### Community 519 - "train_moc.py"
Cohesion: 0.38
Nodes (8): ddp_print(), ddp_setup(), main(), parse_args(), save_checkpoint(), set_seed(), split_train_eval(), unwrap_model()

### Community 520 - "test_chatgpt_qlora_pipeline.py"
Cohesion: 0.43
Nodes (12): build_sft_dataset(), DatasetBuildConfig, FoundryConfig, SubstrateSourceConfig, TrainingConfig, test_dataset_build_is_deterministic_and_preserves_lineage(), test_foundry_frontier_oracle_does_not_enter_direct_sft(), test_foundry_manifest_rollups_and_integration_to_dataset() (+4 more)

### Community 521 - "test_journal_pageindex_mvp.py"
Cohesion: 0.30
Nodes (9): _FakeBundle, _FakeItem, _FakeRecallReply, _run_with_items(), test_logs_report_impl_and_service_status(), test_service_backed_path_is_primary(), test_service_error_falls_back_to_native(), test_service_shape_stays_chat_stance_compatible() (+1 more)

### Community 522 - "memory-crystallization-ui.js"
Cohesion: 0.32
Nodes (14): activate(), apiFetch(), chatTurnCount(), escapeHtml(), loadInbox(), loadRetirementCandidates(), renderDetail(), renderEvidence() (+6 more)

### Community 523 - "test_grammar_atlas_api.py"
Cohesion: 0.30
Nodes (10): _atlas_test_app(), client(), _ensure_hub_scripts_import_path(), _mock_with_session(), Hub Grammar Atlas read API (substrate trace/graph introspection)., test_get_atom_not_found(), test_get_trace_not_found(), test_grammar_atlas_disabled() (+2 more)

### Community 524 - "test_substrate_observability_api.py"
Cohesion: 0.25
Nodes (9): _all_rows(), _fake_engine(), HTTP tests for the self-observability summary route (self-observability v2)., Engine whose execute() keys responses off the table name in the SQL., test_summary_curiosity_signals_capped_and_ranked(), test_summary_each_section_degrades_to_null(), test_summary_full_contract_shape(), test_summary_section_failure_isolated() (+1 more)

### Community 525 - "appraisal.py"
Cohesion: 0.10
Nodes (30): _coerce_features(), _coerce_matter_item(), _coerce_matter_kind(), _coerce_recommended_effect(), _coerce_string_list(), _extract_selected_raw(), _float_field(), _is_bare_frontier_matter_root() (+22 more)

### Community 526 - "test_capability_policy.py"
Cohesion: 0.33
Nodes (7): _goal(), test_capability_policy_allows_episode_journal_at_proposed(), test_capability_policy_allows_readonly_when_goal_proposed(), test_capability_policy_allows_recall_when_goal_proposed(), test_capability_policy_denies_episode_journal_when_disabled(), test_capability_policy_denies_recall_when_auto_readonly_disabled(), test_capability_policy_denies_when_auto_readonly_disabled()

### Community 527 - "test_autonomy_goal_actions.py"
Cohesion: 0.11
Nodes (28): apply_operator_goal_reasoning_promotion(), _autonomy_goal_execution_enabled(), build_goal_graph_query_client(), complete_goal(), dismiss_goal(), execute_goal_action(), fetch_goal_by_artifact_id(), plan_promoted_goal() (+20 more)

=======
>>>>>>> Stashed changes
### Community 529 - "Integrated Memory Cognition Loop Design"
Cohesion: 0.07
Nodes (38): Active-Packet Collector / retrieve_active_packet, CrystallizationDynamicsV1 (activation/decay), Consolidation Crystallization Gate Design, Deterministic Consolidation Gate, MemoryCrystallizationV1, Memory Rail Stack (canonical vs derived), Durable World-Pulse Consumption Design, RedisStreamWorkQueue Durable Stream (+30 more)

### Community 530 - "Docker readiness (§8): run docker compose builds/deploys through scripts/safe_docker_build.sh instead of calling docker compose directly; it refuses to run from the shared/primary checkout and applies the --env-file/-f pattern automatically; raw docker compose examples retained only as reference for one-off logs/ps commands"
Cohesion: 0.60
Nodes (5): Docker readiness (§8): run docker compose builds/deploys through scripts/safe_docker_build.sh instead of calling docker compose directly; it refuses to run from the shared/primary checkout and applies the --env-file/-f pattern automatically; raw docker compose examples retained only as reference for one-off logs/ps commands, Docker readiness (§8): run docker compose builds/deploys through scripts/safe_docker_build.sh instead of calling docker compose directly; it refuses to run from the shared/primary checkout and applies the --env-file/-f pattern automatically; raw docker compose examples retained only as reference for one-off logs/ps commands, docs/superpowers/pr-reports/2026-07-14-agent-git-safety-hooks-pr.md — PR report documenting the full shared-checkout-docker-revert incident story, cited from CLAUDE.md §8, Incident: a concurrent agent session ran docker compose build+up straight from the shared/primary checkout and silently reverted another session's already-verified fix — the concrete motivation for both the safe_docker_build.sh wrapper and the pre-commit shared-checkout guard, scripts/safe_docker_build.sh — required wrapper for all docker compose build/up/deploy in this repo, replacing bare docker compose calls; refuses to run from the shared/primary checkout (worktrees only) and applies the standard --env-file/-f pattern automatically
<<<<<<< Updated upstream

### Community 532 - "render_aitown_tab_blocks"
Cohesion: 0.43
Nodes (5): Server-side AI Town tab HTML fragments for Hub index render., render_aitown_tab_blocks(), _Settings, test_hub_aitown_tab_hidden_when_disabled(), test_hub_aitown_tab_rendered_when_enabled()
=======
>>>>>>> Stashed changes

### Community 533 - "orion-llamacpp-host service (profile-driven llama.cpp GGUF wrapper, Atlas topology)"
Cohesion: 0.19
Nodes (14): config/llm_profiles.yaml (LLM profile/model registry, source of truth), orion-llama-cola-host Docker Compose, Orion Llama-CoLA Host service (dual-path text + latent action indices), orion-llama-cola-host Python dependencies (torch, transformers, deepspeed), orion-llamacpp-host Atlas multi-worker Docker Compose (chat/metacog/fast/agent lanes), orion-llamacpp-host single-worker Docker Compose, orion-llamacpp-host service (profile-driven llama.cpp GGUF wrapper, Atlas topology), orion-llamacpp-host Python dependencies (thin wrapper, no ML libs) (+6 more)

### Community 534 - "config/proposals/proposal_policy.v1.yaml — Layer 7 proposal policy: limits, priority/risk thresholds, dimension weights, and named proposal_templates that turn substrate state into ProposalFrameV1 candidates"
Cohesion: 0.17
Nodes (12): config/proposals/proposal_policy.v1.yaml — Layer 7 proposal policy: limits, priority/risk thresholds, dimension weights, and named proposal_templates that turn substrate state into ProposalFrameV1 candidates, template defer_due_to_low_readiness — no policy gate, defer on self:current to preserve_stability, base_priority 0.25, base_risk 0.0, dimensions uncertainty 0.50 / reliability_pressure 0.50, template inspect_bus_channel_catalog — read_only inspect of orion/bus/channels.yaml as a system target, base_priority 0.38, dimension contract_pressure 0.60, template inspect_execution_pressure — read_only inspect of capability:orchestration, base_priority 0.40, base_risk 0.05, dimension execution_pressure 0.60, template inspect_field_topology_catalog — read_only inspect of config/field/orion_field_topology.v1.yaml, base_priority 0.33, dimension field_intensity 0.45, template inspect_node_resource_pressure — read_only inspect of node:atlas, base_priority 0.34, dimension resource_pressure 0.50, template inspect_transport_status — read_only inspect of capability:transport, base_priority 0.42, dimensions contract_pressure 0.55 / reliability_pressure 0.40, template request_policy_review_for_action — operator_review gate to prepare_for_policy_gate on policy:execution, base_priority 0.20, base_risk 0.30 (highest risk template in the file) (+4 more)

### Community 535 - "Unified Cognitive Substrate Phase 6 (Frontier Expansion / Typed Graph-Delta Generation)"
Cohesion: 0.19
Nodes (14): Materialized Graph State, SubstrateDynamicsEngine, CoherenceAssessmentV1, Unified Cognitive Substrate Phase 6 (Frontier Expansion / Typed Graph-Delta Generation), FrontierContextPackBuilder, FrontierExpansionService, FrontierGraphDeltaBundleV1, Zone-Aware Landing Posture (+6 more)

### Community 536 - "Orion Titanium Contracts"
<<<<<<< Updated upstream
Cohesion: 0.18
Nodes (15): Orion Titanium Contracts, BaseEnvelope, Contract Mismatch Report, Neural Projection (Spark), No Dict Soup Rule, Landing Pad pad.* contracts, Redis Pub/Sub Bus, spark_vector (+7 more)
=======
Cohesion: 0.17
Nodes (16): Orion Titanium Contracts, BaseEnvelope, Dream Contracts (dream.result.v1), Contract Mismatch Report, Neural Projection (Spark), No Dict Soup Rule, Redis Pub/Sub Bus, spark_vector (+8 more)
>>>>>>> Stashed changes

### Community 537 - "Orion Platform Contract"
Cohesion: 0.21
Nodes (14): Channel & Kind Registry, Platform Channel Migration Report, Channel Rename Map, Orion Platform Codex Testing, Drift Detection Checks, Minimal Test Harness (MTH), Orion Platform Contract, Channel Catalog (channels.yaml) (+6 more)

### Community 538 - "MetacogTriggerV1"
Cohesion: 0.23
Nodes (13): Metacognition Pipeline Audit, GenericPayloadV1, Legacy Verb Request Path, MetacognitionTickV1, Orion Metacognition Logging, CollapseMirrorEntryV2, MetacogTriggerV1, trace_metacog.py (+5 more)

### Community 539 - "Vision Grounded Pipeline Design"
Cohesion: 0.19
Nodes (14): Vision Grounded Pipeline Design, enforce_evidence_grounding (council), Trigger-gated selective VLM captioning, VisionEdgeActivityPayload / edge activity channel, Vision Host Pipe Edge Decouple Design, Reply-fed host triggers (host DINO authority), Vision Scene Belief Design (window habituation), believed_hard_labels belief tier (+6 more)

### Community 540 - "Orion Unified Turn (canonical spec)"
Cohesion: 0.18
Nodes (14): Answer Contract From Stance Design (ABANDONED), Imperative-First Motor Design, GrammarReceiptV1 action trace, ThoughtEventV1.imperative (efference copy), No pre-motor classification cathedral, Orion Unified Turn (canonical spec), HarnessPostTurnClosureV1 (prediction_error learning loop), SubstrateFinalizeAppraisalV1 (5a interoception) (+6 more)

### Community 541 - "Felt-State Arc Roadmap Spec"
Cohesion: 0.22
Nodes (14): Felt-State Arc Roadmap Spec, field_channel_corpus.v1 (raw channel pressures), HDBSCAN Attractor/Cluster Discovery, Inner State Registry (REHEARSAL/COMPOSED), mood_arc_corpus.v1 Collector, Windowed Mood-Arc Sequence Autoencoder, Phi Encoder Train/Promote Lifecycle, Two-Tier Shuffle/AR(1) Surrogate Gate (+6 more)

<<<<<<< Updated upstream
### Community 542 - "VerbRegistry"
Cohesion: 0.30
Nodes (5): Loads and caches VerbConfig objects from verbs/*.yaml, Return all registered verbs. Use reload=True to refresh the cache., Lightweight filtering helper used by supervisors/planners., VerbRegistry, VerbConfig

### Community 543 - "test_query.py"
Cohesion: 0.24
Nodes (13): get_trace_graph(), _atom_row(), _chain_query(), _edge_row(), _hop_row(), Regression: BFS must track visited atoms, not hop ids (diamond-safe)., Wire session.query(Model).filter(...).<terminal> for trace graph tests., test_get_temporal_path_follows_atoms_not_hop_ids() (+5 more)

### Community 545 - "ChatResponseFeedbackV1"
Cohesion: 0.15
Nodes (3): _FakeSession, test_handle_envelope_routes_feedback_payload_to_feedback_table(), test_write_row_persists_feedback_id_and_linkage_fields()

### Community 546 - "test_drive_pressure_probe.py"
Cohesion: 0.26
Nodes (13): _probe_records(), Unit tests for the Phase 4 drive-pressure measurement probe (2026-07-12).  Measu, Call site 1 (~line 717, _handle_signal_drive_tick via handle_envelope)., Call site 2 (~line 847, handle_envelope's non-homeostatic drive-update rail)., A probe logging failure at call site 2 must not break save_drive_state or     pr, _signal_env(), test_handle_envelope_call_site_logs_probe(), test_handle_envelope_call_site_survives_probe_failure() (+5 more)

### Community 547 - "_clean_raw_llm_content"
Cohesion: 0.24
Nodes (3): _clean_raw_llm_content(), Strips common LLM preambles and code fences to expose raw JSON., TestExecutorCleaning

### Community 548 - "test_self_study_graphdb.py"
Cohesion: 0.22
Nodes (8): _bindings(), _FakeResponse, _graphdb_post(), test_graphdb_conceptual_retrieval_returns_persisted_concept_profiles(), test_graphdb_factual_retrieval_returns_authoritative_only(), test_graphdb_reflective_retrieval_preserves_all_tiers_and_links(), test_graphdb_retrieval_is_stable_across_repeated_reads(), test_graphdb_retrieval_never_upcasts_in_factual_mode()

### Community 551 - "mind_routes.py"
Cohesion: 0.40
Nodes (10): get_mind_run(), list_mind_runs(), list_recent_mind_runs(), _mind_run_row_dict(), _need_session(), _pool(), _raise_mind_store_http(), Read-only Mind run introspection (same Postgres pool as Hub memory cards). (+2 more)

### Community 553 - "test_hub_grammar_emit.py"
Cohesion: 0.26
Nodes (13): _build(), Tests for hub chat turn grammar event emitter (TDD — written before implementati, test_atom_types_are_valid(), test_build_returns_grammar_event_v1_instances(), test_relation_types_are_valid(), test_repair_signal_absent_when_flag_false(), test_repair_signal_present_when_flag_true(), test_stable_event_ids_same_inputs_same_ids() (+5 more)

### Community 554 - "test_mind_routes.py"
Cohesion: 0.27
Nodes (11): _mock_pool(), Hub Mind run read APIs (session-gated; mock PG pool)., test_get_mind_run_allows_context_session_id(), test_get_mind_run_not_found_when_other_session(), test_get_mind_run_returns_row(), test_list_mind_runs_allows_context_session_without_current_match(), test_list_mind_runs_excludes_other_non_null_session_without_context(), test_list_mind_runs_returns_null_session_fallback_rows() (+3 more)

### Community 555 - "profiles.py"
Cohesion: 0.33
Nodes (5): GPUConfig, LlamaCppConfig, LLMProfile, LLMProfileRegistry, Settings

### Community 556 - "AlertPayload"
Cohesion: 0.07
Nodes (33): AlertPayload, AlertSnapshot, Detection, Detection coming from the vision edge service.      - kind: "face", "motion", "y, Raw event from vision edge, as seen on orion:vision:edge:raw., Simple persistent security state., Summary of a logical 'visit' (a contiguous episode of humans present).     v1 is, A single snapshot image captured around the time of an alert. (+25 more)

=======
>>>>>>> Stashed changes
### Community 557 - "Orion Signals Roster v1 (mesh service tiers)"
Cohesion: 0.14
Nodes (14): orion-biometrics (tier1, organ: biometrics), orion-bus (core, required), orion-collapse-mirror (tier1, organ: collapse_mirror), orion-cortex-exec (tier1, organs: cortex_exec, autonomy, chat_stance), orion-cortex-gateway (routing), orion-cortex-orch (routing), orion-equilibrium-service (tier1, organ: equilibrium), orion-llm-gateway (routing) (+6 more)

<<<<<<< Updated upstream
### Community 558 - "Phase 4 Cluster-Weighting Research"
Cohesion: 0.40
Nodes (6): Field-topology Edge Weights, Phase 4 Cluster-Weighting Research, BIOMETRICS_ROLE_WEIGHTS_JSON, CLUSTER_ROLE_WEIGHTS, metacog_biometrics_cue, orion-state-service Aggregation

### Community 559 - "test_worker_prediction_error_node.py"
Cohesion: 0.22
Nodes (9): _FakeStore, _make_worker(), _RaisingStore, Unit tests for the Rung 1 bridge: worker writes prediction_error onto a durable, Records upsert_node calls without touching a real graph store., test_write_prediction_error_node_fail_open_on_raise(), test_write_prediction_error_node_noop_when_flag_off(), test_write_prediction_error_node_upserts_when_flag_on() (+1 more)

### Community 560 - "test_cortex_gateway_error_reply.py"
Cohesion: 0.23
Nodes (7): _Bus, _Codec, _Decoded, _Env, test_gateway_disabled_recall_forwards_supervised_request_and_replies_to_hub(), test_gateway_replies_when_orch_rpc_raises(), test_gateway_replies_when_result_validation_fails()

### Community 561 - "_ev"
Cohesion: 0.23
Nodes (3): _ev(), TestBuildFactSheet, TestReduceDriveHistory

=======
>>>>>>> Stashed changes
### Community 562 - "services/orion-memory-consolidation/README.md — subscribes to orion:memory:turn:persisted, classifies each chat turn via LLM gateway quick-lane logprobs, patches chat_history_log.spark_meta, tracks consolidation windows, and on boundary closure runs a deterministic consolidation gate (default) or legacy graph suggest"
Cohesion: 0.17
Nodes (13): Deterministic gates over repeated yelling: if Juniper has to repeat a rule twice, turn it into a script/test/check/hook/make target instead of a louder prompt, scripts/concept_relation_digest.py — standalone script (not a live service loop) run every 30 minutes via Athena cron, turning memory_concept_relation_decisions rows into reflection crystallizations; make check-concept-relation-digest-liveness is the fail-safe, querying the real undigested backlog age (not a heartbeat file) and exiting non-zero past MAX_AGE_HOURS (default 3h) — a deterministic gate against the cron entry silently dying or not persisting across a host migration, Cross-window concept-relation resolution (off by default): vector-similarity candidate retrieval plus one bounded structured-output LLM call judging same/refines/contradicts/unrelated against nearest existing active crystallizations of the same kind; refines/contradicts only attach a typed link to the new candidate, never mutate the existing target — that stays a human decision via the links/supersede API; gated by CONCEPT_RELATION_RESOLUTION_ENABLED plus embed/chroma host config, both required or the feature does nothing, MEMORY_CONSOLIDATION_OUTPUT modes: crystallization_propose (default, runs consolidation_memory_gate and skips low-signal windows or proposes MemoryCrystallizationV1), graph_draft (legacy LLM graph-suggest path), skip_only (gate runs for traceability, always marks skipped); gate thresholds require the window not be all low-info-social, since a bare novelty/significance float alone previously let a noisy classifier score crystallize a short greeting-only turn, scripts/drive_history_reflection_synthesis.py (manual/on-demand only, NOT cron'd) — reads real persisted DriveAuditV1 ticks from the Fuseki drives graph and synthesizes one reflection-kind MemoryCrystallizationV1 via a deterministic reducer stage (reduce_drive_history, a pure unit-tested function, same bar as orion/spark/concept_induction/drive_tension.py) followed by a narrow LLM-phrasing stage that only ever sees a pre-computed fact sheet; parse_and_validate_narrative enforces that every cited fact's literal tokens appear verbatim in the output, and the script refuses to synthesize below MIN_EVENTS=5 or MIN_DISTINCT_DAYS=2, Known recurring footgun: 'flag on, dependency not wired' — CONCEPT_RELATION_RESOLUTION_ENABLED=true alone does not imply the digest is scheduled, and this exact pattern has already caused silent no-ops twice in this repo (CONCEPT_RELATION_RESOLUTION_ENABLED itself missing its embed/chroma hosts on first activation, and RECALL_GRAPHITI_IN_CHAT missing its adapter URL) — checking both halves explicitly every time is cheaper than re-discovering this, services/orion-memory-consolidation/README.md — subscribes to orion:memory:turn:persisted, classifies each chat turn via LLM gateway quick-lane logprobs, patches chat_history_log.spark_meta, tracks consolidation windows, and on boundary closure runs a deterministic consolidation gate (default) or legacy graph suggest, Turn change appraisal: each persisted turn after the first in a window gets a logprob-calibrated turn_change_appraisal patch (novelty score, shift kind, confidence, baseline mode); high-confidence novel turns emit OrionSignalV1 on orion:signals:memory_consolidation (+5 more)

### Community 563 - "services/orion-execution-dispatch-runtime/README.md — Layer 9 of the Orion cognition substrate: converts PolicyDecisionFrameV1 + ProposalFrameV1 + SelfStateV1 into ExecutionDispatchFrameV1 envelopes, the motor-nerve service that can actually send real actions"
Cohesion: 0.25
Nodes (9): No empty-shell cognition: do not ship cognition-shaped output with no substance — empty semantic projections, placeholder memory cards, fallback text as generated cognition, raw_len=0 treated as success, stale reducer cursors are all invalid success states requiring inspectable evidence, Experience loop (P2): every real send (success, empty observation, or RPC failure) publishes an ActionOutcomeEmitV1 event onto orion:autonomy:action:outcome (BUS_ACTION_OUTCOME_OUT) — the same always-on route orion-spark-concept-induction already produces onto for curiosity-fetch outcomes, consumed by orion-sql-writer into the durable action_outcomes table; subject always 'orion' (self-directed, never relationship-scoped); replay-safe since sql-writer upserts by dispatch_id via merge(), EXECUTION_DISPATCH_MODE gate — default dry_run (build, no send); real sends require both this env set to dispatch_read_only AND config/execution_dispatch/execution_dispatch_policy.v1.yaml's mode.allow_dispatch_read_only true, plus per-tick and daily send budgets enforced before any send happens, ExecutionDispatchFrameV1 / ExecutionDispatchCandidateV1 — the schema this service builds; dispatch_status vocabulary (prepared, dry_run, blocked, skipped, prepared_for_dispatch, dispatched) enforced at the schema level, with dispatched requiring dispatched_at plus a result_ref or dispatch_error, Idempotency: dispatch_id is deterministic per proposal+policy, so if the process dies between a successful send and frame persistence, the next tick replays the stored substrate_dispatch_results row instead of resending — a real cortex-exec call never fires twice for the same candidate, Rollback: set EXECUTION_DISPATCH_MODE=dry_run and restart this one container — single kill switch for all real sending, services/orion-execution-dispatch-runtime/README.md — Layer 9 of the Orion cognition substrate: converts PolicyDecisionFrameV1 + ProposalFrameV1 + SelfStateV1 into ExecutionDispatchFrameV1 envelopes, the motor-nerve service that can actually send real actions, Theater tripwire: if more than half of the trailing 10 real dispatch results have status=empty (a real send that produced no usable observation), the worker stops sending for the rest of its process lifetime, visible via GET /latest's theater_tripwire_active field; deliberately does not self-clear (a self-clearing tripwire could resume sending on a coincidentally non-empty sample without anyone deciding that was safe) — requires a restart to re-arm (+1 more)

### Community 564 - "config/llm_profiles.yaml (LLM profile registry)"
Cohesion: 0.38
Nodes (7): Gemma 4 31B/E4B multimodal (text+image[+audio]) profiles, llama3-1-cola CoLA intention host profile, config/llm_profiles.yaml (LLM profile registry), qwen36-35b-a3b-udq5km-2xv100-32gb-deep-cognition profile (monster FCC), qwen3-30b-a3b-instruct-2507-q5km-atlas-metacog-16k profile, qwen3-8b-q5km-v100-16gb-atlas-metacog-16k profile, qwen3-coder-next agent breadth/depth profiles

### Community 565 - "MemoryCardV1"
Cohesion: 0.19
Nodes (13): Recall Profiles, Memory Cards v1 Offboarding Guide, Orion Memory Cards v1 Design Spec, Always-On Known Facts Injection, Auto-Extractor (Stage 1, off by default), Cards Recall Backend, Recall Intent Router, MemoryCardV1 (+5 more)

### Community 566 - "Orion Heartbeat Research Charter"
Cohesion: 0.17
Nodes (13): Orion Heartbeat Research Charter, Active Inference / Free-Energy Principle, Baars 1988 - Global Workspace Theory, Friston 2010 - Free-Energy Principle, Gray 2018 - quimb library, Stoudenmire & Schwab 2016 - Tensor Networks for ML, Tononi 2004 - Integrated Information Theory, Functional Minimal Selfhood (+5 more)

### Community 567 - "SuggestDraftV1"
Cohesion: 0.15
Nodes (13): Memory Graph from Chat - Soul-Purpose Review, Generous Extraction, Selective Persistence, Role-Grounded Extraction, SuggestDraftV1, llama.cpp Structured Output Probe, orion-llamacpp-host, orion-llm-gateway, response_format schema enforcement (+5 more)

### Community 568 - "Runtime Trace Signal Nexus Design"
Cohesion: 0.18
Nodes (13): Runtime Introspection Interrogation Design, Anti-bullshit claim-to-evidence guardrail, Strict evidence contract (hop topology), interrogate_runtime_state verb, Chat Stance Signal Adapter Contract, ChatStanceBrief (no-PII signal mapping), OrionSignalV1, Runtime Trace Signal Nexus Design (+5 more)

<<<<<<< Updated upstream
### Community 569 - "findings_bundle_synth.py"
Cohesion: 0.24
Nodes (10): attach_findings_to_debug(), merge_findings_bundle_dicts(), Synthesize a serializable FindingsBundle from planner trace + contract (Phase 2, Merge two FindingsBundle-shaped dicts (exec supervisor aggregation)., synthesize_findings_bundle(), _trace_has_repo_evidence(), _trace_has_runtime_evidence(), test_merge_findings_merges_lists() (+2 more)

### Community 570 - "_FakeProc"
Cohesion: 0.23
Nodes (5): _FakeProc, _FakeStream, test_cancel_before_register_kills_on_spawn(), test_cancel_fcc_turn_kills_registered_process(), test_run_fcc_turn_registers_and_unregisters_process()

=======
>>>>>>> Stashed changes
### Community 571 - "orion-signal-gateway (normalizes organ-bus events into OrionSignalV1)"
Cohesion: 0.28
Nodes (9): orion-signal-gateway (normalizes organ-bus events into OrionSignalV1), orion-signals (organ signal mesh launcher, not a runtime service), orion-signal-gateway docker-compose.yml, orion-signal-gateway otel/collector-config.yaml (OTLP collector), orion-signal-gateway otel/grafana-datasources.yaml (Tempo datasource provisioning), orion-signal-gateway otel/tempo.yaml (Grafana Tempo single-binary config), orion-signal-gateway README.md, orion-signal-gateway requirements.txt (+1 more)

<<<<<<< Updated upstream
### Community 572 - "test_attention_ack.py"
Cohesion: 0.40
Nodes (5): _persisted_state(), The fix routes acks through sql-writer's direct write endpoint     (proxy_post), test_attention_ack_calls_sql_writer_proxy_not_bus(), test_attention_ack_propagates_not_found_from_sql_writer(), test_attention_ack_rejects_mismatched_attention_id()

### Community 573 - "proposal_review_routes.py"
Cohesion: 0.31
Nodes (7): _disabled_payload(), proposal_review_action(), proposal_review_health(), proposal_review_pending(), ProposalReviewActionRequest, Hub routes proxying the context-exec proposal review API., _unavailable_payload()

### Community 574 - "generate_descriptions.py"
Cohesion: 0.32
Nodes (12): _collapse(), compose_identity(), compose_presence_blurb(), main(), patch_juniper(), Splice Juniper (human player) into constants.ts (DEFAULT_NAME) and     world.ts, Rich, third-person identity injected into the agent's prompts., Rich blurb for players seeded outside `Descriptions` (Juniper human join,     Or (+4 more)

### Community 575 - "aitown-panel.js"
Cohesion: 0.36
Nodes (12): activate(), clearReadyTimer(), deactivate(), el(), finishLoading(), loadIframe(), readFrameState(), refreshStatus() (+4 more)

### Community 578 - "test_hub_ui_polish.py"
Cohesion: 0.20
Nodes (3): _render_hub_index(), test_render_hub_index_html_injects_proposal_review_when_enabled(), test_render_hub_index_html_omits_proposal_review_when_disabled()

### Community 581 - "main.py"
Cohesion: 0.24
Nodes (7): main(), pull_model(), Wait for Ollama server to be ready., Pull the specified model using Ollama API., wait_for_ollama(), Config, Settings

### Community 586 - "test_roster.py"
Cohesion: 0.20
Nodes (5): _compose_service_names(), _load_roster(), Gate tests for orion-signals roster contract., roster(), test_every_compose_service_in_compose_file()

### Community 587 - "FakeWorker"
Cohesion: 0.19
Nodes (4): FakeWorker, test_debug_endpoint_returns_structured_json_when_worker_present(), test_lifespan_attaches_worker_and_starts_task(), test_shutdown_stops_worker_once()

### Community 588 - "order"
Cohesion: 0.50
Nodes (4): description, minimum, type, order

### Community 589 - "main.py"
Cohesion: 0.22
Nodes (7): build_vllm_command_and_env(), main(), Build the vLLM OpenAI server command + environment based on Settings + profiles., Config, Resolve (model_id, gpu_cfg) using:           1) Explicit env: VLLM_MODEL_ID (opt, Load raw profile dicts from YAML (same file used by llm-gateway)., Settings

=======
### Community 585 - "orion-spark-introspector: Spark metacognitive streaming service driving the phi/EKG chart"
Cohesion: 0.15
Nodes (13): orion-spark-introspector (tier1, organ: spark_introspector), Orion Cognitive Dashboard UI (index.html + tissue_viz.js), orion-spark-introspector docker-compose.yml, orion-spark-introspector README (Spark Engine v2), CoLA-derived novelty signal (orion-llama-cola-host /v1/understand, per-session distribution novelty), Mood-arc corpus collector (roadmap item 1: MoodArcCorpusRowV1 JSONL sink), scripts/fit_mood_arc_encoder.py (roadmap item 2: windowed sequence-autoencoder over mood-arc corpus), OrionState (compact self-state representation: mood, focus_themes, latent_state) (+5 more)

>>>>>>> Stashed changes
### Community 590 - "Pipeline: Retina Dense"
Cohesion: 0.24
Nodes (12): Capability: vision (declared, unwired), Pipeline: Retina Dense, Pipeline: Retina Fast, Profile: Image Embedding, Profile: Identity/Face Embedding, Profile: Pose Estimation, Profile: Open-Vocab Detector, Profile: Segmentation (SAM2-class) (+4 more)

### Community 591 - "World Pulse Sources Policy v1"
Cohesion: 0.17
Nodes (12): Source: BBC World RSS, Source: CISA KEV Catalog (Manual URL), Source: Hugging Face Blog RSS, Source: Utah Government News (disabled, 404), Source: NASA News RSS, Source: NOAA Newsroom, Source: NPR Politics RSS, World Pulse Sources Policy v1 (+4 more)

### Community 592 - "LLM Services and Agentic Flow"
Cohesion: 0.26
Nodes (12): LLM Services and Agentic Flow, AgentChainService, Cortex-Exec, Cortex-Orch, CouncilService (stub), Dual-Lobe Host (llamacpp), LLM Gateway (Reflective), PlannerReactService (+4 more)

<<<<<<< Updated upstream
### Community 593 - "up_all_services_batched.sh"
Cohesion: 0.27
Nodes (7): collect_up_failures(), is_excluded(), print_failed_logs(), up_all_services_batched.sh script, up_one(), up_one_bg(), wait_for_bus_ready()

### Community 597 - "VectorStore"
Cohesion: 0.17
Nodes (7): ingest_file(), Reads a text file, splits it into chunks, and ingests them into the vector store, A client that connects to the standalone Orion Vector DB service.     This write, Loads the embedding model and initializes the ChromaDB HTTP client         to co, Embeds a list of document texts and adds them to the collection., # NOTE: This is a simple incremental ID. For production, you'd, VectorStore

### Community 598 - "check_service_env_compose_parity.py"
Cohesion: 0.33
Nodes (8): main(), Returns (keys exposed via `environment:`, has env_file directive) for the     se, Raised when the target service can't be unambiguously identified inside a     mu, Pick the compose `services:` entry that corresponds to `service_dirname`     (e., _read_compose_env_keys(), _read_env_example_keys(), _select_service_block(), ServiceSelectionError

### Community 599 - "plan"
Cohesion: 0.50
Nodes (4): description, minItems, type, plan

### Community 600 - "trace_hub_skill_runner_e2e.py"
Cohesion: 0.36
Nodes (9): _ensure_sys_path_stdlib_safe(), _http_to_ws_url(), main(), phase0_offline(), phase1_http(), phase2_ws(), When launched as ``python scripts/this.py``, sys.path[0] is ``.../scripts`` and, _repo_root() (+1 more)

### Community 601 - "test_generate_descriptions.py"
Cohesion: 0.24
Nodes (9): _cards(), Backtick / ${...} / backslash must be escaped for a TS backtick literal., Drift guard: the composed Juniper blurb must be spliced into world.ts., test_cards_have_all_expected_ids(), test_compose_identity_is_rich_and_collapsed(), test_compose_presence_blurb_orion_uses_they(), test_juniper_blurb_present_in_world_ts(), test_render_descriptions_emits_eight_valid_sprites() (+1 more)

### Community 602 - "test_worker_social.py"
Cohesion: 0.47
Nodes (10): _perc(), test_engage_accepts_invite(), test_engage_does_not_stop_when_not_pathfinding(), test_engage_initiates_with_nearby_player(), test_engage_stops_once_to_face_partner_when_pathfinding(), test_engage_walks_to_partner_when_walking_over(), test_initiate_off_when_distance_zero(), test_initiate_respects_cooldown() (+2 more)

### Community 603 - "Settings"
Cohesion: 0.19
Nodes (8): get_settings(), Configuration for the Orion Hub service, loaded from environment variables., Settings, substrate_atlas_page(), test_situation_enabled_defaults_true(), test_situation_enabled_false_values(), test_situation_enabled_true_values(), test_hub_voice_timeout_settings_defaults()

### Community 604 - "agent-claude-trace.js"
Cohesion: 0.38
Nodes (11): appendLiveClaudeStep(), basename(), clip(), contextRiskSuffix(), ensurePanel(), finalizeLiveClaudeTrace(), formatHarnessHeading(), formatToolInput() (+3 more)

### Community 607 - "test_mind_provenance_normalizer.py"
Cohesion: 0.30
Nodes (11): Historical runs may omit mind.llm_synthesis_attempted on the success path., _run_node(), test_fail_open_fixture_normalizer(), test_fail_open_fixture_phase_rows_expose_semantic_filter_counts(), test_fail_open_fixture_still_renders_provenance_sections(), test_filtered_empty_semantic_only_warns_not_failed_callout(), test_orch_http_failed_fixture_normalizer(), test_shadow_fixture_renders_shadow_fields() (+3 more)

### Community 608 - "test_turn_stop_command.py"
Cohesion: 0.18
Nodes (5): General 'stop chat' command: a per-connection active-turn registry in websocket_, A connection is registered at setup even before any turn starts (correlation_id, The registry stores the active_turn dict by reference: mutating it directly, test_cancel_active_turn_for_connection_with_no_turn_in_flight_is_noop(), test_cancel_active_turn_reflects_live_mutation_no_manual_sync()

### Community 609 - "timeout_ms"
Cohesion: 0.50
Nodes (4): timeout_ms, description, minimum, type

### Community 610 - "_body"
Cohesion: 0.38
Nodes (9): _maybe_publish_spark_introspect(), Chat-turn candidates feed spark-introspector; internal RPCs (introspect_spark,, _run_async(), _should_publish_spark_candidate(), _body(), test_maybe_publish_allows_hub_like_chat(), test_maybe_publish_skips_introspect_spark(), test_should_not_publish_internal_rpc_completions() (+1 more)

### Community 612 - "test_action_outcome_sql_shape.py"
Cohesion: 0.29
Nodes (8): Idempotent spark_telemetry persistence keyed by correlation_id., upsert_spark_telemetry(), Tests for idempotent spark_telemetry persistence., Regression: ON CONFLICT SET must target DB column 'metadata', not Python attr 'm, test_on_conflict_update_uses_metadata_column_not_python_attr(), test_upsert_filters_unknown_keys(), test_upsert_with_correlation_id_uses_on_conflict(), test_upsert_without_correlation_id_uses_plain_insert()

### Community 615 - "test_worker_attention_broadcast_tick.py"
Cohesion: 0.35
Nodes (10): _graph_node(), _make_worker(), Unit tests for the continuous attention broadcast tick (rung 3).  Verifies the w, Each broadcast tick appends a dwell row alongside the projection., test_broadcast_disabled_is_noop(), test_broadcast_fails_open_on_snapshot_error(), test_broadcast_fails_open_on_store_init_error(), test_broadcast_persists_winning_coalition() (+2 more)

### Community 616 - "claim:orion:substrate-telemetry:0001 — orion-substrate-telemetry persists tier outcomes"
Cohesion: 1.00
Nodes (3): claim:orion:substrate-telemetry:0001 — orion-substrate-telemetry persists tier outcomes, claim:orion:substrate-telemetry:0002 — orion-cortex-orch optionally merges telemetry facet, source:2026-05-14-substrate-tier-telemetry-design-ref (metadata)

=======
>>>>>>> Stashed changes
### Community 617 - "Reasoning Schema Phase 1"
Cohesion: 0.09
Nodes (30): Reasoning Promotion Phase 3, Contradiction-Aware Gating, HITL Escalation Policy, Dynamic Entity/Domain Lifecycle Governance, Deterministic Transition Evaluator, PromotionEvaluationResultV1, Explicit Promotion Transition Matrix, Reasoning Schema Phase 1 (+22 more)

### Community 618 - "Unified Cognitive Substrate Phase 1 (Shared Ontology + Canonical Contracts)"
Cohesion: 0.25
Nodes (11): Unified Cognitive Substrate Phase 1 (Shared Ontology + Canonical Contracts), Canonical Substrate Ontology, ConceptNodeV1, DriveNodeV1, StateSnapshotNodeV1, SubstrateEdgeV1, Unified Cognitive Substrate Phase 2 (Domain Adapter Mappings), Domain-to-Substrate Adapter Layer (+3 more)

### Community 619 - "Unified Cognitive Substrate Phase 11 (Narrow Runtime Review Execution)"
Cohesion: 0.22
Nodes (11): Unified Cognitive Substrate Phase 10 (Review Scheduling and Revisit Cadence), GraphReviewCyclePolicyV1, GraphReviewQueue, Unified Cognitive Substrate Phase 11 (Narrow Runtime Review Execution), GraphReviewRuntimeResultV1, Single-Cycle Bounded Execution, Unified Cognitive Substrate Phase 12 (Review Telemetry and Calibration), Advisory-Only Calibration (No Auto Policy Mutation) (+3 more)

### Community 620 - "FCC-Cortex GWT Dispatch Design"
Cohesion: 0.22
Nodes (11): Hub Agent Claude Design (FCC harness in chat), fcc_claude_bridge (subprocess motor), stream-json harness event relay, FCC-Cortex GWT Dispatch Design, Redis bus-native RPC (no HTTP Hub-to-organ), Global Workspace Theory (GWT) mapping, orion-harness-governor (orion/harness library), Trust rupture refusal threshold + boundary register (+3 more)

### Community 622 - "execution_trajectory reducer / ExecutionRunStateV1"
Cohesion: 0.20
Nodes (11): Phi Cognitive Motor Unification Design, execution_trajectory reducer / ExecutionRunStateV1, harness grammar_emit.py collector, Motor-agnostic execution grammar, seed-v3 encoder input contract, execution_trajectory projection cap/prune, seed-v4 trainable feature set, Orch Route-Grammar Lane Design (+3 more)

### Community 623 - "source:2026-05-20-knowledge-forge-v1-merge (metadata)"
Cohesion: 0.29
Nodes (10): Orion Memory Ontology (GraphDB), orionmem-v2026-05.ttl, shapes-orionmem-v2026-05.ttl, claim:orion:knowledge-forge:0001 — Forge v1 FastAPI endpoints on port 8630, claim:orion:knowledge-forge:0003 — git-tracked orion-knowledge/ remains source of truth, claim:orion:knowledge-forge:0004 — compile excludes disputed/stale/superseded claims by default, claim:orion:knowledge-forge:0002 — Hub Forge tab proxies /api/knowledge/*, claim:orion:knowledge-forge:0005 — v1 excludes GraphDB/vector search/autonomous rewriting/silent mutation (+2 more)

### Community 624 - "Service: orion-notify"
Cohesion: 0.18
Nodes (11): Channel "orion:notify:config:preference" (kind=event, schema=NotificationPreferencesUpdate) producers=[orion-notify] consumers=[orion-sql-writer], Channel "orion:notify:config:recipient" (kind=event, schema=RecipientProfileUpdate) producers=[orion-notify] consumers=[orion-sql-writer], Channel "orion:notify:in_app" (kind=event, schema=HubNotificationEvent) producers=[orion-notify] consumers=[orion-hub], Channel "orion:notify:persistence:receipt" (kind=event, schema=NotificationReceiptEvent) producers=[orion-notify] consumers=[orion-sql-writer], Channel "orion:notify:persistence:request" (kind=event, schema=NotificationRecord) producers=[orion-notify] consumers=[orion-sql-writer], Schema: HubNotificationEvent, Schema: NotificationPreferencesUpdate, Schema: NotificationReceiptEvent (+3 more)

### Community 625 - "Cognition Packs (Memory, Executive, Emergent)"
Cohesion: 0.20
Nodes (11): delivery_pack: user-facing artifact verbs, emergent_pack: introspection/reflection/dream/counterfactual verbs, executive_pack: classify/plan/prioritize/evaluate verbs, memory_pack: memory retrieval/contextualization/narrative verbs, Cognition Packs (Memory, Executive, Emergent), Orion Cognition Layer README, Semantic Planner (verb -> ExecutionPlan), Verb exec_step bus contract (orion-exec:request/result:<Service>) (+3 more)

<<<<<<< Updated upstream
### Community 626 - "walkable_tiles"
Cohesion: 0.24
Nodes (9): _map(), objectTiles indexed [layer][x][y]; -1 = empty. `blocked` is a set of (x,y)., test_walkable_empty_on_malformed(), test_walkable_excludes_blocked_tiles(), test_walkable_multiple_layers_any_block(), Walkability derived from an AI Town ``worldMap``.  AI Town stores object/collisi, Return the set of walkable integer tiles ``(x, y)`` for a worldMap.      Fail-op, walkable_tiles() (+1 more)

### Community 627 - "test_fcc_motor_summarize.py"
Cohesion: 0.33
Nodes (9): Regression: harness step summaries must carry tool_use names and tool_result bod, Replay the failing turn's stream: tool evidence must survive into summaries., _step(), test_pr_title_stream_receipts_carry_tool_evidence(), test_summarize_assistant_text_preserved(), test_summarize_system_subtype(), test_summarize_tool_result_carries_body(), test_summarize_tool_result_list_content_and_error_flag() (+1 more)

### Community 628 - "recommend_actions_from_alerts"
Cohesion: 0.44
Nodes (7): recommend_actions_from_alerts(), summarize_recommended_actions(), main(), test_policy_coherence_drop_error_actions(), test_policy_dedup_and_ordering(), test_policy_novelty_spike_warn_actions(), test_policy_summary()

### Community 629 - "test_signal_drive_consumer.py"
Cohesion: 0.33
Nodes (10): Task 6 live: homeostatic signal channels ride a drive-only rail.  Asserts the wo, A bad prior state / store fault in the drive-update section must degrade to, _signal_env(), test_failure_channel_mints_tension(), test_homeostatic_source_classification(), test_never_raises_when_drive_update_breaks(), test_pubsub_patterns_include_specific_channels_not_wildcard(), test_real_biometric_drop_updates_drives_no_induction() (+2 more)

### Community 630 - "model.py"
Cohesion: 0.14
Nodes (18): _artanh_clamped(), expmap0(), mobius_add(), poincare_distance(), poincare_distance_pairs(), project_to_ball(), Exponential map at origin: tangent vector -> point on ball., d_c(x, y) = 2/sqrt(c) * artanh( sqrt(c) * || (-x) ⊕_c y || ) (+10 more)

### Community 631 - "refit_salience_weights.py"
Cohesion: 0.31
Nodes (8): test_refit_consumes_label_rows_and_returns_weights(), test_refit_handles_empty_labels(), candidate_weights_from_labels(), load_labels(), main(), Salience weight refit — DOCUMENTED STUB. Not run in production this round.  The, Read outcome label rows. Best-effort; [] if no DB configured., STUB: prove the label table is consumable; return seeded weights.      A real fi

### Community 635 - "compact_vision_scene_interpretation_json_schema"
Cohesion: 0.23
Nodes (10): compact_vision_scene_interpretation_json_schema(), Compact JSON Schema for VisionSceneInterpretationV1 (llama.cpp json_object+schem, Inline schema without $ref — suitable for Atlas metacog json_object+schema., build_interpretation_llm_options(), Gateway options for deterministic VisionSceneInterpretationV1 JSON., test_build_interpretation_llm_options_wires_structured_schema(), Tests for VisionSceneInterpretationV1 compact JSON schema contract., test_compact_schema_has_required_top_level_keys() (+2 more)

### Community 636 - "agent-trace.js"
Cohesion: 0.35
Nodes (9): appendLiveAgentStep(), buildTimelineRows(), ensureLiveTracePanel(), extractAgentTrace(), formatDuration(), normalizeSummary(), resolveLiveTraceAnchor(), shouldShowAgentTrace() (+1 more)

### Community 637 - "organ-signals-graph-ui.js"
Cohesion: 0.33
Nodes (9): attach(), buildCorrelationGraphElements(), buildGraphElements(), destroyCy(), filterSignalsByLayer(), isPlaceholderDimensions(), isStubSignal(), organClassColor() (+1 more)

### Community 638 - "_Store"
Cohesion: 0.29
Nodes (6): _Store, test_presence_invalid_payload_is_422(), test_presence_roundtrip(), test_situation_brief_reflects_manual_presence(), test_situation_status_and_brief(), test_situation_status_and_brief_disabled()

### Community 641 - "context.py"
Cohesion: 0.12
Nodes (16): VisionEdgeArtifact, VisionGuardAlert, VisionGuardSignal, bus_worker(), _handle_envelope(), Subscribe to vision artifacts., _source(), AppContext (+8 more)

### Community 643 - "_reload_settings"
Cohesion: 0.31
Nodes (9): Regression for fix/mind-enrichment-wall-budget: the shipped default wall must, _reload_settings(), test_config_warnings_silent_when_disabled(), test_config_warns_on_http_timeout_not_above_wall(), test_config_warns_on_sub_viable_wall(), test_default_wall_is_viable_for_three_phase_synthesis(), test_mind_enrichment_defaults_off(), test_mind_enrichment_reads_env() (+1 more)

### Community 644 - "readiness_payload"
Cohesion: 0.31
Nodes (6): lifespan(), ready(), check_embedding(), check_model_dir(), check_postgres(), readiness_payload()

### Community 646 - "models.py"
Cohesion: 0.22
Nodes (5): CollapseTriageEvent, Config, RAGDocumentEvent, Defines the schema for explicitly adding a document to the RAG store,     coming, Converts the RAG document event into the standard format.

### Community 647 - "test_curiosity.py"
Cohesion: 0.45
Nodes (10): _coverage(), _fake_backend(), test_backend_error_degrades_to_no_followup(), test_capability_denied_returns_empty(), test_disabled_returns_empty(), test_dry_run_skips_fetch(), test_fetches_under_covered_section(), test_gate_evaluation_error_degrades_to_empty() (+2 more)

### Community 650 - "test_vision_retina_settings.py"
Cohesion: 0.36
Nodes (9): Docker compose passes empty RETINA_WIDTH/HEIGHT; must not crash Settings()., _reload_settings(), test_blank_retina_dimensions_coerced_to_none(), test_blank_retina_width_and_height_together(), test_jpeg_quality_clamped(), test_retina_dimensions_explicit_integers(), test_retina_source_explicit_overrides_path_alias(), test_retina_source_path_alias() (+1 more)

### Community 651 - "Endogenous Drive Origination Design"
Cohesion: 0.27
Nodes (10): Autonomy Arc Measurement Gate (Step 0), Endogenous Drive Origination Design, Origination Dynamics Under Leaky Math, Homeostatic Drives Real Tensions Design, Internal Economy Scarcity Allocation Design, Voluntary Attention Override Design, Effort Budget (bounded act of will), DriveEngine Leaky Integrator (cadence-invariant) (+2 more)

### Community 652 - "orion-social-memory service"
Cohesion: 0.22
Nodes (10): Social Claim Revision and Peer-Claim Tracking, Social Consensus, Divergence, and Attribution, Speaker-Aware Consensus States, Social Conversational Commitments, SocialCommitmentResolutionV1, SocialCommitmentV1, Social Epistemic Stance and Uncertainty Signaling, Epistemic Claim Kinds (+2 more)
=======
### Community 652 - "orion-social-memory service"
Cohesion: 0.06
Nodes (40): Social Artifact Dialogue, Conservative Dialogue Scope Handling, SocialArtifactConfirmationV1, SocialArtifactProposalV1, SocialArtifactRevisionV1, Social Claim Revision and Peer-Claim Tracking, Claim Stance Lifecycle, Social Consensus, Divergence, and Attribution (+32 more)
>>>>>>> Stashed changes

### Community 653 - "Unified Cognitive Substrate Phase 13 (GraphDB-Backed Persistence)"
Cohesion: 0.27
Nodes (10): Unified Cognitive Substrate Phase 13 (GraphDB-Backed Persistence), GraphDBSubstrateStore, SubstrateGraphStore Interface, Unified Cognitive Substrate Phase 14 (GraphDB Read/Query Layer and Hub Wiring), Source-Honest Explicit Fallback (No Silent Swaps), SubstrateQueryResultV1, Unified Cognitive Substrate Phase 15 (GraphDB Cognitive Re-anchoring), build_graph_views_from_store (+2 more)

### Community 654 - "Brainstorming Session #1 - Appendix Ideas 3-10"
Cohesion: 0.22
Nodes (10): Self-State Continuity and Live Dimensions v1 (spec, ideas 1-2 implemented), Brainstorming Session #1 - Appendix Ideas 3-10, Idea 10: Predictive substrate (forward model), Idea 3: Close the action outcome feedback loop, Idea 4: Bridge autonomy drive pressures to self-state, Idea 5: Substrate-level surprise signal (prediction error), Idea 6: Rolling self-state archive, Idea 7: Drive audit loop (+2 more)

### Community 655 - "orion-mesh-guardian service"
Cohesion: 0.22
Nodes (10): Actions Scheduler Periodic Docker Health Design, _scheduler_docker_findings unhealthy detection, Mesh Bus Resilience + Auto-Remediation Design, Half-death pattern (process up, bus broken), orion-mesh-guardian service, PUBSUB NUMSUB readiness probe, Tiered remediation state machine (recreate then build), Mesh Critical Failure Email Notifications Design (+2 more)

### Community 656 - "Orion Relational Stance Design (v1)"
Cohesion: 0.29
Nodes (10): Orion Relational Stance Design (v1), ChatStanceBrief, enforce_chat_stance_quality, interface_cost vs connection_seek semantic dimensions, Orion Relational Stance v2 Design, compile_speech_contract, interaction_regime (relational/minimal/instrumental), Late contract injection (TURN CONTRACT block) (+2 more)

### Community 657 - "Concept Relation Resolution Design"
Cohesion: 0.22
Nodes (10): CONCEPT_AUTONOMOUS_TRIGGER_ENABLED off decision, ConceptWorker / concept extraction, Concept Relation Resolution Design, ConceptRelationDecision (bounded LLM judgment), Memory crystallization pipeline, scope_overlap cross-window dedup structural bug, Crystallization Event Observability Design, ConceptRelationDecisionEventV1 (+2 more)

### Community 659 - "Channel "orion:kg:edge:ingest.v1" (kind=event, schema=KgEdgeIngestV1) producers=[orion-topic-foundry] consumers=[orion-rdf-writer, orion-graphdb]"
Cohesion: 0.20
Nodes (10): Channel "orion:kg:edge:ingest.v1" (kind=event, schema=KgEdgeIngestV1) producers=[orion-topic-foundry] consumers=[orion-rdf-writer, orion-graphdb], Channel "orion:topic:foundry:drift:alert.v1" (kind=event, schema=TopicFoundryDriftAlertV1) producers=[orion-topic-foundry] consumers=[*], Channel "orion:topic:foundry:enrich:complete.v1" (kind=event, schema=TopicFoundryEnrichCompleteV1) producers=[orion-topic-foundry] consumers=[*], Channel "orion:topic:foundry:run:complete.v1" (kind=event, schema=TopicFoundryRunCompleteV1) producers=[orion-topic-foundry] consumers=[*], Schema: KgEdgeIngestV1, Schema: TopicFoundryDriftAlertV1, Schema: TopicFoundryEnrichCompleteV1, Schema: TopicFoundryRunCompleteV1 (+2 more)

<<<<<<< Updated upstream
### Community 662 - "organ_layer"
Cohesion: 0.20
Nodes (12): layers_export(), organ_layer(), Organ layer taxonomy for Organ Signals mesh filters (Milestone B0)., Return layer for organ_id; unknown organs default to cognition., JSON-serializable layer map for Hub API., test_layers_export_includes_all_registry_organs(), test_organ_layer_cognition_organs(), test_organ_layer_runtime_organs() (+4 more)

### Community 664 - "check_daily_schedule_collisions.py"
Cohesion: 0.36
Nodes (8): _find_collisions(), _format_time_of_day(), _load_cadences(), main(), Minimal distance in minutes between two minute-of-day values, generic over     s, Returns {cadence_name: minute_of_day} for all four named cadences (including the, _read_env_example_values(), _time_of_day_distance_minutes()

### Community 666 - "_attempt_mind_handoff_chat_stance_shortcut"
Cohesion: 0.53
Nodes (8): _attempt_mind_handoff_chat_stance_shortcut(), If Orch supplied a validated Mind handoff, skip the LLM stance synthesis step., _exec_prep(), _step(), test_shortcut_returns_none_when_orch_did_not_authorize_skip(), test_shortcut_returns_none_when_payload_invalid(), test_shortcut_returns_none_when_skip_flag_without_authorization(), test_shortcut_returns_result_when_orch_authorized_meaningful_handoff()

### Community 667 - "endogenous_runtime.py"
Cohesion: 0.09
Nodes (46): EndogenousTriggerDecisionV1, EndogenousWorkflowPlanV1, EndogenousCalibrationProfileV1, EndogenousCalibrationRecommendationV1, EndogenousEvaluationRequestV1, EndogenousEvaluationResultV1, EndogenousMetricSummaryV1, PromotionCalibrationSummaryV1 (+38 more)

### Community 669 - "correlation_chain_from_cognition_trace"
Cohesion: 0.22
Nodes (9): api_signals_correlation(), Signal chain keyed by ``source_event_id`` / correlation id (Runtime Trace Nexus, correlation_chain_from_cognition_trace(), _map_step_organ(), Synthesize Organ Signals correlation chains from cached cognition traces., Build a correlation graph chain when signal inspect cache missed the turn., _step_services(), Correlation API fallback from cognition trace cache. (+1 more)

### Community 670 - "renderMemoryDebugModal"
Cohesion: 0.31
Nodes (10): applyDebugTextLayout(), buildMemoryDebugRecallEntryNode(), clearMemoryDebugPanel(), collectRecallEntries(), normalizeMemoryDebugModel(), renderMemoryDebugModal(), summarizeInlineText(), toPrettyText() (+2 more)

### Community 671 - "substrate.js"
Cohesion: 0.29
Nodes (5): fetchSection(), metaForSection(), refresh(), renderError(), renderSection()

### Community 672 - "test_mind_enabled_contract.py"
Cohesion: 0.31
Nodes (6): _build(), test_context_metadata_mind_enabled_true_passes_through(), test_grounded_small_payload_sets_metadata_mind_enabled_true(), test_missing_mind_enabled_reports_not_requested(), test_string_true_in_context_metadata_is_normalized(), test_top_level_mind_enabled_true_is_normalized()

### Community 674 - "test_projection_starvation.py"
Cohesion: 0.28
Nodes (8): client(), _empty_projection_with_diagnostics(), _mind_prep(), Regression: populated projection must not downgrade to starved fallback silently, Regression for fix/mind-enrichment-wall-budget: a light Mind run that sends NO, test_light_path_without_projection_does_not_claim_orch_starvation(), test_mind_run_emits_projection_starvation_diagnostics(), test_rich_projection_cannot_become_zero_items_without_diagnostics()

### Community 680 - "test_embodiment_c_hook.py"
Cohesion: 0.42
Nodes (9): _drive(), _make_worker(), Unit tests for the Orion embodiment C producer hook.  Verifies the substrate wor, test_cache_drive_state_fails_open_on_bad_decode(), test_emit_fails_open_when_publish_raises(), test_flag_off_publishes_nothing(), test_flag_on_publishes_one_involuntary_intent(), test_no_drive_state_publishes_nothing() (+1 more)

=======
>>>>>>> Stashed changes
### Community 683 - "Channel "orion:dream:log" (kind=event, schema=DreamResultV1) producers=[orion-cortex-exec] consumers=[orion-sql-writer, orion-dream]"
Cohesion: 0.29
Nodes (7): Channel "orion:dream:compaction-delta" (kind=event, schema=MemoryCompactionDeltaV1) producers=[orion-dream] consumers=[none], Channel "orion:dream:log" (kind=event, schema=DreamResultV1) producers=[orion-cortex-exec] consumers=[orion-sql-writer, orion-dream], Channel "orion:dream:trigger" (kind=event, schema=DreamTriggerPayload) producers=[orion-dream] consumers=[orion-cortex-orch], Schema: DreamResultV1, Schema: DreamTriggerPayload, Schema: MemoryCompactionDeltaV1, Service: orion-dream

<<<<<<< Updated upstream
### Community 685 - "STTEngine"
Cohesion: 0.07
Nodes (30): Payload for Speech-to-Text (ASR) request.     Kind: stt.transcribe.request, Payload for TTS synthesis request.     Kind: tts.synthesize.request, STTRequestPayload, STTResultPayload, TTSRequestPayload, TTSResultPayload, Sends a TTSRequestPayload to the TTS Service and waits for a TTSResultPayload., Sends an STTRequestPayload to the Speech Service and waits for an STTResultPaylo (+22 more)

=======
>>>>>>> Stashed changes
### Community 687 - "concept_induction_pass Workflow"
Cohesion: 0.22
Nodes (9): Active Verbs Manifest, Verb Enforcement (no silent chat_general fallback), concept_induction_pass Workflow, Append-only journal.entry.write.v1 Boundary, Chat-Invoked Cognitive Workflow Registry, Workflow vs Actions Skill Distinction, Bounded Deterministic Synthesis, Concept Induction Details Modal (+1 more)

### Community 690 - "Unified Cognitive Substrate Phase 17 (Operator-Controlled Policy Adoption and Rollback)"
Cohesion: 0.39
Nodes (9): Unified Cognitive Substrate Phase 16 (GraphDB Query Planning and Reuse), Bounded Query-Result Reuse Cache, Unified Cognitive Substrate Phase 17 (Operator-Controlled Policy Adoption and Rollback), SubstratePolicyProfileStore, SubstratePolicyProfileV1, Unified Cognitive Substrate Phase 18 (Durable Policy Store and Cache/Runtime Wiring), query_cache_enabled Knob, Unified Cognitive Substrate Phase 19 (SQL-Backed Policy Control-Plane Persistence) (+1 more)

### Community 692 - "Biometrics Reference Adapter"
Cohesion: 0.22
Nodes (9): Biometrics Reference Adapter, Causal DAG Organ Registry, EwmaBand / NormalizationContext, Biometrics Stage 0-5 Metric Lineage, Multicollinearity as Integration Surface, OTEL Trace Propagation, Orion Errors OTEL + system.error Bridge Design, Error-Pressure Projector (+1 more)

### Community 693 - "orion-memory-consolidation service"
Cohesion: 0.25
Nodes (9): Memory Consolidation Pipeline Design, Logprob turn classify (MEMORY/BOUNDARY), orion-memory-consolidation service, Situational phase window bounds, Turn Change Appraisal v1 Design, Remove tissue phi theater (no shadow tissue), turn_change_appraisal (logprob novelty/shift), Turn-change Classify Hardening Design (+1 more)

### Community 694 - "Metacog Prompt Slim Context Design"
Cohesion: 0.22
Nodes (9): Metacog Prompt Slim Context Design, metacog_biometrics_cue (compact cue), Worker context preflight / trim, Metacog Two-Pass Native Logprob Probe Design, Collapse mirror (MetacogDraftTextPatchV1), Native logprob uncertainty probe (pass 2), Metacog Substrate Funnel Design, phi-grounded causal_density scoring (+1 more)

### Community 697 - "Stance React"
Cohesion: 0.22
Nodes (9): Reverie Narrate (Spontaneous Felt-Layer Narration), SpontaneousThoughtV1 (schema), Simulation, Skills — Mesh Presence (Tailscale), Stance React, Story Weave, Substrate Inspect, Substrate Observe (+1 more)

<<<<<<< Updated upstream
### Community 699 - "dataset.py"
Cohesion: 0.48
Nodes (6): _chat_template(), _iter_jsonl(), _load_from_jsonl(), _normalize_record(), _stable_bucket(), CanonicalSftExample

### Community 700 - "Settings"
Cohesion: 0.32
Nodes (4): _parse_list(), Settings, Docker compose may pass empty strings when .env keys are missing., test_settings_ignore_empty_substrate_env()

### Community 702 - "scan_cognition_library"
Cohesion: 0.38
Nodes (5): get_cognition_library(), Returns the scanned list of Packs and Verbs available in the system.     Used by, find_repo_root(), Scans the orion/cognition folder for packs and verbs.     Returns:         {, scan_cognition_library()

### Community 704 - "social_room_inspection_cache.py"
Cohesion: 0.29
Nodes (7): get(), get_latest(), In-memory store for the latest social-room routing_debug per room.  Hub writes a, Record the latest routing_debug for a completed social-room turn., Return the latest snapshot for a specific room, or None., Return the most recently stored snapshot across all rooms., store()

### Community 706 - "test_memory_graph_structured_output.py"
Cohesion: 0.32
Nodes (6): _client_result(), _ensure_hub_scripts_import_path(), hub_settings(), _msg_payload(), Hub memory_graph_suggest structured-output request wiring., test_suggest_request_includes_structured_options()

### Community 707 - "test_substrate_biometrics_debug_api.py"
Cohesion: 0.31
Nodes (5): _atlas_emission(), _atlas_pressure_receipt(), Newest global receipt for another node must not appear on atlas latest., test_latest_chain_shape(), test_latest_chain_skips_non_matching_global_receipt()

### Community 708 - "test_substrate_execution_dispatch_debug_api.py"
Cohesion: 0.33
Nodes (5): _candidate(), _sample_dispatch_frame(), test_latest_includes_status_summary_counts(), test_latest_returns_frame(), test_latest_status_summary_all_zero_on_empty_frame()

### Community 709 - "test_substrate_field_debug_api.py"
Cohesion: 0.42
Nodes (6): _atlas_field_state(), _fake_engine_with_field(), test_field_capability_llm_inference(), test_field_latest_not_found(), test_field_latest_returns_parsed_state(), test_field_node_atlas_returns_vector_and_capabilities()

### Community 711 - "test_http_contract.py"
Cohesion: 0.32
Nodes (4): client(), _mind_prep(), _projection_with_item(), test_mind_run_emits_shadow_synthesis_from_projection_items_without_authority()

=======
>>>>>>> Stashed changes
### Community 712 - "orion-rag: retrieval-augmented generation orchestrator, enriches queries with vector-db context before delegating to LLM host"
Cohesion: 0.22
Nodes (9): orion-ollama-host docker-compose (GPU runtime, port 11434, healthcheck, model volume), orion-ollama-host: wrapper around official Ollama container, auto-pulls model on startup, exposes port 11434, orion-ollama-host dependencies (pydantic, pydantic-settings, PyYAML, requests), orion-pageindex docker-compose (port 8360, PageIndex build args, journal_entry_index table wiring), orion-pageindex: standalone journals PageIndex service (corpora rebuild/status/query API for journals and chat_episodes), orion-pageindex dependencies (fastapi, uvicorn, pydantic, psycopg2-binary, SQLAlchemy, pytest), orion-rag docker-compose (bus request/reply channels, VECTOR_DB_URL, EMBEDDING_MODEL), orion-rag: retrieval-augmented generation orchestrator, enriches queries with vector-db context before delegating to LLM host (+1 more)

### Community 716 - "orion-sql-db: PostgreSQL database + pgAdmin client"
Cohesion: 0.10
Nodes (22): orion-social-memory docker-compose.yml, orion-social-memory README, orion:chat:social:stored (social-memory input channel), orion-social-memory requirements.txt, orion-social-memory: relational continuity synthesizer for social-room turns, orion-social-room-bridge docker-compose.yml, orion-social-room-bridge requirements.txt, orion-social-room-bridge: social platform bridge (Callsyne/Hub) (+14 more)

<<<<<<< Updated upstream
### Community 717 - "test_phi_reward_sql_shape.py"
Cohesion: 0.29
Nodes (7): PhiIntrinsicRewardV1, _normalize_phi_reward_payload(), _make_reward(), Shape checks for the phi_reward SQL write path (no Postgres required)., test_model_map_registers_phi_reward_sql_with_schema(), test_normalize_phi_reward_payload_maps_to_real_columns(), test_phi_reward_row_constructs_without_raising()

### Community 719 - "test_mind_http_client.py"
Cohesion: 0.53
Nodes (8): _ok_result_json(), _req(), _settings(), test_empty_base_url_fails_open(), test_http_500_fails_open(), test_ok_returns_result(), test_oversized_body_fails_open(), test_timeout_fails_open()

### Community 720 - "introspect.py"
Cohesion: 0.61
Nodes (7): _allowed_schemas(), _get_cached(), list_columns(), list_schemas(), list_tables(), _set_cached(), table_fingerprint()

### Community 722 - "main.py"
Cohesion: 0.36
Nodes (6): join_openai_message_content(), Normalize OpenAI-compatible message.content (str or part list) to plain text., _extract_chat_result_text(), Tests for join_openai_message_content helper., test_join_list_content_skips_reasoning_parts(), test_join_string_content()

### Community 724 - "orion-social-memory service"
Cohesion: 0.25
Nodes (8): Social Artifact Dialogue, Conservative Dialogue Scope Handling, SocialArtifactConfirmationV1, SocialArtifactProposalV1, SocialArtifactRevisionV1, orion-social-memory service, Social Style and Rituals, Style Adaptation Snapshot

=======
>>>>>>> Stashed changes
### Community 725 - "GraphDB Semantic vs SQL Operational Ownership Split"
Cohesion: 0.25
Nodes (8): GraphDB Semantic vs SQL Operational Ownership Split, Substrate vs Control-Plane vs Operational SQL Split, Unified Cognitive Substrate Phase 20 (Policy Comparison Operationalization), Policy Comparison (baseline_vs_active / previous_vs_current / selected_pair), Unified Cognitive Substrate Phase 20c (Postgres Comparison/Control-Plane Parity), Postgres Operational/Control-Plane Truth, Unified Cognitive Substrate Phase 21 (Control-Plane Parity and Wiring Verification), Bus/Schema/SQL-Writer Wiring Verification

### Community 726 - "Unified Cognitive Substrate Phase 4 (Graph Dynamics and Pressure Propagation)"
Cohesion: 0.29
Nodes (8): Unified Cognitive Substrate Phase 4 (Graph Dynamics and Pressure Propagation), ActivationUpdateV1, Bounded Deterministic Dynamics (No Learned/GNN), Contradiction Amplification, PressureUpdateV1, Unified Cognitive Substrate Phase 5 (Graph Cognition V1), extract_graph_features, Metacognitive Perception Brief

### Community 727 - "Recall Service (orion-recall)"
Cohesion: 0.29
Nodes (8): Recall Service (orion-recall), MemoryBundleV1, RecallDecisionV1 telemetry, RecallQueryV1, Titanium Envelope, Organ Reducer Registry (13 reducers), SurfaceEncoding v2, Lane Visibility Scoping

### Community 728 - "AutonomyStateV2"
Cohesion: 0.20
Nodes (11): AutonomyStateV2 Reducer Design, AutonomyStateV2, Drive Pressures & Tensions, Orion's Mind Control-Plane Service Design, ChatStanceBrief, Bounded Cognition Loops (feedforward), MindHandoffBriefV1, Orch Canonical Entrypoint (Hub to Orch to Mind) (+3 more)

### Community 729 - "SubstrateGraphRecordV1"
Cohesion: 0.25
Nodes (8): SubstrateGraphRecordV1, ProducerRegistryV1, Trust tiers + merge policy, Graph Compression v1 Design, CompressionRegionV1, Leiden clustering + community-summary (native GraphRAG pattern), orion-graph-compression service, orion-recall graph_compression adapter

### Community 730 - "DriveEngine + Concept-Induction Deactivation Design"
Cohesion: 0.29
Nodes (8): DriveEngine + Concept-Induction Deactivation Design, DriveEngine leaky-integrator, endogenous_origination.py NO-GO gate, Six Drives Conceptual Audit, DRIVE_KEYS six-drive taxonomy, Autonomy Gate Instrument (P6) Design, measure_autonomy_gate.py CLI, UNMEASURABLE verdict distinction

### Community 731 - "Inner-State Unification Design"
Cohesion: 0.25
Nodes (8): Inner-State Unification Design, FieldStateV1 / FieldAttentionFrameV1, SelfStateV1, Reverie Chat Bridge + Resonance Monitoring Design, Resonance monitor (HealthMonitor port), reverie glimpse chat bridge (internal-only), Resonance alert window-echo (not live bug) root cause, _project_recent_dispatch_actions template projection

### Community 732 - "Vector Audit"
Cohesion: 0.36
Nodes (8): Vector Audit, RECALL_VECTOR_COLLECTIONS Zero-Result Root Cause, VectorDocumentUpsertV1 Schema/Registry, vector-host / vector-writer / recall Pipeline, Vector Memory Cleanup, Chroma Metadata Sanitizer, query_embeddings (no-MiniLM BGE retrieval), Session-Scoped Recall (session_id metadata fix)

### Community 733 - "Orion Node Bootstrap README (Ubuntu 24.04)"
Cohesion: 0.25
Nodes (8): Tailscale auth key placeholder file (empty), Admin SSH authorized_keys policy, Script: orion-bootstrap.sh, Orion Node Bootstrap README (Ubuntu 24.04), Requirement: docker-compose.repo SSH URL, Requirement: Tailscale auth key (env or file), Script: verify-agent.sh, Script: verify-gpu.sh

<<<<<<< Updated upstream
### Community 734 - "run_attention_bound_proposal_eval.py"
Cohesion: 0.38
Nodes (6): fetch_attended_target_ids(), open_readonly_connection(), Kill-criterion eval: attention-bound proposal target diversity (P5).  `inspect_a, Open a psycopg2 connection and force a read-only session.      Returns None on a, Return the target_id of every inspect_attended_target candidate whose     parent, run()

### Community 735 - "enum"
Cohesion: 0.25
Nodes (8): enum, ExecutiveControl, Generative, MemoryAccess, MetaCognition, Perception, SelfModification, Transform

### Community 738 - "chat.py"
Cohesion: 0.29
Nodes (6): ChatResultPayload, EnrichedChat, Standardized payload for LLM generation results., Enriched metadata for a chat message (tags, summary, embeddings pointer)., Minimal payload capturing a raw chat exchange., RawChat

### Community 739 - "Canonical phi: _phi_from_self_state() / _get_phi_stats"
Cohesion: 0.50
Nodes (4): Canonical phi: _phi_from_self_state() / _get_phi_stats(), orion-equilibrium-service heartbeat trigger, OrionTissue: fallback-only tensor (cold-start/outage path), Rationale: spark_engine.py/integration.py/strategies.py deleted (zero production consumers, third φ implementation)

### Community 741 - "smoke_all_notifications.sh"
Cohesion: 0.39
Nodes (6): add_test(), smoke_all_notifications.sh script, SKIP_COMPOSE, start_stack(), usage(), wait_for_health()

### Community 745 - "test_health.py"
Cohesion: 0.60
Nodes (5): _load_app(), test_health(), test_health_proposal_review_block_parent_writable_before_first_write(), test_health_proposal_review_block_store_configured(), test_health_proposal_review_block_store_missing()

### Community 746 - "conftest.py"
Cohesion: 0.32
Nodes (7): _ensure_cortex_exec_paths(), _purge_app_modules_if_wrong_service(), pytest_sessionstart(), Make the top-level ``app`` package resolve to orion-cortex-exec.      In a multi, Avoid importing full spaCy when tests only need app.router (pulls autonomy -> co, Service .env often sets large LLM_CHAT_* dev budgets; unit tests expect canonica, _stub_spacy_for_router_imports()

### Community 749 - "run_dream"
Cohesion: 0.08
Nodes (29): enrich_from_graphdb_ids(), _avg(), fetch_recent_sql_fragments(), Returns fragments from:       - collapse_mirror (timestamp is string → cast to t, Input columns:       - gpu (JSON): {"latest_file":"...", "gpus":[             {", _summarize_biometrics(), _to_float(), enrich_from_chroma() (+21 more)

### Community 750 - "record_turn"
Cohesion: 0.28
Nodes (8): presence_snapshot(), Hub presence — Orion's chat liveness as a self-state observable.  Records chat-t, Test helper: clear in-process presence state., Presence from in-process turn history; None before the first turn., Record one chat turn; best-effort, never raises, never blocks chat.      The Pos, record_turn(), reset(), _write_snapshot_to_postgres()

### Community 751 - "test_hub_local_time_naive_utc.py"
Cohesion: 0.43
Nodes (7): _denver_env(), _extract_format_hub_local_time_source(), Behavioral regression test for `formatHubLocalTime` in app.js.  Bug: orion-notif, Pull the real `formatHubLocalTime` (and its helpers) straight out of app.js., _run_node_denver(), test_naive_utc_created_at_renders_correct_denver_day_and_time(), test_offset_suffixed_value_is_not_double_converted()

### Community 752 - "test_memory_graph_bridge_ui.py"
Cohesion: 0.32
Nodes (4): _bridge_suggest_block(), _memory_suggest_block(), test_app_js_wires_memory_graph_bridge_handlers(), test_memory_js_listens_for_bridge_import_event()

### Community 753 - "test_mind_hub_tab.py"
Cohesion: 0.25
Nodes (4): mindRunsModal must not live under #scheduleModal.hidden or it never paints., Bare `global` is undefined in browsers and aborts DOMContentLoaded before tab wi, test_app_js_lane_api_uses_global_this_not_node_global(), test_mind_runs_modal_is_sibling_of_schedule_modal_not_nested()

### Community 754 - "test_presence_chat_injection.py"
Cohesion: 0.05
Nodes (85): parse_compactor_digest_json(), Parse an LLM digest JSON payload into the given compactor digest model.      Sha, assert_digest_within_budget(), build_quiet_day_digest(), parse_github_compactor_digest_json(), Bound digest LLM input size while preserving total merge count metadata., stable_github_compactor_journal_entry_id(), trim_github_compactor_input() (+77 more)

### Community 755 - "_FakeBus"
Cohesion: 0.36
Nodes (4): _FakeBus, test_api_chat_response_feedback_publishes_valid_payload(), test_api_chat_response_feedback_rejects_invalid_payload(), test_feedback_downvote_emits_pressure_event_telemetry()

### Community 756 - "test_self_brain_routes.py"
Cohesion: 0.32
Nodes (3): _fake_engine(), _frame(), test_tail_returns_ascending_and_200()

### Community 757 - "test_stop_chat_ui_smoke.py"
Cohesion: 0.25
Nodes (4): Regression guard: the cancel request must key off the per-connection id     capt, The Stop button must be shown for every WS send path that starts a     server-ca, test_app_js_shows_stop_button_on_every_turn_kind_send(), test_app_js_stop_request_uses_connection_id_not_session_id()
=======
### Community 739 - "Canonical phi: _phi_from_self_state() / _get_phi_stats"
Cohesion: 0.50
Nodes (4): Canonical phi: _phi_from_self_state() / _get_phi_stats(), orion-equilibrium-service heartbeat trigger, OrionTissue: fallback-only tensor (cold-start/outage path), Rationale: spark_engine.py/integration.py/strategies.py deleted (zero production consumers, third φ implementation)
>>>>>>> Stashed changes

### Community 758 - "orion-notify: minimal notification host centralizing email delivery, attention requests, chat messages, recipient preferences, escalation"
Cohesion: 0.29
Nodes (8): orion-mesh-guardian docker compose deployment (port 7161, mounts /repo ro + docker.sock, NOTIFY_BASE_URL, equilibrium snapshot), orion-mesh-guardian: detects half-dead bus consumers on chat critical path, publishes Hub Pending Attention cards via notify, optional auto-remediation via docker compose, orion-mesh-guardian Python dependencies (fastapi, pydantic, httpx, pyyaml, redis, loguru, requests), orion-notify-digest docker compose deployment (port 7150, shared postgres with notify, drift alert env), orion-notify-digest: builds daily summary of notification activity, sends via orion-notify, integrates Topic Foundry topics/drift alerts, orion-notify docker compose deployment (port 7140, SMTP env, in-app channel, escalation poll), orion-notify: minimal notification host centralizing email delivery, attention requests, chat messages, recipient preferences, escalation, orion-notify Python dependencies (fastapi, requests, pyyaml, loguru, redis, httpx)

<<<<<<< Updated upstream
### Community 760 - "test_rdf_chatturn_windowing.py"
Cohesion: 0.32
Nodes (4): RDF chat-turn recall must honor the profile time window.  The graph stores no us, _run_window(), test_windowing_drops_out_of_window_and_stamps_kept(), test_windowing_noop_when_since_minutes_non_positive()

### Community 763 - "test_phase21_wiring_verification.py"
Cohesion: 0.43
Nodes (6): _load_settings_module(), test_evidence_index_wiring_exists_in_settings_env_compose_bus_and_registry(), test_feedback_wiring_exists_in_settings_env_and_compose(), test_journal_index_wiring_exists_in_settings_env_compose_bus_and_registry(), test_markdown_adapter_channel_wiring_exists_in_settings_env_compose_bus_and_registry(), test_parsed_document_channel_wiring_exists_in_settings_env_compose_bus_and_registry()
=======
### Community 762 - "orion-social-memory: relational continuity synthesizer for social-room turns"
Cohesion: 0.25
Nodes (8): orion-social-memory docker-compose.yml, orion-social-memory README, orion:chat:social:stored (social-memory input channel), orion-social-memory requirements.txt, orion-social-memory: relational continuity synthesizer for social-room turns, orion-social-room-bridge docker-compose.yml, orion-social-room-bridge requirements.txt, orion-social-room-bridge: social platform bridge (Callsyne/Hub)
>>>>>>> Stashed changes

### Community 767 - "orion-llm-gateway manual smoke tests: bus chat/exec_step envelopes, ollama vs vllm backend selection"
Cohesion: 0.38
Nodes (7): Bus channel orion-exec:request:LLMGatewayService (LLM gateway intake), OrionBusAsync class (orion.core.bus.async_service.OrionBusAsync) used to publish/subscribe on Redis bus, orion-llm-gateway manual smoke tests: bus chat/exec_step envelopes, ollama vs vllm backend selection, orion-memory-consolidation docker compose deployment (port 8635, consolidation/crystallizer/concept-relation env), orion-mind router profiles: default (route_kind brain, chat_general, advisory) and conservative (route_kind no_chat, mandatory), orion-mind docker compose deployment: semantic/appraisal/stance LLM synthesis routes, evidence limits, LLM gateway intake channel, orion-mind Python dependencies (fastapi, pydantic, pyyaml, redis)

### Community 768 - "config/autonomy/capability_policy.v1.yaml — policy config gating which autonomy capabilities may auto-execute per cycle, by side-effect class, required goal status, required drive origins/signal kinds, and per-cycle budget"
Cohesion: 0.33
Nodes (7): config/autonomy/capability_policy.v1.yaml — policy config gating which autonomy capabilities may auto-execute per cycle, by side-effect class, required goal status, required drive origins/signal kinds, and per-cycle budget, capability journal.compose.episode — write side-effect, auto_execute true, requires goal status 'proposed', predictive drive origin, no required signal kinds, budget 1/cycle, capability recall.query.readonly — readonly side-effect, auto_execute true, requires goal status 'proposed', predictive drive origin, world_coverage_gap signal, budget 2/cycle (P4 addition, mirrors web.fetch.readonly's gate), capability web.fetch.readonly — readonly side-effect, auto_execute true, requires goal status 'proposed', predictive drive origin, world_coverage_gap signal, budget 2/cycle, capability web.fetch.write — external side-effect, auto_execute false, requires goal status 'planned', budget 0/cycle (blocked from autonomous execution), capability world_pulse.run — readonly side-effect, auto_execute true, requires no goal status, no drive origin gating, budget 1/cycle, Readonly capabilities (recall.query.readonly, P4): ConceptWorker is the sole production call site for maybe_execute_substrate_act_after_metabolism, which gates a Firecrawl fetch and a new inline RecallService RPC under config/autonomy/capability_policy.v1.yaml per cycle, trying recall first so a successful recall leaves that cycle's fetch budget unconsumed; degrades to a no-op if recall_bus/recall_source kwargs aren't wired

### Community 769 - "Social GIF Expression Layer"
Cohesion: 0.29
Nodes (7): Social GIF Expression Layer, SocialGifIntentV1, SocialGifPolicyDecisionV1, SocialGifUsageStateV1, Social GIF Interpretation Proxy, Non-Visual GIF Text Proxy, GIF Proxy Reaction Classes

<<<<<<< Updated upstream
### Community 770 - "Social Scenario Replay Harness"
Cohesion: 0.29
Nodes (7): Social Scenario Replay Harness, SocialScenarioEvaluationResultV1, SocialScenarioExpectationV1, SocialScenarioFixtureV1, Social Shakedown Workflow, SocialShakedownFixV1, SocialShakedownIssueV1
=======
### Community 771 - "Orion Landing Pad Guide"
Cohesion: 0.48
Nodes (7): Landing Pad pad.* contracts, Orion Landing Pad Guide, PadEventV1, Salience Pulse (pad.signal.v1), Landing Pad Reducer, Salience Scorer, StateFrameV1 (Specious Present)
>>>>>>> Stashed changes

### Community 772 - "live_state vs recovery_state"
Cohesion: 0.29
Nodes (7): Orion Vision Host Production Readiness Design, Vision Health/Readiness Probe, VisionScheduler / GpuInspector, Orion Vision Window Projection Service Design, Ephemeral-First, Not a Memory Organ, live_state vs recovery_state, Vision Window Snapshot Envelope v1

### Community 773 - "Hub OTEL Traces + Metrics Observability Design"
Cohesion: 0.29
Nodes (7): Hub OTEL Traces + Metrics Observability Design, Grafana Tempo + Prometheus default stack, orion-signal-gateway, otel_trace_id join key, Orion Meta-Services Architecture Graph Design, Epistemic classes (Observed/Declared/Documented/Inferred), ExtractionScope

### Community 774 - "CognitiveUnificationLayer"
Cohesion: 0.33
Nodes (7): Cognitive Unification Design, CognitiveUnificationLayer, substrate.tier_outcomes Mind bus telemetry, UnifiedRelationalBeliefSetV1, Substrate Tier Telemetry Persistence Design, Dumb Hub constraint (HTTP only, no bus), orion-substrate-telemetry service

### Community 775 - "Phi seed-v4 Feature Set Design"
Cohesion: 0.43
Nodes (7): Phi Corpus Hygiene Design, Phi seed-v4 Feature Set Design, token-based execution_load feature, Phi Truthful Corpus Overview, Reasoning Telemetry Adapter Design, ReasoningActivityV1 windowed projection, ReasoningCallV1 per-call event

### Community 776 - "Self-State & Mesh Substrate Redesign"
Cohesion: 0.29
Nodes (7): Scarcity Economy Brainstorm v2, Interoceptive recalibration / pinned-sensor gate, Queue-wait scarcity sense organ, resource_pressure saturated at 1.0, Self-State & Mesh Substrate Redesign, Metric substrate design invariants (no field without producer, preserve provenance), Mesh embodiment via field-topology edges

### Community 778 - "Channel "orion:evidence:index:upsert" (kind=event, schema=EvidenceUnitV1) producers=[orion-sql-writer, *] consumers=[orion-evidence-index, orion-sql-writer, *]"
Cohesion: 0.29
Nodes (7): Channel "orion:evidence:index:upsert" (kind=event, schema=EvidenceUnitV1) producers=[orion-sql-writer, *] consumers=[orion-evidence-index, orion-sql-writer, *], Channel "orion:evidence:markdown:ingest" (kind=event, schema=MarkdownSpecIngestV1) producers=[*] consumers=[orion-sql-writer, orion-evidence-index], Channel "orion:evidence:parsed:ingest" (kind=event, schema=ParsedDocumentIngestV1) producers=[*] consumers=[orion-sql-writer, orion-evidence-index], Schema: EvidenceUnitV1, Schema: MarkdownSpecIngestV1, Schema: ParsedDocumentIngestV1, Service: orion-evidence-index

### Community 779 - "Channel "orion:graph:compression:stale" (kind=event, schema=CompressionStalenessMarkV1) producers=[orion-rdf-writer, orion-graph-compression] consumers=[orion-graph-compression]"
Cohesion: 0.29
Nodes (7): Channel "orion:graph:compression:events" (kind=event, schema=GraphCompressionRegionMaterializedV1) producers=[orion-graph-compression] consumers=[*], Channel "orion:graph:compression:stale" (kind=event, schema=CompressionStalenessMarkV1) producers=[orion-rdf-writer, orion-graph-compression] consumers=[orion-graph-compression], Channel "orion:substrate:mutation:pressure" (kind=event, schema=MutationPressureEvidenceV1) producers=[orion-graph-compression] consumers=[none], Schema: CompressionStalenessMarkV1, Schema: GraphCompressionRegionMaterializedV1, Schema: MutationPressureEvidenceV1, Service: orion-graph-compression

### Community 780 - "orion-rdf-writer (bus → triples → RDF store service)"
Cohesion: 0.29
Nodes (7): orion-rdf-writer (bus → triples → RDF store service), orion-recall (memory retrieval / MemoryBundleV1 fusion service), orion-rdf-writer docker-compose.yml, orion-rdf-writer requirements.txt, orion-recall docker-compose.yml, orion-recall README.md, orion-recall requirements.txt

<<<<<<< Updated upstream
### Community 783 - "smoke_memory_cognition_loop_e2e.sh"
Cohesion: 0.43
Nodes (5): fail(), _health_curl(), PYTHONPATH, run_pytest(), smoke_memory_cognition_loop_e2e.sh script

=======
>>>>>>> Stashed changes
### Community 784 - "town_cards.yaml (cast source of truth)"
Cohesion: 0.33
Nodes (7): Generated Juniper Feld join description, town_cards.yaml (cast source of truth), Juniper Feld (AI Town human player character card), Orion (AI Town character card), orion-ai-town docker-compose.yml, orion-ai-town README, orion-ai-town service (AI Town mesh wrapper)

### Community 787 - "orion-dream Service"
Cohesion: 0.29
Nodes (7): orion-cortex-orch Python Dependencies, orion-dream Docker Compose Service, orion-dream Service, orion-dream Python Dependencies, orion-embodiment Docker Compose Service, orion-embodiment Service, orion-embodiment Python Dependencies

<<<<<<< Updated upstream
### Community 789 - "fcc_model_mapping.py"
Cohesion: 0.38
Nodes (4): label_to_claude_model_id(), Map FCC env key labels to stable Claude tier model ids for claude CLI --model., test_label_to_tier_model_ids(), test_unknown_label_raises()

### Community 792 - "_Store"
Cohesion: 0.43
Nodes (4): _Store, test_inject_session_presence_fills_empty_client_dict_from_store(), test_inject_session_presence_preserves_client_payload(), test_inject_session_presence_uses_store_when_payload_missing()

### Community 793 - "test_substrate_attention_debug_api.py"
Cohesion: 0.43
Nodes (4): _fake_engine_with_frame(), _sample_attention_frame(), test_attention_latest_not_found(), test_attention_latest_returns_frame()

### Community 794 - "test_substrate_policy_debug_api.py"
Cohesion: 0.43
Nodes (4): _fake_engine_with_frame(), _sample_policy_frame(), test_policy_latest_not_found(), test_policy_latest_returns_frame()

### Community 795 - "test_substrate_proposal_debug_api.py"
Cohesion: 0.43
Nodes (4): _fake_engine_with_frame(), _sample_proposal_frame(), test_proposals_latest_not_found(), test_proposals_latest_returns_frame()

### Community 801 - "test_registry_dag.py"
Cohesion: 0.29
Nodes (3): ``ORGAN_REGISTRY`` integrity (phase-2 causal DAG contract)., DFS from each organ must not revisit nodes on the stack (no cycles)., test_registry_acyclic()

### Community 805 - "test_route_map_completeness.py"
Cohesion: 0.33
Nodes (3): _load_sql_writer_channel_kinds(), Ensure every subscribed bus kind has a sql-writer persistence path (prevents sil, test_subscribed_catalog_kinds_are_routable_or_explicitly_special()

=======
>>>>>>> Stashed changes
### Community 807 - "whisper-tts (docker-compose service, GPU TTS/STT)"
Cohesion: 0.33
Nodes (7): vllm-host (docker-compose service, GPU-backed vLLM runtime), orion-vllm-host Python dependencies (pydantic, pydantic-settings, pyyaml), voip-endpoint (docker-compose service, Asterisk/SIP host-networked), orion-voip-endpoint Python dependencies (fastapi, redis, pydantic), whisper-tts (docker-compose service, GPU TTS/STT), Orion Whisper TTS README (TTS/STT bus contracts, Coqui XTTS-v2, Whisper), orion-whisper-tts Python dependencies (TTS, openai-whisper, transformers pin)

<<<<<<< Updated upstream
### Community 808 - "test_tts_engine_settings.py"
Cohesion: 0.38
Nodes (3): _load_settings_module(), test_tts_settings_xtts_defaults(), test_whisper_tts_timeout_settings_defaults()

### Community 809 - "test_execution_dispatch_bus_catalog.py"
Cohesion: 0.43
Nodes (6): _channel_entry(), Bus catalog coverage for orion-execution-dispatch-runtime as a cortex-exec produ, recall.query.readonly (P4): the autonomy tick issues an inline recall RPC     fr, test_concept_induction_is_recall_service_request_producer(), test_execution_dispatch_runtime_is_action_outcome_producer(), test_execution_dispatch_runtime_is_background_exec_request_producer()

### Community 813 - "Reasoning Promotion Phase 3"
Cohesion: 0.40
Nodes (6): Reasoning Promotion Phase 3, Contradiction-Aware Gating, HITL Escalation Policy, Deterministic Transition Evaluator, PromotionEvaluationResultV1, Explicit Promotion Transition Matrix

### Community 814 - "Phase 5 Research Findings"
Cohesion: 0.40
Nodes (6): Dream Contracts (dream.result.v1), Substrate Ladder L7-L11, Phase 5 Research Findings, compaction_applier.py (inert), L7-L11 Ladder is Rehearsal, Reverie Consolidation Grounding

=======
>>>>>>> Stashed changes
### Community 815 - "Information-Dynamics Pillars"
Cohesion: 0.33
Nodes (6): Ablation Baseline (heartbeat disabled), Kraskov et al. 2004 - Estimating Mutual Information (KSG), Maldacena 1998 - AdS/CFT Correspondence, Pastawski et al. 2015 - Holographic Quantum Codes, Information-Dynamics Pillars, Pre-Registered Hypotheses H1-H4

### Community 816 - "OrionSignalV1"
Cohesion: 0.47
Nodes (6): Organ Signal Gateway Offboarding Guide, Organ Signal Gateway Design (Phase 1), OrionSignalV1, Organ Signal Gateway Phase 2 Design, Production-Ready Gate (Critical/Important/Minor), Deterministic 64-hex signal_id

### Community 817 - "memory.turn.persisted outbox"
Cohesion: 0.33
Nodes (6): Spark Introspection Lane Isolation Design, Chat/spark/background lane model, Queues for work, PubSub for broadcasts, orion-spark-introspector, correlation_id is sacred (trace_id end-to-end), memory.turn.persisted outbox

### Community 818 - "Repair Pressure v2 + Pre-Turn Appraisal Rail Design"
Cohesion: 0.40
Nodes (6): Repair Pressure v2 + Pre-Turn Appraisal Rail Design, Kill phrase_match_v1 keyword tables, logprob_probe_v2 (seven-kind evidence), AppraisalParadigm plugin rail, PreTurnAppraisalRequestV1 / TurnAppraisalBundleV1, orion-thought organ (ThoughtV1)

### Community 820 - "inner_state_registry.py / InnerStateSignal"
Cohesion: 0.33
Nodes (6): is_corpus_row_healthy ingestion gate, Cognition Metric Lineage & Liveness Registry Ideas, MetricRegistration dataclass, variance-gate liveness classifier, check_inner_state_registry.py gate, inner_state_registry.py / InnerStateSignal

### Community 821 - "AutonomyStateV2 Closed-Loop Wiring Design"
Cohesion: 0.40
Nodes (6): AutonomyStateV2 Closed-Loop Wiring Design, AutonomySliceV1 model, compile_harness_prefix consumer, AutonomyStateV2 Postgres persistence store, Autonomy Experience Loop (P2) Design, ActionOutcomeEmitV1 Layer 9 emit

### Community 822 - "Topic Foundry (Windowing v2, Micro/Macro, Enrichment)"
Cohesion: 0.40
Nodes (6): Topic Foundry (Windowing v2, Micro/Macro, Enrichment), Topic Foundry Hub Contract Deep-Dive, Topic Foundry /capabilities Contract, Hub Topic Studio UI, Micro/Macro Run Scope, Windowing v2 Modes

<<<<<<< Updated upstream
### Community 823 - "verb.schema.json"
Cohesion: 0.33
Nodes (5): description, $id, $schema, title, type

### Community 824 - "services"
Cohesion: 0.33
Nodes (6): description, services, description, items, minItems, type

### Community 825 - "enum"
Cohesion: 0.33
Nodes (6): enum, type, priority, high, low, medium

=======
>>>>>>> Stashed changes
### Community 827 - "Recall Memory"
Cohesion: 0.40
Nodes (6): Pattern Detection, Build RDF Triples, Recall Memory, Recall Profile: reflect.v1, Reflect, Web Search (Simulated)

### Community 828 - "Perceive: Retina Fast Pipeline (Embed, Detect, Caption)"
Cohesion: 0.40
Nodes (6): Perceive: Caption Frame, Perceive: Detect Open-Vocabulary Objects, Perceive: Embed Image, Perceive: Retina Fast Pipeline (Embed, Detect, Caption), Perceive Vision Events (Host -> Window -> Council), Perceive Vision Memory (Host -> Window -> Council -> Scribe)

<<<<<<< Updated upstream
### Community 829 - "recall.py"
Cohesion: 0.40
Nodes (4): Standardized memory fragment returned by Recall service., Request to recall memory from various sources (Vector, SQL, RDF)., RecallRequest, RecallResult

### Community 830 - "Spark organ (salience, change, concept formation)"
Cohesion: 0.29
Nodes (7): orion-meta-tags service, orion-notify service, orion-notify-digest service, Spark organ (salience, change, concept formation), orion-spark-concept-induction service, orion-spark-introspector service, orion-topic-foundry service

### Community 832 - "context_exec_beta_gate.sh"
Cohesion: 0.47
Nodes (4): check_beta_key(), print_env_file(), PYTHONPATH, context_exec_beta_gate.sh script

### Community 834 - "migrate_graphdb_to_fuseki.py"
Cohesion: 0.73
Nodes (5): _count_triples(), _export_batch(), _import_batch(), main(), _request()

### Community 835 - "smoke_actions_daily.sh"
Cohesion: 0.33
Nodes (5): ACTIONS_DAILY_RUN_ONCE_DATE, NOTIFY_API_TOKEN, NOTIFY_BASE_URL, ORION_BUS_URL, smoke_actions_daily.sh script

### Community 836 - "smoke_vision_caption_provenance.sh"
Cohesion: 0.53
Nodes (4): main(), run_fake_mode(), run_real_mode(), smoke_vision_caption_provenance.sh script

### Community 837 - "test_scheduler_cursor_state_path.py"
Cohesion: 0.53
Nodes (5): _load_compose(), _parse_env_example(), Guards the fix for the ephemeral-/tmp cursor durability gap.  `services/orion-ac, test_compose_mounts_volume_covering_state_paths(), test_env_example_store_paths_not_under_tmp()
=======
### Community 830 - "Spark organ (salience, change, concept formation)"
Cohesion: 0.33
Nodes (6): orion-meta-tags service, orion-notify-digest service, Spark organ (salience, change, concept formation), orion-spark-concept-induction service, orion-spark-introspector service, orion-topic-foundry service
>>>>>>> Stashed changes

### Community 840 - "orion-attention-runtime service"
Cohesion: 0.33
Nodes (6): FieldAttentionFrameV1 (substrate_attention_frames), FieldStateV1 (substrate_field_state), orion-attention-runtime service, orion-field-digester service (referenced upstream writer), orion-notify service (referenced alert sink), orion-athena-sql-db / orion-sql-db (referenced Postgres)

<<<<<<< Updated upstream
### Community 841 - "get_settings"
Cohesion: 0.29
Nodes (3): health(), get_settings(), Settings

### Community 842 - "filter_world_context_capsule"
Cohesion: 0.60
Nodes (3): filter_world_context_capsule(), test_filter_world_context_capsule_fail_open_missing(), test_filter_world_context_capsule_filters_expired_and_low_confidence()

### Community 843 - "up-with-tailscale.sh"
Cohesion: 0.33
Nodes (5): ORION_ACTIONS_TAILSCALE_PATH, ORION_CONTAINER_TAILSCALE_BIN, ORION_HOST_TAILSCALE_BIN, ORION_HOST_TAILSCALE_RUN, up-with-tailscale.sh script

=======
>>>>>>> Stashed changes
### Community 846 - "orion-equilibrium-service Docker Compose Service"
Cohesion: 0.33
Nodes (6): orion-equilibrium-service Docker Compose Service, orion-equilibrium-service Service, orion-equilibrium-service Python Dependencies, orion-field-digester Docker Compose Service, orion-field-digester Service, orion-field-digester Python Dependencies

<<<<<<< Updated upstream
### Community 848 - "test_aitown_proxy.py"
Cohesion: 0.60
Nodes (3): _request(), test_aitown_proxy_disabled_returns_404(), test_aitown_proxy_forwards_path()

### Community 855 - "test_world_pulse_proxy_routes.py"
Cohesion: 0.60
Nodes (3): _request(), test_world_pulse_alias_routes_forward_expected_paths(), test_world_pulse_proxy_returns_controlled_http_error()

### Community 856 - "verify_mind_llm_e2e.sh"
Cohesion: 0.73
Nodes (5): check_url(), fail(), pass(), verify_mind_llm_e2e.sh script, warn()

### Community 857 - "test_llm_uncertainty_telemetry.py"
Cohesion: 0.50
Nodes (3): _mind_prep(), _prep(), Mind llm_uncertainty telemetry for semantic synthesis.

### Community 861 - "settings.py"
Cohesion: 0.40
Nodes (3): Config, Helper to parse the subscription channels from env var., Settings

### Community 862 - "test_context_exec_proposal_storage_defaults.py"
Cohesion: 0.40
Nodes (3): _active_doc_text(), Assert context-exec proposal ledger smoke scripts use durable storage defaults., test_proposal_review_doc_active_examples_use_durable_storage()

### Community 864 - "test_recall_profiles_cards_knobs.py"
Cohesion: 0.60
Nodes (4): _profiles_dir(), Every shipped recall profile must declare cards_top_k for the memory-cards rail., test_all_recall_profiles_define_cards_backend_weight(), test_all_recall_profiles_define_cards_top_k()

### Community 865 - "test_recent_turn_effect_alerts.py"
Cohesion: 0.53
Nodes (5): _load_core_event_cache(), _load_executor_module(), test_core_event_cache_filters_turn_effect_alerts(), test_format_recent_turn_effect_alerts_summary(), test_system_alert_tags_merge()

=======
>>>>>>> Stashed changes
### Community 866 - "orion-memory-crystallizer: governed cognitive memory crystallization worker; proposes/validates MemoryCrystallizationV1, projects to Chroma/Graphiti/FalkorDB, never canonical without governor"
Cohesion: 0.40
Nodes (5): Bus channel orion:memory:crystallization:proposed (memory.crystallization.proposed.v1), MemoryCrystallizationV1 artifact schema: governed cognitive memory, proposal/approval workflow, orion-memory-crystallizer docker compose deployment (port 8634, crystallization channel + Graphiti/FalkorDB env), orion-memory-crystallizer: governed cognitive memory crystallization worker; proposes/validates MemoryCrystallizationV1, projects to Chroma/Graphiti/FalkorDB, never canonical without governor, orion-memory-crystallizer Python dependencies (fastapi 0.111, asyncpg, psycopg2-binary)

### Community 867 - "Agent Git Safety Mechanism Stack"
Cohesion: 0.50
Nodes (5): Defense in Depth Principle, destructive_git_guard.py, Agent Git Safety Mechanism Stack, orion-git-shim, Worktree Hygiene Tooling

### Community 870 - "Workflow Schedule Production Hardening v1"
Cohesion: 0.50
Nodes (5): Attention Notification Philosophy, Workflow Schedule Production Hardening v1, JSON Schedule Store v1 Rationale, Lightweight In-Process Counters, Structured Schedule Error Codes

### Community 871 - "Substrate Trace Map Template"
Cohesion: 0.50
Nodes (5): Substrate Trace Map Template, grammar_events table, PUBLISH_SERVICE_GRAMMAR flag, Trace Redaction Rules, Trace ID Format

### Community 872 - "Phi Snapshot"
Cohesion: 0.29
Nodes (7): Spark Metrics v2, Channelized Tissue Expectations, Novelty z-score (baseline-relative), Phi Snapshot, Proxy Telemetry Not Canonical State, reduce_autonomy_state reducer, v0 Tissue Shadow Comparison

### Community 873 - "Memory Graph Annotator (Hub) + Dual-Write GraphDB"
Cohesion: 0.40
Nodes (5): Memory Graph Annotator (Hub) + Dual-Write GraphDB, AffectiveDisposition / Situation / TypedEntity, orionmem Ontology (RDF/PROV-O/Schema.org), Escaping Shallow Reference-less Recall, Memory Graph Chat Entry Bridge Design

### Community 875 - "GoalProposalEngine v2 (dedupe + semantic goals)"
Cohesion: 0.40
Nodes (5): chat_stance.py unified beliefs integration, Autonomy Goals v2 Design, Goal lifecycle state machine (promote/plan/execute), GoalProposalEngine v2 (dedupe + semantic goals), Stance mode decoupled from goals

### Community 876 - "AutonomyStateV2 Evidence signal_tension Design"
Cohesion: 0.60
Nodes (5): AutonomyStateV2 Evidence signal_tension Design, AutonomyEvidenceRefV1 typed evidence contract, chat_evidence_to_tension adapter, AutonomyStateV2 hard isolation from phi/SelfState/DriveEngine, signal_drive_map.yaml

<<<<<<< Updated upstream
### Community 877 - "install-nvidia.sh"
Cohesion: 0.70
Nodes (4): add_generic_repo_list(), add_hardcoded_repo_list(), install_from_ubuntu_repo(), install-nvidia.sh script

### Community 878 - "cortex_exec_fleet_helpers.sh"
Cohesion: 0.50
Nodes (3): cortex_exec_fleet_helpers.sh script, up_cortex_exec_fleet(), verify_cortex_exec_fleet()

### Community 879 - "test_autonomy_isolation.py"
Cohesion: 0.83
Nodes (3): _imports_autonomy_v2(), _python_files(), test_autonomy_state_v2_not_wired_into_phi_or_self_state()

=======
>>>>>>> Stashed changes
### Community 881 - "Chat (Generalist)"
Cohesion: 0.40
Nodes (5): Chat (Generalist), Chat (Kids Story), Chat (Quick), Chat (Social Room), Orion Voice Finalize

### Community 882 - "Daily Metacog v1"
Cohesion: 0.50
Nodes (5): Daily Metacog v1, Daily Pulse v1, Journal Compose, Log (Collapse Mirror), Log Orion Metacognition

### Community 883 - "Self Repo Inspect"
Cohesion: 0.70
Nodes (5): Self Concept Induce, Self Concept Reflect, Recall Profile: self.factual.v1, Self Repo Inspect, Self Retrieve (Trust-Mode Lane Retrieval)

### Community 885 - "Orion Journaler Service Boundaries and Semantics"
Cohesion: 0.50
Nodes (5): Chat Discussion Window Journaling Workflow, Journal Dispatch Registry (trigger_kind -> policy, fail-closed), Orion Journaler Service Boundaries and Semantics, JournalEntryWriteV1.trigger_kind Propagation, chat.continuity.v1 Recall Profile (Recent sql_chat Only)

### Community 886 - "orion-security-watcher (Guard: vision presence/alert debounce service)"
Cohesion: 0.40
Nodes (5): orion-security-watcher (Guard: vision presence/alert debounce service), orion-security-watcher index.html template, orion-security-watcher docker-compose.yml, orion-security-watcher README.md, orion-security-watcher requirements.txt

### Community 887 - "Bounded trigger loop in ConceptWorker.handle_envelope"
Cohesion: 0.40
Nodes (5): Rationale: concept_induction_pass stayed a bounded reader; missing runtime behavior was autonomous triggering, ConceptInductionTrigger contract (source_kind, subjects, trigger_reason, ...), Bounded trigger loop in ConceptWorker.handle_envelope(), Graph mapping shape (SparkConceptProfile/Subject/Concept/Cluster/StateEstimate RDF nodes), Rationale: graph write additive/isolated, LocalProfileStore stays source of truth

<<<<<<< Updated upstream
### Community 888 - "Phase 3B: Parity Evidence and Cutover-Readiness Model"
Cohesion: 0.40
Nodes (5): Phase 2: Spark ConceptProfile Graph Read Model, Phase 3A: Shadow Rollout for Spark ConceptProfile Repository, Phase 3B: Parity Evidence and Cutover-Readiness Model, Phase 4: ConceptProfile Runtime Cutover (concept_induction_pass), Phase 0: Spark Concept Profile Repository Seam

### Community 889 - "diagnose_cortex_bus_stack.py"
Cohesion: 0.83
Nodes (3): _ensure_sys_path_stdlib_safe(), main(), _repo_root()

### Community 891 - "smoke_active_verbs.sh"
Cohesion: 0.60
Nodes (3): fail(), pass(), smoke_active_verbs.sh script

### Community 892 - "smoke_presence_grounding.py"
Cohesion: 0.83
Nodes (3): _headers(), main(), _print_step()

### Community 893 - "verify-bound-capability-live.sh"
Cohesion: 0.70
Nodes (3): fail(), need_cmd(), verify-bound-capability-live.sh script

### Community 896 - "_exec_import_guard.py"
Cohesion: 0.60
Nodes (4): ensure_orion_cortex_exec_app(), _purge_app_tree(), Side-effect: put ``services/orion-cortex-exec`` first and drop a foreign ``app``, _strip_other_service_paths()

### Community 897 - "test_metacog_trigger_lineage.py"
Cohesion: 0.70
Nodes (4): _load_executor_module(), test_metacog_trigger_lineage_chat_turn_overrides_trigger_kind(), test_metacog_trigger_lineage_passes_baseline_from_trigger_payload(), test_metacog_trigger_lineage_passes_dense_from_trigger_payload()

### Community 899 - "test_situation_prompt_integration.py"
Cohesion: 0.80
Nodes (4): _base_ctx(), _render(), test_prompts_render_with_situation_fragment_scenarios(), test_prompts_render_without_situation_fragment()

### Community 900 - "_orch_import_guard.py"
Cohesion: 0.60
Nodes (4): ensure_orion_cortex_orch_app(), _purge_app_tree(), Side-effect: put ``services/orion-cortex-orch`` first and drop a foreign ``app``, _strip_other_service_paths()

### Community 904 - "conftest.py"
Cohesion: 0.60
Nodes (4): _ensure_hub_paths(), _hub_service_isolation(), pytest_configure(), Ensure Hub ``scripts`` package resolves to ``services/orion-hub/scripts`` (not r

### Community 908 - "_ensure_hub_scripts_import_path"
Cohesion: 0.60
Nodes (4): _ensure_hub_scripts_import_path(), Repo-root ``scripts/`` shadows Hub when pytest mixes repo tests with Hub tests (, test_memory_graph_approve_requires_graph_or_named_graph(), test_memory_graph_validate_fixture_roundtrip()

### Community 915 - "test_crystallization_repository_import.py"
Cohesion: 0.40
Nodes (4): orion-memory-consolidation ships only asyncpg; the crystallization     repositor, The sync DDL helper is allowed to require psycopg2, but only when called., test_apply_schema_still_uses_psycopg2_lazily(), test_repository_imports_without_psycopg2()

### Community 916 - "_mind_import_guard.py"
Cohesion: 0.60
Nodes (4): ensure_orion_mind_app(), _purge_app_tree(), Side-effect: put ``services/orion-mind`` first and drop a foreign ``app`` packag, _strip_other_service_paths()

### Community 918 - "test_sql_chat_windowing.py"
Cohesion: 0.60
Nodes (4): Post-fetch SQL chat windowing drops stale sql_timeline/sql_chat rows., _run_window(), test_sql_windowing_drops_out_of_window_uuid_rows(), test_sql_windowing_keeps_recent_rows_with_real_ts_when_id_not_uuid()

### Community 919 - "test_webhook_auth.py"
Cohesion: 0.70
Nodes (4): _fake_process(), _sign(), test_webhook_accepts_valid_hmac_when_secret_configured(), test_webhook_rejects_invalid_hmac_when_secret_configured()

### Community 923 - "conftest.py"
Cohesion: 0.60
Nodes (4): _ensure_thought_paths(), pytest_configure(), Ensure orion-thought ``app`` resolves to this service during tests., _thought_service_isolation()

### Community 924 - "test_stance_prompt_renders_coloring.py"
Cohesion: 0.70
Nodes (4): _render(), test_block_absent_without_coloring(), test_block_does_not_introduce_output_keys(), test_block_present_with_coloring()

### Community 927 - "test_agent_trace_js.py"
Cohesion: 0.70
Nodes (4): _run_node(), test_agent_trace_helpers_gate_group_and_timeline(), test_agent_trace_helpers_gate_message_level_debug_sections(), test_live_agent_step_anchors_to_conversation_not_body()

### Community 929 - "test_memory_graph_core_pure.py"
Cohesion: 0.60
Nodes (4): _import_top_levels(), Memory-graph core stays deterministic: no LLM vendor SDKs or bus client imports., test_json_extract_has_no_nonstdlib_imports(), test_memory_graph_core_modules_have_no_llm_vendor_or_bus_imports()

### Community 933 - "hub_quick_playwright_live.py"
Cohesion: 0.60
Nodes (4): main(), Quick (fast) only: send probe1, wait for reply, immediately send probe2, wait ag, _run_fast_two_turns(), _run_one()
=======
### Community 921 - "orion-sql-writer: durable bus-to-Postgres persistence consumer"
Cohesion: 0.40
Nodes (5): orion-sql-writer docker-compose.yml, orion-sql-writer README, action.outcome.emit.v1 persistence (action_outcomes table, idempotent upsert on action_id), orion-sql-writer requirements.txt, orion-sql-writer: durable bus-to-Postgres persistence consumer
>>>>>>> Stashed changes

### Community 934 - "orion-rdf-writer Canonical Writer"
Cohesion: 0.50
Nodes (4): Fuseki / SPARQL Active Graph Backend, orion-gdb-client Legacy Quarantine, orion-rdf-writer Canonical Writer, RDF Store V1 Cutover

### Community 935 - "Social Context Window Selection"
Cohesion: 0.67
Nodes (4): Social Context Window Selection, SocialContextCandidateV1, SocialContextSelectionDecisionV1, SocialContextWindowV1

### Community 936 - "Social Thread Choreography"
Cohesion: 0.67
Nodes (4): Social Thread Choreography, SocialHandoffSignalV1, SocialThreadRoutingDecisionV1, SocialThreadStateV1

### Community 937 - "Daily Delivery Burst After Restart Design"
Cohesion: 0.50
Nodes (4): Daily Delivery Burst After Restart Design, Durable Scheduler Cursors, Hub NotificationCache, orion-actions scheduler

### Community 938 - "Memory Graph Draft Viz + Bridge Turns Design"
Cohesion: 0.50
Nodes (4): Memory Graph Draft Viz + Bridge Turns Design, Cytoscape draft graph UI, hub-utterance id strategy (hybrid C), SuggestDraftV1

### Community 939 - "chat_kids_story verb"
Cohesion: 0.50
Nodes (4): Kids Story Chat Verb Design, chat_kids_story verb, FAST_SINGLE_PASS_CHAT_VERBS frozenset, No child PII in repo (firewall-backed listeners)

### Community 940 - "Substrate-Fed Motivation Design (v1)"
Cohesion: 0.67
Nodes (4): Substrate-Fed Motivation Design (v1), Capability policy (C on A+B decision gate), substrate_metabolism adapter (metabolize_substrate_signals), world_coverage_gap signal

### Community 941 - "Chat History Compactor Design"
Cohesion: 0.83
Nodes (4): Chat History Compactor Design, chat_history_compactor_pass workflow, ChatHistoryCompactorDigestV1, compactor_index (indexed upsert window key)

### Community 942 - "CortexOrchAdapter dispatch_failure signal"
Cohesion: 0.50
Nodes (4): Cortex-Orch Dispatch-Failure Signal Design, CortexOrchAdapter dispatch_failure signal, orch_dispatch_failure metadata flag (dedup mechanism), OrionSignalV1 / Organ Signals tab

### Community 943 - "Journal/Notification Flood Fix Design"
Cohesion: 0.50
Nodes (4): Journal/Notification Flood Fix Design, Scheduler Cursor Durability (volume mount), Declarative Journal Dispatch Registry, RDF Recall Recency Decay Fix

### Community 947 - "Fact Extraction"
Cohesion: 0.67
Nodes (4): Evaluate, Fact Extraction, Goal Formulation, Memory graph suggest (brain draft JSON)

### Community 951 - "orion-self-experiments (typed self-experiment registry + context-exec dispatcher)"
Cohesion: 0.50
Nodes (4): orion-self-experiments (typed self-experiment registry + context-exec dispatcher), orion-self-experiments docker-compose.yml, orion-self-experiments README.md, orion-self-experiments requirements.txt

<<<<<<< Updated upstream
### Community 953 - "dependencies"
Cohesion: 0.50
Nodes (3): puppeteer, dependencies, puppeteer

### Community 954 - "check_fcc_context_parity.py"
Cohesion: 0.83
Nodes (3): main(), _max_profile_ctx_size(), _read_env_int()

### Community 956 - "grammar_production_truth.sh"
Cohesion: 0.83
Nodes (3): check_truth(), grammar_production_truth.sh script, validate_truth_json()

### Community 958 - "run_answer_depth_proof_suite.py"
Cohesion: 0.83
Nodes (3): _ensure_test_runtime_deps(), _has_import(), main()

### Community 959 - "smoke_council_debug.sh"
Cohesion: 0.83
Nodes (3): fail(), pass(), smoke_council_debug.sh script

### Community 963 - "smoke_metacog_phase_contract.py"
Cohesion: 0.83
Nodes (3): _ensure_platform_system(), _load_executor_module(), main()

### Community 964 - "smoke_orion_bus_transport_full_stack.sh"
Cohesion: 0.83
Nodes (3): psql_run(), smoke_orion_bus_transport_full_stack.sh script, usage()

### Community 965 - "smoke_telemetry_normalization.py"
Cohesion: 0.83
Nodes (3): _ensure_platform_system(), _load_module(), main()

### Community 966 - "smoke_topic_foundry_bertopic.sh"
Cohesion: 0.83
Nodes (3): require_jq(), run_train(), smoke_topic_foundry_bertopic.sh script

### Community 967 - "smoke_topic_foundry_remote.sh"
Cohesion: 0.83
Nodes (3): curl_api(), require_cmd(), smoke_topic_foundry_remote.sh script

### Community 972 - "conftest.py"
Cohesion: 0.67
Nodes (3): _dream_service_isolation(), _ensure_dream_paths(), Ensure orion-dream ``app`` resolves to this service during evals.

### Community 973 - "conftest.py"
Cohesion: 0.67
Nodes (3): _dream_service_isolation(), _ensure_dream_paths(), Ensure orion-dream ``app`` resolves to this service during tests.

### Community 974 - "test_speech_settings_defaults.py"
Cohesion: 0.67
Nodes (3): _load_embodiment_settings(), Regression: town speech must not default to legacy cortex intake., test_speech_defaults_use_chat_lane_and_skip_unified()

### Community 975 - "entrypoint.sh"
Cohesion: 0.50
Nodes (3): FCC_OPEN_BROWSER, LOG_FILE, entrypoint.sh script

### Community 978 - "conftest.py"
Cohesion: 0.67
Nodes (3): _ensure_governor_paths(), _governor_service_isolation(), Ensure orion-harness-governor ``app`` resolves to this service during tests.

### Community 979 - "verify_agent_claude_stream_live.py"
Cohesion: 0.67
Nodes (3): main(), Live smoke: Hub agent-claude WS streams claude_step + final llm_response.  Usage, run()

### Community 980 - "verify_agent_repl_stream_live.py"
Cohesion: 0.67
Nodes (3): main(), Gate 2 live proof: drive the Hub chat WS in agent mode and assert live step fram, run()

### Community 981 - "self_observability.js"
Cohesion: 0.83
Nodes (3): activatePanel(), deactivatePanel(), styleTabButton()

### Community 987 - "validate_llamacpp_upgrade.sh"
Cohesion: 0.83
Nodes (3): run_infer(), run_server_boot(), validate_llamacpp_upgrade.sh script

=======
>>>>>>> Stashed changes
### Community 990 - "orion-policy-runtime: Layer 8 substrate service evaluating ProposalFrameV1 against SubstratePolicyV1, persists PolicyDecisionFrameV1 (policy is not execution)"
Cohesion: 0.50
Nodes (4): orion-policy-runtime docker-compose (port 8120, SUBSTRATE_POLICY_PATH, POLICY_POLL_INTERVAL_SEC), orion-policy-runtime: Layer 8 substrate service evaluating ProposalFrameV1 against SubstratePolicyV1, persists PolicyDecisionFrameV1 (policy is not execution), substrate_policy_decision_frames table (PolicyDecisionFrameV1 records), orion-policy-runtime dependencies (fastapi, sqlalchemy, psycopg2-binary, PyYAML)

<<<<<<< Updated upstream
### Community 991 - "smoke.sh"
Cohesion: 0.83
Nodes (3): fail(), pass(), smoke.sh script

### Community 994 - "conftest.py"
Cohesion: 0.67
Nodes (3): _ensure_substrate_paths(), Ensure substrate-runtime ``app`` resolves to this service during tests., _substrate_service_isolation()

### Community 998 - "claim:test:0001 (accepted claim fixture)"
Cohesion: 0.67
Nodes (3): claim:test:0001 (accepted claim fixture), source:test:fixture (design_doc source fixture, primary trust), spec:test:compile (execution_ready spec fixture, component orion-test)

### Community 999 - "test_channel_prefix_guardrail.py"
Cohesion: 0.83
Nodes (3): _is_allowed_channel(), test_channel_catalog_prefixes(), test_literal_publish_subscribe_prefixes()

### Community 1003 - "test_recall_alert_profile.py"
Cohesion: 0.83
Nodes (3): _load_fusion_module(), test_tag_prefix_boost_ranks_alert_tag(), test_turn_effect_boost_ranks_high_delta()

### Community 1005 - "test_workflow_ui_js.py"
Cohesion: 0.83
Nodes (3): _run_node(), test_workflow_ui_normalizes_metadata_and_chip_label(), test_workflow_ui_run_again_visibility_rules_and_non_workflow_passthrough()
=======
### Community 998 - "claim:test:0001 (accepted claim fixture)"
Cohesion: 0.67
Nodes (3): claim:test:0001 (accepted claim fixture), source:test:fixture (design_doc source fixture, primary trust), spec:test:compile (execution_ready spec fixture, component orion-test)
>>>>>>> Stashed changes

### Community 1006 - "Substrate Atlas (grammar-atom Cytoscape.js graph)"
Cohesion: 0.67
Nodes (3): Substrate Atlas (grammar-atom Cytoscape.js graph), substrate-atlas.js, substrate-lattice.js (sibling viz, uninspected)

### Community 1007 - "Channel triage heuristic report (audit_001/reports_postfix, remediated)"
Cohesion: 0.67
Nodes (3): Channel triage heuristic report (audit_001/reports), Channel triage heuristic report (audit_001/reports_postfix, remediated), Channel triage heuristic report (audit_002/reports)

### Community 1008 - "Topology Node: prometheus"
Cohesion: 0.67
Nodes (3): Mesh Node: prometheus (observability), Capability: memory, Topology Node: prometheus

### Community 1009 - "Qwen3 Thinking-Off Policy"
Cohesion: 0.67
Nodes (3): Live Verification Gates A/B, orion-llamacpp-host Rebuild Design, Qwen3 Thinking-Off Policy

### Community 1010 - "resolve_subject_identity"
Cohesion: 0.67
Nodes (3): Dyadic Chat to Relationship Routing, resolve_subject_identity(), Autonomy Subject Routing Contract

### Community 1011 - "CallSyne Handoff Spec"
Cohesion: 1.00
Nodes (3): CallSyne Handoff Spec, media_hint (bounded GIF cue), Social-room Bridge

### Community 1012 - "Hub Mind Tab v1 Completion"
Cohesion: 0.67
Nodes (3): Hub Mind Tab v1 Completion, mind_hub.js client modularization, GET /api/mind/runs/recent aggregates

### Community 1013 - "SmolagentsCodeEngine REPL reasoning loop"
Cohesion: 0.67
Nodes (3): context-exec SmolagentsCodeEngine Design, Read-only repo/recall organ tools + sandbox, SmolagentsCodeEngine REPL reasoning loop

### Community 1014 - "agent_repl context-exec mode"
Cohesion: 1.00
Nodes (3): Agent-Lane Reasoning Loop (smolcode REPL) Design, agent_repl context-exec mode, SmolagentsCodeEngine (CodeAgent loop)

### Community 1015 - "Graphiti Rail Activation Design (A-B-C slices)"
Cohesion: 1.00
Nodes (3): Graphiti Rail Activation Design (A-B-C slices), canonical_mutated always false (projection-only invariant), GraphitiAdapter / orion-graphiti-adapter

### Community 1016 - "ENABLE_TOOL_SEARCH env contract"
Cohesion: 0.67
Nodes (3): FCC Motor ToolSearch Design, Eager MCP schema injection problem, ENABLE_TOOL_SEARCH env contract

### Community 1017 - "Reverie Narration Continuity Design"
Cohesion: 0.67
Nodes (3): Reverie Narration Continuity Design, Reverie Chain Continuity (prior_thoughts/next_focus), Verdict-Aware Narration (loop_outcomes)

### Community 1018 - "requirements-dev.txt (repo-wide dev/test deps)"
Cohesion: 0.67
Nodes (3): orion-sql-writer-tests CI workflow, Memory-graph / SHACL validation dependency (rdflib+pyshacl), requirements-dev.txt (repo-wide dev/test deps)

### Community 1026 - "Assess Runtime State"
Cohesion: 0.67
Nodes (3): Assess Mesh Presence, Assess Runtime State, Assess Storage Health

### Community 1027 - "Dream Cycle"
Cohesion: 0.67
Nodes (3): Dream Cycle, Dream Preprocess, Dream Simple

### Community 1028 - "Finalize Response"
Cohesion: 1.00
Nodes (3): Finalize Response, Generate Code Scaffold, GitHub Compactor Digest

### Community 1032 - "Orion Recall Profiles Overview (Multi-Backend Ensemble Policy)"
Cohesion: 0.67
Nodes (3): assist.light.v1 Recall Profile (Latency Lane, No Vector), chat.general.v1 Recall Profile (UX Continuity / Lightweight), Orion Recall Profiles Overview (Multi-Backend Ensemble Policy)

### Community 1033 - "graph.compressions.v1 recall profile (unified)"
Cohesion: 0.67
Nodes (3): graph.compressions.global.v1 recall profile, graph.compressions.local.v1 recall profile, graph.compressions.v1 recall profile (unified)

### Community 1035 - "Autonomous Event-Driven Concept Induction Trigger Loop note"
Cohesion: 1.00
Nodes (3): Autonomous Event-Driven Concept Induction Trigger Loop note, Phase 1: Spark ConceptProfile Graph Materialization note, orion/spark README

### Community 1088 - "orion-fcc Docker Compose Service"
Cohesion: 0.67
Nodes (3): orion-fcc Docker Compose Service, orion-fcc Service (Anthropic-compatible FCC proxy), orion-fcc Python Dependencies (free-claude-code)

### Community 1089 - "orion-feedback-runtime Docker Compose Service"
Cohesion: 0.67
Nodes (3): orion-feedback-runtime Docker Compose Service, orion-feedback-runtime Service (Layer 10), orion-feedback-runtime Python Dependencies

### Community 1100 - "orion-meta-tags: LLM-based enrichment of collapse events (entities, sentiment, tags) via orion:collapse:triage -> orion:tags:enriched"
Cohesion: 0.67
Nodes (3): orion-meta-tags docker compose deployment (bus enabled, triage/tagged channels, health check, startup delay), orion-meta-tags: LLM-based enrichment of collapse events (entities, sentiment, tags) via orion:collapse:triage -> orion:tags:enriched, orion-meta-tags Python dependencies (fastapi, spacy, sentence-transformers, pydantic, redis)

### Community 1102 - "orion-notify service (notification policy owner)"
Cohesion: 0.67
Nodes (3): orion-notify policy rules (quiet hours, recipient groups, event-kind/severity routing, throttling), orion-notify service (notification policy owner), orion-notify-digest dependencies (fastapi, uvicorn, pydantic, SQLAlchemy, requests, redis, loguru)

### Community 1120 - "orion-world-pulse (docker-compose service, Firecrawl-backed curiosity fetch)"
Cohesion: 0.67
Nodes (3): orion-world-pulse (docker-compose service, Firecrawl-backed curiosity fetch), orion-world-pulse Python dependencies (fastapi, requests, redis, pytest), sample_section.html test fixture (HTML link extraction, internal vs external hrefs)

### Community 1121 - "claim:test:bad-ref (disputed claim fixture with dangling references)"
Cohesion: 0.67
Nodes (3): claim:test:bad-ref (disputed claim fixture with dangling references), claim:does:not:exist (dangling depends_on reference, no fixture defines it), source:missing:0001 (dangling source_refs reference, no fixture defines it)

## Ambiguous Edges - Review These
- `agent-trace.js (plain-text step consumer)` → `Agent Trace inspection modal (Hub UI, fail case screenshot)`  [AMBIGUOUS]
  .verify-run/hub_agent_trace_timeline.png · relation: conceptually_related_to
- `orion-cortex-gateway Docker Compose Service` → `orion-cortex-orch Docker Compose Service`  [AMBIGUOUS]
  services/orion-cortex-gateway/docker-compose.yml · relation: references
- `Orion Cortex Orchestrator Service` → `orion-cortex-orch Docker Compose Service`  [AMBIGUOUS]
  services/orion-cortex-orch/README.md · relation: references
- `orion-gpu-cluster-power service (compose)` → `orion-hub service (README) — browser gateway into the mesh`  [AMBIGUOUS]
  services/orion-gpu-cluster-power/docker-compose.yml · relation: conceptually_related_to
- `Self Brain UI panel — brain map + region EKG canvas visualizer` → `Hub main dashboard template (index.html) — tab nav + chat/EKG/debug panels`  [AMBIGUOUS]
  services/orion-hub/templates/index.html · relation: references
- `Substrate Inspector UI (orion-hub)` → `Substrate Atlas UI (orion-hub)`  [AMBIGUOUS]
  services/orion-hub/templates/substrate.html · relation: conceptually_related_to
- `orion-substrate-runtime compose config` → `orion-substrate-telemetry compose config`  [AMBIGUOUS]
  services/orion-substrate-telemetry/docker-compose.yml · relation: conceptually_related_to
- `orion-vision-edge perception node service` → `orion-vision-window rolling window aggregator service`  [AMBIGUOUS]
  services/orion-vision-window/README.md · relation: shares_data_with
- `claim:test:bad-ref (disputed claim fixture with dangling references)` → `claim:does:not:exist (dangling depends_on reference, no fixture defines it)`  [AMBIGUOUS]
  tests/fixtures/knowledge_forge/claims/disputed/claim-test-bad-ref.yaml · relation: references
- `claim:test:bad-ref (disputed claim fixture with dangling references)` → `source:missing:0001 (dangling source_refs reference, no fixture defines it)`  [AMBIGUOUS]
  tests/fixtures/knowledge_forge/claims/disputed/claim-test-bad-ref.yaml · relation: references
- `Substrate Trace (design concept)` → `Typed Frames (compiled artifacts)`  [AMBIGUOUS]
  docs/context-engineering/00_substrate_trace_doctrine.md · relation: derived_from
- `SelfStateV1` → `Phase 3 Capacity vs Continuity Real Data`  [AMBIGUOUS]
  docs/notes/2026-07-12-phase3-capacity-vs-continuity-real-data.md · relation: references
- `dominant_attention_targets (salience)` → `_node_salience raw-vs-decayed race`  [AMBIGUOUS]
  docs/notes/2026-07-14-attention-salience-decay-bypass-investigation.md · relation: conceptually_related_to

## Knowledge Gaps
<<<<<<< Updated upstream
- **1183 isolated node(s):** `install-docker.sh script`, `install-utils.sh script`, `orion-bootstrap.sh script`, `GIT_SSH_COMMAND`, `setup-node.sh script` (+1178 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **348 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.
=======
- **850 isolated node(s):** `CognitionTracePayload schema`, `HarnessRunStepV1 schema`, `substrate-atlas.js`, `substrate-lattice.js (sibling viz, uninspected)`, `websocket_handler.py` (+845 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **103 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.
>>>>>>> Stashed changes

## Work-memory lessons

**Preferred sources** — corroborated by past sessions; start here.
- `SubstrateGraphStore Interface` (2× useful, score=1.961334539)
- `graphiti_core Backend (hybrid vector+graph search)` (2× useful, score=1.961334539)

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `agent-trace.js (plain-text step consumer)` and `Agent Trace inspection modal (Hub UI, fail case screenshot)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `orion-cortex-gateway Docker Compose Service` and `orion-cortex-orch Docker Compose Service`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `Orion Cortex Orchestrator Service` and `orion-cortex-orch Docker Compose Service`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `orion-gpu-cluster-power service (compose)` and `orion-hub service (README) — browser gateway into the mesh`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Self Brain UI panel — brain map + region EKG canvas visualizer` and `Hub main dashboard template (index.html) — tab nav + chat/EKG/debug panels`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `Substrate Inspector UI (orion-hub)` and `Substrate Atlas UI (orion-hub)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `orion-substrate-runtime compose config` and `orion-substrate-telemetry compose config`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._