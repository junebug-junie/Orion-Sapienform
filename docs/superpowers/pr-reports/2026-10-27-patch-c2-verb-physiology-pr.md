# Patch C2: every cortex verb declares its physiology

## Summary

- Every cortex verb now says who pays for it. Each of the 106 verb files in `orion/cognition/verbs/` gains a `physiology:` block with four fields: tier (0 vital, 1 upkeep, 2 discretionary), want family, GPU pool lanes, and how long it can wait (`deferrable_window_s`, 0 = perishable).
- New schema `VerbPhysiologyV1` (`orion/cognition/verb_physiology.py`). Tier 0 never bids, so tier 0 must use family `turn` or `autonomic` with a window of 0, and those two families are tier 0 only. That keeps the gate total without letting tier 0 become a way for an Orion-chosen act to skip the market.
- `plan_loader.load_verb_physiology(verb)` parses and exposes the block. `build_plan_for_verb` is unchanged, so plans and the Orch -> Exec payload are byte-identical. Nothing acts on the block yet; Patch C3 reads it in cortex-orch.
- New CI gate `scripts/check_verb_physiology.py` (in `orion-static-gates.yml` and `make check-verb-physiology`): every verb has a valid block, and every lane names a role or class in `config/gpu_pool.yaml`.

## Outcome moved

Before: no verb said what tier it belongs to, so the C3 bid site would have had to guess or hunt call sites. Now the registry answers it structurally, and CI refuses a new verb without an answer. The spec's Phase 2 Gate 1 row "Every verb declares its physiology" is live.

## Current architecture

Verb YAMLs carried `priority`, `requires_gpu`, `timeout_ms`; `plan_loader.build_plan_for_verb` turned them into an `ExecutionPlan`. Body economy tiers (spec Organ 3b, decided 10-23) existed only in the design doc.

## Architecture touched

- `orion/cognition` (new module + one loader function). No service code, no bus, no schema registry, no `orion/physiology/*`, no durable-runs, no sql-writer.
- CI: one new step in `orion-static-gates.yml`; one Makefile target.

## Files changed

- `orion/cognition/verb_physiology.py`: the schema, parser, and pool-lane loader.
- `orion/cognition/plan_loader.py`: `load_verb_physiology()`.
- `orion/cognition/verbs/*.yaml` (106 files): appended `physiology:` block. Verified by script that every file's other keys parse identically before and after.
- `scripts/check_verb_physiology.py`: the gate (`--summary` prints counts).
- `tests/test_check_verb_physiology.py`: gate tests (missing block, invalid tier, unknown lane, valid block, every field required, tier/family rule, act-ledger family match, plan unchanged).
- `.github/workflows/orion-static-gates.yml`, `Makefile`: wiring.

## Schema / bus / API changes

- Added: verb YAML key `physiology` and `VerbPhysiologyV1` (a config model, not a bus schema; not in `orion/schemas/registry.py`).
- Removed / Renamed: none.
- Behavior changed: none at runtime.
- Compatibility notes: existing verb-YAML readers ignore unknown keys (`planner/models.py VerbConfig` has no `extra="forbid"`; `verb_catalog`, `ActionsSkillRegistry`, `self_inventory` read named keys only).

## Env/config changes

- Added / Removed / Renamed keys: none.
- `.env_example` updated: no. Local `.env` sync: not needed.

## Classification method

Tier follows the verb's **self-initiated** invoker when one exists in live code, because Juniper's turns never bid in C3 whatever the verb's tier. A verb only reachable from Juniper's turn (Hub chat, agent-mode supervisor tool catalog, Hub skill runner) is `tier 0 / turn`. A no-LLM, no-pool-GPU step run by a scheduler or as a workflow's fetch step is `tier 0 / autonomic`. Families use the act-ledger actor name where one exists (`actions.daily_metacog`, `actions.journal`, `dream.cycle`, `thought.painting`). Lanes are pool classes the verb's LLM step lands on via its route (`quick*` -> fast, `metacog*` -> metacog, `chat` -> chat, `agent`/durable hold -> agent); `[]` = no pool GPU. Windows: 0 for tier 0; tier 1/2 use the act horizon where one exists (painting 5400), otherwise an authored default for C3 to calibrate.

### Counts

```text
tier counts: tier 0=71, tier 1=25, tier 2=10
  tier 0  autonomic                        12
  tier 0  turn                             59
  tier 1  actions.daily_metacog            1
  tier 1  actions.daily_pulse              1
  tier 1  actions.journal                  1
  tier 1  compactor.digest                 2
  tier 1  curiosity.compaction             1
  tier 1  curiosity.supervisor             1
  tier 1  dream.cycle                      3
  tier 1  execution_dispatch.maintain      3
  tier 1  memory.consolidation             1
  tier 1  metacog.log                      2
  tier 1  orion_day.letter                 2
  tier 1  self_study                       6
  tier 1  spark.introspect                 1
  tier 2  concept_induction                2
  tier 2  execution_dispatch.inspect       1
  tier 2  execution_dispatch.substrate     3
  tier 2  thought.painting                 2
  tier 2  thought.reverie                  2
check_verb_physiology: OK (106 verbs declare a valid physiology block)
```

### Full classification (106 verbs)

| verb | tier | family | lanes | window_s | call-site evidence |
|---|---|---|---|---|---|
| `perceive_caption_frame` | 0 | `autonomic` | (none) | 0 | no dispatcher found; perception runs as the vision pipeline, not as verbs (spec Execution seam) |
| `perceive_detect_open_vocab` | 0 | `autonomic` | (none) | 0 | no dispatcher found; perception runs as the vision pipeline, not as verbs (spec Execution seam) |
| `perceive_embed_image` | 0 | `autonomic` | (none) | 0 | no dispatcher found; perception runs as the vision pipeline, not as verbs (spec Execution seam) |
| `perceive_retina_fast` | 0 | `autonomic` | (none) | 0 | no dispatcher found; perception runs as the vision pipeline, not as verbs (spec Execution seam) |
| `perceive_vision_events` | 0 | `autonomic` | (none) | 0 | no dispatcher found; perception runs as the vision pipeline, not as verbs (spec Execution seam) |
| `perceive_vision_memory` | 0 | `autonomic` | (none) | 0 | no dispatcher found; perception runs as the vision pipeline, not as verbs (spec Execution seam) |
| `skills.biometrics.snapshot.v1` | 0 | `autonomic` | (none) | 0 | orion-actions logic.py:21 periodic scheduler; also a turn tool |
| `skills.chat.discussion_window.v1` | 0 | `autonomic` | (none) | 0 | no-GPU SQL fetch step of chat_history_compactor_pass / journal_discussion_window_pass (workflow registry.py:132/231) |
| `skills.docker.ps_status.v1` | 0 | `autonomic` | (none) | 0 | orion-actions logic.py:24/43 periodic scheduler findings; also a turn tool |
| `skills.gpu.nvidia_smi_snapshot.v1` | 0 | `autonomic` | (none) | 0 | orion-actions logic.py:22 periodic scheduler; also a turn tool |
| `skills.repo.github_recent_prs.v1` | 0 | `autonomic` | (none) | 0 | no-GPU fetch step of the github_compactor_pass workflow (workflow registry.py:206); the LLM half is github_compactor_digest_v1 |
| `skills.system.notify_chat_message.v1` | 0 | `autonomic` | (none) | 0 | orion-actions logic.py:23 periodic scheduler delivery; also a turn tool |
| `analyze_text` | 0 | `turn` | chat, fast | 0 | cortex-orch decision_router.py:212/335 fallback primary verb for an auto-routed Hub turn |
| `answer_current_datetime` | 0 | `turn` | (none) | 0 | capability-backed tool on agent-mode Hub turns (supervisor.py, capability_bridge.py) |
| `answer_direct` | 0 | `turn` | chat, fast | 0 | supervisor.py:296 terminal tool inside an agent-mode Hub turn |
| `ask_camera` | 0 | `turn` | (none) | 0 | capability-backed tool on agent-mode Hub turns (supervisor.py, capability_bridge.py); executor.py capability route; vision is not a pool lane |
| `assess_mesh_presence` | 0 | `turn` | (none) | 0 | capability-backed tool on agent-mode Hub turns (supervisor.py, capability_bridge.py) |
| `assess_risk` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher |
| `assess_runtime_state` | 0 | `turn` | (none) | 0 | capability-backed tool on agent-mode Hub turns (supervisor.py, capability_bridge.py) |
| `assess_storage_health` | 0 | `turn` | (none) | 0 | capability-backed tool on agent-mode Hub turns (supervisor.py, capability_bridge.py) |
| `auto_depth_select` | 0 | `turn` | chat, fast | 0 | decision_router.py ORCH_INTERNAL_DENY: orch-internal routing on a Hub turn |
| `auto_route` | 0 | `turn` | chat, fast | 0 | cortex-orch main.py/decision_router.py/orchestrator.py:382: mode=auto Hub turn routing |
| `chat_general` | 0 | `turn` | chat, fast | 0 | Hub websocket_handler.py:231 / api_routes.py:972 every chat turn; embodiment worker.py:1152 town speech (someone waiting); orion-actions main.py:2010 only as the carrier envelope of a scheduled workflow |
| `chat_kids_story` | 0 | `turn` | fast | 0 | Hub turn, executor.py FAST_SINGLE_PASS_CHAT_VERBS -> quick |
| `chat_quick` | 0 | `turn` | fast | 0 | Hub Quick mode + embodiment speech (quick_embodied, interactive), executor.py FAST_SINGLE_PASS_CHAT_VERBS |
| `chat_social_room` | 0 | `turn` | chat | 0 | Hub social_room.py:54 SOCIAL_ROOM_VERB, a live room turn |
| `compare_options` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher |
| `counterfactual` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher |
| `evaluate` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher |
| `extract_facts` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher |
| `finalize_response` | 0 | `turn` | chat, fast | 0 | supervisor.py:296 terminal tool inside an agent-mode Hub turn |
| `generate_code_scaffold` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher |
| `goal_formulate` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher |
| `harness_finalize_reflect` | 0 | `turn` | chat, agent | 0 | orion/harness/finalize.py resolve_finalize_llm_lane: finalizes Hub/FCC turns on the owner lane |
| `housekeep_runtime` | 0 | `turn` | (none) | 0 | capability-backed tool on agent-mode Hub turns (supervisor.py, capability_bridge.py) |
| `inspect_docker_container_status` | 0 | `turn` | (none) | 0 | capability-backed tool on agent-mode Hub turns (supervisor.py, capability_bridge.py) |
| `inspect_gpu_status` | 0 | `turn` | (none) | 0 | capability-backed tool on agent-mode Hub turns (supervisor.py, capability_bridge.py) |
| `introspect` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher |
| `list_biometrics_recent_readings` | 0 | `turn` | (none) | 0 | capability-backed tool on agent-mode Hub turns (supervisor.py, capability_bridge.py) |
| `look_at_camera` | 0 | `turn` | (none) | 0 | capability-backed tool on agent-mode Hub turns (supervisor.py, capability_bridge.py) |
| `orion_response_repair` | 0 | `turn` | chat, agent | 0 | orion/hub/turn_orchestrator.py + orion/harness/finalize.py: repairs a Hub reply |
| `pattern_detect` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher |
| `plan_action` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher; agent_chain_guards.py |
| `rdf_build` | 0 | `turn` | (none) | 0 | no dispatcher found; only reachable as an agent-mode supervisor tool |
| `recall` | 0 | `turn` | (none) | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher (recall itself also runs as a step service, not as this verb) |
| `reflect` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher (the durable 'reflect' workflow runs self_study.reflect, not this verb) |
| `search_web` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher |
| `self_critique` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher |
| `send_operator_notification` | 0 | `turn` | (none) | 0 | capability-backed tool on agent-mode Hub turns (supervisor.py, capability_bridge.py) |
| `show_biometrics_snapshot` | 0 | `turn` | (none) | 0 | capability-backed tool on agent-mode Hub turns (supervisor.py, capability_bridge.py) |
| `simulate` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher |
| `skills.biometrics.raw_recent.v1` | 0 | `turn` | (none) | 0 | capability_bridge.py + Hub skill_runner_catalogue.py (Juniper's skill runner) |
| `skills.docker.compose_service_bringup.v1` | 0 | `turn` | (none) | 0 | Hub api_routes.py:1291 (Juniper's bring-up button) |
| `skills.mesh.mesh_ops_round.v1` | 0 | `turn` | (none) | 0 | Hub skill_runner_catalogue.py (Juniper's skill runner) |
| `skills.mesh.refresh_service_envs.v1` | 0 | `turn` | (none) | 0 | Hub skill_runner_catalogue.py (Juniper's skill runner) |
| `skills.mesh.tailscale_mesh_status.v1` | 0 | `turn` | (none) | 0 | Hub skill_runner_catalogue.py (Juniper's skill runner) |
| `skills.mesh.up_all_services.v1` | 0 | `turn` | (none) | 0 | Hub skill_runner_catalogue.py (Juniper's skill runner) |
| `skills.perception.ask_camera.v1` | 0 | `turn` | (none) | 0 | capability_bridge.py (ask_camera tool on a turn) |
| `skills.storage.disk_health_snapshot.v1` | 0 | `turn` | (none) | 0 | Hub skill_runner_catalogue.py (Juniper's skill runner) |
| `skills.system.time_now.v1` | 0 | `turn` | (none) | 0 | capability_bridge.py + Hub skill_runner_catalogue.py |
| `stance_react` | 0 | `turn` | chat | 0 | orion-thought mind_enrichment.py:343 / bus_listener.py: the stance step of every unified Hub turn; executor.py default route chat |
| `story_weave` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher |
| `summarize_context` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher |
| `summarize_recent_changes` | 0 | `turn` | (none) | 0 | capability-backed tool on agent-mode Hub turns (supervisor.py, capability_bridge.py) |
| `synthesize_patterns` | 0 | `turn` | chat, fast | 0 | no dispatcher found; only reachable as an agent-mode supervisor tool |
| `tag_enrich` | 0 | `turn` | (none) | 0 | no dispatcher found; only reachable as an agent-mode supervisor tool |
| `triage` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher |
| `write_guide` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher |
| `write_recommendation` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher |
| `write_runbook` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher |
| `write_tutorial` | 0 | `turn` | chat, fast | 0 | cortex-exec supervisor tool catalog (supervisor.py VerbRegistry.list) for agent-mode Hub turns; no other live dispatcher |
| `daily_metacog_v1` | 1 | `actions.daily_metacog` | metacog | 21600 | orion-actions main.py:108 scheduled job (ACTIONS_DAILY_LLM_ROUTE=metacog); act actor actions.daily_metacog (act_emit.py:57) |
| `daily_pulse_v1` | 1 | `actions.daily_pulse` | metacog | 21600 | orion-actions main.py:107 scheduled job (ACTIONS_DAILY_LLM_ROUTE=metacog); opens no act (main.py:1575) |
| `journal.compose` | 1 | `actions.journal` | fast, agent | 3600 | orion-actions _dispatch_journal -> cortex-orch journal.compose (ACTIONS_JOURNAL_LLM_ROUTE=quick_background), actor actions.journal; ALSO durable journal.compose runs (world_pulse_digest, actor durable.journal_compose, agent hold) |
| `chat_history_compactor_digest_v1` | 1 | `compactor.digest` | agent | 86400 | chat_history_compactor_pass scheduled by orion-actions (main.py:96) -> compactor.digest durable run (DIGEST_LLM_ROUTE=agent) |
| `github_compactor_digest_v1` | 1 | `compactor.digest` | agent | 86400 | github_compactor_pass scheduled by orion-actions (main.py:96) -> compactor.digest durable run (DIGEST_LLM_ROUTE=agent) |
| `curiosity_compactor_digest_v1` | 1 | `curiosity.compaction` | agent | 86400 | orion-dream curiosity_compaction_submit.py -> durable curiosity.compaction graph (llm_route agent) |
| `curiosity_hop_reading` | 1 | `curiosity.supervisor` | chat | 86400 | orion/curiosity/supervisor.py:87 via scripts/report_curiosity_supervisor_readings.py (operator-run), route chat |
| `dream_cycle` | 1 | `dream.cycle` | metacog | 21600 | cortex-orch orchestrator.py:1012 dream trigger (DREAM_LLM_ROUTE=metacog_background); actor dream.cycle (orion-dream act_emit.py) |
| `dream_preprocess` | 1 | `dream.cycle` | metacog | 21600 | legacy dream verb, denied in verbs/active.yaml; no live dispatcher |
| `dream_simple` | 1 | `dream.cycle` | metacog | 21600 | legacy dream verb, denied in verbs/active.yaml; no live dispatcher |
| `skills.runtime.builder_prune.v1` | 1 | `execution_dispatch.maintain` | (none) | 21600 | execution_dispatch_policy.v1.yaml maintain -> Orion's own disk upkeep |
| `skills.runtime.docker_prune_stopped_containers.v1` | 1 | `execution_dispatch.maintain` | (none) | 21600 | execution_dispatch_policy.v1.yaml prune_stopped_containers |
| `skills.runtime.image_prune.v1` | 1 | `execution_dispatch.maintain` | (none) | 21600 | execution_dispatch_policy.v1.yaml prune_dangling_images |
| `memory_graph_suggest` | 1 | `memory.consolidation` | fast | 3600 | orion/memory_graph/suggest_runner.py:68 used by orion-memory-consolidation (autonomous); also Hub memory UI api_routes.py:3080; executor.py default route quick |
| `log_collapse_mirror` | 1 | `metacog.log` | metacog | 3600 | only in decision_router.py ORCH_INTERNAL_DENY; no live dispatcher found |
| `log_orion_metacognition` | 1 | `metacog.log` | metacog | 3600 | cortex-orch orchestrator.py:907 metacog trigger handler; execution_lanes.py:50 background; llm_lane.py _BACKGROUND_VERBS |
| `orion_day_carry_forward_v1` | 1 | `orion_day.letter` | agent | 21600 | Hub orion_day_letter.py -> durable orion_day.letter run (ORION_DAY_LLM_ROUTE=agent), orion_day_graph.py:325 |
| `orion_day_note_v1` | 1 | `orion_day.letter` | agent | 21600 | Hub orion_day_letter.py -> durable orion_day.letter run (ORION_DAY_LLM_ROUTE=agent), orion_day_graph.py:322 |
| `self_concept_induce` | 1 | `self_study` | (none) | 21600 | cortex-exec self_study.py/self_study_harness.py source kind |
| `self_concept_reflect` | 1 | `self_study` | (none) | 21600 | self_review workflow (registry.py:160, autonomous_invocable) + self_study.py |
| `self_repo_inspect` | 1 | `self_study` | (none) | 21600 | cortex-exec self_study.py:1749/1893 source kind |
| `self_retrieve` | 1 | `self_study` | (none) | 21600 | cortex-exec verb_adapters.py adapter; self-study retrieval |
| `self_study.reflect` | 1 | `self_study` | agent | 21600 | cortex-exec self_study.py -> durable self_study.reflect graph (SELF_STUDY_REFLECT_LLM_ROUTE=agent) |
| `skills.self_study.analyze.v1` | 1 | `self_study` | (none) | 21600 | execution_dispatch_policy.v1.yaml analyze_self_study_source |
| `introspect_spark` | 1 | `spark.introspect` | fast | 600 | cortex-orch execution_lanes.py:47 spark lane, executor.py default route quick; no live sender found in this tree |
| `concept_induction` | 2 | `concept_induction` | fast | 86400 | no dispatcher found (the string is a producer id in chat_stance/self_study_harness); category spark |
| `concept_induction_journal_synthesize` | 2 | `concept_induction` | agent | 86400 | concept_induction_pass workflow (registry.py:166, autonomous_invocable), workflow_runtime.py:2993 |
| `skills.perception.look_at_camera.v1` | 2 | `execution_dispatch.inspect` | (none) | 300 | execution_dispatch_policy.v1.yaml observe_tension_via_camera (Orion's chosen look); also a turn tool; reads the vision window, no pool GPU |
| `substrate.inspect` | 2 | `execution_dispatch.substrate` | fast | 1800 | execution_dispatch_policy.v1.yaml proposal_kind inspect; no route stamped -> gateway default quick |
| `substrate.observe` | 2 | `execution_dispatch.substrate` | fast | 1800 | execution_dispatch_policy.v1.yaml proposal_kind observe |
| `substrate.summarize` | 2 | `execution_dispatch.substrate` | fast | 1800 | execution_dispatch_policy.v1.yaml proposal_kind summarize |
| `skills.imagination.render_scene.v1` | 2 | `thought.painting` | diffusion, metacog | 5400 | execution_dispatch_policy.v1.yaml render_scene/express, act actor thought.painting, horizon 5400 |
| `visual_context_interpret` | 2 | `thought.painting` | metacog | 5400 | orion-thought visual_chain.py:488 (route metacog), the painting chain's interpret step |
| `reverie_expectation_judge` | 2 | `thought.reverie` | metacog | 900 | orion-thought reverie.py loop (metacog route) |
| `reverie_narrate` | 2 | `thought.reverie` | metacog | 900 | orion-thought reverie.py loop (metacog route); llm_lane.py _BACKGROUND_VERBS |

### Act-ledger actors with no verb YAML

These act families exist in code, but no verb file carries them. Their work does not enter cortex through a YAML verb today, so C3 cannot join them by verb family yet:

- `durable.curiosity_investigate`, `durable.self_inquiry`, `durable.self_sense_eval`: durable runs drive a Hub harness turn (`runner._run_turn`), not a named verb.
- `durable.journal_compose`: uses `journal.compose`, whose family is `actions.journal` (see below).
- `hub.world_pulse_read`, `hub.retro_leisure`: their dispatch routes name code-registered verbs (`orion.world_pulse_read.pull.v1`, `orion.retro_leisure.start.v1`) with no YAML.
- `topic_foundry.run`, `hub.outreach`, `hub.retro_coplay`: no verb exists yet. The spec plans `topic_foundry.run` and `outreach.compose`.

### Verbs I could not classify confidently

- **`journal.compose`**: two actors compose through it. orion-actions runs it in-process (`actions.journal`, the common path). Durable `journal.compose` runs use it too (`durable.journal_compose`, e.g. world_pulse_digest). The family names the first. C3 must take the actor from the caller. A YAML comment records this.
- **`self_study` family (6 verbs)**: tier 1, to agree with `services/orion-durable-runs/app/run_acts.py:111-114`. That file calls `self_study.reflect` upkeep (Organ 3b Tier 1). The task brief listed self-study among Orion-chosen acts, so Juniper may want tier 2.
- **`memory_graph_suggest`**: two callers. Autonomous memory-consolidation runs it (`orion/memory_graph/suggest_runner.py:68`), and so does Juniper's Hub memory UI. It is tier 1, as the spec table says.
- **No live dispatcher found**: `introspect_spark`, `log_collapse_mirror`, `concept_induction`, `dream_preprocess`, `dream_simple` (denied in `active.yaml`), the 6 `perceive_*` verbs, `rdf_build`, `tag_enrich`, `synthesize_patterns`. Each is classified by the role it would play if invoked.
- **Biometrics and periodic health skills** (`skills.biometrics.snapshot.v1`, `skills.gpu.nvidia_smi_snapshot.v1`, `skills.docker.ps_status.v1`, `skills.system.notify_chat_message.v1`): tier 0 `autonomic`, not tier 1. This deviates from the spec's tier table, which names biometrics as tier 1. They use no LLM and no pool GPU, and the orion-actions periodic scheduler runs them, so there is nothing for them to bid with.
- **`curiosity_hop_reading`**: tier 1 grading upkeep. It grades curiosity runs. Today only an operator script calls it.
- **Windows**: every non-zero `deferrable_window_s` is an authored prior, not a measurement. Painting uses its 5400 s act horizon. Daily jobs use 6 h, digests 24 h, reverie 15 min. C3 should calibrate them from the act ledger before pricing on them.

## Tests run

```text
python scripts/check_verb_physiology.py --summary        -> OK, 106 verbs
pytest tests/test_check_verb_physiology.py -q            -> 27 passed
  (missing block, invalid tier x6, unknown lane, valid block, class-as-lane, empty lanes,
   every field required x4, bad window x3, unknown field, tier0 family rule, tier0 window rule,
   turn family on tier 2, non-mapping block, real repo passes, families read from the act
   producers + dispatch policy, plan_loader exposes block with plan unchanged)
orion-cortex-orch tests (repo root)    -> 31 failed / 231 passed, identical to main baseline
orion-cortex-exec verb-YAML-sensitive test files (10 files, run one by one from the service dir)
                                       -> identical results to main baseline
pytest orion/cognition/tests + verb-adjacent tests/ files -> only pre-existing failures
  (orion_cognition import errors; execution-dispatch policy assertions), unrelated to verbs
All orion-static-gates.yml steps run locally after merging main -> 38/38 PASS
check_definition_drift.py --gate       -> PASS (no re-lock needed)
```

## Evals run

```text
None. This patch is a structural contract with no behaviour to evaluate. The CI gate is the check.
Patch C3 (bidding) is where an eval belongs.
```

## Docker/build/smoke checks

```text
None. No runtime behaviour changes. cortex-orch/exec load the YAML through
build_plan_for_verb, which is unchanged.
```

## Review findings fixed

- Finding: `self_study.reflect` was on tier 2, but `run_acts.py:111-114` already calls it tier 1 upkeep.
  - Fix: all 6 `self_study` verbs moved to tier 1.
  - Evidence: `--summary` now shows tier 1 `self_study` 6.
- Finding: `journal.compose` has two act actors, and its family names only one.
  - Fix: a YAML comment and this report record that C3 must take the actor from the caller.
  - Evidence: `orion/cognition/verbs/journal.compose.yaml`.
- Finding: the family/actor test copied the actor strings, so a rename would not fail it.
  - Fix: the test now reads `DAILY_METACOG_ACTOR`/`JOURNAL_ACTOR`/`ACTOR` from the producers' source. It also reads every `act.actor` in `execution_dispatch_policy.v1.yaml` whose `cortex_verb` has a YAML.
  - Evidence: `test_families_match_act_ledger_actors`.
- Finding: biometrics tier deviated from the spec without saying so, and the windows had no stated basis.
  - Fix: both are documented here. The module docstring says windows are authored priors.
- Finding (nit): a tier-0 verb could declare a non-zero window.
  - Fix: tier 0 now requires `deferrable_window_s: 0`, with a test.
- Finding (nit): a lane name can be both a role and a class.
  - Fix: documented. A name that is a class reads as the class.
- Finding (nit): `look_at_camera`'s dispatch kind is `inspect`.
  - Fix: its family is now `execution_dispatch.inspect`.
- Finding (nit): `curiosity_hop_reading` is grading upkeep.
  - Fix: moved to tier 1.
- Not fixed (nit): a duplicate `physiology:` key in one YAML is silently last-wins (`yaml.safe_load`). Low risk, because the block is appended once per file.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low.
  - Concern: these classifications become pricing inputs in C3.
  - Mitigation: the table above cites a call site for every verb. The uncertain ones are listed. C3 should review the tier 1/2 rows before acting on them.
- Severity: low.
  - Concern: a new verb without a block fails CI.
  - Mitigation: that is the intended gate. The error message says how to fix it.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2922

🤖 Generated with [Claude Code](https://claude.com/claude-code)
