## Summary

The chat prompt's DURABLE MEMORY section was mostly noise. This PR cuts four noise sources from the chat recall path. Real recent Hub chat with Juniper still provides continuity.

- AI Town NPC chatter no longer enters chat continuity. The chat recall scans the Hub chat table only.
- Bus hop-latency z-scores ("hub -> cortex-exec hop latency |z|=58.9") are no longer fetched for chat.
- The `active_packet` "durable beliefs" collector is deleted. It showed the same few stale user requests on almost every turn.
- Topic-foundry concepts ("sub-concept-topicfoundry-... associated_with sub-concept-seed-orion") are dropped from the concept-region collector.
- An empty continuity or belief digest now renders as nothing, not as a bare header.

Fixes #3461.

## Outcome moved

Replay of the 20 most recent real Hub chat queries through both chat recall phases, live Postgres and FalkorDB, read-only (writes stubbed), same 900-minute continuity window for both runs:

| Kind | Before (lines) | After (lines) |
|---|---|---|
| `active_packet` | 68 (in 20 of 20 turns) | 0 |
| `bus_synaptic_anomaly` | 12 (in 11 of 20 turns) | 0 |
| `concept_region` topic-foundry | 3 (in 1 of 20 turns) | 0 |
| AI Town (`sql_chat:aitown_chat_history_log`) | 0 in this replay (see below) | 0 |
| **Noise total** | **83** | **0** |
| Real Hub chat (`sql_chat:chat_history_log`) | 80 (20 of 20 turns) | 80 (20 of 20 turns) |
| `falkor_chat` chat turns | 28 | 28 |
| `concept_region` (non topic-foundry) | 3 | 4 |
| Turns with an empty DURABLE MEMORY | 0 | 0 |

AI Town did not reach the top 6 in this replay because Hub turns were newer. Direct live check of the same fetch over 24 hours, top 50 rows: with the AI Town union, 40 of 50 rows were AI Town; without it, 17 of 17 rows were Hub chat.

Live 7-day prompt scan before the fix (1,463 chat prompts, 2026-11-08..15): `active_packet` in 1,461, bus anomaly in 889, AI Town in 263, `concept_region` in 477 (1,056 of the concept_region lines were topic-foundry ids).

## Current architecture

orion-cortex-exec `pcr_chat_memory.py` calls orion-recall twice per chat turn: `recall_phase=continuity` (profile `chat.continuity.v1`) and `recall_phase=purposeful` (profile `chat.belief.<intent>.v1`). Their rendered digests become DURABLE MEMORY in the prompt.

- Continuity: the `sql_chat` feed unioned `chat_history_log` with `aitown_chat_history_log`. The `bus_synaptic_anomaly` feed ran on every recall whose profile did not disable it; no chat profile disabled it.
- Purposeful: the PCR collectors `active_packet` (memory crystallizations, with Graphiti search on the contradiction intent) and `concept_region` (substrate concept graph, label match plus edges). `active_packet` ranked first in the belief digest. Every live turn derived intent `open_loop` (#3486).

## Architecture touched

- orion-recall only. One shared seam: `orion/recall/profiles/chat.belief.*.yaml` (the dead `active_packet` weight is removed).
- New `pcr_collectors.apply_chat_memory_exclusions()`, applied in `worker.process_recall` to both PCR phases before any fetch.
- `sql_chat.fetch_chat_history_pairs(include_aitown=...)`, default `True` for other profiles.
- `concept_region` filters `sub-concept-topicfoundry-*` nodes and the edges that touch them, before render and before the activation reinforcement write.
- Not touched: `orion/harness/prefix.py`, `orion/memory/pages/` (#3460 in flight), any table or stored data. Core pages (RELATIONSHIP from `page:juniper`) are a separate prompt slot and are unaffected.

## Files changed

- `services/orion-recall/app/pcr_collectors.py`: intent plans without `active_packet`/`graphiti`; new `apply_chat_memory_exclusions`.
- `services/orion-recall/app/worker.py`: apply exclusions for both PCR phases; pass `include_aitown`; drop the `active_packet` collector unit and the `active_packet_refs` debug field.
- `services/orion-recall/app/sql_chat.py`: `include_aitown` parameter.
- `services/orion-recall/app/collectors/concept_region.py`: topic-foundry filter.
- `services/orion-recall/app/collectors/active_packet.py`: deleted.
- `services/orion-recall/app/fusion.py`: `active_packet` rank removed; empty digests render "".
- `services/orion-recall/app/settings.py`, `.env_example`, `docker-compose.yml`: removed keys only `active_packet` read.
- `orion/recall/profiles/chat.belief.*.yaml`: removed `active_packet` backend weight.
- `scripts/sync_local_env_from_example.py`, `scripts/report_dead_env_keys.py`: removed the two retired `RECALL_GRAPHITI_*` keys from the never-sync list.
- `services/orion-recall/README.md`: new section 16; stale `active_packet` docs removed.
- Tests: new `tests/test_chat_durable_memory_noise_retired.py`; updated `test_pcr_belief.py`, `test_pcr_purposeful_entity_injection.py`, `test_recall_pcr_block_timing.py`, `test_concept_region_wiring.py`, `test_active_packet_activation_floor.py` (library tests kept), `test_docker_compose_numeric_defaults.py`, `tests/scripts/test_sync_local_env_from_example.py`; deleted `test_active_packet_retrieval_event_logging.py`.
- Eval: new `evals/run_chat_durable_memory_noise_eval.py` + `evals/test_chat_durable_memory_noise_eval.py`.

## Schema / bus / API changes

- Added: none.
- Removed: `recall_debug.pcr.active_packet_refs` (no consumer found in the repo).
- Renamed: none.
- Behavior changed: chat continuity and belief digests no longer carry the four sources; empty digests are "" instead of a header line.
- Compatibility notes: no bus channel or schema registry change. Hub's `/api/memory/active-packet` route and the `orion.memory.crystallization` library are unchanged.

## Env/config changes

- Added keys: none.
- Removed keys (orion-recall): `RECALL_ACTIVE_PACKET_ENABLED`, `RECALL_GRAPHITI_IN_CHAT`, `RECALL_GRAPHITI_ADAPTER_URL`, `RECALL_GRAPHITI_FALKORDB_URI`, `RECALL_GRAPHITI_TIMEOUT_SEC`, `RECALL_CRYSTALLIZATION_VECTOR_COLLECTION`, `RECALL_CARDS_EMBEDDING_URL`.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (nothing to add). The sync script does not delete keys, so the live `services/orion-recall/.env` still lists the removed keys. They are ignored (`extra="ignore"` in settings). Deleting them by hand is optional cleanup.
- Skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. python -m pytest services/orion-recall/tests services/orion-recall/evals -q
  384 passed, 2 failed
  The 2 failures also fail on origin/main before this change:
    test_recall_policy_harness.py::test_process_recall_diagnostic_contains_gating_suppression_and_selection
    test_recall_vector_amputation.py::test_worker_and_recall_v2_import_without_vector_adapter
New regression tests: 10 passed. Mutation check: with apply_chat_memory_exclusions removed from worker.py, 2 of them fail.
python -m pytest tests/scripts/test_sync_local_env_from_example.py scripts/tests/test_check_env_template_parity.py tests/test_report_dead_env_keys.py tests/test_check_service_env_compose_parity.py tests/test_no_personal_terms.py tests/test_check_settings_defaults.py -q
  97 passed
python scripts/check_env_template_parity.py   PASS
python scripts/check_env_key_single_source.py OK
python scripts/check_metric_lineage.py --gate PASS
python scripts/check_definition_drift.py --gate PASS
git diff --check                               clean
```

## Evals run

```text
services/orion-recall/evals/run_chat_durable_memory_noise_eval.py, run inside the orion-athena-recall
container (cwd /app, live .env) against code copies under /tmp (removed afterwards), 20 real Hub queries,
--continuity-minutes 900, writes stubbed (no concept reinforcement, no crystallization boost or retrieval event):
  before (origin/main): noise_lines_total=83, sql_chat:chat_history_log=80
  after  (this branch): noise_lines_total=0,  sql_chat:chat_history_log=80
Query text is private chat and is not stored in the repo; the script prints counts only.
```

## Docker/build/smoke checks

```text
No image build run. The replay above imported both code trees inside the running recall container,
which proves the code paths against live Postgres and FalkorDB. Live chat-prompt proof is UNVERIFIED
until orion-recall is rebuilt from main.
```

## Review findings fixed

Review ran in a subagent. It found no must-fix issues. Its own mutation checks confirmed each new test fails when its fix is removed.

- Finding: the `timings_ms` contract description still listed `pcr_active_packet`.
  - Fix: removed it from `orion/core/contracts/recall.py`.
  - Evidence: `tests/test_recall_pcr_contract.py` passes.
- Finding: README lines still described `pcr_active_packet` and the `RECALL_GRAPHITI_*` keys as live.
  - Fix: rewritten as retired (#3461). Section 16 now says the exclusions apply only under `RECALL_PCR_ENABLED`.
  - Evidence: diff.
- Finding: a header could still render over an empty body when the render budget dropped every item.
  - Fix: both digests now check the rendered body, not the item list.
  - Evidence: recall suite 384 passed (same 2 pre-existing failures).
- Finding: the new chat-phase test loop could pass when sql_chat was never called.
  - Fix: added an explicit guard for the continuity phase. The purposeful plan disables sql_chat.
  - Evidence: test passes.
- Finding: two `concept_region.py` docstrings pointed at the deleted file.
  - Fix: reworded.
  - Evidence: diff.
- Finding (deferred): `seed_crystallization_id` has no reader left in recall. cortex-exec still sends it and uses it for its own intent rule.
  - Fix: added to #3487, so it is decided together with the stale count gate.
- Finding (deferred): the topic-foundry filter runs after the 500-row Falkor read budget.
  - Fix: filed #3488 (push the exclusion into the query). This is not a regression.
- Finding: stale keys remain in the live `.env`.
  - Fix: listed under Env/config changes as optional cleanup.

## Restart required

```bash
scripts/safe_docker_build.sh orion-recall up -d --build
```

Run from the clean primary `main` checkout after merge. Then check the next chat prompt:
`docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select raw->>'prompt' from cockpit_turn_sighting where stage='motor_boot' order by created_at desc limit 1"` must show no `[active_packet]`, `[bus_synaptic_anomaly`, `aitown_chat_history_log`, or `topicfoundry` lines.

## Risks / concerns

- Severity: medium. Concern: the spec's retirement table says retire `active_packet` only after `page:lately` is live. `page:lately` does not exist yet. Mitigation: live data shows `active_packet` only ever rendered the `open_loops` bucket (every turn's intent was `open_loop`, #3486): the same few stale user requests. Removing it loses no real content. Stored crystallizations are untouched; nothing retires them automatically (`should_retire` is display-only in Hub).
- Severity: low. Concern: crystallizations no longer get a recall boost on each chat turn, so their activation decays. Mitigation: no reaper acts on it; Hub's retirement-candidate column may show more rows.
- Severity: low. Concern: Graphiti's only chat read rode on `active_packet` (contradiction intent, which never fired live). Mitigation: the spec plans Graphiti as the fact layer under pages, a different reader.
- Severity: low. Concern: `concept_region` still renders seed and news concepts and raw-id edges between non-topic-foundry nodes. Mitigation: out of scope for #3461; none of the remaining lines were noise kinds named in the issue.
- Issues opened: #3486 (intent is open_loop on every turn), #3487 (eligible_belief_count gate and seed_crystallization_id are stale after this retirement), #3488 (topic-foundry rows still use the concept_region read budget).
- Issues closed: #3461 (on merge; live proof after deploy).

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
