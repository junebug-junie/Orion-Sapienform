# Memory episodes, Stage 1 PR 2: the shadow memory writer

Follows #2479 (boundary fixes and the episode close event), merged 2026-10-02 05:16 UTC.

## Summary

- **Orion now writes its own memories from each finished conversation, in shadow.**
  - When a conversation episode closes, orion-durable-runs reads the whole episode word for word.
  - It holds the 27B model for one call and asks it what is worth carrying forward, written in Orion's own voice.
  - It saves the memories to new tables that nothing else reads yet.
- **Code checks every memory before it is saved.**
  - Every quote must appear word for word in the full text of the turn it cites. The 160-character trap from spec revision 1 has a test.
  - A memory labelled "Juniper said" must quote one of her messages.
  - Nothing Orion turned over on its own (reverie, curiosity) can be labelled as something Juniper said, or something the two of them discussed.
  - Memories that fail are logged and not saved.
- **The writer runs at `system` priority**, ahead of curiosity's background runs, on its own route `memory_distill`, so the GPU pool's telemetry can see it.
- **A daily report sets old against new.** orion-memory-consolidation writes a markdown file each day: for each episode, the old intake's rows next to the new memories. No notification is sent.
- **An offline eval ran the writer on 6 real past episodes on the live 27B lane.** Numbers below.

## Outcome moved

Before this PR, a "memory" was Juniper's last message copied verbatim. After it, each closed episode produces grounded, voiced memories in shadow. The live eval measured what the writer actually produces (see Evals run). Nothing Orion uses at runtime changes.

## Current architecture

- **PR #2479:** orion-memory-consolidation closes shadow episodes under Rule 3 and publishes `memory.episode.closed.v1`. Nothing consumed it.
- **orion-durable-runs** already runs admitted LangGraph workflows under GPU pool holds (`journal.compose` is the pattern). Holds could only be `background` or `urgent`, and background and system runs shared 4 driver slots.

## Architecture touched

- **orion-durable-runs:** subscribes `orion:memory:episode:closed`; new graph `memory.episode_distill`; a direct LLM gateway call under the hold; a system-priority driver slot.
- **Contracts:**
  - `DurableWorkflowV1` gains `memory.episode_distill`, with `EpisodeDistillBriefV1`;
  - `ResourceRequirementV1.priority` gains `system`;
  - new LLM output models (`EpisodeDistillationV1` and its parts).
- **Routes:** `memory_distill` in `config/gpu_pool.yaml` (`{class: agent, priority: system}`) and `orion/llm/routes.py` (accepted, display order, system-only).
- **Postgres:** new shadow tables `episode_memory`, `episode_memory_evidence`, `episode_memory_referent`, `episode_memory_event`, `memory_tension_shadow`, `episode_distill_run`.
- **orion-memory-consolidation:** daily report writer on a named volume.

## Files changed

- `services/orion-durable-runs/app/episode_distill_graph.py` (new): the graph (`load_episode -> resource_request -> resource_wait -> distill -> persist -> finish`) and `request_from_closed_event`.
- `services/orion-durable-runs/app/{admission_runtime.py,runner.py,main.py,settings.py}`:
  - graph registration and the load/persist dependencies;
  - `_call_memory_distill_llm`, a direct gateway RPC with `options.gpu_lease`, JSON output and thinking off;
  - the closed-event subscriber;
  - `MAX_CONCURRENT_SYSTEM_DRIVERS=1`;
  - settings.
- `services/orion-durable-runs/{.env_example,docker-compose.yml,README.md,requirements.txt}`: kill switch and keys; jinja2 for the `.j2` prompt.
- `orion/cognition/prompts/memory_episode_distill.j2` (new): the distiller prompt (v2 in code, `MEMORY_EPISODE_DISTILL_PROMPT_VERSION`).
- `orion/memory/episode/{validate.py,distill.py,store.py,report.py}` (new): the deterministic checks, prompt rendering and JSON parsing, the idempotent shadow writer, and the report renderer.
- `orion/schemas/{memory_episode.py,durable_run.py,resource_admission.py}`, `orion/llm/routes.py`, `config/gpu_pool.yaml`, `orion/bus/channels.yaml`, `config/metrics/metric_definitions.lock.json`.
- `services/orion-sql-db/manual_migration_episode_memory_v1.sql` (new).
- `services/orion-memory-consolidation/app/{episode_report.py,main.py,settings.py,episode_shadow.py}`, `.env_example`, `docker-compose.yml`: the daily report. `episode_shadow` now shares one command detector with the distiller.
- `services/orion-memory-consolidation/evals/run_episode_distill_eval.py` (new) and `evals/results/2026-10-02-episode-distill-eval-summary.json` (counts only).
- Tests:
  - `orion/memory/episode/tests/*`;
  - `services/orion-durable-runs/tests/test_episode_distill_graph.py`;
  - a system-priority driver test in `test_urgent_preemption.py`;
  - `services/orion-memory-consolidation/tests/test_episode_report_pg.py`.
- Fixtures: the route golden fixture `orion/gpu_pool/tests/fixtures_routes_compat_golden.json` (additions only) and the gateway models-list test.

## Schema / bus / API changes

- **Added:**
  - workflow `memory.episode_distill` and brief `EpisodeDistillBriefV1`;
  - priority value `system` on `ResourceRequirementV1`;
  - route `memory_distill`;
  - orion-durable-runs as a consumer of `orion:memory:episode:closed`, a producer on `orion:exec:request:LLMGatewayService`, and a consumer of `orion:exec:result:LLMGatewayService:*`.
- **Behavior changed:**
  - durable-runs now drives at most one `system` run outside the background cap of 4;
  - urgent runs still go first.
- **Compatibility:** additive values on `extra="forbid"` / `Literal` contracts.
  - orion-sql-writer validates every `DurableRunStateV1` row, so it must run this build before durable-runs emits `memory.episode_distill` states.
  - Hub, orion-actions, cortex-orch and cortex-exec also parse durable state. An old build logs a validation error on these rows; whether each one drops them quietly is UNVERIFIED.
- **Not added (deviation):** `memory.episode.distilled.v1`. It has no consumer in Stage 1 and would fail the orphan-metric ratchet. `episode_distill_run` plus the run's state rows are the trace.

## Env/config changes

- **Added** (orion-durable-runs):
  - `MEMORY_EPISODE_WRITER_ENABLED=true` (kill switch);
  - `CHANNEL_MEMORY_EPISODE_CLOSED`;
  - `MEMORY_EPISODE_DISTILL_ROUTE=memory_distill`;
  - `MEMORY_EPISODE_DISTILL_TIMEOUT_SEC=600.0`;
  - `MEMORY_EPISODE_DISTILL_MAX_TOKENS=4096`;
  - `MEMORY_EPISODE_DISTILL_DEADLINE_HOURS=20.0`;
  - `CHANNEL_LLM_INTAKE`;
  - `MEMORY_EPISODE_RECONCILE_INTERVAL_SEC=900.0`, `MEMORY_EPISODE_DISTILL_MAX_ATTEMPTS=3` (review: reconciler).
- **Added** (orion-memory-consolidation): `MEMORY_EPISODE_REPORT_ENABLED=true`, `MEMORY_EPISODE_REPORT_DIR=/data/memory-episode-reports`, `MEMORY_EPISODE_REPORT_TZ=America/Denver`.
- `.env_example` and compose updated for both services.
- **Local `.env` synced** with `python scripts/sync_local_env_from_example.py --all-keys orion-durable-runs orion-memory-consolidation` (written to the primary checkout). Pre-existing divergences (`POSTGRES_URI`, `DURABLE_RUNS_GRAPH_HOST`, `CONCEPT_RELATION_RESOLUTION_ENABLED`) were not touched.
- **Deviation:** the spec put `MEMORY_EPISODE_WRITER_ENABLED` in orion-memory-consolidation. It lives in orion-durable-runs, where the writer actually runs.
- **Deviation:** `MEMORY_EPISODE_SHADOW_COMPARE_ROUTE` (the live 8B comparison) was not added. The 8B comparison runs in the offline eval only.

## Tests run

```text
pytest orion/memory/episode/tests orion/llm/tests orion/gpu_pool/tests orion/situational/tests \
       services/orion-memory-consolidation/tests services/orion-memory-consolidation/evals
  (ORION_MEMORY_EPISODE_TEST_DATABASE_URL -> disposable postgres:16-alpine)       -> 761 passed (after merging main)
PYTHONPATH=. pytest services/orion-durable-runs/tests  -> 260 passed, 71 skipped (main: 253 passed, 71 skipped)
pytest services/orion-llm-gateway/tests               -> 384 passed
pytest services/orion-gpu-pool/tests                  -> 147 passed, 15 skipped
Postgres-backed (skipped in CI without the env var): store persist + replay, the load SQL on full text,
  the daily report, the PR 1 boundary tests.
Static gates: check_metric_lineage --gate PASS; check_definition_drift --gate PASS after re-lock
  (2 HIGH routing_changed: orion-durable-runs added as producer of orion:exec:request:LLMGatewayService
  and consumer of orion:exec:result:LLMGatewayService:* -- the real contract change);
  check_chat_route_poachers, check_circe_worker_refs, check_env_template_parity, check_service_hostname_refs,
  check_compose_no_relative_mounts, check_async_routes_not_blocking, check_inner_state_registry: pass.
check_service_env_compose_parity orion-durable-runs: OK. orion-memory-consolidation: 14 keys missing,
  the same 14 as on main; none of this PR's keys.
Pre-existing failure, also on main: tests/test_cortex_route_resolution.py::test_live_config_declares_exactly_the_shipped_template_routes.
```

## Evals run

Offline, on the live 27B lane through the gateway (route `agent`, the same class as `memory_distill`; the new route only exists once this deploys). Episodes: the Austin morning plus the five most recent other Rule 3 episodes with at least 3 content turns. Two 27B runs per episode, one 8B run (`quick_background`) for comparison. Turn text was read read-only from Postgres. Private output stayed under `/tmp`. The committed summary is counts only: `services/orion-memory-consolidation/evals/results/2026-10-02-episode-distill-eval-summary.json`.

```text
python services/orion-memory-consolidation/evals/run_episode_distill_eval.py --out /tmp/...   (shipped prompt v2)

27B (11 successful runs of 12; 1 failed with gpu_pool_unavailable:deadline while the lane was busy)
  memories proposed 75, kept 70, rejected 5 (all no_verified_quote), voice downgrades 0
  juniper_said grounded in a verified prompt quote:   37/37   (acceptance 4: 100%)   PASS
  statements naming "Orion" (third person):            0/70   (first prompt: 7/58)
  voices: juniper_said 37, orion_thought 27, worked_out_together 6; every channel "chat"
  coverage of non-command turns: mean 0.81, min 0.50; 5 of 11 runs >= 0.80   (acceptance 6) NOT MET per run
  junk after validation: 0 statements <= 5 words, 0 duplicates, 0 command-only   (acceptance 7) PASS
  about_juniper with a word found in none of its quotes: 11/11 -> all forced high   (acceptance 5) see finding 2
  referent-set Jaccard between two runs: 0.875, 0.5, 0.625, 0.8, 0.25, n/a   (acceptance 8, target >= 0.8) NOT MET
  tokens per episode: 2.2k-3.4k in, 1.7k-3.1k out; model latency p50 107 s, max 267 s
Austin property checks (two runs):
  event referent with alias "austin"          yes, yes
  juniper_said with verified "introvert" quote yes, no
  follow_up due on/after 2026-09-30            yes, yes
  no memory resting only on command turns      yes, yes
8B (quick_background): 5 of 6 runs returned invalid JSON; the one valid run kept 3 memories, coverage 0.33.
  On these numbers the 27B is better on grounding and coverage (acceptance 9); the 8B mostly cannot answer.
Hold wait p95 under 2 h (acceptance 10): UNVERIFIED (needs the deployed graph).
Report renders 7 consecutive days (acceptance 11): UNVERIFIED (needs deploy); renders in a Postgres test.
```

**What the 5 rejections were:** quotes the model stitched together with "..." from distant parts of a turn, and one quote that shares only 5 characters with the turn it cites. Strict verification caught real problems on real data.

## Docker/build/smoke checks

```text
docker compose --env-file .env --env-file services/<svc>/.env -f services/<svc>/docker-compose.yml config
  orion-durable-runs: renders MEMORY_EPISODE_WRITER_ENABLED, MEMORY_EPISODE_DISTILL_ROUTE=memory_distill, ...
  orion-memory-consolidation: renders MEMORY_EPISODE_REPORT_DIR and the memory-episode-reports volume
Not built or deployed (instructed). Both migrations applied cleanly to a disposable Postgres in tests.
```

## Review findings fixed

Code review of Stage 1, 2026-10-02. Each fix has a regression test that fails on the code before it.

- **BLOCKER: source monitoring went the wrong way.**
  - **Finding:** `orion_read` / `orion_self_knowledge` were upgraded to `juniper_said` when only a Juniper prompt quote verified. The internal-channel check ran on the claimed voice, the channel was forced to `chat`, and quotes had no minimum length. The reported repro (a reverie "I dreamed that Juniper secretly wants to leave her job soon." with quote "I") was kept as `('juniper_said','chat')`.
  - **Fix:** voices only move away from Juniper's (worked_out_together -> juniper_said -> orion_thought). `orion_read` / `orion_self_knowledge` become `orion_thought` (Orion's reply quoted) or are rejected (`voice_unsupported_by_evidence`). The internal-channel check runs on the claimed voice AND on the final voice and channel. The channel is never rewritten. Quotes need ≥3 words, or ≥15 characters in scripts written without spaces.
  - **Evidence:** `orion/memory/episode/tests/test_validate_source_monitoring.py`: the exact repro, plus a 5 voices × 8 channels × 3 evidence-role matrix (120 cases). 37 tests failed before the fix; all pass after.
- **SHOULD 1: the novel-word stakes backstop was a word list.**
  - **Fix:** removed (no `intake_junk` / `_STOPWORDS` import). The self-conclusion upgrade is removed too. Stakes are the distiller's own field; `high` only means `pending_confirmation`. **Juniper is deciding the stakes policy.**
  - **Evidence:** `test_stakes_are_the_distillers_own`, `test_no_word_list_is_imported`.
- **SHOULD 2: fold quotes before matching.**
  - **Fix:** NFKC, curly→straight quotes, dash variants, casefold and whitespace collapse, on both sides.
  - **Evidence:** `test_quotes_are_folded_on_both_sides` (`don't` vs `don’t`, em/en dash, NBSP, full-width).
  - **Eval:** re-validating the same 70 saved model answers gives 70 kept, 5 rejected (all `no_verified_quote`), identical to before. Folding rescued none of the five: they were stitched with "..." or invented (one shares 5 characters with its turn). One memory moved worked_out_together → orion_thought (its prompt quote was under 3 words). Juniper voice on an internal channel: 0. LIVE_RERUN_PLACEHOLDER
- **SHOULD 3a: Fix 2's live effect, stated honestly.**
  - **Fix:** a correction section in `docs/superpowers/pr-reports/2026-10-02-memory-episode-boundary-pr.md`.
  - **What it says:** Fix 2 changes live inputs: the novelty and significance the crystallization gate reads (`consolidation_gate.py`), the provenance values `intake_consolidation_window.py` copies, the `chat_history_log` scores, and the turn-change signal count.
  - **Evidence:** measured on 107 closing turns, mean novelty 0.930 vs 0.651 and significance 0.472 vs 0.554. Live window closing is unchanged.
- **SHOULD 3b: sql-writer double publish at the source.** Separate PR #2487:
  - every subscriber enumerated; none relies on two copies;
  - the guards listed, with which can be retired;
  - the turn envelope (with `spark_meta`) wins over the row read-back.
- **SHOULD 4: lost close events and failed runs.**
  - **Fix:** `services/orion-durable-runs/app/episode_distill_reconcile.py`, a 15 min loop. Closed direct episodes with no `episode_distill_run` (15 min grace, 72 h lookback, 10 per pass) are resubmitted as ordinary admitted durable runs:
    - the base id when there was no attempt (lost event);
    - `memdistill-<ep>-a<N>` when every attempt failed and the latest is over 1 h old;
    - at most `MEMORY_EPISODE_DISTILL_MAX_ATTEMPTS=3`.
  - No side queue; idempotent run ids.
  - **Evidence:** `test_episode_distill_reconcile.py`: a decision matrix plus a Postgres pass (lost → base, failed → a2; busy, done, fresh and skipped untouched; a second pass submits nothing).
- **SHOULD 5: pin to the 27B on gpu1.**
  - **Fix:** a dedicated pool class `memory_distill: {roles: [agent]}` (gpu1 only); route `{class: memory_distill, priority: system}`.
  - **Evidence:** `orion/gpu_pool/tests/test_memory_distill_placement.py`, which failed on the old config. In the golden route view, only the `memory_distill` entries change (lent-chat and gpu2-spill states now read `down`, not chat/gpu2). `check_gpu_pool_config` ok; `check_chat_route_poachers` PASS.
- **SHOULD 6: run the Postgres-backed tests in CI.**
  - **Fix:** `.github/workflows/orion-memory-episode-tests.yml` with a Postgres 16 service and `ORION_MEMORY_EPISODE_TEST_DATABASE_URL`. It fails if any test reports SKIPPED. Locally: 438 + 14 passed, 0 skipped.
- **SHOULD 7: the deploy-order comment was backwards.**
  - **Fix:** `orion/schemas/durable_run.py` now says consumer-first: sql-writer, gateway and gpu-pool BEFORE durable-runs.
  - **Evidence:** `test_deploy_order_note.py`.
- **Nit, daily report:**
  - **Fix:** yesterday's file is rewritten each pass until every closed episode has a distill run, or 12 h after local midnight; then it is marked final.
  - **Evidence:** `test_report_is_rewritten_until_every_episode_is_distilled`, which failed before.
- **Nit, evidence primary key:**
  - **Fix:** keyed on `quote_sha256`, with the full quote stored.
  - **Evidence:** `test_a_3kb_quote_is_stored`. Before: `index row size 3952 exceeds btree version 4 maximum 2704`.
- **Nit, #2479 migration:**
  - **Fix:** the GIN index moves to `manual_migration_memory_episode_v1_gin.sql` (`CREATE INDEX CONCURRENTLY`, outside a transaction). Rollback scripts were added for both migrations.
  - **Evidence:** `test_episode_migrations_pg.py` (apply → rollback → re-apply).
- **Nit, outreach stamp:**
  - **Fix:** removed the `client_meta.conversation_phase` write; nothing read it.
  - **Evidence:** `test_outreach_writes_no_conversation_phase_nobody_reads`.

## Restart required

Deploy order (consumer-first; each step after the one before):
1. #2479's migration, then its GIN index:
   - `manual_migration_memory_episode_v1.sql`;
   - `manual_migration_memory_episode_v1_gin.sql`, run with plain `psql -f` (outside a transaction).
2. `manual_migration_episode_memory_v1.sql` (edited in this PR: `quote_sha256` key; not yet applied anywhere).
3. orion-sql-writer. With #2487 merged, deploy that build; it is independent but natural here.
4. orion-gpu-pool and orion-llm-gateway (the `memory_distill` class and route).
5. orion-durable-runs (the subscriber, the graph and the reconciler).
6. orion-memory-consolidation (#2479's shadow plus the daily report).
7. orion-hub (#2479's stamp; the outreach stamp is removed).
8. When convenient, rebuild orion-actions, orion-cortex-orch and orion-cortex-exec (they parse durable state rows).

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_memory_episode_v1.sql
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_memory_episode_v1_gin.sql
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_episode_memory_v1.sql
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-memory-consolidation up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Rollback:
- `MEMORY_EPISODE_WRITER_ENABLED=false` and `MEMORY_EPISODE_SHADOW_ENABLED=false`;
- then `manual_migration_episode_memory_v1_rollback.sql` and `manual_migration_memory_episode_v1_rollback.sql`.

## Risks / concerns

- **Severity:** (resolved) **Was:** the spec's novel-word backstop pushed every `about_juniper` memory to high stakes; removed in review. The old text follows for the record: 11 of 11 had a paraphrase word not in their quote ("told", "draining" vs "drains"). At Stage 3 that would ask Juniper to confirm every self-description. **Mitigation:** shadow only; it is reported here. Proposed fix for Juniper to decide: compare against the full text of the cited turns, not only the quote, or drop the backstop and rely on the model's stakes plus the stakes floor.
- **Severity:** medium. **Concern:** run-to-run agreement on what a memory is about is low (referent Jaccard 0.25-0.875, target ≥ 0.8) at temperature 0.2. **Mitigation:** shadow only. Options: temperature 0, or the candidate referent list (empty in this eval because the shadow tables were empty).
- **Severity:** medium. **Concern:** a run takes ~1.5-4.5 min of agent-lane time, about twice the spec's ~45 s estimate. At 1-2 episodes a day, that is ~3-9 min a day at system priority. **Mitigation:** one system driver at a time; the kill switch is `MEMORY_EPISODE_WRITER_ENABLED=false`.
- **Severity:** low. **Concern:** the self-description ("introvert") was kept as its own memory in 1 of 2 Austin runs. **Mitigation:** reported. The daily report will show the live rate.
- **Severity:** low. **Concern:** `system` is a new value on `ResourceRequirementV1.priority`. An older build that parses stored requests rejects that row. **Mitigation:** only durable-runs submits it; follow the deploy order.
- **Severity:** low. **Concern:** the daily report contains Juniper's words. **Mitigation:** it lives on a named Docker volume, is never committed or published, and the code says so.
- **Deviations from the spec:**
  - no Hub report page (markdown only, which the spec allows: "a Hub page and/or markdown");
  - no `memory.episode.distilled.v1` event (no consumer yet);
  - no crosswalk or Graphiti projection (Stage 2);
  - no live 8B comparison route (eval only);
  - the kill switch lives in durable-runs;
  - Juniper-voiced memories in a chat episode are forced to channel `chat`.
- **UNVERIFIED:**
  - the graph on the live bus;
  - the hold on the live pool at system priority;
  - the daily report on real days;
  - whether Hub, actions and orch drop unknown-workflow state rows quietly.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2484

🤖 Generated with [Claude Code](https://claude.com/claude-code)
