## Summary

Lane C of "Orion plays Zelda" (spec `docs/superpowers/specs/2026-10-22-orion-plays-zelda-design.md`, PR #2846): sections E (global contract) and F (memory export), the proposal-mode record, acceptance checks 5 (allocator side) and 6.

- Orion can now *choose* to play. A new act verb, `play`, has a dispatch route, `retro_leisure_session`. The motor allocator admits or refuses it like any other act, and a refusal is stamped `allocator:<reason>` on the dispatch frame.
- Hub consumes the admission. It asks Lane B's leisure runner for its gate verdict, starts the session, and opens a `play` act. When the session ends, Hub exports it.
- One finished session becomes one memory, built by fixed rules with no model writing it:
  - The summary holds facts and counts only, such as "Played The Legend of Zelda solo for 22 min, 9 new rooms, died twice, 14 of 31 predictions met, 5 unknown."
  - It says "played together" when a human drove more than half of the frames.
  - It says "with a human" whenever the share is not known.
  - The session becomes one `memory_cards` row (provenance `retro_play`) and one `retro_play_sessions` row.
- The same day, the Temporal Self chronicle reads that row as source `retro_play`, as one `play` arc. The chat stance cue then says "Played The Legend of Zelda solo, 14 of 31 predictions met (10:02 to 10:24) [arc:...]".
- Every switch ships ON.

**Dependency order: A (#2851) merges first, then B (#2850), then C (this PR).** Without Lane B's `scripts/retro_leisure.py`, every start is refused `runner_unavailable`. A fake session is never made.

## Outcome moved

- Before: Retro gave Orion's memory nothing, and the allocator scheduled no Retro act.
- After: an idle Orion can be admitted to play, under every Hub gate.
- Each session leaves one recall-able card and one same-day arc.
- The play act is graded on whether the session produced a summary with at least one real planner decision.

## Current architecture

- ActVerb had 11 values, with no `play`.
- Act-backed routes were limited to `render_scene` (painting) and `read_world_pulse` (the reading pull, which is the closest precedent: an executor key that becomes one bus RPC to Hub).
- Hub had no Retro consumer for memory.
- Temporal Self had no game source.

## Architecture touched

- **Contract (global):**
  - `ActVerb` gains `play`.
  - `MemoryProvenance` gains `retro_play`.
  - `TemporalSourceKind` gains `retro_play`.
  - `ArcKind` gains `play`.
  - New RPC `orion:retro_leisure:start:request` / `:reply:*`.
  - `ActRouteRefV1` gains `cost_source` and `cold_cost_ceiling_sec`.
- **Dispatch runtime:** a mixin, `app/retro_leisure_dispatch.py`, wired into `worker.py`.
- **Hub:**
  - `scripts/retro_leisure_listener.py` (the consumer).
  - `scripts/retro_memory_export.py` (the builder and writer).
  - The play act in `scripts/act_emit.py`.
  - Wiring in `main.py`.
- **Durable-runs:** the SQL source for `retro_play`. It is skipped via `to_regclass` while the table is missing, so it can never fail the whole read.
- **Temporal Self:**
  - Adapter, process arc, frame pin, stance cue clause, recent-arcs wording, and the Day-letter count.
- **SQL:** `services/orion-sql-db/manual_migration_retro_play_sessions_v1.sql`. It is additive (CREATE TABLE/INDEX) with `REQUIRED-BY: orion-hub, orion-durable-runs`. It was verified against the post-pull auto-apply allowlist (`auto_apply_refusals` returns []).

## Files changed

- `orion/schemas/grammar.py`, `tests/test_act_ledger_contract.py`: the `play` verb.
- `orion/schemas/retro_leisure.py`, `orion/schemas/registry.py` (both `_REGISTRY` and `SCHEMA_REGISTRY`, verified via `resolve()`), `orion/bus/channels.yaml`: the start RPC.
- `orion/schemas/retro_play.py`: the summary input and output models.
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml`, `config/proposals/proposal_policy.v1.yaml`, `tests/test_proposal_scoring.py`: the route and the no-dimension template.
- `orion/execution_dispatch/policy.py`, `orion/autonomy/act_candidates.py`: the cost-source fields.
- `services/orion-execution-dispatch-runtime/app/{retro_leisure_dispatch,worker,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`.
- `services/orion-hub/scripts/{retro_leisure_listener,retro_memory_export,act_emit,main}.py`, `app/settings.py`, `.env_example`, `README.md`.
- `orion/core/contracts/memory_cards.py`: provenance `retro_play`.
- `orion/temporal_self/{sources,arcs,frame,stance_cue,recent_arcs,day_block}.py`, `orion/schemas/temporal_self.py`, `README.md`.
- `services/orion-durable-runs/app/{temporal_self_sources,settings,main}.py`, `.env_example`, `docker-compose.yml`, `README.md`, `evals/temporal_self_fixture_db.py`.
- `orion/temporal_self/evals/{run_play_cue_eval.py,fixtures/retro_play_sessions.json,export_fixture_day.sql}`.
- `services/orion-sql-db/manual_migration_retro_play_sessions_v1.sql`.
- Tests: `tests/test_rung3_retro_leisure.py`, `services/orion-hub/tests/test_retro_{memory_export,memory_export_pg,leisure_listener}.py`, `orion/temporal_self/tests/test_play_arcs.py`, plus one case added to `services/orion-durable-runs/tests/test_temporal_self_chronicle_postgres.py`.
- CI:
  - `.github/workflows/retro-memory-tests.yml` (new, with real Postgres).
  - New steps in `thinking-rung3.yml` and `temporal-self-tests.yml`.

## Schema / bus / API changes

- **Added:**
  - `RetroLeisureStartRequestV1` / `RetroLeisureStartResultV1` (`retro_leisure.start.{request,result}.v1`).
  - `RetroPlaySessionSummaryV1` (not on the bus).
  - Table `retro_play_sessions`.
  - The literal values listed above.
- **Removed:** none.
- **Renamed:** none.
- **Behavior changed:**
  - `build_frame` always keeps today's latest play arc in `arcs_today`.
  - The stance cue's play clause ranks above the thread and sleep clauses. When the 400-char cap binds, those clauses are dropped first. This is intended: as the last clause, the play clause was dropped on both real 10-11 frames.
- **Compatibility, consumer-first:** these models forbid unknown values, so the order matters.
  - **First:** deploy orion-cortex-orch, orion-hub, orion-thought and orion-sql-writer. These are the readers of memory cards, frames and grammar acts.
  - **Then:** deploy orion-durable-runs, which folds `play` arcs.
  - **Before the first act:** Hub must not emit a `play` act until sql-writer has `play` in `ActVerb`.
  - Rows can only appear once Lane B is merged, so today's risk is low.

## Env/config changes

- Added keys, all `true`/`15` (ON):
  - `ORION_DISPATCH_RETRO_LEISURE_ENABLED`, `ORION_DISPATCH_RETRO_LEISURE_RPC_TIMEOUT_SEC` (dispatch runtime).
  - `HUB_RETRO_MEMORY_EXPORT_ENABLED` (Hub).
  - `TEMPORAL_SELF_RETRO_PLAY_ENABLED` (durable-runs).
- Removed or renamed keys: none.
- `.env_example` updated: yes, for all 3 services. The settings and compose files match.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-hub orion-execution-dispatch-runtime orion-durable-runs`: the 4 keys were added to the live main-checkout `.env` files. Dry-run first: only these 4 were missing.
- Skipped keys requiring operator action: none.
- Rollback levers:
  - `ORION_DISPATCH_RETRO_LEISURE_ENABLED=false` (the allocator refuses `retro_leisure_disabled`).
  - `RETRO_LEISURE_ENABLED=false` (Lane B).
  - `HUB_RETRO_MEMORY_EXPORT_ENABLED=false`.
  - `TEMPORAL_SELF_RETRO_PLAY_ENABLED=false`.
- Deviation from the spec: the spec names "route `enabled: false`" as a lever. I used a dispatch env kill switch instead, matching the reading pull. The policy model has no per-route `enabled` flag, and adding one would change route resolution for every route.

## Tests run

```text
pytest orion/temporal_self/tests                                   213 passed
pytest tests/test_rung3_retro_leisure.py tests/test_rung3_reading_pull.py tests/test_rung3_act_candidates.py \
  tests/test_motor_allocator.py tests/test_act_ledger_contract.py tests/test_proposal_scoring.py \
  tests/test_grammar_event_producer_catalog.py tests/test_execution_dispatch_bus_catalog.py \
  tests/test_bus_reply_channel_catalog_coverage.py                 156 passed
services/orion-hub: tests/test_retro_memory_export.py tests/test_retro_leisure_listener.py \
  tests/test_act_ledger_producers.py                               52 passed
services/orion-hub: tests/test_retro_memory_export_pg.py (throwaway postgres:16, real memory_cards.sql
  + migration + real insert_card)                                  1 passed
services/orion-durable-runs: tests (throwaway postgres:16)         562 passed, 1 skipped
services/orion-hub full suite: 36 failed / 3998 passed. The same 36 fail on origin/main in a clean
  detached worktree (diffed by name): 0 new failures.
scripts/check_migration_required_by.py   OK
scripts/check_env_template_parity.py     PASS
scripts/check_bus_reply_channels.py      22 prefixes, 0 uncovered
scripts/check_env_key_single_source.py   OK
scripts/check_inner_state_registry.py    OK
git diff --check                         clean
```

Note: `scripts/check_schema_registry.py` and `scripts/check_bus_channels.py` (named in AGENTS.md) do not exist in this repo. The registries were verified with `resolve()` and the catalog tests above.

## Evals run

```text
python orion/temporal_self/evals/run_play_cue_eval.py
  play clause survived the 400-char cap in 6 of 6 renders -- PASS
  (real 2026-10-11 "talking" and "busy" frames + 3 fixture sessions; before the priority change: 0 of 6)
python orion/temporal_self/evals/run_arc_precision_eval.py   passed: true (unchanged)
```

The eval fixture sessions are synthetic because no Zelda session has run yet. Replace them with a real export once sessions exist.

## Docker/build/smoke checks

```text
No image build or deploy (instructed: do not deploy). Postgres-backed tests ran against a throwaway local
ghcr.io/junebug-junie/postgres:16 container (stopped after).
```

## Review findings fixed

Review: orion-repo-agent, review-only, target `origin/main...HEAD`.

- **Finding (BLOCKER):** leisure summaries said "solo" with human share 0 even when Lane B reported `human_may_be_in_control`.
  - **Fix:** the flag is read. The control basis becomes `unknown`, and the summary says "with a human (control share not recorded)". The Temporal Self label says "solo" only for a leisure session with a recorded human share of 0. Optional per-port frame attribution is accepted if Lane B sends it.
  - **Evidence:** `test_lane_b_session_end_maps_onto_the_builder`, `test_label_says_who_played_from_the_control_share`.
- **Finding:** unknown decision counts were written as 0, and all-unknown verdicts read as failures.
  - **Fix:** `decisions` and `predictions_unknown` are nullable and left out of the text when not recorded. Unknown verdicts are named ("..., 5 unknown").
  - **Evidence:** `test_no_decisions_and_unknown_game_render_plainly` and related tests.
- **Finding:** `cost_source` was declared but nothing read it.
  - **Fix:** the worker branches on `ref.cost_source == "act_duration"` with generic ceiling logic.
  - **Evidence:** `test_cold_session_is_charged_the_cap_and_admitted_cold`.
- **Finding:** a Hub restart graded the act 0 with `hub_stopped`.
  - **Fix:** the act is no longer closed on cancel. It is left to the writer's expiry (see Risks).
- **Finding:** a replayed export never closed the act.
  - **Fix:** the duplicate path closes the act again, with the same stable act id.
  - **Evidence:** `test_export_claims_the_row_writes_one_card_links_it_and_closes_the_act`.
- **Finding:** an RPC timeout left the proven-path episode without its object.
  - **Fix:** the deterministic object is named on no-answer.
  - **Evidence:** `test_no_answer_still_names_the_proven_episode_object`.
- **Finding:** a second admission was possible before the open act was stored.
  - **Fix:** `started_within_sec=horizon` for act-duration routes.
  - **Evidence:** `test_a_started_session_holds_the_route_before_its_act_is_stored`.
- **Finding:** the consumer-first order for `MemoryProvenance` was not written down.
  - **Fix:** the order is now in the compatibility notes above.
- **Finding (NIT):**
  - **Fixes:** the dispatch id is passed explicitly to the export, card slugs carry a hash, an inferred share gives card confidence `likely`, and a stray blank line was removed.

## Restart required

Build and deploy from the clean primary main after A, B and then C merge, in consumer-first order. The post-pull rebuild applies the migration before the Hub and durable-runs rebuilds.

```bash
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- **Severity: medium.**
  - **Concern:** Lane B's `RetroSessionEnded` carries no atlas session counts and no per-frame control attribution.
  - **What happens now:** leisure summaries omit room and death counts. They are left out, never written as 0. The control share comes from the sponsor rule: any human acquire preempts the leisure grant and ends the session, and `human_may_be_in_control` overrides this.
  - **Mitigation:** the builder already reads optional `atlas_session` (Lane A's `zelda.atlas.session` from `GET /sessions/{id}`) and `control` fields. Follow-up for Lane B: capture both before stopping the session. Lane A's applied-input journal has no controller field per row, so frame attribution needs Lane A and Lane B work.
- **Severity: medium.**
  - **Concern:** the co-play session-end call site (`export_session(kind="co_play", ...)`) is not wired. It lives in Lane B's files.
  - **Mitigation:** follow-up for Lane B.
- **Severity: low.**
  - **Concern:** `cold_cost_ceiling_sec` (1800 s) is the one exception to "no typed fallback".
  - **Mitigation:** it is used only when the act has never closed, never on a read error. It equals the doer's enforced session cap, so a cold act is charged its worst case.
- **Severity: low.**
  - **Concern:** a Hub restart mid-session leaves the play act open, and the writer then expires it at 0. Juniper rebuilds Hub mid-game by choice (#2836), so this pulls the `play` belief down for reasons unrelated to play.
  - **Mitigation:** none in this patch. It is recorded here as a known confound.
- **Severity: low.**
  - **Concern:** `PLAY_HORIZON_SEC` (2400) assumes `RETRO_LEISURE_SESSION_MAX_SEC` is 1800.
  - **Mitigation:** re-derive it if the cap is raised.
- **UNVERIFIED:**
  - No live leisure session has run.
  - The card being cited by live chat recall ("did you play anything today?") is unproven. Recall's full-text query ANDs every term, so that exact question may not match the card. The same-day path is the stance cue.
  - The allocator admitting the route live is unproven.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
