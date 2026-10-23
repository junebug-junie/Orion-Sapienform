# Episodes from the street (walkway follow-ups spec #2763, item 1)

## Summary

- Notable things that happen outside now become episodes in Orion's day (Temporal Self arcs). Before, episodes came from the room camera only.
- "Notable" reuses gates that already exist. A walkway or doorbell motion story counts when it is published, clean, and its rules version passed Juniper's spot-check gate. A doorbell ring counts. A graded handoff counts when its p-value is under 0.05 (the street attention gate's event rule).
- Every frame, every parked car, every unpublished glimpse, every moving car, and plain Ring motion stays out.
- Each street episode is labelled "Outside (walkway camera): ..." or "Outside (doorbell camera): ...". People are only ever given as the honest range from #2781 ("people: at least 2, at most 6"), never as a bare count.
- A kill switch (ON) and a daily cap (12) live in orion-durable-runs. A log line and a stored row prove where each street episode came from.

## Outcome moved

Before: `ROOM_STREAMS = {"cam0"}`, so Orion's day had no record of anything on the street. Live read-only replay of the last 72 h through the real reader and reducer: **27 notable street events** (24 motion stories after merging walking-together groups, 3 doorbell rings), **24 street episodes** with the cap at 12 (10-20: 10, 10-21: 12 (15 seen, the cap kept 3 out), 10-22: 2). With no change, all of them were 0.

## Proposal mode (CLAUDE.md §0A: memory change)

- **Capability change:** Orion's own chronology (Temporal Self arcs, the recent-arcs block reverie reads, the Day letter arc count) now includes outside events. A street arc can also show in "Recently closed today" (only the latest street arc, so a busy walkway cannot crowd out Orion's own arcs). The chat stance cue does not render street arcs.
- **Data touched:** reads `vision_motion_story`, `vision_motion_story_verdict`, `vision_rollup`, `vision_events` (rings only), `vision_handoff_street_outbox`. Writes only the chronicle's own tables (`temporal_self_event`, `temporal_self_arc`, frames, days). No new tables, no migration.
- **Privacy boundary:** there is no boundary between Juniper and Orion. Toward the outside, the label is built from row fields only: no image, no crop, no face, no caption text. A name appears only when the story's strict naming rule already set `name_label` (Juniper-given names); a handoff's `person_guess` is marked "(a guess)". The arc is `orion_internal`, like every other non-chat arc.
- **Trace that proves it worked:** a `temporal_self_arc` row with `kind = 'street'` whose `evidence_refs` name the source row (`vision_motion_story:<id>`, `vision_events:<id>`, `vision_handoff_street_outbox:<id>`), and the chronicle log line `temporal_self_street_episode arc_id=... source=vision_motion_story:...`. When the cap holds events back: `temporal_self_street_capped count=N`.
- **Dangerous failure mode:** a passer-by read as company in the room, or as Juniper. Guarded: street arcs are a separate kind, the label always starts "Outside (", `presence_transition` and `vision_percept` stay room-only (tested), and nothing binds to a street arc by time, so room percepts and Orion's attention rows never attach to it (tested with a mutation check). Second risk: the street floods Orion's day. Guarded by the cap and the one-line limit in recent arcs.
- **Disable / roll back:** `TEMPORAL_SELF_STREET_EPISODES_ENABLED=false` in `services/orion-durable-runs/.env` and restart orion-durable-runs: the source is not read and new episodes are room-only again. Arcs already written stay (retention 90 days) and are harmless. Full rollback: revert this PR (deploy orion-durable-runs before orion-thought / orion-hub on the way back, the reverse of the rollout order).

## Current architecture

Temporal Self (`orion/temporal_self/`) is a pure reducer: one adapter per source turns a row into a `TemporalSelfEventV1`, and `arcs.fold` turns events into arcs. The orion-durable-runs chronicle (`app/temporal_self_sources.py`, `app/temporal_self_chronicle.py`) reads Postgres by window, folds, and stores arcs. Vision reached it only as room presence (`presence_transition`, `company` arcs) and room percepts (`vision_percept`, context only); both drop any stream not in `ROOM_STREAMS`.

## Architecture touched

- Contract: `TemporalSourceKind` gains `street_event`; `ArcKind` gains `street` (stored models, `_REGISTRY` only, not bus payloads). Same consumer-first rollout as `play` / `strain`.
- Reducer: `street_event` is a process kind (one closed arc per event), in `_NO_CONTEXT_KINDS` and `NO_BODY_KINDS`, capped by `ReducerConfig.street_arcs_per_day`.
- Driver: three `SourceQuery` reads of one kind; a read is skipped while any of its tables is missing (`also_requires`).

## Files changed

- `orion/schemas/temporal_self.py`: `street_event` source kind, `street` arc kind, rollout note.
- `orion/temporal_self/sources.py`: `street_event` adapter (stories, rings, handoffs), `street_story_label`.
- `orion/temporal_self/arcs.py`: process kind, lane rule, no time binding, daily cap.
- `orion/temporal_self/body.py`, `day_block.py`, `recent_arcs.py`: street is not a body window; Day letter counts it; recent arcs show only the latest street line.
- `orion/temporal_self/README.md`, `services/orion-durable-runs/README.md`: how it works, proof, kill switch.
- `orion/temporal_self/tests/test_street_arcs.py`: new gate tests.
- `services/orion-durable-runs/app/temporal_self_sources.py`: the three reads; multi-table optional check.
- `services/orion-durable-runs/app/temporal_self_chronicle.py`: street episode / capped log lines.
- `services/orion-durable-runs/app/settings.py`, `main.py`, `.env_example`, `docker-compose.yml`: kill switch and cap.
- `services/orion-durable-runs/evals/temporal_self_fixture_db.py`: the street tables' migrations in the test schema.
- `services/orion-durable-runs/tests/test_temporal_self_chronicle_postgres.py`: the reads and the fold on real Postgres.
- `.github/workflows/temporal-self-tests.yml`, `orion-durable-runs-tests.yml`: run when the gate / rollup / street migrations change.

## Schema / bus / API changes

- Added: `TemporalSourceKind "street_event"`, `ArcKind "street"`.
- Removed: none.
- Renamed: none.
- Behavior changed: the chronicle reads street tables; frames may contain `street` arcs.
- Compatibility notes: frame / arc readers use `extra="forbid"` Literals. **Rebuild orion-thought and orion-hub before orion-durable-runs**, or they reject a frame holding a street arc (the stance cue and recent arcs go dark until they are rebuilt). No bus channel changes.

## Env/config changes

- Added keys: `TEMPORAL_SELF_STREET_EPISODES_ENABLED=true`, `TEMPORAL_SELF_STREET_EPISODES_PER_DAY=12` (orion-durable-runs).
- Removed keys: none. Renamed keys: none.
- `.env_example` updated: yes (`services/orion-durable-runs/.env_example`), plus `settings.py`, compose, README.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, both keys added.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/temporal_self/tests                                  -> 227 passed (9 new)
pytest orion/temporal_self/tests orion/thought/tests orion/orion_day/tests -> 406 passed
cd services/orion-durable-runs && pytest tests (ORION_ADMISSION_TEST_DSN = disposable postgres:16) -> 578 passed, 1 skipped
pytest services/orion-thought/tests/test_stance_temporal_self_cue.py tests/test_personality_cue.py -> 32 passed
pytest services/orion-hub/tests/test_chat_turn_trace_routes.py tests/test_orion_day_temporal_day.py -> 18 passed
Mutation check: removing "street" from _NO_CONTEXT_KINDS fails test_room_company_is_unchanged... (room percept binds)
Static: check_env_template_parity PASS, check_inner_state_registry OK, check_metric_lineage --gate PASS,
check_definition_drift --gate PASS, check_service_hostname_refs OK, static pytest bundle 32 passed
```

## Evals run

```text
services/orion-durable-runs/evals/temporal_self_live_replay.py --step-sec 600 -> passed (patch2_passed true)
orion/temporal_self/evals/run_arc_precision_eval.py -> passed
orion/temporal_self/evals/run_play_cue_eval.py -> PASS
orion/orion_day/evals/run_orion_day_eval.py -> PASS
Live read-only 72 h replay (real SourceReader + fold against conjourney, READ ONLY transaction):
  27 events pass (24 motion_story, 3 doorbell_ring); arcs per day at cap 12: 10-20 10, 10-21 12, 10-22 2
```

The fixture day (10-09) predates motion stories, so the arc-precision replay has no street rows; the street path is covered by the gate tests, the Postgres test, and the live 72 h replay.

## Docker/build/smoke checks

```text
docker compose (orion-durable-runs) config with the primary .env files -> both new keys present
No image build or deploy (not requested).
```

## Review findings fixed

- See the follow-up commit on this branch (filled in after the review subagent).

## Restart required

```bash
scripts/safe_docker_build.sh orion-thought up -d --build && scripts/safe_docker_build.sh orion-hub up -d --build && scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: low. Concern: live labels rarely carry the people range, because the hour's rollup is written after the hour closes and the chronicle reads with a 300 s lag. Mitigation: no count is shown at all in that case (never a bare count); replays and backfills get the range.
- Severity: low. Concern: a story is written a median 49 s after it ends (p99 20 min); 1 of 54 published stories in 7 days lands after the read lag and is stored as a late row, never folded. Mitigation: counted in `temporal_self_late_rows`; acceptable loss for a capped source.
- Severity: low. Concern: the spot-check gate is read as of now, not as of the story's time, so a replay after Juniper's tally changes can differ from the live fold. Mitigation: the same rule the walkway rollups use; stated in the README.
- Severity: low. Concern: the cap is first come, first kept, so a late ring on a busy day can be capped out. Mitigation: 12 bound once in 72 h (10-21: 15 seen). Revisit if rings get dropped (the capped log line names them).
- Interaction check: the parked episode idle-close work concerns chat episodes in orion-memory-consolidation (`episode_shadow.py`), not Temporal Self arcs. No interaction.

## PR link

(filled in after `gh pr create`)
