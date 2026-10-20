# Strain arcs: Orion remembers each strained episode in its day (strain loop part 1)

## Summary

- Orion has been strained since 10-18 12:07 UTC. The cabinet is at 34.6 °C (critical) and 11 GPU jobs are waiting. Until now, no part of Orion's day story knew it.
- The regulate step (orion-durable-runs) already writes a row when the strain level changes. It now also writes:
  - one row when a new cause joins a strained episode (for example, the cabinet going critical mid-episode)
  - one seed row for an episode the day story has no start row for (the first step after deploy)
  - a keepalive row every 30 min while strained
- The day story (Temporal Self chronicle) turns those rows into one `strain` arc per episode. The arc carries its causes and carries over midnight, keeping when the episode really began.
- Orion sees the arc in the unified chat turn's stance cue, for example "Strained since Sun 06:07 (GPU jobs backed up, cabinet too hot) [arc:…]". It is also in the Day letter ("Strained: …"), in reverie's recent-arc block, and in the citation truth check.
- Spec: `docs/superpowers/specs/2026-10-20-strain-loop-design.md` (PR #2800), part 1 of 3. Part 2 (strain news in attention) is a separate branch.

## Outcome moved

Before, Orion's strain was a reflex only. Seven gates held work back, and no awareness stream (chat, Day letter, reverie) could say why or since when. Now the chat turn carries a cited "Strained since …" line while strained, and the Day letter records each episode.

Replay of the real rows (all 74 `arousal_transition` rows, 10-10 to now, read-only) gives **24 strain arcs for the 24 real episodes**. The current one is open with the cue "Strained since Sun 06:07 (GPU jobs backed up)".

## Current architecture

- `orion/regulation/arousal.py:classify_arousal` gives engaged / idle / strained / unknown. It runs in the durable-runs `temporal_self.update` thread every 120 s.
- A row is written to `temporal_self_event` only on a level change. A cause that joined mid-episode was never recorded.
- The chronicle reducer (`orion/temporal_self/arcs.py`) excluded `arousal_transition` on purpose ("regulation lane, not here").

## Architecture touched

- **orion-durable-runs:**
  - The regulate node writes cause, seed and keepalive rows (`_strain_causes_step`, `regulation_store.cause_row`). It carries checkpoint keys `strain_causes_seen` and `strain_row_at`.
  - The chronicle reads the regulation's own rows (`SourceQuery("arousal_transition")`).
- **orion/temporal_self:**
  - `arousal_transition` adapter. It **keeps the row's own `event_id`**: the chronicle stores every folded event in the same table, so a new id would re-insert a copy that the next read folds again (a mutation test reproduces exactly that runaway: 45 nested copies).
  - `strain` lane in the reducer.
  - Readers: stance cue, Day letter, recent arcs, citation truth.
- **orion-dream:** `strain` added to `ARC_EXCLUDED_KINDS`. A days-long arc would saturate the replay weight and fill every dream.

## Files changed

- `orion/regulation/arousal.py`: `STRAIN_CAUSES` / `strain_causes()`, one owner for the cause names.
- `orion/schemas/temporal_self.py`: source kind `arousal_transition`, arc kind `strain`, optional `episode_began_at` on the arc and its summary.
- `orion/temporal_self/sources.py`: the adapter.
- `orion/temporal_self/arcs.py`: `_fold_strain`, stale timeout, midnight carry, no context binding.
- `orion/temporal_self/frame.py`: passes `episode_began_at`.
- `orion/temporal_self/stance_cue.py`: strain clause and cause words.
- `orion/temporal_self/day_block.py`, `recent_arcs.py`, `citation_truth.py`: the readers.
- `orion/temporal_self/tests/test_strain_arcs.py`: new, 16 tests.
- `orion/temporal_self/evals/export_fixture_day.sql`, `services/orion-durable-runs/evals/temporal_self_fixture_db.py`: fixtures include the rows.
- `services/orion-durable-runs/app/{temporal_self_graph,temporal_self_driver,regulation_store,temporal_self_sources,main,settings}.py`: producer, reader and flag.
- `services/orion-durable-runs/{.env_example,docker-compose.yml,README.md}`: `TEMPORAL_SELF_STRAIN_ARCS_ENABLED`.
- `services/orion-durable-runs/tests/test_temporal_self_{regulate,chronicle_postgres}.py`: producer and real-Postgres tests.
- `services/orion-dream/app/replay.py` and its test: strain excluded from dreams.
- `scripts/analysis/measure_arousal_reader_replay.py`: ignores same-level rows, which would restart a level's age.
- `orion/inner_state_registry.py`, `config/metrics/metric_definitions.lock.json`: chronicle listed as a reader of `regulation.state.v1` (re-locked).

## Schema / bus / API changes

- **Added:**
  - `TemporalSourceKind` `arousal_transition`
  - `ArcKind` `strain`
  - `TemporalSelfArcV1.episode_began_at`, `ArcSummaryV1.episode_began_at` (optional)
- **Removed / renamed:** none.
- **Behaviour changed:** `temporal_self_event` gets same-level `arousal_transition` rows (event id `…:strained:at:<time>`, `payload.joined` / `payload.seed`). The label is still the level, so level readers still read `strained`.
- **Compatibility:** the models forbid unknown values. **Consumer-first deploy:** orion-thought and orion-hub before orion-durable-runs. orion-dream reads columns, not models.
- No bus channel. No migration: same table and columns.

## Env/config changes

- **Added keys:** `TEMPORAL_SELF_STRAIN_ARCS_ENABLED=true` (orion-durable-runs). With it off there are no cause rows and the chronicle stops reading the kind. Any open strain arc then closes by the stale timeout.
- **Removed / renamed:** none.
- **`.env_example` updated:** yes, and docker-compose passthrough added.
- **Local `.env` synced:** the key was added to the primary checkout's `services/orion-durable-runs/.env`. The sync script only targets its own checkout, and the worktree has no `.env`.
- **Skipped keys:** none.

## Tests run

```text
orion/temporal_self/tests                                   206 passed
orion/temporal_self orion/regulation orion/personality     324 passed (before review fixes)
services/orion-durable-runs/tests (disposable Postgres)     547 passed, 1 skipped (before review fixes)
durable-runs regulate + chronicle (+postgres) after fixes   64 passed
services/orion-thought stance/recent-arcs/personality cue   44 passed
services/orion-hub day/temporal-self/curiosity fact         45 passed
services/orion-dream test_replay_arc_candidates             20 passed
orion/thought + settings/metric/no-personal-terms tests     233 passed
Mutation checks: no-context rule removed -> test fails; adapter new id -> Postgres test fails (45 copies)
```

## Evals run

```text
python orion/temporal_self/evals/run_arc_precision_eval.py                 passed
python services/orion-durable-runs/evals/temporal_self_live_replay.py --step-sec 600   "passed": true
Live replay of 74 real arousal_transition rows (read-only) -> 24 strain arcs = 24 real episodes;
  open now: since 2026-10-18T12:07:36Z, cue "Strained since Sun 06:07 (GPU jobs backed up) [arc:…]"
```

## Docker/build/smoke checks

```text
Not built here. Gates: check_inner_state_registry OK, check_definition_drift --gate PASS (re-locked),
check_metric_lineage --gate PASS, --prompt-semantics PASS, check_env_template_parity PASS,
check_env_key_single_source OK, check_sentience_instruments --static-only OK, git diff --check OK.
```

## Review findings fixed

- **Finding (HIGH):** the flag off, or a lost closing row (MEDIUM), left a strain arc open forever, so the cue said "strained since" indefinitely.
  - **Fix:** a 30-min keepalive while strained, plus a reducer stale timeout (4,500 s) that closes at the last row. Same design as `company`.
  - **Evidence:** `test_keepalives_hold_an_episode_and_silence_closes_it_at_the_last_row`, `test_strain_keepalive_every_30_min_while_strained_and_none_after`.
- **Finding (MEDIUM):** after an outage, the end time was wrong in both directions.
  - **Fix:** the adapter carries `prev_dropped:gap`, and a not-strained row after a gap closes at the last row (`source_stale`). Keepalives bound the restart-while-strained case to 30 min.
  - **Evidence:** `test_not_strained_after_an_outage_ends_the_arc_at_its_last_row`.
- **Finding (LOW-MED):** `measure_arousal_reader_replay.py` restarted level ages on same-level rows.
  - **Fix:** it filters `event_id NOT LIKE '%:at:%'`.
- **Finding (LOW):** orion-dream silently gained strain arcs.
  - **Fix:** excluded, with a test.
- **Finding (LOW):** bare "was strained" matched ordinary English.
  - **Fix:** the phrase is anchored to a clock time.
  - **Evidence:** `test_citation_phrase_ignores_ordinary_strained`.
- **Finding (LOW):** a state reset loses a carried episode.
  - **Fix:** documented in the README. The keepalive now bounds the loss to 30 min.

## Restart required

Consumer-first, from the clean primary `main` checkout after merge. One command per line:

```bash
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-dream up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- **Severity: medium.**
  - **Concern:** deploying durable-runs before thought/hub makes the frame carry `strain` and `episode_began_at`, which older readers reject, and the stance cue and Day letter drop out.
  - **Mitigation:** the deploy order above.
- **Severity: low.**
  - **Concern:** until the first post-deploy step, the open episode's label shows only `gpu_queue_sustained`.
  - **Mitigation:** the seed row on that step adds `cabinet_hot`.
- **Live verification is UNVERIFIED until deploy.** Check:
  - a `strain` row in `temporal_self_arc`
  - a seed row (`payload.seed = true`) in `temporal_self_event`
  - "Strained since" in a live turn's stance cue (`cockpit_turn_sighting`)

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2817

🤖 Generated with [Claude Code](https://claude.com/claude-code)
