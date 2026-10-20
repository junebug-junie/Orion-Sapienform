# Street continuity: one passage across the door and walkway cameras (handoff loop)

**Merge order: this PR must merge AFTER #2822 (`fix/motion-story-frame-edges`) and #2816
(`fix/doorbell-ring-tracking`).** Both are merged into this branch; it reads their
motion-story edge columns, `StreetEventV1`, the `doorbell_ring` / `doorbell_motion` events
and the retina's `orion:vision:look:doorbell` self-look key.

Spec: `docs/superpowers/specs/2026-10-20-street-continuity-handoff-design.md` (PR #2803),
sections 2, 3, 4, the Hub line from section 8, the asks, and the metric-gate replay.

## Summary

- When someone leaves one camera toward the other, Orion now writes down when it expects
  them on the other camera, points that camera's attention there if there is still time,
  and later grades the guess: met, early, late, missed, or "could not look".
- When the two cameras see the same moment on the central walkway, Orion records the
  pair as a "same person?" candidate. A strong pair opens a capped question to Juniper.
  The handoff code never names anyone and never merges anyone.
- A watched miss gets five checkable explanations (my detector missed them, they were
  late, they turned back, they went into a place, unexplained). An unexplained miss goes
  to curiosity through the existing `vision_unresolved` table and to attention as a
  street event.
- The Hub story card now says "Came from the door camera 38 s earlier." (or the reverse,
  or "The door camera saw them at the same moment.").
- A read-only replay eval over the stored history is the metric gate. Today's data is
  too thin for a verdict (details below).

## Outcome moved

- Before: two cameras, two unrelated stories. Nothing could be expected, so nothing could
  be surprising, and a missing walker read the same as a camera that was off.
- After: every departure toward the other camera is a graded prediction with a p-value,
  and "camera was not looking" is its own outcome (`unscorable`), never `missed`.
  Replay: 46 of 46 doorbell-target windows came out `unscorable`, 0 false misses.

## Current architecture

- Motion stories per camera (`app/vision_motion_loop.py`), now with entry/exit edges and
  exit speed (#2822). 35 of 36 published walkway stories end `inside` the frame.
- `vision_rhythm.py` grades daily routines and owns `orion:vision:expect:<stream>`.
- Retina records Ring pushes as `doorbell_ring` / `doorbell_motion` vision events and opens
  a capped live view when `orion:vision:look:doorbell` changes (#2816).
- No cross-camera link of any kind existed.

## Architecture touched

- `orion-sql-writer`: new clocked loop `vision_handoff_loop` (15 s, advisory lock
  `0x0A1C_0005`, cursor on motion-story `created_at`, backs off while the migration is
  missing). Pure rules in `app/vision_handoff.py`.
- `vision_rhythm._set_expect_keys`: honours an open handoff's `handoff_until` before it
  shortens or deletes `orion:vision:expect:<stream>` (the two writers of that key no longer
  clobber each other).
- Bus: `orion-sql-writer` becomes a producer of `orion:vision:street:events`
  (`StreetEventV1`), through an outbox on `vision_handoff` (marked only after a successful
  publish).
- Hub: `/api/vision/motion-stories` adds `handoff` per story (fail-open without the table);
  `walkway-stories.js` renders it.

## Files changed

- `config/vision_topology.yaml`: the four portals (central_walk overlap, approach_door,
  door_to_walkway, walkway_to_door) and place zones. Priors only.
- `orion/vision/topology.py`: loader with validation (`TopologyError` with the reason).
- `services/orion-sql-db/manual_migration_vision_handoff_v1.sql`: `vision_handoff`,
  `vision_handoff_hypothesis`, `vision_handoff_cursor` (additive, auto-apply safe).
- `services/orion-sql-writer/app/vision_handoff.py`: projection from `inside`, prior and
  conformal bands, window, match, grade, overlap, hypotheses, p-values, salience table,
  street events, Redis value merge.
- `services/orion-sql-writer/app/vision_handoff_loop.py`: open / grade / overlap / rings /
  hypotheses / asks / prune steps, Redis steering, outbox publish.
- `services/orion-sql-writer/app/vision_rhythm.py`: `handoff_hold` + key coexistence.
- `services/orion-sql-writer/app/main.py`, `settings.py`, `.env_example`,
  `docker-compose.yml`, `Dockerfile` (copies the topology), `README.md`.
- `orion/schemas/vision.py`: `VisionUnresolvedV1.reason` gains `handoff_missed`;
  StreetEvent comment names the new producer.
- `orion/curiosity/study_material.py`: plain words for the new reason.
- `orion/bus/channels.yaml`, `config/metrics/metric_definitions.lock.json` (regenerated),
  `tests/test_street_event_bus_catalog.py`.
- `services/orion-hub/scripts/motion_story_routes.py`, `static/js/walkway-stories.js` + tests.
- Tests: `tests/test_vision_handoff.py` (pure), `tests/test_vision_handoff_pg.py` (real
  Postgres), `tests/test_vision_expect_refresh.py` (key coexistence).
- Evals: `evals/run_vision_handoff_replay_eval.py` (live, read-only),
  `evals/test_vision_handoff_replay_eval.py` (synthetic street, CI).
- CI: `orion-sql-writer-tests.yml`, `orion-memory-episode-tests.yml`.

## Schema / bus / API changes

- Added: tables `vision_handoff`, `vision_handoff_hypothesis`, `vision_handoff_cursor`;
  `vision_events` types `handoff_met`, `handoff_early`, `handoff_late`, `handoff_missed`,
  `overlap_pair` (written by direct SQL in the sql-writer, the same path as the rhythm
  loop's `arrived_as_expected`; no scribe or type filter sits on that path); producer
  `orion-sql-writer` on `orion:vision:street:events`; `VisionUnresolvedV1.reason`
  value `handoff_missed`; Hub JSON field `handoff` on each story.
- Removed / renamed: none. The metric lock entry for the street channel is re-keyed from
  `orion-vision-retina/...` to `orion-sql-writer/...` because the drift tool keys a
  channel by its first producer alphabetically; `all_producers` lists both, and nothing
  in the repo referenced the old key.
- Behavior changed: the rhythm loop no longer deletes or shortens the expect key while a
  handoff window is open.
- Compatibility: `StreetEventV1` is unchanged (extra=forbid), only new producers of
  existing kinds. `orion-substrate-runtime` does not subscribe yet (spec phase 3).

## Env/config changes

- Added keys (all ON): `VISION_HANDOFF_ENABLED=true`, `VISION_HANDOFF_INTERVAL_SEC=15`,
  `VISION_HANDOFF_KINDS`, `VISION_HANDOFF_LOOK_KINDS`, `VISION_HANDOFF_MIN_MET=10`,
  `VISION_HANDOFF_MIN_COVERAGE=0.8`, `VISION_HANDOFF_MIN_FACE_COSINE=0.50`,
  `VISION_HANDOFF_MIN_BODY_COSINE=0.92`, `VISION_HANDOFF_MIN_OSNET_COSINE=0.82`,
  `VISION_HANDOFF_ATTENTION_P=0.05`, `VISION_HANDOFF_SAME_ASK_DAILY_CAP=2`,
  `VISION_HANDOFF_INITIAL_LOOKBACK_SEC=3600`, `VISION_HANDOFF_STEER_ENABLED=true`,
  `VISION_HANDOFF_SELF_LOOK_ENABLED=true`, `VISION_HANDOFF_STREET_EVENTS_ENABLED=true`.
- Removed / renamed: none.
- `.env_example` updated: yes (`services/orion-sql-writer/.env_example`).
- local `.env` synced: default run reported "no changes" (keys outside the default feature
  list), so `python scripts/sync_local_env_from_example.py --all-keys orion-sql-writer`
  added all 15 keys to `/mnt/scripts/Orion-Sapienform/services/orion-sql-writer/.env`.
- Skipped keys: none of mine. Three pre-existing unrelated keys are reported "diverged"
  (`SQL_WRITER_SUBSCRIBE_CHANNELS`, `SQL_WRITER_ROUTE_MAP_JSON`,
  `GRAMMAR_RETENTION_PERIODIC_MAX_CYCLE_SEC`); left alone.
- Appearance bars were measured live 2026-10-20 on walkway clusters (23 individuals):
  different people's body centroids 95th pct 0.90, 99th 0.93; OSNet 95th 0.79, 99th 0.82.

## Tests run

```text
pytest tests/test_vision_handoff.py                      56 passed (pure rules)
pytest tests/test_vision_handoff_pg.py                   10 passed (real Postgres 16, all migrations)
pytest tests/test_vision_expect_refresh.py tests/test_vision_rhythm.py ...  129 passed (sql-writer unit list)
pytest services/orion-hub/tests/test_motion_story_routes.py   25 passed (real Postgres)
pytest services/orion-hub/tests/test_walkway_stories_js.py    4 passed (node)
static gates (orion-static-gates.yml list)               388 passed; metric lineage, definition drift,
                                                          migration REQUIRED-BY, env parity: PASS
orion-vision-retina tests                                146 passed, 2 failed locally only (the local
                                                          venv lacks ring_doorbell; untouched by this PR)
Mutation check (11 mutants of the key rules)             11 killed, 0 survived:
  coverage->missed, band switch at 10, projection cap, overlap skew, unexplained always,
  lone candidate, approach depth, departure direction, prior p two-sided, missed p, priors start at 1
```

## Evals run

```text
pytest evals/test_vision_handoff_replay_eval.py   2 passed (synthetic street: learned g within 10% of
  truth, spread < 50%, band covers 64/80 = 80% (acceptance 70-90%), 0 false misses, met p KS < 0.25,
  quiet-hour gate passes at chance, unwatched doorbell -> unscorable)
python evals/run_vision_handoff_replay_eval.py --since 2026-10-16T00:00:00Z   (live, read-only)
```

## Metric gate: handoff surprise (`vision_handoff.p_value`)

1. **Provenance.** `p_value_arrival` / `p_value_missed` in `app/vision_handoff.py`, called by
   `step_grade` in `app/vision_handoff_loop.py` from motion-story timestamps
   (`started_at`, `ended_at`), exit speed and exit foot point (stored by #2822, or derived
   from story evidence boxes by the same `vision_motion` functions when the edges migration
   is not applied).
2. **Independence.** Built from story timing and geometry across two cameras. Not derived
   from `perception_prediction_error` (whole-frame embedding drift) or the rhythm model
   (time of day). Shares the detector with every vision signal; that is why
   `detector_miss` is its own hypothesis.
3. **Theory.** Split-conformal predictive interval on a physical travel time (rank p-values
   are uniform under exchangeability); on the prior band, a log-normal whose 10%/90%
   quantiles are the band ends. A miss's p is the Laplace-smoothed counted chance of no
   arrival.
4. **Live sanity.** Replay over 2026-10-16 .. 2026-10-20 (all stored stories: walkway
   187 person / 34 dog / 32 stroller / 20 bicycle; doorbell 5 person; 95 doorbell census
   spans; 1 doorbell_motion event; 0 rings):
   - Outcomes: approach_door 25 unscorable; walkway_to_door 21 unscorable;
     door_to_walkway 0 after the fix below. **0 met, 0 missed.**
   - Overlap pairs: 0 paired, 1 ambiguous (16:24 UTC, 3 walkway candidates, best body
     centroid cosine 0.74), 4 unpaired.
   - Learned g per portal: none (no met handoff).
   - p-value uniformity on met: no met p-values (n = 0).
   - Rest state: 17 of 17 hours with handoffs were quiet; 0 attention-gate passes, even
     with "known person" and "unusual hour" forced true. **This is zero because nothing
     was graded met or missed, not because calm p-values were observed.**
   - 80% rule: every doorbell-target window (46) was unscorable; 0 doorbell misses.
   - Steering: 21 of 46 windows were still ahead when the story landed (stories land a
     median 48.6 s after the walker left), so Redis steering can help about half the time.
   - **Verdict: TOO THIN FOR A VERDICT.** 0 met handoffs (< 10): no learned band, no
     uniformity test. The doorbell sees only during Ring bursts (95 census spans in 5 days),
     so walkway -> door is unscorable unless the self-look fires in time. The gate must be
     re-run after the self-look has been live for a week and #2822's edges are applied.
   - Structural limit found: with `n` calibration values the conformal p cannot go below
     `2/(n+1)`; under n = 39 no arrival can pass p < 0.05. Documented in the module and
     pinned by a test. Until then only misses can reach the gate.
5. **Existing mechanism.** None for cross-camera handoff (searched handoff, transit,
   cross-camera, markov, kalman, scenario). Reused: rhythm's grading rule and
   `coverage_fraction`, the expect key, `vision_unresolved` (table + ORM model), the person
   model's channels, `unit` and `load_crops_by_id`, the same-individual ask
   (`_insert_ask`, `pair_ref`, `same_groups`, `ask_image_refs`).
6. **Reversibility.** Three additive tables; `VISION_HANDOFF_ENABLED=false` stops the loop;
   street events, self-look and steering each have their own flag; asks have a cap (0 = off).
   Nothing reads `p_value` yet outside this loop and the street event.

Findings from the replay that changed the code:

- The only replay "miss" (door_to_walkway, 21:02 UTC) was a doorbell visitor walking right
  and toward the door, projected as a sidewalk departure. It would have become a false
  "unexplained" miss. Fix: `exclude_dir_depth: 1` on `door_to_walkway` (topology), with a test.
- The synthetic street showed that the spec's match window (band plus late margin) grades
  the ~10% of ordinary walkers who arrive before the band as `missed`. Fix: an early margin
  (one window width, never before departure) and an `early` outcome that trains the gap.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-sql-writer build -> refused: worktree has no .env (the wrapper
  reads the live .env from the primary checkout only for deploys)
docker build -f services/orion-sql-writer/Dockerfile -t orion-sql-writer:handoff-test .  -> OK
docker run ... python -c "load_topology(); import app.vision_handoff_loop, app.main; Settings()"
  -> ['central_walk', 'approach_door', 'door_to_walkway', 'walkway_to_door'] / handoff True 15.0
Live loop: UNVERIFIED (not deployed; the migration is not applied live).
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_handoff_v1.sql
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Run from the clean primary `main` checkout after #2822, #2816 and this PR are merged (the
post-pull applier may already have applied the migration; it is additive-DDL only). The
#2822 edges migration and #2816's retina deploy are prerequisites for useful data.

## Risks / concerns

- Severity: medium. Concern: the metric gate has no verdict (0 met handoffs). Mitigation:
  nothing reads `p_value` for decisions yet; `orion-substrate-runtime` does not subscribe to
  street events. Re-run the replay after a week live.
- Severity: medium. Concern: cross-camera body similarity looks low (0.74 on the one
  same-moment pair) against a 0.92 bar, so "same person?" asks from handoffs may almost
  never fire. Mitigation: bars are env keys; the replay prints pair scores; the cap stays.
- Severity: low. Concern: the handoff "same person?" asks also count against the face-ask
  budget (`VISION_CONFIRM_ASK_DAILY_CAP`, it counts every `vision_same_individual` ask).
  Mitigation: both caps are small and explicit.
- Severity: low. Concern: about half of the windows are already past when a story lands
  (~49 s story latency), so steering and the self-look help only for longer windows.
  Mitigation: counted as `steer_too_late` in each tick summary.
- Not built: the "where did they go?" ask. `OrionAskV1` has no choices field, so a
  buttons-style ask is not supported cleanly; skipped. Section 5's "never at night" rule for
  self-looks is not built (the retina's daily cap applies).

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
