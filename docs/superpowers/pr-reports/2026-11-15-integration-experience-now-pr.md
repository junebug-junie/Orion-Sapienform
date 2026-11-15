# Orion's now: experiences, salience, cites, integration metric (integration patch 1)

Refs #3423 (epic #3434). Spec: `docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md`
(PR #3416, signed off 2026-11-15), Part A, "Patch 1". Patch 2 (the integration gate) is stacked on this branch.

## Summary

- Three organs that were not in Orion's experience stream now are: every finished reading
  (`reading_turn`), every scored heat forecast (`prediction_surprise`, with how surprised Orion should be
  by the outcome), and every Prime thought (`prime_thought`, labelled with Prime's own note).
- Every new experience gets a **salience**: how unusual it is against the same source's last 30 days.
  Too little history means "unknown" (NULL), never 0.
- New: **Orion's now**. Any organ can ask "what did the other organs live since I last looked?" and gets at
  most 12 items, at most 2 per organ, never its own, most unusual first. Every such read is stored.
- An organ that used an item cites it. The cite is checked against what that organ was actually shown;
  only valid cites are stored. Bad cites are counted and dropped.
- The **integration metric** (share of experiences a different organ used within 24 h, and the "closed
  organs") has one definition, a route, and a live read. Today it is 0, as it must be: nothing reads the
  now until patch 3.

## Outcome moved

Before: no shared, checked record of which organ used which experience; readings, predictions and Prime
were not experiences at all. After: the record exists end to end (emit -> salience -> now -> cite ->
stored use -> metric), proven by a lifecycle test on real Postgres. Live proof of the stored path needs the
deploy (UNVERIFIED until then; see "Live evidence").

## Current architecture

Temporal Self's chronicle (orion-durable-runs, `temporal_self_chronicle.py`) reads 25 source kinds into
`temporal_self_event` and folds arcs. The citation gate (`orion/temporal_self/citations.py`) checks the chat
stance's and the letter's `[arc:]` cites, and records nothing durable about use.

## Architecture touched

- `orion/schemas/temporal_self.py`: `TemporalSourceKind` + `reading_turn`, `prediction_surprise`,
  `prime_thought`; new models `TemporalNowV1`, `TemporalNowItemV1`, `ExperienceCitesV1`,
  `ExperienceUseRowV1`, `ExperienceUseResultV1`, `ExperienceUseRateV1`, `OrganUseV1` (all registered).
- `orion/temporal_self/sources.py`: three adapters; `action_outcome` carries its row's `surprise_nats`.
- `orion/temporal_self/now.py` (new, pure): registry parse, magnitude rules, salience, `build_now`,
  `record_use`, `use_rate`. `orion/integration/organs.py` (new) is the one file reader.
- `config/integration/organs.yaml` (new): every source kind's organ, and each kind's salience magnitude.
- orion-durable-runs: the three source reads, `SalienceBook` (salience at insert), the INSERT's two new
  columns, `ExperienceStore`, three routes, retention of the two new tables, the flag, the Dockerfile copy.
- `services/orion-sql-db/manual_migration_experience_now_v1.sql` (new, additive).

## Code that disagreed with the spec (trusted the code)

- **`reading_turn` reads closed `hub.world_pulse_read` acts, not `reading_durable_turn`.** Live, that table
  is the stage request queue (two stages and retries per reading; 702 rows, 106 consumed in 7 days) with no
  finished state. The act close is the one row per finished reading, and it carries `surprise_nats`.
- **`action_outcome` already held every act close**, readings and Prime included (act:look 373, act:infer
  105 in 7 days), as content-free rows. The spec's map called readings and Prime "not a Temporal Self
  source". Kept both: `action_outcome` is the prediction organ's per-act outcome scoring; the new kinds
  carry content (label, act ref).
- **Three of the six new kinds, not six.** The spec's patch order puts `intention_close`, `day_letter`,
  `personality_choice` in patch 7. Naming them now with no reader would be a keyword cathedral.
- **Organ registry lands here, not in patch 2.** The now needs "which organ owns this kind" and salience
  needs each kind's magnitude, so the file exists from patch 1 with exactly those fields. Patch 2 adds the
  gate's fields. `salience_magnitude` is a map (kind -> rule), since an organ owns several kinds.
- **`POST /temporal-self/now/use`.** The spec names `record_use` but not how readers in other services
  call it; they cannot write durable-runs' tables, so the check and the insert happen behind one route.
- **`GET /temporal-self/integration`.** The metric's live read and debug surface; patch 2's coverage
  report uses the same SQL.
- **`GET /temporal-self/now` writes a row.** The spec requires each call to write `experience_shown`.

## Metric quality gate (salience and the experience use rate)

1. **Provenance.** Salience: `SalienceBook.score` (chronicle, before commit) with
   `orion.physiology.event_wants.source_salience` over magnitudes from `MAGNITUDE_RULES` (each rule reads
   one field of the source row, named in `now.py`). Use rate: `experience_use` rows, written only by
   `POST /temporal-self/now/use` after `record_use` checks them against `experience_shown`.
2. **Independence.** Salience comes from each source's own magnitude only; use never feeds salience
   (spec danger: a citation feedback loop). The use rate is the successor of `act_downstream_use`
   (spec), not a sibling; it shares no input with salience.
3. **Theory anchor.** Salience: the source's own strictly-below percentile, the rule event wants already
   use (rest point 0). Use rate: global workspace theory (Baars; Dehaene and Changeux): content is
   integrated when other specialist processes consume it; a checked cite is the observable trace.
4. **Live data** (`evals/experience_now_live_read.py`, 2026-11-15 05:37Z, last 7 days):

   | kind | rule | scored share NULL | at rest (0) | >= 0.9 | distinct |
   |---|---|---|---|---|---|
   | curiosity_run | moved + formed | 0.0 | 0.40 | 0.10 | 43 |
   | dream_cycle | hypothesis count | 0.0 | 0.36 | 0.0 | 10 |
   | field_dominance_run | ticks | 0.0 | 0.01 | 0.12 | 661 |
   | gpu_wait | waited ms | 0.01 | 0.0 | 0.03 | 1181 |
   | metacog_observation | severity | 0.0 | 0.58 | 0.0 | 147 |
   | prediction_surprise | surprisal | 0.0 | 0.99 | 0.004 | 40 |
   | action_outcome | surprisal | 0.0 | 0.002 | 0.05 | 1187 |
   | reading_turn | surprisal | 0.0 | 0.30 | 0.0 | 171 |
   | reverie_chain | thought count | 0.0 | 0.17 | 0.0 | 192 |
   | office_session | visit seconds | 0.9 (10 visits, under 20 history) | - | - | - |

   Two magnitudes failed and were **not** wired: `vision_percept` entity count (a producer step on 11-09
   from 1 to 4-5 entities per percept made 88% of ordinary scenes read >= 0.9 against the 30-day history;
   they could not read calm), and `prime_thought` surprisal (drifts 1e-6 per thought whatever the thought
   did). Both kinds read NULL and rank after scored items. The use rate's rest point is 0 and is reachable:
   it is 0 today.
5. **Existing mechanism.** Reuses `temporal_self_event`, `source_salience`, the citation gate's rule.
6. **Reversibility.** Two nullable columns, two tables, one module, three routes, one flag. No training
   default.

## Live evidence (read-only, before deploy)

From `ORION_LIVE_DSN=... python services/orion-durable-runs/evals/experience_now_live_read.py --check`
(exit 0), 2026-11-15 05:37Z:

- New sources through the chronicle's own SQL, last 7 days: `reading_turn` 372, `prediction_surprise`
  4,010, `prime_thought` 36. Example labels: "no_read_evidence: Internalizing Investment Mandates: ...",
  "athena.cabinet_temp_c experiment_guard > 32.0: forecast 0%, crossed (max 32.37)".
- Prime's now over the last 24 h (12 items from 2,975 candidates): 2 action outcomes (salience 1.0), 2
  attention dominance runs (0.99), 2 curiosity runs ("tested 1, moved 1, formed 4", 0.95), 2 GPU waits
  (1.7 h waits, 0.92), 2 reverie chains (0.76), 2 readings that failed to read (0.66). Never a Prime item.
- **Baseline integration metric:** 2,980 experiences in 24 h, 0 used by another organ, rate 0.0%, closed
  organs 0 of 13. (`experience_use` does not exist live yet.)

UNVERIFIED until deploy: rows of the three new kinds in `temporal_self_event`, non-null salience from
`GET /temporal-self/now?organ=prime`, and its `experience_shown` row (the spec's live acceptance). Proof
goes on #3423 after deploy.

## Files changed

- `orion/schemas/temporal_self.py`, `orion/schemas/registry.py`: kinds and models.
- `orion/temporal_self/sources.py`, `orion/temporal_self/now.py`, `orion/integration/*`: adapters and rules.
- `config/integration/organs.yaml`: organs and magnitudes.
- `services/orion-durable-runs/app/{temporal_self_sources,temporal_self_store,temporal_self_chronicle,main,settings}.py`.
- `services/orion-durable-runs/{.env_example,docker-compose.yml,Dockerfile,README.md}`.
- `services/orion-sql-db/manual_migration_experience_now_v1.sql`.
- `config/physiology/contract.yaml`: `now.py` is a display reader of `event_wants` (body-economy gate).
- `orion/schema_skew_discovery.py`: `ExperienceCitesV1`'s writer is an HTTP body (schema-skew gate).
- Tests: `orion/temporal_self/tests/test_now.py`, `services/orion-durable-runs/tests/test_experience_now.py`,
  `.../test_experience_now_postgres.py`, retention case in `.../test_temporal_self_chronicle_postgres.py`.
- Eval: `services/orion-durable-runs/evals/experience_now_live_read.py`; fixture DB applies the migration.
- `orion/temporal_self/README.md`.

## Schema / bus / API changes

- Added: 3 `TemporalSourceKind` values; 7 models; `GET /temporal-self/now`, `POST /temporal-self/now/use`,
  `GET /temporal-self/integration`; tables `experience_shown`, `experience_use`; columns
  `temporal_self_event.salience`, `.surprise_nats`.
- Removed / renamed: none.
- Behavior changed: `action_outcome` events carry `surprise_nats` in their payload.
- Compatibility: only orion-durable-runs builds `TemporalSelfEventV1`/state (checked: no other service
  validates them; frames carry no source kind), and nothing else reads the new models yet, so no consumer
  has to deploy first. No bus channel.

## Env/config changes

- Added keys: `TEMPORAL_SELF_NOW_ENABLED=true` (orion-durable-runs; ships on).
- `.env_example` updated: yes. Compose and settings: yes.
- Local `.env`: added by hand to the primary checkout's `services/orion-durable-runs/.env`.
- Skipped keys requiring operator action: none.

## Tests run

```text
orion/temporal_self/tests                                    302 passed
services/orion-durable-runs/tests (incl. Postgres, local disposable DB)   all passed (see CI)
tests/scripts/test_schema_skew_discovery.py, tests/test_check_body_economy_contract.py   passed
orion-static-gates.yml, temporal-self-tests.yml, temporal-self-readers-tests.yml run step by step locally: all ok
```

## Evals run

```text
services/orion-durable-runs/evals/temporal_self_live_replay.py --step-sec 600    passed (patch 2 replay unchanged)
services/orion-durable-runs/evals/experience_now_live_read.py --check (live, read-only)   exit 0, numbers above
```

The spec's ablation eval (hollow cites) belongs to patch 3, when cites exist.

## Docker/build/smoke checks

```text
Not built here. The Dockerfile change is one COPY of config/integration/organs.yaml, pinned by
test_dockerfile_copies_the_organ_registry.
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

Deploy order: the migration first, then orion-durable-runs. The post-pull rebuild applies the migration
itself (additive, `REQUIRED-BY: orion-durable-runs`, indexes `CONCURRENTLY`); the deploy wrapper refuses
`up` while it is unapplied.

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: low. Concern: the chronicle INSERT names the two new columns; a deploy before the migration
  fails every window (retried, nothing lost). Mitigation: the wrapper refuses `up` until applied.
- Severity: low. Concern: first-window salience history reads up to 31 days of one kind per kind (largest:
  `prediction_surprise` ~17k small rows once accumulated). Mitigation: one read per kind per process, then
  memory; indexed by `(source_kind, occurred_at)`.
- Severity: info. `prediction_surprise` adds ~570 rows a day to `temporal_self_event` (one series, two
  thresholds), 99% at the rest point.
- Issues: refs #3423. Opened: none. Closed: none (live proof after deploy).

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
