## Summary

Orion now asks Juniper to confirm what the walkway camera saw, in the "Orion is asking" box on the Hub, with the pictures on the card.

- **"Are these the same person?"** Two people whose faces look alike (face similarity at least 0.40) get one card that shows both faces side by side. Juniper presses Same person, Different people, or Not sure.
- **The answer is kept as learning data.** Each yes/no answer is saved with the face similarity at that moment (`vision_individual_pair_verdict`). These rows are the labeled data for tuning the face match threshold, which nobody has calibrated yet. "Same" links the two individuals (nothing is merged or deleted) and copies a name across. "Different" means the pair is never asked again.
- **"Who is this?" the same day** for a person whose face was seen clearly at least 5 times. Before this, every ask had to wait for 10 sightings over 5 days. The card shows up to three pictures, face first.
- **Several pictures per ask.** `OrionAskV1.image_refs` (additive) and the `orion_ask.image_refs` column. Old rows still show their single `image_ref`.
- **Their own daily cap**, `VISION_CONFIRM_ASK_DAILY_CAP=6`. The 10-sightings rule keeps `ORION_ASK_DAILY_CAP=2`.

## Outcome moved

The walkway camera has a face pipeline (live about one day), but Orion had never asked a single vision question (`orion_ask` has 0 `vision_individual` rows). The reason: the only rule needed 5 separate days. The replay eval on today's live data shows **3 asks open on deploy**, each with both faces on the card. Each answer gives the face threshold one labeled calibration point.

## Current architecture

- The opener (`services/orion-sql-writer/app/vision_individuals.py` `_open_asks`) had one rule: unlabeled, 10 or more sightings, 5 or more local days, `ORION_ASK_DAILY_CAP=2` (counting `vision_individual` asks only). One picture (`image_ref`).
- Answers: the Hub `POST /api/asks/{id}/answer` updates the row and publishes `orion:ask:answered`. sql-writer polls `status='answered' AND applied_at IS NULL` and sets `vision_individual.label`. orion-substrate-runtime writes an entity node for `vision_individual` answers only.
- The Hub card (`static/js/vision-asks.js`) showed one image.
- Faces cluster by `VISION_INDIVIDUALS_FACE_MATCH_THRESHOLD=0.60`. This value is uncalibrated, and the live data shows one person split across several individuals.

## Architecture touched

- Contract: `orion/schemas/ask.py` (`image_refs`, documented `vision_same_individual`).
- DB: new migration `manual_migration_vision_confirm_asks_v1.sql` (required by orion-sql-writer, orion-hub).
- orion-sql-writer: the opener, answer apply, prune guard, settings, and a migration gate.
- orion-hub: the list route (`image_refs`), yes/no enforcement, and the card UI.
- No bus changes. Answers still travel through the table, and the substrate listener ignores the new `source_kind`.

## Files changed

- `orion/schemas/ask.py`: additive `image_refs`. Documents the new `source_kind`.
- `services/orion-sql-db/manual_migration_vision_confirm_asks_v1.sql`: `orion_ask.image_refs` and `vision_individual_pair_verdict`. Additive and idempotent.
- `services/orion-sql-writer/app/vision_individuals.py`: `ask_image_refs`, face rule, pair finder with same-person grouping, yes/no parse, `_open_face_asks`, `_apply_pair_answers`, label propagation, separate cap SQL, migration check, prune guard.
- `services/orion-sql-writer/app/settings.py`, `app/vision_individuals_loop.py`, `.env_example`, `README.md`: 3 new keys and docs.
- `services/orion-sql-writer/tests/test_vision_confirm_asks.py`: pure rules and the full answer path against real Postgres (opt-in).
- `services/orion-sql-writer/tests/test_vision_walkway_faces_stationary.py`: picks the slow-rule SQL explicitly and also checks that the face query skips stationary individuals.
- `services/orion-sql-writer/evals/run_vision_confirm_asks_replay_eval.py`: reads live data read-only and runs the real opener on a throwaway database.
- `services/orion-hub/scripts/ask_routes.py`: selects `image_refs` (backfilled from `image_ref` on old rows). Same-person cards accept only `yes` / `no` (422 otherwise).
- `services/orion-hub/static/js/vision-asks.js` (+ tests): pictures side by side with inline layout styles, alt text, "Picture no longer available." fallback, and the same-person card.
- `services/orion-hub/tests/test_ask_routes.py`, `tests/test_orion_is_asking_browser_smoke.py`: route and Chromium coverage.
- `scripts/report_vision_individuals.py`: prints the same-person verdicts with their face similarity (the calibration readout).

## Schema / bus / API changes

- Added: `OrionAskV1.image_refs: list[str] = []`; `orion_ask.image_refs JSONB NOT NULL DEFAULT '[]'`; table `vision_individual_pair_verdict`; `source_kind="vision_same_individual"`; `GET /api/asks` returns `image_refs`.
- Removed / renamed: none.
- Behavior changed: `POST /api/asks/{id}/answer` on a `vision_same_individual` card returns 422 `answer_must_be_yes_or_no` unless the answer is yes or no (case-insensitive, stored lowercase).
- Compatibility: producers build `OrionAskV1`, and nothing validates it from foreign payloads, so the additive field is safe on `extra="forbid"`. The Hub and sql-writer both need the column, and the migration header declares both, so `safe_docker_build.sh up` refuses until the migration is applied. Without it, sql-writer's individuals loop backs off with a clear error instead of crashing.

## Env/config changes

- Added keys (orion-sql-writer): `VISION_ASK_FACE_MIN_FACES=5`, `VISION_SAME_ASK_MIN_FACE_COSINE=0.40`, `VISION_CONFIRM_ASK_DAILY_CAP=6`.
- Removed / renamed: none.
- `.env_example` updated: yes. Compose loads them via `env_file`.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-sql-writer`: yes, all 3 keys were added to the primary checkout's `.env`.
- Skipped keys requiring operator action: none from this change. `SQL_WRITER_SUBSCRIBE_CHANNELS` and `SQL_WRITER_ROUTE_MAP_JSON` were already diverged and are untouched. This change adds no channel or route.

## Measured face-similarity band (metric gate)

1. **Where it comes from:** `vision_individual.face_centroid` (running mean of `vision_crop_observation.face_embedding`, 512-d, from orion-vision-host), via `apply_batch` in `vision_individuals.py`.
2. **Independence:** this is the same signal that drives face clustering. It is not a new metric. It is used only to pick which pairs to ask about.
3. **Why it measures what it claims:** cosine similarity between face embeddings is what face-recognition models are trained to separate. Two faces in one frame are, by construction, different people.
4. **Live data** (108 face crops, 5,778 pairs, 2026-10-15). Two faces in the same frame (known different people), n=37: p10 0.06, p25 0.12, **p50 0.27**, p75 0.41, p90 0.52. The same face 3 s or less apart in the same place (almost always the same person), n=72: **p10 0.39**, p25 0.55, **p50 0.65**, p75 0.71, p90 0.76. The two overlap between about 0.40 and 0.55, so 0.40 is where Orion is genuinely unsure. One same-frame pair read 0.986, which is probably two boxes on one person.

   All crop pairs (lower edge of each 0.05 bin; the end bins also hold everything beyond them):

   ```text
   -0.20 46 | -0.15 56 | -0.10 110 | -0.05 196 | +0.00 288 | +0.05 385 | +0.10 481 | +0.15 495 | +0.20 487 | +0.25 482
   +0.30 464 | +0.35 428 | +0.40 340 | +0.45 312 | +0.50 284 | +0.55 274 | +0.60 238 | +0.65 213 | +0.70 110 | +0.75+ 89
   ```

   Pairs of individual face centroids (face_n of 5 or more): `62cb5fcc`/`f5242d40` **0.771**, `4354c877`/`9bea16d5` 0.533, `4354c877`/`f5242d40` 0.411, the rest 0.19–0.33. The first pair is *above* the 0.60 match threshold, but the clusterer still kept the two apart, because it matches one crop at a time against a centroid that was still forming.
5. **Existing mechanism:** none existed. The match threshold was set by hand.
6. **Reversibility:** set `VISION_ASK_FACE_MIN_FACES=0` to turn both kinds off. The verdict table is additive.

## Tests run

```text
services/orion-sql-writer: pytest tests/test_vision_confirm_asks.py (ORION_TEST_PG_URI=throwaway postgres:16) -> 20 passed
services/orion-sql-writer: pytest tests/test_vision_walkway_faces_stationary.py tests/test_vision_individuals.py tests/test_vision_walkway_zones_thumbs_cursor.py -> 69 passed
services/orion-sql-writer: pytest tests evals -> 14 failures, all also failing on main in this environment (biometrics, chat_history, notify, journal, gpu_pool, subscription ratchet); none touched here
services/orion-hub: pytest tests/test_ask_routes.py (with ORION_TEST_PG_URI) -> 44 passed
services/orion-hub: pytest tests/test_ask_routes.py tests/test_hub_ui_layout_pass.py tests/test_dream_routes.py tests/test_curiosity_unresolved_material_read.py -> 89 passed
services/orion-hub: node --test static/js/vision-asks.test.js -> 27 passed
services/orion-hub: pytest tests/test_orion_is_asking_browser_smoke.py (playwright 1.49.1, Chromium) -> 1 passed
services/orion-substrate-runtime: pytest tests/test_ask_answered_listener.py -> 16 passed
tests/test_curiosity_unresolved_percepts.py -> 10 passed; tests/test_agent_trace_schema_registry.py -> 2 passed
gates: check_env_template_parity PASS, check_env_key_single_source OK, check_metric_lineage --gate PASS,
       check_definition_drift --gate PASS, check_migration_required_by OK, check_async_routes_not_blocking OK, git diff --check clean
```

The browser smoke caught a real defect on its first run. The side-by-side layout depended only on Tailwind classes, and the cards stacked when the CDN was not loaded. The fix sets the layout styles inline as well.

## Evals run

```text
python evals/run_vision_confirm_asks_replay_eval.py --live <live, READ ONLY> --scratch <throwaway postgres:16>
copied: 42 individuals, 337 sightings, 17 asks
OK: 3 ask(s) would open
  vision_same_individual 62cb5fcca03b435398da0871cefe1753 | f5242d408e814b89b347b1df580c5dbf  face_n 19/18  cosine 0.771  images 2
  vision_same_individual 4354c877c4974b8385a74bc52cab348b | 9bea16d559c442b38be8c23d0c5e7809  face_n 47/21  cosine 0.533  images 2
  vision_same_individual 4354c877c4974b8385a74bc52cab348b | f5242d408e814b89b347b1df580c5dbf  face_n 47/18  cosine 0.411  images 2
  "who is this?": 0 on the first tick. All 4 face individuals are in an open pair ask; they are asked after Juniper answers (3 slots left today).
```

## Docker/build/smoke checks

```text
No image build: no dependency or Dockerfile change. Integration ran against a throwaway postgres:16-alpine container (removed afterwards)
with the three walkway migrations applied; the new migration also applied cleanly a second time (idempotent).
Live deploy: not done (Juniper deploys). Live path: UNVERIFIED until deployed.
```

## Review findings fixed

Code review ran in a subagent against `origin/main...HEAD`.

- Finding (BLOCKER): the Hub's new answer/dismiss UPDATE could not be prepared by Postgres (`AmbiguousParameterError: could not determine data type of parameter $3`). Every Answer and Dismiss button would have failed, for every ask kind, including the existing ones. The route tests passed only because their fake pool never sends SQL to Postgres.
  - Fix: explicit `::text` casts. The statement now lives in the module constant `CLOSE_ASK_SQL`.
  - Evidence: the new `test_close_ask_sql_runs_on_real_postgres` prepares the real statement on a throwaway Postgres and covers yes / maybe (refused) / dismiss / a "who is this?" name. The old statement reproduces the `AmbiguousParameterError`; the new one passes.
- Finding (should-fix): a name was copied only one hop. With verdicts b=c and then a=b, naming a left c unnamed, and the opener would never ask about c.
  - Fix: `_propagate_label` walks the whole confirmed group with a recursive CTE. It still never overwrites a name Juniper gave directly.
  - Evidence: `test_a_name_reaches_the_whole_confirmed_group`.
- Finding (nit): "Not sure" blocked a pair forever.
  - Fix: dismissed pairs come back after `VISION_ASK_COOLDOWN_DAYS`, like expired ones.
  - Evidence: `test_not_sure_pair_comes_back_after_the_cooldown`.
- Not changed (noted):
  - A copied name gets no substrate entity (see Risks).
  - A missing migration stops the whole individuals loop. The deploy wrapper's `REQUIRED-BY` gate prevents that state.
  - "Yes"/"No" answers are lowercased for every ask kind. This is harmless.

## Restart required

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_confirm_asks_v1.sql
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low. Concern: the 0.40 band comes from one day and 108 face crops (mostly Juniper and neighborhood kids). Mitigation: it is a knob, and every answer now produces a labeled calibration point (`scripts/report_vision_individuals.py`).
- Severity: low. Concern: "same" links individuals but does not merge them, so future crops can still split. Mitigation: deliberate (no history is deleted). The verdict table is the input for a later threshold or merge change.
- Severity: low. Concern: a name copied across a "same" link does not create a substrate entity for the linked individual (the substrate listener only sees the asked individual). Mitigation: the Postgres label is correct; follow-up if the substrate needs it.
- Severity: low. Concern: the memory confirmation starvation in the brief is not real. `ORION_ASK_DAILY_CAP` has counted only vision asks since 2026-10-06. Mitigation: the new cap is still separate, so face asks do not use up the 10-sightings rule's budget either.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
