## Summary

- Every walkway person crop that already gets a SigLIP body vector now also gets a second body vector from OSNet, a model built to tell people apart (person re-identification). It is stored next to the SigLIP vector for about a week so the two can be scored against each other. **Shadow only:** the matcher still uses SigLIP. Nothing about who a crop is assigned to changes.
- The body model that actually runs is now visible. The profile asks for SigLIP2, but SigLIP2 does not load, so the host silently runs SigLIP v1. That is now an ERROR log line at load, a `models.body_embedder` block on `/health`, a `body_embedder_fallback:<id>` entry in `/ready`'s `degraded_reasons`, and an `embedding_model_id` on every stored crop. The live body model is **not** switched.
- A kill switch, `VISION_OSNET_SHADOW_ENABLED` (on), stops it. If OSNet cannot load or fails on a frame, only the OSNet vector is lost. Detection, the SigLIP vector and faces are unchanged.
- A read-only eval, `services/orion-sql-writer/evals/run_osnet_vs_siglip_eval.py`, scores both models on the same crops. It says "insufficient data" until about a day of crops carry both vectors.

## Outcome moved

Before: there was no way to test whether a person re-identification model separates walkway people better than SigLIP, because no crop had both vectors and only 390 of 4,988 person crops had a thumbnail (design doc, "A better body vector"). The body model was silently the fallback.

After deploy: every embedded person crop carries both vectors, and the eval scores them on geometry-labelled pairs and on Juniper's picture verdicts. The fallback is named in health, readiness and every row.

## Current architecture

- orion-vision-host `crop_embeddings.py` crops each tracked box in a zone and embeds it with the `embed_image` profile (SigLIP). The loader asks for SigLIP2, fails under transformers 4.49.0, and falls back to SigLIP v1 with one WARNING line. The crop result reported the requested model id.
- Vectors ride `VisionObject` -> orion-vision-window `VisionCropV1` (`extra="forbid"`) -> orion-sql-writer `vision_crop_observation`.
- The individuals matcher (`vision_individuals.py`) reads `embedding` and `face_embedding` by explicit column list.

## Architecture touched

- orion-vision-host: new `app/osnet_shadow.py` (load, checksum, preprocess, embed, status) and `app/osnet_ain.py` (vendored network); `crop_embeddings.py` attaches the OSNet vector; `runner.py` wires it and exposes `model_status()`; `model_manager.py` records requested vs loaded body model; `main.py` adds `models` to `/health` and `/ready`; Dockerfile downloads the pinned weights.
- Contract: three additive optional fields on `VisionObject` and `VisionCropV1`.
- orion-vision-window: forwards the fields; leaves them out of the payload while empty.
- orion-sql-writer: three nullable columns, written by the crop handler. The matcher is not touched.

## Files changed

- `services/orion-vision-host/app/osnet_shadow.py`: OSNet loader (sha256 check, `weights_only=True`), Torchreid test transform, failure state with a 10-minute retry, `/health` status, VRAM delta.
- `services/orion-vision-host/app/osnet_ain.py`: OSNet-AIN network vendored from Torchreid (MIT, commit f8cd150), classifier head and gdown download removed.
- `services/orion-vision-host/app/crop_embeddings.py`: `_attach_osnet` on embedded person crops; `embedding_model_id` on every embedded crop.
- `services/orion-vision-host/app/runner.py`: builds `OsnetShadow`, warms it where `embed_image` is enabled, passes the actually loaded body model id, strips the new keys on a crop failure, `model_status()`.
- `services/orion-vision-host/app/model_manager.py`: `embedder_status` (requested, loaded, fallback, error), ERROR line on fallback.
- `services/orion-vision-host/app/main.py`: `models` on `/health` and `/ready`; `model_degraded_reasons`.
- `services/orion-vision-host/app/artifacts.py`: carries the new fields onto `VisionObject`.
- `services/orion-vision-host/app/settings.py`, `.env_example`, `docker-compose.yml`: `VISION_OSNET_SHADOW_ENABLED`, `VISION_OSNET_WEIGHTS_PATH`.
- `services/orion-vision-host/Dockerfile`: downloads the weights at a pinned Hugging Face revision and checks the sha256.
- `services/orion-vision-host/tests/test_osnet_shadow.py`: new.
- `orion/schemas/vision.py`: additive fields.
- `services/orion-vision-window/app/crops.py`, `app/main.py`, `tests/test_crop_observations.py`: forward, omit while empty, test.
- `services/orion-sql-db/manual_migration_vision_osnet_shadow_v1.sql`: three nullable columns.
- `services/orion-sql-writer/app/vision_crop_persist.py`: writes the columns; the no-embed strip clears the OSNet vector too.
- `services/orion-sql-writer/evals/run_osnet_vs_siglip_eval.py`: new read-only eval.
- `services/orion-sql-writer/tests/test_vision_osnet_shadow.py`: new (pure + real Postgres).
- `.github/workflows/orion-sql-writer-tests.yml`: applies the migration and runs the new Postgres tests in the must-not-skip step.
- READMEs of the three services.

## Schema / bus / API changes

- Added: `VisionObject.embedding_model_id`, `.osnet_embedding`, `.osnet_model_id`; the same on `VisionCropV1`. Columns `vision_crop_observation.osnet_embedding real[]`, `osnet_model_id text`, `embedding_model_id text`. `/health` -> `models`; `/ready` -> `models` and new `degraded_reasons` entries (`ready` is unchanged by them).
- Removed: none.
- Renamed: none.
- Behavior changed: none for matching. `crop_embeddings` detect stats gain `osnet` (count) and `osnet_error`.
- Compatibility notes: `VisionCropV1` is `extra="forbid"`. The window omits the new keys while empty, so an older writer accepts crops until vision-host starts producing them. **Deploy order: migration, then orion-sql-writer, then orion-vision-window, then orion-vision-host.** No bus channel change.

## Env/config changes

- Added keys: `VISION_OSNET_SHADOW_ENABLED=true`, `VISION_OSNET_WEIGHTS_PATH=/app/models/osnet/osnet_ain_x1_0.pt` (orion-vision-host).
- Removed / renamed keys: none.
- `.env_example` updated: yes (orion-vision-host).
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-vision-host`: yes, both keys added; no diverged lines reported.
- skipped keys requiring operator action: none.

## Metric quality gate (new signal: the OSNet body vector)

1. Provenance: `app/osnet_shadow.py` `OsnetShadow.embed_fn` -> `crop_embeddings._attach_osnet`, on the same crop pixels SigLIP receives.
2. Independence: same input pixels as SigLIP, different model. It is meant to be compared with SigLIP, not added to it. It feeds nothing.
3. Theory anchor: OSNet-AIN (Zhou et al., ICCV 2019; TPAMI 2021) is trained with identity labels on person re-identification data (MSMT17, DukeMTMC-reID, CUHK03) to separate people; SigLIP is trained to match pictures to captions.
4. Live data: real weights load strictly (every layer, nothing extra); on real walkway thumbnails different crops gave cosines 0.32 to 0.78 and the output is deterministic. Separation on live pairs is exactly what the eval measures; **UNVERIFIED until deployed**.
5. Existing mechanism: none in the repo.
6. Reversibility: kill switch; three nullable columns that nothing reads except the eval; rollback SQL in the migration header.

## Tests run

```text
orion-vision-host (inside image orion-vision-host-vision-host:osnet-shadow, worktree mounted read-only, pinned to the T10):
  python3 -m pytest -q tests                         356 passed
  python3 -m pytest -q tests/test_osnet_shadow.py    22 passed (real-weights test ran, not skipped)
orion-vision-window: pytest -q tests                 160 passed
orion-sql-writer (throwaway postgres:16 with the six walkway migrations, ORION_TEST_PG_URI set):
  pytest tests/test_vision_osnet_shadow.py tests/test_vision_split.py tests/test_vision_confirm_asks.py
         tests/test_vision_walkway_persist.py        all passed
  pytest tests evals                                 874 passed, 14 failed -- the same 14 fail on origin/main
                                                     in this venv (notify/biometrics/chat-history env issues)
Gates: check_env_template_parity PASS, check_migration_required_by OK, check_env_key_single_source OK,
       check_metric_lineage --gate PASS, check_definition_drift --gate PASS,
       check_compose_no_relative_mounts PASS, check_service_env_compose_parity N/A (env_file), git diff --check clean
```

## Evals run

```text
services/orion-sql-writer/evals/run_osnet_vs_siglip_eval.py: exercised against the throwaway Postgres by
tests/test_vision_osnet_shadow.py (insufficient -> exit 2 with "INSUFFICIENT DATA"; 30 synthetic frames ->
58 track pairs, 30 same-frame pairs, exit 0, OSNet wins; verdict pairs expanded). Not run against live data:
the columns do not exist live until the migration is applied, and no crop has an OSNet vector yet.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-vision-host build      -> built; weights sha256 "OK" in the build log
  (image retagged orion-vision-host-vision-host:osnet-shadow; :latest restored to the live image 52f1035b9d06)
Throwaway container on the T10 (--gpus device=00000000:D8:00.0), live captioner sharing the card:
  OSNet load 0.2 s, weights on GPU 8.4 MB
  peak allocated 188 MB, reserved 292 MB at batch 8; nvidia-smi delta 310 MiB
  latency: 1 crop 15.5 ms (p50), 4 crops 18.5 ms, 8 crops 30.4 ms (3.8 ms per crop)
Runner smoke (real SigLIP on CPU, real OSNet, real walkway frame, two person boxes + one dog box):
  stats osnet=2; persons: siglip 1152-d + osnet 512-d; dog: siglip only
  body_model = google/siglip-so400m-patch14-384 (requested google/siglip2-so400m-patch14-384, fallback=true)
  ERROR line "[MODEL] BODY EMBEDDER IS THE FALLBACK ..." printed at load
```

## Review findings fixed

Review ran in a separate read-only subagent. No must-fix findings. Fixed:

- Finding: the eval named a winner on a strict AUC `>` (noise could decide it), and its docs claimed more than short-term pairs show.
  - Fix: paired bootstrap (same resampled pairs for both models, 200 rounds); a winner only when the 95% interval of the let-through difference excludes zero, else "tie". Let-through is the headline. Docstring and README now say geometry pairs test short-term consistency only; only verdict pairs test cross-visit matching.
  - Evidence: `test_bootstrap_calls_a_tie_when_the_models_score_the_same`; real-Postgres eval test still names OSNet on the synthetic street.
- Finding: track-pair labels ignored person boxes without vectors, so a never-embedded box could not veto an ambiguous pair.
  - Fix: the eval loads geometry for every person crop in the window (vectors only where both exist); pairs are scored only when both ends have both vectors. Rows are streamed (`yield_per=2000`) and the engine is `postgresql_readonly`.
  - Evidence: `test_a_box_without_vectors_still_vetoes_an_ambiguous_track_pair`.
- Finding: nothing pinned "OSNet never makes /ready false".
  - Fix: `test_ready_stays_true_when_osnet_fails_and_body_falls_back` calls `readiness_payload()`.
- Finding: `_attach_osnet` did not check vector length; NaN path untested.
  - Fix: shape check requires 512-d (`OSNET_DIM`, pinned equal to `FEATURE_DIM`); tests for a NaN row and a wrong dimension.
- Finding: vision-host README did not state the deploy order (an old window silently drops the fields).
  - Fix: "Deploy last" line in the vision-host README.
- Finding (nit): the body embedder was force-loaded even when a frame had nothing to embed.
  - Fix: `embedding_model_id` is read lazily after the embedder runs; test `test_body_model_id_is_read_after_the_embedder_ran`.
- Finding (nit): OSNet counters updated without the lock; one lock across devices.
  - Fix: counters under the lock; the cross-device lock is commented as accepted (0.2 s once per device).
- Finding (nit): window omit test validated against the new schema only.
  - Fix: also validates the crop against a frozen pre-change `VisionCropV1` (`extra="forbid"`).
- Not fixed here: sql-writer's write-failure fallback log puts `str(exc)` in the detail, which for a SQLAlchemy `StatementError` can include bound vectors (pre-existing; now one more vector). Needs `hide_parameters=True` on that engine in a separate PR.

## Restart required

Do not run these from a worktree. From the clean primary `main` checkout after merge:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_osnet_shadow_v1.sql
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-vision-window up -d --build
scripts/safe_docker_build.sh orion-vision-host up -d --build
```

(The post-pull auto-apply step should apply the migration by itself; the first line is the manual fallback.)

Post-deploy check:

```bash
curl -s localhost:32797/health | python3 -m json.tool | grep -A12 '"models"'
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "SELECT count(*) FILTER (WHERE osnet_embedding IS NOT NULL) osnet, count(*) person, max(observed_at) FROM vision_crop_observation WHERE label='person' AND embedding IS NOT NULL AND observed_at > now() - interval '15 minutes'"
python services/orion-sql-writer/evals/run_osnet_vs_siglip_eval.py --live postgresql://postgres:...@localhost:55432/conjourney   # after ~1 day
```

Expected: `models.osnet_shadow.state = loaded`, `models.body_embedder.*.loaded = google/siglip-so400m-patch14-384`, `osnet` close to `person` in the SQL count.

## Risks / concerns

- Severity: low. Concern: the build now downloads 8.5 MB from Hugging Face; if it is unreachable the vision-host build fails. Mitigation: Docker layer cache keeps it after the first build; the sha256 check is the point.
- Severity: low. Concern: about 0.3 GB more on the T10 (about 4 GB free before). Mitigation: kill switch; measured, not estimated.
- Severity: low. Concern: the weights are a third-party conversion of the Torchreid checkpoint. Mitigation: pinned revision and sha256; strict load proves every layer matched the network; `weights_only=True`.
- Severity: low. Concern: pre-existing -- a failed crop write logs `str(exc)`, which can carry bound vectors. Mitigation: separate PR (`hide_parameters=True`).
- Severity: info. Concern: the geometry pair labels are proxies (same as the design doc). Mitigation: they come from box geometry, not from either model, so neither model grades itself; Juniper's verdict pairs are scored separately.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2720

🤖 Generated with [Claude Code](https://claude.com/claude-code)
