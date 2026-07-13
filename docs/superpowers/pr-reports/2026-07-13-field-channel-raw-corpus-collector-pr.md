## Summary

- Corrects Item 1 of `docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md`: the shipped collector (`mood_arc_corpus.v1`, PR #989) captured `_phi_from_self_state()`'s output — 4 hand-tuned heuristic scalars (`coherence`/`energy`/`novelty`/`valence`), already smoothed by `orion-field-digester`'s `apply_decay(0.92)` and additionally hand-weighted. A full session of downstream autoencoder work (`feat/mood-arc-encoder-cli`, merged as PR #1019) found that any "trajectory structure" detected there was almost entirely explained by the known decay mechanism, not anything emergent.
- Adds `FieldChannelCorpusRowV1` (`orion/schemas/telemetry/field_channel_corpus.py`, new): captures `collect_field_channel_pressures()`'s flat channel dict straight from `FieldStateV1` — raw per-node/per-capability pressures (`cpu_pressure`, `gpu_pressure`, `memory_pressure`, `thermal_pressure`, `execution_load`, etc.), before any coherence/novelty/valence hand-weighting. Still carries the known `apply_decay(0.92)` (produced at the point `FieldStateV1` itself is computed — out of scope to remove here), but not additionally composited.
- New producer: `services/orion-field-digester/app/worker.py`'s `_tick()`, gated on `FIELD_CHANNEL_CORPUS_PATH` (empty/off by default, same convention as `MOOD_ARC_CORPUS_PATH`).
- Extracts `InnerStateCorpusSink` from `services/orion-spark-introspector/app/inner_state_sink.py` to `orion/telemetry/corpus_sink.py` — it had zero real service-specific coupling (only import was the already-shared `orion.telemetry.corpus_rotation`) and exactly one real call site (`orion-spark-introspector/app/worker.py`). `orion-field-digester` is a different service and must not reach into another service's internals per this repo's architecture rules — promoting an already-generic, already-decoupled class to the shared `orion/telemetry/` package (alongside `corpus_rotation.py`) is the correct fix, not a workaround.
- Registers `field_channel_corpus.v1` in `orion/self_state/inner_state_registry.py`; updates the existing `mood_arc_corpus.v1` entry to note supersession (that sink keeps running, untouched — not a gap to close by disabling it).
- Updates the roadmap spec doc's Item 1 section with the corrected design and a dated correction note explaining why (ties to the traced `apply_decay(0.92)` finding).
- Updates three READMEs (`orion-field-digester`, `orion/self_state`, `orion-spark-introspector`) to describe what's actually running.

## Outcome moved

Item 1 of the roadmap now targets the substrate layer the whole initiative actually needed to test for emergent structure, instead of a pre-smoothed, pre-composited summary. `mood_arc_corpus.v1` (the superseded sink) keeps collecting, unaffected — this is additive, not a rollback.

## Current architecture

Before this patch: `orion-field-digester` had no corpus-export mechanism at all — `FieldStateV1` was computed, decayed, diffused, and persisted to Postgres, but never exported as a flat time series. `InnerStateCorpusSink` lived inside `orion-spark-introspector`'s `app/` folder despite being fully generic.

## Architecture touched

`orion-field-digester` (new sink, settings, env, compose), `orion/telemetry/` (new shared `corpus_sink.py`), `orion/schemas/telemetry/` (new schema), `orion/self_state/inner_state_registry.py`, the roadmap spec doc, three service/self-state READMEs.

## Files changed

- `orion/telemetry/corpus_sink.py` (new, moved from `services/orion-spark-introspector/app/inner_state_sink.py`): generic append-only JSONL corpus sink, unchanged behavior.
- `services/orion-spark-introspector/app/inner_state_sink.py`: deleted (no re-export shim — exactly one real caller, fixed in the same patch).
- `services/orion-spark-introspector/app/worker.py`: one import line updated.
- `services/orion-spark-introspector/tests/test_inner_state_sink_rotation.py`, `tests/test_inner_state_features.py`: import paths updated to the new shared location (the old `importlib.util` file-loading hack, needed only because the class was buried in a service folder, is now a normal import — a real simplification, not just a mechanical fix).
- `orion/schemas/telemetry/field_channel_corpus.py` (new): `FieldChannelCorpusRowV1` — `generated_at`, `tick_id`, `channels: dict[str, float]` (variable-width, not fixed — documented for future Item 2 rework).
- `orion/telemetry/corpus_rotation.py`: minor doc/typing touch-up alongside the sink move.
- `services/orion-field-digester/app/worker.py`: `_FIELD_CHANNEL_SINK` instance, append call in `_tick()` using `collect_field_channel_pressures(state)`, guarded `try/except Exception`.
- `services/orion-field-digester/app/settings.py`: `field_channel_corpus_path`, `corpus_sink_max_bytes`, `corpus_sink_rotated_keep` (mirrors `orion-spark-introspector`'s existing fields exactly).
- `services/orion-field-digester/.env_example`, `docker-compose.yml`: new keys, empty/off by default.
- `scripts/sync_local_env_from_example.py`: `FIELD_CHANNEL_` added to `SYNC_PREFIXES`.
- `orion/self_state/inner_state_registry.py`: new `field_channel_corpus.v1` entry; `mood_arc_corpus.v1` entry's notes updated to reference supersession.
- `docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md`: Item 1 corrected, dated correction note added.
- `services/orion-field-digester/README.md`, `orion/self_state/README.md`, `services/orion-spark-introspector/README.md`: updated.
- `services/orion-field-digester/tests/test_field_channel_corpus.py`, `tests/test_field_channel_corpus_schema.py` (new): coverage for append/disabled/failure-doesn't-abort-tick and schema round-trip.

## Schema / bus / API changes

- Added: `FieldChannelCorpusRowV1` (file-only, no bus channel).
- Compatibility notes: `InnerStateCorpusSink`'s move is a pure relocation (identical class body), only import paths change. No existing schema shape changed.

## Env/config changes

- Added keys (`services/orion-field-digester/.env_example`): `FIELD_CHANNEL_CORPUS_PATH=`, `CORPUS_SINK_MAX_BYTES=200000000`, `CORPUS_SINK_ROTATED_KEEP=5`.
- `.env_example` updated: yes.
- Local `.env` synced: **not done — cannot be, from an isolated worktree with no local `.env`.** Run `python scripts/sync_local_env_from_example.py` on the real deployment host after this merges.
- Skipped keys requiring operator action: none beyond the sync step above.

## Tests run

```text
.venv/bin/python -m pytest services/orion-field-digester/tests -q
=> 37 passed

.venv/bin/python -m pytest services/orion-spark-introspector/tests -q
=> 165 passed, 1 skipped, 1 failed (test_inner_state_emit.py::test_inner_features_settings_defaults,
   seed-v3 vs seed-v4 features_version default — pre-existing, environment-dependent,
   unrelated to this patch; same failure documented against a clean checkout in the
   sibling PR's report for feat/mood-arc-encoder-manifest-schema)

.venv/bin/python -m pytest tests/test_field_channel_corpus_schema.py -q
=> 4 passed

.venv/bin/python scripts/check_inner_state_registry.py
=> inner_state_registry gate OK (13 entries checked)

.venv/bin/python -c "import yaml; yaml.safe_load(open('services/orion-field-digester/docker-compose.yml'))"
=> OK
```

## Evals run

None applicable — instrumentation/schema/registry patch, no model or training component.

## Docker/build/smoke checks

Not deployed this session. `FIELD_CHANNEL_CORPUS_PATH` defaults to empty everywhere (settings default, compose's `${FIELD_CHANNEL_CORPUS_PATH:-}`), so this ships inert until an operator explicitly opts in.

## Review findings fixed

Code-review skill run in a subagent (8 parallel angles) against the diff before commit; the review pass ran across a session interruption (usage-limit reset) and was resumed to completion. All acceptance checks above were re-verified after the review pass, immediately before this commit.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-field-digester/.env \
  -f services/orion-field-digester/docker-compose.yml up -d --build
```
Not run this session — not deployed. Ships inert (empty default) even once deployed, until `FIELD_CHANNEL_CORPUS_PATH` is explicitly set. `orion-spark-introspector` also needs a restart once merged, since its `worker.py` import path changed (mechanical only, no behavior change):
```bash
docker compose --env-file .env --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low
- Concern: no rotation-story validation yet at this service's real growth rate — same class of risk flagged for `mood_arc_corpus.v1` in its own PR, mitigated the same way (off by default).
- Mitigation: `CORPUS_SINK_MAX_BYTES`/`CORPUS_SINK_ROTATED_KEEP` reused (200MB/5 rotated), inert until opted in.
- Severity: low
- Concern: `FieldChannelCorpusRowV1.channels` is variable-width (channel set can vary tick to tick) — a future Item 2 rework consuming this corpus must handle a variable key set, not assume a fixed schema. Documented in the schema's own docstring; not solved here (no consumer exists yet).

## Merge order

No dependency on the two mood-arc schema/CLI branches — this branches clean from `origin/main`. `feat/mood-arc-encoder-cli` has already merged (PR #1019); this is independent and can merge on its own.

## PR link

`gh` is unauthenticated in this environment — open manually at:
https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/field-channel-raw-corpus-collector
