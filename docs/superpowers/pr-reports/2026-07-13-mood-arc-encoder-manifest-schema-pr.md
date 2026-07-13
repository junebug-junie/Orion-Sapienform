## Summary

- Adds `MoodArcEncoderManifestV1` to `orion/schemas/telemetry/mood_arc.py` — the manifest schema for Item 2 (windowed felt-state-trajectory autoencoder) of `docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md`. Sibling to `PhiEncoderManifestV1` (reuses `CorpusStatsV1`/`TrainingStatsV1` as-is from `orion.schemas.telemetry.phi_encoder`, not a subclass — different input/window semantics per spec).
- Fixes a pre-existing gap: `MoodArcCorpusRowV1` (shipped in PR #989) was never registered in `orion/schemas/registry.py`'s `_REGISTRY` or `SCHEMA_REGISTRY`, unlike its phi-encoder siblings. Both `MoodArcCorpusRowV1` and `MoodArcEncoderManifestV1` are now registered.
- Adds `tests/test_mood_arc_encoder_schema.py` covering registry resolution, `kind` string assertions, a `model_dump_json()`/`model_validate_json()` round-trip, and `extra="forbid"` rejection.
- No producer, no consumer, no runtime behavior change — schema-only slice, disk-only artifact, dark by design.

## Outcome moved

Establishes the schema contract Item 2's actual training CLI (`scripts/fit_mood_arc_encoder.py`, follow-up branch `feat/mood-arc-encoder-cli`) is built against, and closes a real, pre-existing registry gap for `MoodArcCorpusRowV1` from PR #989.

## Current architecture

Before this patch, `orion/schemas/telemetry/mood_arc.py` had only `MoodArcCorpusRowV1` (Item 1's per-tick corpus row schema), and neither it nor any mood-arc schema was registered in `orion/schemas/registry.py`.

## Architecture touched

`orion/schemas/telemetry/mood_arc.py`, `orion/schemas/registry.py`, new test file. No service, config, or runtime files.

## Files changed

- `orion/schemas/telemetry/mood_arc.py`: added `MoodArcEncoderManifestV1` class + import of `CorpusStatsV1`/`TrainingStatsV1`
- `orion/schemas/registry.py`: added import line, `_REGISTRY` entries, and `SCHEMA_REGISTRY` entries (with `kind`) for both `MoodArcCorpusRowV1` and `MoodArcEncoderManifestV1`
- `tests/test_mood_arc_encoder_schema.py`: new test file

## Schema / bus / API changes

- Added: `MoodArcEncoderManifestV1` (file-only schema, no bus channel)
- Registered (gap fix, not newly introduced): `MoodArcCorpusRowV1` in `_REGISTRY` / `SCHEMA_REGISTRY` with kind `self.mood_arc_corpus.v1`
- Registered (new): `MoodArcEncoderManifestV1` in `_REGISTRY` / `SCHEMA_REGISTRY` with kind `self.mood_arc_encoder.manifest.v1`
- Compatibility notes: purely additive, no existing schema shape changed

## Env/config changes

None — schema-only patch, no `.env`/`.env_example`/compose/requirements touched.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest \
  tests/test_mood_arc_encoder_schema.py \
  tests/test_phi_encoder_schema.py \
  services/orion-spark-introspector/tests/test_mood_arc_corpus.py -q
=> 10 passed, 16 warnings (warnings are pre-existing, unrelated protected-namespace deprecation notices)
```

## Evals run

Not applicable — schema-only patch, no model/training component in this item.

## Docker/build/smoke checks

Not applicable — schema-only patch, no runtime/config/Docker changes.

## Known acceptance-check conflict (flagged, not silently swallowed — resolved by a follow-up branch)

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/check_inner_state_registry.py
=> inner_state_registry gate FAILED:
   - orion/schemas/registry.py declares 'MoodArcEncoderManifestV1', which matches an
     inner-state keyword ("mood") and has no orion/self_state/inner_state_registry.py entry
```

Root cause: `scripts/check_inner_state_registry.py`'s keyword heuristic (`INNER_STATE_KEYWORDS` includes `"mood"`) flags any name added to `orion/schemas/registry.py`'s flat `_REGISTRY` that matches the keyword list and isn't covered by an entry in `orion/self_state/inner_state_registry.py`'s `REGISTRY` or the script's own `_EXTRA_COVERED_SCHEMA_NAMES` allowlist. `MoodArcCorpusRowV1` is already covered (it's the schema behind `signal_id="mood_arc_corpus.v1"`); `MoodArcEncoderManifestV1` was not, since it had no producer or inner-state-registry entry yet by design at the time of this patch.

This task was explicitly scoped to three files (`mood_arc.py`, `registry.py`, the new test file) and forbade touching `orion/self_state/inner_state_registry.py` — fixing this gate required a real producer to exist first. **Resolved in the follow-up branch `feat/mood-arc-encoder-cli`**, which adds `scripts/fit_mood_arc_encoder.py` (the producer) and the corresponding `mood_arc_encoder.v1` registry entry. `check_inner_state_registry.py` passes on that branch.

## Review findings fixed

Code-review skill not run in a subagent for this patch (scope was schema/registry/test-only, explicitly bounded to 3 files). No material issues found by manual inspection; patterns mirror existing `PhiEncoderManifestV1`/`PhiIntrinsicRewardV1` registrations exactly.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: `scripts/check_inner_state_registry.py` gate fails on this branch in isolation, until the follow-up producer branch merges.
- Mitigation: resolved by `feat/mood-arc-encoder-cli` (already pushed, depends on this branch merging first). Documented here rather than silently deferred.

## Merge order

This branch has no dependencies and can merge first. `feat/mood-arc-encoder-cli` branches from this branch's tip (not `main`) and should be rebased onto `main` once this merges.

## PR link

`gh` is unauthenticated in this environment — open manually at:
https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/mood-arc-encoder-manifest-schema
