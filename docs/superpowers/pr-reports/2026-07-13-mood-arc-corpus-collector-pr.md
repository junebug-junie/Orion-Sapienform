## Summary

- Implements Item 1 ("post-fix corpus collector") of `docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md` — the floor of a roadmap for a downstream windowed felt-state autoencoder.
- `handle_self_state()` (`services/orion-spark-introspector/app/worker.py`) now appends one `MoodArcCorpusRowV1` row per tick to an independent JSONL sink, gated solely on `MOOD_ARC_CORPUS_PATH` being configured (empty/unset = off, the default everywhere in this patch).
- No bus channel, no cognition consumer. Registered as `mood_arc_corpus.v1`, `composition_status=REHEARSAL` in `orion/self_state/inner_state_registry.py`.
- 8-angle code review (spawned before commit, per this repo's mandatory review gate) found and fixed real issues in the first draft — detailed below, not glossed over.

## Outcome moved

The roadmap's floor is now real, tested, deployable (currently off by default) infrastructure instead of a design doc. Nothing downstream (item 2's training run) can start until this has collected real hours of data — this patch makes that clock able to start.

## Current architecture

Before this patch, `orion-spark-introspector` had one corpus sink (`_INNER_SINK` / `InnerStateCorpusSink`, writing `InnerStateFeaturesV1` rows for the phi encoder's own training). No mechanism existed for capturing felt-state *trajectories* (as opposed to single-tick snapshots) at all.

## Architecture touched

`services/orion-spark-introspector/app/{worker,settings,inner_state_sink}.py`, `orion/schemas/telemetry/mood_arc.py` (new), `orion/self_state/inner_state_registry.py`, two READMEs, `.env_example`, `docker-compose.yml`, `scripts/sync_local_env_from_example.py`.

## Files changed

- `orion/schemas/telemetry/mood_arc.py` (new): `MoodArcCorpusRowV1` — `generated_at`, `self_state_id`, `coherence`/`energy`/`novelty`/`valence`, `valence_source`, `dominant_node`.
- `services/orion-spark-introspector/app/worker.py`: `_MOOD_ARC_SINK` instance; append call placed before the `_pub_bus.enabled` gate (see Review findings), guarded on `.enabled`, broad `except Exception`.
- `services/orion-spark-introspector/app/inner_state_sink.py`: `InnerStateCorpusSink.append()`'s type hint widened from `InnerStateFeaturesV1` to `BaseModel` — it's now genuinely used for two unrelated schemas.
- `services/orion-spark-introspector/app/settings.py`: `mood_arc_corpus_path` (`MOOD_ARC_CORPUS_PATH`, default `""`), given its own section, not nested under the Plan-1 inner-features comment block.
- `services/orion-spark-introspector/.env_example`, `docker-compose.yml`: new key (compose default deliberately empty/off — see Risks).
- `scripts/sync_local_env_from_example.py`: `MOOD_ARC_` added to `SYNC_PREFIXES`.
- `orion/self_state/inner_state_registry.py`, `orion/self_state/README.md`, `services/orion-spark-introspector/README.md`: registry entry + doc updates, corrected mid-review (see below).
- `docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md`: brought into this PR — it was sitting uncommitted in a different checkout, cited by 4 files in this diff as if already merged.
- `services/orion-spark-introspector/tests/test_mood_arc_corpus.py` (new): 6 tests.

## Schema / bus / API changes

- Added: `MoodArcCorpusRowV1` (file-only, no bus channel).
- Compatibility notes: none — purely additive, no existing schema/bus contract touched.

## Env/config changes

- Added keys: `MOOD_ARC_CORPUS_PATH` (service `.env_example`, `docker-compose.yml`, `sync_local_env_from_example.py`).
- `.env_example` updated: yes.
- Local `.env` synced: not applicable — no local `.env` exists in this worktree (expected; worktrees don't carry gitignored files). No live deployment this session.
- Skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=.:services/orion-spark-introspector pytest \
  tests/test_inner_state_registry_gate.py services/orion-spark-introspector/tests -q \
  --deselect services/orion-spark-introspector/tests/test_inner_state_emit.py::test_inner_features_settings_defaults
161 passed, 1 skipped, 1 deselected

python scripts/check_inner_state_registry.py
inner_state_registry gate OK (10 entries checked)

python -c "import yaml; yaml.safe_load(open('services/orion-spark-introspector/docker-compose.yml'))"
YAML valid
```

The one deselected test is a pre-existing, unrelated worktree artifact (no local `.env`), confirmed identical on a clean `main` checkout — not touched by this patch.

## Evals run

None applicable — instrumentation only, no model/training component in this item.

## Docker/build/smoke checks

Not deployed. `MOOD_ARC_CORPUS_PATH` defaults to empty everywhere (settings default, `docker-compose.yml`'s `${MOOD_ARC_CORPUS_PATH:-}`), so this ships inert until an operator explicitly opts in.

## Review findings fixed

Spawned 8 parallel review angles against the initial implementation (written by a subagent per this repo's subagent-driven-development pattern) before committing. Real, material findings, independently confirmed by multiple angles in several cases:

- Finding (5 of 8 angles, independently): the append call originally sat *after* `if not (_pub_bus and _pub_bus.enabled): return`, silently coupling this bus-independent training-data sink to bus health — contradicting the design's own stated intent and diverging from `_INNER_SINK`'s placement.
  - Fix: moved before that gate. `phi_now`, `valence_source`, `dominant_node` are all already finalized by that point in the function.
  - Evidence: new regression test `test_mood_arc_corpus_appends_even_when_pub_bus_disabled`.
- Finding (3 of 8 angles): `except OSError` only wrapped the file write; `MoodArcCorpusRowV1(...)` construction happened inside the same try block, so a pydantic `ValidationError` (or any non-`OSError`) would propagate past it and abort the rest of `handle_self_state()` — including the real, consumed `spark_state_snapshot` publish that follows.
  - Fix: `except Exception`, plus an `if _MOOD_ARC_SINK.enabled:` guard so disabled deployments pay zero construction cost and inherit zero of this risk.
  - Evidence: new regression test `test_mood_arc_corpus_construction_failure_does_not_abort_tick`, which forces a construction failure and asserts the snapshot still publishes.
- Finding (2 of 8 angles, one tied directly to CLAUDE.md's "runtime truth beats config truth" rule): docs described collection as "gated on `valence_source` being present" — never true in code; `valence_source` is a plain `str` defaulting to `"heuristic"`, never `None` on this path.
  - Fix: corrected in the registry entry and both READMEs to describe the real gate (`MOOD_ARC_CORPUS_PATH` configured).
- Finding (angle H, quoting CLAUDE.md 0A directly): the roadmap spec doc cited by this same diff's registry entry and both READMEs did not exist in the repo — dangling reference, exactly the "no runtime proof" pattern the cited rule bans.
  - Fix: brought the spec doc into this PR.
- Finding (angle C/D): `InnerStateCorpusSink.append()`'s type hint/docstring declared an `InnerStateFeaturesV1`-only contract, silently violated by this patch's second, unrelated schema.
  - Fix: widened to `BaseModel`, docstring updated to state it's now a generic sink.
- Finding (angle D): `MoodArcCorpusRowV1.timestamp` broke the `generated_at` naming convention every sibling telemetry schema uses.
  - Fix: renamed.
- Finding (angle D): `MOOD_ARC_` missing from `sync_local_env_from_example.py`'s `SYNC_PREFIXES` and from `docker-compose.yml`'s environment block, unlike every sibling corpus-path key.
  - Fix: added both.
- Finding (angle G, tied to this project's own incident history): the shared `InnerStateCorpusSink` class has no rotation/retention and has already grown to ~98MB/36k rows in 5 days for its existing use (`INNER_FEATURES_CORPUS_PATH`) — the same class of unbounded-write bug behind a prior host-freeze incident in this project.
  - Disposition: documented prominently (schema docstring, both READMEs), not code-fixed — building real rotation infrastructure is out of proportionate scope for Item 1 and would touch already-shipped, already-running shared infra. `docker-compose.yml`'s new key defaults to empty specifically because of this (unlike `INNER_FEATURES_CORPUS_PATH`'s already-accepted always-on default), keeping this patch's own blast radius at zero until someone deliberately opts in. Flagged explicitly below, not silently left implicit.
- Finding (angle B): `orion/self_state/README.md`'s intro paragraph listing tracked signal types wasn't updated to mention the new entry, even though the REHEARSAL bullet a few lines below named it.
  - Fix: added.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml up -d --build
```

Not run this session — not deployed. Ships inert (empty default) even once deployed, until `MOOD_ARC_CORPUS_PATH` is explicitly set.

## Risks / concerns

- Severity: medium
- Concern: no rotation/retention story for this sink, and the shared sink class it reuses has a real, live precedent of unbounded growth in this exact deployment (~98MB/36k rows/5 days), plus this project has a prior host-freeze incident from the same bug class (unbounded per-event disk writes).
- Mitigation: off by default everywhere (settings, `.env_example`, `docker-compose.yml`). Documented explicitly in the schema docstring and both READMEs as a real, unaddressed gap — not silently deferred. Building rotation is scoped as a legitimate future need, not assumed solved.
- Severity: low
- Concern: this patch reuses `InnerStateCorpusSink` for a second schema by duck-typing (it only ever calls `.model_dump(mode="json")`) rather than a more formal generic-container refactor.
- Mitigation: type hint widened to `BaseModel` (the honest, minimal fix) rather than over-engineering a `Generic[T]` class for two current callers.

## PR link

Branch pushed: `feat/mood-arc-corpus-collector` (link generated on push, below).
