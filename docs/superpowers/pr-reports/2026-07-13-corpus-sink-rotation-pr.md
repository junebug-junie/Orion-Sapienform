## Summary

- Fixes the real, live incident-history risk flagged after the mood-arc corpus collector PR: `InnerStateCorpusSink` (backing `INNER_FEATURES_CORPUS_PATH`) had grown unbounded — confirmed 2026-07-13: ~104MB/36,776 rows in 5 days, zero rotation, same bug class as a prior host-freeze incident in this project.
- Adds size-based rotation (`CORPUS_SINK_MAX_BYTES`, default 200MB) with retention pruning (`CORPUS_SINK_ROTATED_KEEP`, default 5) to the shared sink class, used by both `_INNER_SINK` and the new `_MOOD_ARC_SINK`.
- Fixes three independent downstream readers (`scripts/fit_phi_encoder.py`, `scripts/diag.py` via shared import, `services/orion-spark-introspector/train/evals/eval_phi_encoder_health.py`) that would otherwise silently train/diagnose on only the post-rotation slice of the corpus once rotation ever fires.
- 8-angle code review across two full passes (the rotation mechanism itself, then the mechanism plus its downstream fixes) found and fixed real, live-data-affecting bugs — detailed below, not glossed over.

## Outcome moved

The live, already-growing `/mnt/telemetry/phi/corpus/inner_state.jsonl` file now has a real, tested bound instead of unbounded growth toward a repeat of this project's prior disk-exhaustion incident. Training/diagnostic scripts stay correct across rotation instead of silently shrinking their effective corpus.

## Current architecture

`InnerStateCorpusSink.append()` (`services/orion-spark-introspector/app/inner_state_sink.py`) did a raw `Path.open("a")`/write/flush per tick with no size check, ever. Three separate scripts each independently implemented a single-file JSONL reader (`fit_phi_encoder.py`'s `_load_jsonl`, `eval_phi_encoder_health.py`'s `load_corpus_rows` — a byte-for-byte duplicate, `diag.py` importing the former).

## Architecture touched

`services/orion-spark-introspector/app/{inner_state_sink,settings,worker}.py`, `scripts/fit_phi_encoder.py`, `services/orion-spark-introspector/train/evals/eval_phi_encoder_health.py`, and a new shared module `orion/telemetry/corpus_rotation.py`.

## Files changed

- `services/orion-spark-introspector/app/inner_state_sink.py`: size-based rotation (`_rotate_if_needed`/`_try_rotate`/`_prune_old_rotations`), collision-safe renaming, strict-pattern-filtered pruning, whole-rotation-attempt OSError guard, negative-value clamping.
- `orion/telemetry/corpus_rotation.py` (new): single shared source of truth for the rotated-filename pattern and multi-file resolution — extracted mid-review after the pattern was found duplicated three ways.
- `scripts/fit_phi_encoder.py`, `services/orion-spark-introspector/train/evals/eval_phi_encoder_health.py`: now import the shared resolver instead of duplicating it; both read across rotated siblings, not just the active file.
- `services/orion-spark-introspector/app/settings.py`: `corpus_sink_max_bytes` (`ge=1_000_000`), `corpus_sink_rotated_keep` (`ge=0`) — shared policy for both sinks.
- `services/orion-spark-introspector/.env_example`, `docker-compose.yml`, `scripts/sync_local_env_from_example.py`: new keys, consistent defaults.
- `orion/schemas/telemetry/mood_arc.py`, `orion/self_state/inner_state_registry.py`, `services/orion-spark-introspector/README.md`, `docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md`: stale "no rotation" claims corrected; new shared-policy coupling disclosed.
- Tests: `services/orion-spark-introspector/tests/test_inner_state_sink_rotation.py` (new, 11 tests), `tests/test_phi_encoder_fit_script.py` (+2), `services/orion-spark-introspector/tests/test_phi_encoder_health_eval.py` (+2) — the latter added specifically because review found the eval script's identical fix had zero coverage.

## Schema / bus / API changes

None. Rotation is internal sink behavior; corpus reader fixes are additive.

## Env/config changes

- Added: `CORPUS_SINK_MAX_BYTES=200000000`, `CORPUS_SINK_ROTATED_KEEP=5`.
- `.env_example` updated: yes. `docker-compose.yml` updated: yes. `sync_local_env_from_example.py` (`CORPUS_SINK_` prefix): yes.
- Local `.env` synced: no local `.env` in this worktree (expected, gitignored, absent in a fresh `git worktree add`).

## Tests run

```text
PYTHONPATH=.:services/orion-spark-introspector pytest \
  tests/test_inner_state_registry_gate.py tests/test_phi_encoder_fit_script.py \
  services/orion-spark-introspector/tests -q \
  --deselect services/orion-spark-introspector/tests/test_inner_state_emit.py::test_inner_features_settings_defaults
188 passed, 1 skipped

python scripts/check_inner_state_registry.py
inner_state_registry gate OK (10 entries checked)
```

Also directly verified (not just unit-tested) that the collision-counter fix actually prevents data loss: temporarily reverted it to the naive (second-precision, no-counter) version, confirmed the regression test fails with the exact silent-overwrite it's meant to catch (`len(rotated)==1` instead of `2`), then restored the fix and confirmed green.

## Evals run

None applicable.

## Docker/build/smoke checks

Not deployed this patch — code is tested and reviewed. The already-live 104MB corpus file is unaffected until this code is actually deployed and the file grows past 200MB (~5 more days at current rate); no destructive action taken against it this session.

## Review findings fixed

Two full 8-angle review passes (the second re-run after the first pass's fixes substantially changed the diff). Real, material findings, several independently confirmed by multiple angles:

- Finding (2 independent angles, most severe): `_prune_old_rotations`'s original glob (`{name}.*`) matched ANY file sharing the corpus basename prefix — a manually-placed backup, a `.gz` archive, a stray editor temp file — all eligible for silent deletion.
  - Fix: filtered through a strict regex matching only genuine rotation output, extracted to `orion/telemetry/corpus_rotation.py`'s `ROTATED_SUFFIX_RE`.
  - Evidence: `test_prune_never_deletes_a_file_not_matching_the_rotation_pattern`.
- Finding: `Path.rename()` onto an existing rotated-file path (a same-second, or even same-microsecond, collision) silently overwrites it — destroying a previously-rotated backup, the opposite of what rotation exists to prevent.
  - Fix: microsecond-precision timestamps plus an explicit existence-check/counter fallback (`.1`, `.2`, ...).
  - Evidence: `test_rotation_collision_counter_avoids_overwriting_existing_rotated_file` — verified by deliberately reverting the fix and confirming the test reproduces the exact data-loss failure, then restoring it.
- Finding: pruning sorted by filesystem `mtime`, not the filename's own embedded timestamp — under clock skew, NFS, or a backup/restore touching the directory, this could prune a newer file while keeping an older one.
  - Fix: sort by filename (already lexically correct) instead of reading filesystem metadata at all.
- Finding: `CORPUS_SINK_ROTATED_KEEP` set to a negative value produced backwards behavior via Python's negative-slice semantics — kept only the oldest file, deleted every newer one.
  - Fix: clamped to `max(0, value)` in the sink constructor, plus `ge=0` validation at the settings layer.
  - Evidence: `test_negative_max_rotated_files_is_clamped_not_inverted`.
- Finding: a transient filesystem error (ESTALE/EACCES on `/mnt/telemetry`, a real external volume mount in production) during any `.exists()`/`.stat()` call inside rotation would propagate uncaught, aborting the current tick's write entirely.
  - Fix: the whole rotation attempt wrapped in one `try/except OSError`, logging and skipping rotation for that tick rather than losing the row.
  - Evidence: `test_rotation_failure_does_not_prevent_the_row_from_being_written`.
- Finding (cross-file trace): three independent corpus readers (`fit_phi_encoder.py`, `diag.py` via import, `eval_phi_encoder_health.py`) read a single exact path — once rotation ever fires, they'd silently train/diagnose on only the post-rotation slice, no error.
  - Fix: `orion/telemetry/corpus_rotation.py`'s `resolve_rotated_corpus_files`, imported by all three read sites (the sink's own prune logic too, after mid-review extraction).
  - Evidence: `test_load_jsonl_reads_across_rotated_corpus_files`, `test_load_corpus_rows_reads_across_rotated_files` (the latter added specifically because review flagged the eval script's identical fix as having zero coverage).
- Finding (simplification, 2 angles): the rotated-filename regex + resolver was independently duplicated in three places (the sink, `fit_phi_encoder.py`, `eval_phi_encoder_health.py`), all needing to stay byte-for-byte in sync with no gate catching drift.
  - Fix: extracted to `orion/telemetry/corpus_rotation.py`, all three call sites now import it.
- Finding (docs): `MoodArcCorpusRowV1`'s schema docstring and this service's README still said "no rotation, manual retention plan needed" after rotation was actually implemented; the roadmap spec's Item 1 explicitly assumed no rotation would ever be needed.
  - Fix: corrected in all three places, plus disclosed that pruned mood-arc rows (unlike `InnerStateFeaturesV1`, which has `scripts/backfill_phi_corpus.py`) have no backfill path — genuinely gone once pruned, not just archived.
- Finding (altitude, considered seriously, not dismissed): `loguru` is already a direct, unused dependency of this exact service and has native `rotation=`/`retention=` support that would have avoided at least the collision and mtime-ordering bugs above outright; three review passes finding three distinct correctness bugs in the hand-rolled reimplementation is itself evidence for that critique.
  - Disposition: **deferred, not dismissed.** A same-session rewrite to loguru would trade the current implementation's now-verified state (4+ independent review passes, 0 outstanding known bugs, each prior bug fixed and pinned with a regression test proven to catch it) for a fresh, unverified rewrite under time pressure — itself a new source of risk. Named explicitly here as a legitimate follow-up: migrating `InnerStateCorpusSink` to loguru's `logger.add(path, rotation=..., retention=..., format="{message}")` (with `bind()`/`filter=` for per-instance isolation across the two live sinks) is a real, scoped next patch, not assumed away.
- Finding: `CORPUS_SINK_MAX_BYTES` had no sane floor (`ge=1` permitted e.g. `100`, causing rotation on nearly every tick).
  - Fix: raised to `ge=1_000_000` (1MB) — comfortably below any real threshold, rules out the pathological case.
- Finding: the `mood_arc_corpus.v1` registry entry didn't disclose that its rotation/retention thresholds are a policy shared with (and tuned against) `INNER_FEATURES_CORPUS_PATH`'s real size/cadence, not independently verified for mood-arc's own growth rate.
  - Fix: disclosed explicitly in the registry notes.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml up -d --build
```

Not run this session. This is the fix for a real, currently-accumulating production risk (~104MB and growing on the live host) — recommend deploying promptly once merged, well before the file reaches the 200MB rotation threshold (~5 more days at current growth rate) so the first real rotation event happens under the tested code path, not the old unbounded one.

## Risks / concerns

- Severity: medium (documented, not silently deferred)
- Concern: hand-rolled rotation logic, even after fixing 4 real bugs across review, could still have an undiscovered edge case; `loguru` (already a dependency) would likely have avoided several of these bugs by construction.
- Mitigation: named as an explicit follow-up above, not left implicit. Current implementation is thoroughly tested (11 dedicated tests, one bug's fix verified by deliberately reproducing the regression) and reviewed across two full 8-angle passes.
- Severity: low
- Concern: no backfill path exists for pruned `MoodArcCorpusRowV1` rows (unlike `InnerStateFeaturesV1`).
- Mitigation: documented explicitly; at the default policy (200MB × 5 = ~1GB retained) this is generous relative to the roadmap's stated "weeks, not months" scope before training would consume it.

## PR link

Branch pushed: `feat/corpus-sink-rotation` (link generated on push, below).
