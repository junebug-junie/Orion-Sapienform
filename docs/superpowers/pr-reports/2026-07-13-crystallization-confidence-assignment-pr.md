# PR report: deterministic confidence assignment (formation + reinforcement)

Implements Item 1 of `docs/superpowers/specs/2026-07-13-recall-epistemic-honesty-and-observability-spec.md`.

## Summary

- `CrystallizationConfidence` (`certain | likely | possible | uncertain`) has been a dead field since inception — every one of 164 historical rows in `memory_crystallizations` sits on the schema's `"likely"` default; nothing anywhere ever set it explicitly. `salience.py::CONFIDENCE_BOOST`, a real downstream consumer, has therefore never varied.
- Adds `infer_confidence()` (deterministic, no LLM, no I/O) to `orion/memory/crystallization/salience.py`, reading evidence count/strength and `dynamics.reinforcement_count`.
- `apply_salience()` now calls `infer_confidence()` as its first step — both real formation call sites (`proposer.py::propose`, `intake_pipeline.py::process_consolidation_crystallization`) get it for free, no call-site duplication.
- `dynamics.py::reinforce()` recomputes confidence after `reinforcement_count` increments — recurrence (the same fact independently re-derived) is real evidentiary support.
- **Hard invariant enforced by a dedicated test:** `recall_boost()` never touches confidence — being retrieved is not evidence something is true, only that it's relevant. This guards against the exact conformity-bias failure mode the companion reinforcement/decay spec named.

## Outcome moved

`score_salience()`'s confidence term goes from a silent flat constant (always `+0.05`, since `confidence` was always `"likely"`) to a real, evidence-driven signal. Two otherwise-identical crystallizations with different evidence support now get different salience.

## Current architecture (before this patch)

`apply_salience()` computed salience by reading `crystallization.confidence` — but nothing ever set that field to anything but the Pydantic model default. One whole dimension of the salience formula was inert.

## Architecture touched

`orion/memory/crystallization/salience.py`, `orion/memory/crystallization/dynamics.py`. No schema, bus, or API contract changes — `CrystallizationConfidence`/`dynamics.reinforcement_count` already existed; this patch makes them causally connected for the first time.

## Files changed

- `orion/memory/crystallization/salience.py`: new `infer_confidence()`; `apply_salience()` calls it first.
- `orion/memory/crystallization/dynamics.py`: `reinforce()` recomputes confidence after incrementing `reinforcement_count`; one new import (`salience.py::infer_confidence`).
- `tests/test_memory_crystallization.py`: `TestInferConfidence` (12 cases covering the full tiering table) + 2 formation/salience-movement acceptance tests (spec's acceptance checks 1 and 2).
- `tests/test_memory_crystallization_dynamics.py`: `TestReinforceConfidence` (confidence climbs tiers across repeated reinforcement, monotonic — never regresses on a stale recompute) + `test_recall_boost_never_touches_confidence` (the hard invariant).

## Design decisions worth flagging

**Where `infer_confidence()` lives:** in `salience.py`, next to `CONFIDENCE_BOOST`/`score_salience()` (they're conceptually paired). Checked for an import cycle first — `dynamics.py` importing from `salience.py` is safe (`salience.py` has no `dynamics.py` import), confirmed by direct import.

**Call-site consolidation:** the spec suggested calling `infer_confidence()` "wherever `apply_salience()` is currently called." Instead, it's called once, inside `apply_salience()` itself, so every real caller gets it automatically with no duplication. Verified the full list of real `apply_salience()` callers: `proposer.py::propose()` (used by Hub's manual propose route and `orion-memory-crystallizer`'s worker) and `intake_pipeline.py::process_consolidation_crystallization()` (used by `orion-memory-consolidation`'s worker, including the window-intake path). Also checked `intake_autonomy_episode.py::build_crystallization_from_episode()` — it does **not** call `apply_salience()` and has no live service consumer today (only referenced from a smoke script) — nothing to wire there, consistent with the spec's own arsonist summary.

**Tiering resolution:** the spec's table lists conditions in presentational (ascending-confidence) order, but the row conditions overlap — e.g. a single moderate-strength source with zero reinforcement literally matches both the "possible" and "likely" rows as written. Implemented as strength-ordered evaluation (floor → certain → possible → likely → fallback) so each case lands on the tier the spec's intent actually describes. Documented inline in the function's docstring.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
$ source venv/bin/activate && python -m pytest tests/test_memory_crystallization.py tests/test_memory_crystallization_dynamics.py -q
1 failed, 58 passed, 6 warnings in 3.03s
```

The 1 failure (`TestMemoryCardBackwardCompat::test_memory_card_v1_unchanged_in_registry_gap`) is pre-existing and unrelated — independently confirmed by running the identical test against clean `origin/main`:

```text
$ python -m pytest tests/test_memory_crystallization.py::TestMemoryCardBackwardCompat::test_memory_card_v1_unchanged_in_registry_gap -q
1 failed, 6 warnings in 3.09s
```

Both runs independently re-executed by the orchestrator, not taken from the implementing agent's report alone. The 16-test `test_memory_crystallization_dynamics.py` suite specifically, including the two new confidence tests and the hard-invariant test, passes 16/16:

```text
$ python -m pytest tests/test_memory_crystallization_dynamics.py -v
... 16 passed
```

## Evals run

No eval harness applies to this deterministic formula. Acceptance checks 1 and 2 from the spec (real variance appears post-deploy; salience actually moves between differently-evidenced crystallizations) are implemented as unit tests above rather than a live-deploy check, since the logic is deterministic and fully covered by construction — unlike the companion reinforcement/decay spec's saturation gate, there's no live-traffic-dependent behavior to separately verify here.

## Docker/build/smoke checks

Not applicable — this is a pure Python library change (`orion/memory/crystallization/`), consumed by `orion-hub`, `orion-memory-crystallizer`, and `orion-memory-consolidation` without any of their own code changing. No new dependency, no Docker rebuild required for those services to pick this up (they already `COPY orion /app/orion` from the shared package).

## Review findings fixed

**Worktree-isolation anomaly caught and corrected by the orchestrator, not the implementing agent.** During implementation, the agent reported observing a "concurrent sibling agent's" uncommitted edits (from the parallel `feat/recall-reinforcement-decay-wiring` patch) appearing in this worktree's working tree — despite the two worktrees being confirmed-distinct physical directories (different inodes, verified via `stat`). The agent's own commit (`f39ec431`) was already correctly scoped to only its intended 4 files at the time it reported this, but left 5 stray uncommitted files (byte-identical to the sibling patch's changes) sitting in the working tree afterward. Orchestrator independently verified via `git show f39ec431 --stat` that the commit itself was clean, then discarded the stray uncommitted files with `git restore` (safe — they were uncommitted duplicates of work the sibling patch owns and commits separately in its own branch). Final pushed state contains only this patch's intended 4 files, confirmed via `git diff origin/main --stat` post-rebase.

## Restart required

```text
No restart required.
```
Pure library change; consuming services (`orion-hub`, `orion-memory-crystallizer`, `orion-memory-consolidation`) pick it up on their next normal rebuild/restart, no special action.

## Risks / concerns

- Severity: Low — the tiering thresholds are a first-cut heuristic (explicitly named as such in the spec). Real-world calibration against post-deploy data (spec acceptance check 1: query the confidence distribution on new rows after deployment) is a natural follow-up, not blocking for this patch.
- Severity: Low — no retroactive backfill of the 164 existing `"likely"` rows, matching the spec's explicit non-goal. They'll pick up real confidence only when/if they're independently reinforced going forward.

## PR link

Not opened via `gh` (no token, SSH-only remote, consistent with every PR this session). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/crystallization-confidence-assignment?expand=1`
