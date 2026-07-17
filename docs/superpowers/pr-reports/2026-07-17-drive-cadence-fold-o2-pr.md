# PR report: rate-normalize the drive integrator (O2)

Branch: `feat/drive-cadence-fold-o2`
Series: follows `docs/superpowers/pr-reports/2026-07-16-drive-economy-desaturation-o1-o4-pr.md`
(O1+O4) and `docs/superpowers/pr-reports/2026-07-17-drive-predictive-reground-o3-pr.md`
(O3, PR #1114, merged). This is O2 -- the last of the two named follow-ups
from that diagnosis, and the last piece of the drive-economy desaturation
sequence (O1/O3/O4 already merged).

## Summary

- `DriveEngine.update()` (`orion/spark/concept_induction/drives.py`, untouched
  by this patch) was called once per raw bus event (~13/min, ~4.6s apart) from
  `orion/spark/concept_induction/bus_worker.py`. With `decay_tau_sec=1800.0`,
  decay between consecutive calls is negligible (~0.3%) -- repeated
  same-direction impulses converge pressure toward 1.0 within seconds
  regardless of tau (verified concretely during O3's review: a repeated
  impulse of ~0.5 pins pressure above 0.95 within ~5 ticks, ~23 seconds).
  This is the event-rate/decay mismatch mechanism behind the cpu/gpu_pressure
  saturation bugs (PRs #1108-1111), reproduced by any repeated same-direction
  impulse at bus cadence.
- New `ConceptWorker._update_drive_pressures(subject, tensions, now)` method
  splits read from write: every bus event still gets a fresh drive-pressure
  snapshot (a decay-only projection, `tensions=[]`, on non-fold ticks -- so
  goal proposal, dossier, identity snapshot, and the publish itself all keep
  getting a live `drive_state` every event, unchanged from before), but NEW
  tension impulses are buffered per-subject in memory and only actually
  applied + persisted to the integrator at most once per
  `_DRIVE_FOLD_INTERVAL_SEC` (900.0s, a first-pass calibration -- see the
  interval-math table in this report's history, not a new env key, same
  convention as O3's/O4's own threshold constants).
- Both existing call sites (`_handle_signal_drive_tick`, `handle_envelope`)
  refactored to call the shared method instead of inlining load/update/save.
- `DriveEngine` itself is completely unchanged -- this is purely a call-site
  cadence fix in the stateful caller, not a change to the integrator math.
  Every tension is still individually published to the bus exactly as
  before, on every event, regardless of fold status -- only whether it gets
  folded into the *persisted pressure* is throttled.

## Outcome moved

This closes the drive-economy desaturation diagnosis started 2026-07-15/16.
O1 fixed dominance-attribution poisoning; O3 gave `predictive` its first
primary tension source; O2 fixes the remaining pressure-pinning mechanism
(event-rate vs decay-rate mismatch) both of them shared. Expect
`measure_autonomy_gate.py`'s SATURATED verdict (added in O4) to clear for
honest reasons post-deploy, and `drive_audits` to show real, differentiated
per-drive variance instead of pinned-near-1.0 pressures -- this is the
post-deploy re-measurement this whole series has been building toward.

## Current architecture

`handle_envelope` and `_handle_signal_drive_tick` each independently loaded
`prior_drive_state`, computed `previous_ts`, called
`DriveEngine.update(tensions=<this event's tensions>, ...)`, and immediately
persisted the result via `store.save_drive_state(...)` -- once per raw bus
event, with no cadence awareness.

## Architecture touched

`orion/spark/concept_induction/bus_worker.py` only. No changes to
`DriveEngine`/`drives.py`, `store.py`, `ConceptSettings`, schema registry,
bus channels, or env.

## Files changed

- `orion/spark/concept_induction/bus_worker.py`:
  - New module constants `_DRIVE_FOLD_INTERVAL_SEC = 900.0` and
    `_MAX_PENDING_DRIVE_TENSIONS = 500` (see Review findings below for the
    latter).
  - `ConceptWorker.__init__`: two new in-memory per-subject dicts,
    `self._pending_drive_tensions` and `self._last_drive_fold_at`.
  - New `ConceptWorker._update_drive_pressures()` method.
  - `_handle_signal_drive_tick` and `handle_envelope`: both refactored to
    call the new method instead of inlining load/update/save.
  - `_log_drive_pressure_probe`'s docstring corrected (see Review findings).
- `orion/spark/concept_induction/tests/test_drive_pressure_fold.py`: new
  file, 5 tests (cold-start always folds, in-interval call neither folds nor
  persists, post-interval call folds everything buffered since the last real
  fold, a 30-tick steady-state saturation-prevention regression, and a
  buffer-overflow-caps-and-drops-oldest test added during review).

## Schema / bus / API changes

None. `TensionEventV1` publishing is unaffected -- every tension is still
published to the bus on every event; only persistence-into-the-integrator
cadence changed.

## Env/config changes

None. Both new thresholds are plain module-level constants, not env keys,
matching this patch series' established convention (O3's `0.30`/`0.05`
thresholds, O4's `SATURATION_*` constants).

## Tests run

```text
cd /mnt/scripts/Orion-Sapienform-drive-cadence-fold-o2
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/spark/concept_induction/tests -q
165 passed, 20 warnings in 12.57s
```

(164 before the review-driven overflow-cap addition, 165 after.)

## Evals run

No eval harness exists for this surface (tracked as issue #1066, same gap
noted in O1/O3/O4). The post-deploy live re-measurement below is the
behavioral check.

## Docker/build/smoke checks

Not run -- pure Python call-cadence change in an already-running consumer
path (`ConceptWorker`), no config/dependency/port/compose changes. Restart of
`orion-spark-concept-induction` picks this up.

## Review findings fixed

Full 8-angle `/code-review medium` pass (3 correctness angles, 3 cleanup
angles, altitude, conventions; 1-vote verify where needed).

- **Fixed (efficiency, real gap): unbounded pending-tension buffer.**
  `_pending_drive_tensions[subject]` had only a time-based bound (900s), no
  numeric cap -- each buffered `TensionEventV1` carries a full
  `ArtifactProvenance`, and this repo has an established convention of
  capping exactly this class of collection (`evidence_event_ids <= 200` in
  the pressure reducer and execution merge, cited precedent from prior
  fixes). Added `_MAX_PENDING_DRIVE_TENSIONS = 500`, drop-oldest-first on
  overflow (same "keep most recent" convention as `_prune_window`'s window
  trimming), with a `drive_pressure_fold_buffer_overflow` warning log so an
  overflow is observable, not silent. New regression test
  (`test_pending_buffer_caps_and_drops_oldest`) confirms the cap and the
  drop-oldest ordering.
- **Fixed (simplification, real): duplicate `DriveEngine.update()` calls.**
  The original fold/non-fold branches each called `self.drive_engine.update()`
  with nearly-identical arguments (differing only in `tensions=`). Collapsed
  to `tensions_to_apply = pending if should_fold else []` computed once, a
  single `update()` call, branching only on the persist/clear/log side --
  removes the duplication-drift risk the reviewer flagged. No behavior
  change (165/165 still passing after the refactor).
- **Fixed (cosmetic, cross-file trace): stale `_log_drive_pressure_probe`
  docstring.** Claimed to log "right after every `save_drive_state`," but
  post-O2 it's called on every event including non-fold ticks where
  `save_drive_state` is not invoked. Docstring corrected to point readers at
  the `drive_pressure_fold folded=` log line for that distinction, and to
  flag that the original `AutonomyStateV2` comparison this probe was built
  for is itself historical (V2 retired 2026-07-16, independent of this
  patch).
- **Checked and cleared: `goals.py` priority-scoring pressure lag.**
  `GoalProposalEngine._priority` weights `drive_state.pressures.get(drive_origin)`
  at 0.7 and a same-tick tension-magnitude term at 0.3. On non-fold ticks,
  the 0.7-weighted pressure term is a decay-only projection that doesn't yet
  reflect this tick's own new tension (buffered, not applied) -- up to 900s
  of lag on that one term specifically. This is an accepted, documented
  consequence of the fix (`dominant_drive` and the tension-weight term both
  still react immediately every tick; only the persisted-pressure component
  is throttled) -- not a bug, and not fixed, because "fold on every
  goal-proposal-relevant tick" would defeat the entire point of this patch.
  Documented directly in `_update_drive_pressures`'s docstring so it's not
  lost.
- **Checked and cleared: `scripts/drive_state_divergence_audit.py` /
  `services/orion-hub/scripts/drives_analytics_queries.py` reading a
  less-frequently-updated value.** The divergence-audit script's own
  docstring already documents that its `AutonomyStateV2` comparison side is
  "frozen/historical" (V2 retired 2026-07-16) -- it was already a
  point-in-time snapshot against a static baseline, not a live trend tool,
  so `DriveEngine`'s side updating at most every 900s doesn't introduce new
  noise; if anything it makes back-to-back script reruns more stable, not
  less. The Hub analytics panel showing a value that's honestly stale by up
  to 900s (vs. a value that updates every event but is pinned near 1.0, the
  actual pre-fix state) is a strict improvement, not a regression.
- **Checked and cleared: in-memory-only buffer lost on worker restart.**
  `_pending_drive_tensions` has no persistence -- a restart mid-fold-window
  silently drops buffered-but-unfolded tensions (their pressure effect
  vanishes for that window; the raw `TensionEventV1`s are still published to
  the bus first, so the audit trail is intact). This is the identical
  restart-loses-state risk profile every other in-memory per-subject buffer
  in this same file already carries (`self.window`, `self.last_run`,
  `self.recent_event_seen`) -- not a new class of fragility introduced by
  this patch, and adding persistence for this one buffer while the others
  stay in-memory would be an inconsistent, out-of-scope special case.
- **Checked and cleared (verified via trace, not assumed): no
  race/aliasing/mutable-default bugs.** `_update_drive_pressures` is a plain
  sync method with no `await` inside it, so no interleaving is possible
  between the two call sites even though they can share the same subject
  (`"orion"`). The `fold_tensions = pending` / `self._pending_drive_tensions[subject]
  = []` sequence rebinds the dict key to a *new* list object; the local
  variable retains the old reference, already consumed by
  `drive_engine.update()` before the reassignment -- no aliasing bug,
  confirmed by the passing `test_third_call_after_interval_folds_all_buffered_tensions_together`.
- **Checked and cleared: `previous_ts` tz-naive-guard asymmetry.** The old
  `handle_envelope` inline block lacked the `tzinfo is None` guard that
  `_handle_signal_drive_tick`'s inline block had; the new shared method now
  applies the guard uniformly. Confirmed this is a strict improvement, not a
  behavior change in practice -- `save_drive_state` always serializes from a
  tz-aware `datetime.now(timezone.utc)`, so the guard was already dead code
  on the `handle_envelope` path either way.
- **Checked and cleared (altitude): right layer, right generality.**
  `DriveEngine` is deliberately kept a pure, externally-stated function (no
  internal clock, no subject-awareness) -- it's directly instantiated by
  `orion/autonomy/evals/run_homeostatic_drives_eval.py` for deterministic
  math testing, and pushing cadence state into it would contaminate that
  determinism while duplicating the per-subject-dict pattern `ConceptWorker`
  already owns. Confirmed `bus_worker.py` is the *only* production caller of
  `DriveEngine.update()` in the repo, so a generic reusable "cadence-gated
  integrator" primitive would be premature abstraction against a population
  of one. Confirmed this fix is orthogonal to (not a second special case
  stacked on) the earlier `leaky_math_enabled` fix -- that one fixed the
  decay function's *shape* (removed the ~0.731 fixed point); this one fixes
  *call frequency vs. tau*.
- **Checked and cleared (reuse): no existing "buffer + fold" primitive
  duplicated.** `TensionRateLimiter` is a sliding-window drop-over-cap
  limiter (discards excess, never queues) -- a genuinely different contract
  from accumulate-and-flush-later. `DeviationGate` has no cadence concept.
  `endogenous_origination.py`'s `cooldown_sec` gate is structurally closest
  but is suppress-and-discard, not accumulate-and-flush. No existing
  primitive does what `_update_drive_pressures` needed; it's a legitimately
  new, correctly-scoped mechanism.
- **Checked and cleared (conventions): no CLAUDE.md violation.** Real test
  coverage attached to the new mechanism (including an explicit regression
  test for the saturation bug this patch fixes, per section 11's "every bug
  fix ships with a regression test that would have caught the bug"), a
  documented tunable constant rather than a bare magic number, no env/schema
  surface touched, minimal thin-seam patch riding the existing
  `DriveEngine.update()` call sites.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-spark-concept-induction/.env \
  -f services/orion-spark-concept-induction/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: LOW. Concern: `_DRIVE_FOLD_INTERVAL_SEC=900.0` is a first-pass
  calibration with no live pressure-distribution data to validate against
  (same honesty standard as every other threshold in this series). Interval
  math worked out during design: `p* = impulse / (1 - decay*(1-impulse))`
  with `decay = exp(-interval/1800)` -- at 900s, decay≈0.607, giving
  p*≈0.63 for a representative ~0.4 folded impulse (right at the 0.62
  activation threshold). Mitigation: plain module constant, trivially
  tunable; post-deploy re-measurement against `drive_audits` (same command
  used to verify O1/O3/O4) is the real check.
- Severity: LOW. Concern: if the *folded* impulse itself gets clamped near
  1.0 (many co-firing tensions summed within the 900s window before the
  `_clamp_signed([-1,1])` ceiling in `DriveEngine.update()`), decay stops
  mattering and pressure still approaches 1.0 regardless of interval choice.
  Cannot be ruled out without live data on how many co-firing, same-drive
  tensions typically land within one 900s window. Mitigation: same
  post-deploy re-measurement; if this manifests, the next lever is the fold
  interval itself, already isolated to one named constant.
- Severity: LOW. Concern (documented, not fixed): goal-proposal priority
  scoring's pressure term can lag a freshly-dominant drive by up to 900s.
  Mitigation: `dominant_drive` and the tension-weight term react
  immediately; only 70% of one scoring formula's inputs are throttled, and
  that's the intended trade this whole patch makes.

## PR link

(filled after push)
