# PR report: DriveEngine fold-batch clamp collapse fix

**Read this before assuming the live symptom is closed.** This PR fixes the aggregation-math
bug that *creates* drive-pressure ties. It does **not** un-stick the specific
`coherence`/`capability`/`predictive` tie already observed live in production — that tie
persists under pure decay and will keep replaying indefinitely until either a genuine
differentiating tension reaches those three drives, or a separate, newly-discovered, unstarted
gap (§ below) is investigated and fixed. Status: **DONE_WITH_CONCERNS**, not DONE — see Concerns.

## Summary

- `DriveEngine.update()`'s live path summed every tension's impact into `impact_sum[drive]`
  unbounded across a whole fold batch, then clamped the final sum once. When that clamp
  saturated to exactly `±1.0`, the result collapsed to exactly `1.0`/`0.0` regardless of starting
  pressure — any two drives whose batch-summed impact both exceeded the bound landed on an
  identical value.
- Caught live, mid-collapse: a live gate re-run (post-deploy of the field-digester fix, PR
  #1144) showed the dominance-attribution fix holding (32.05% top share, unchanged) but the gate
  still reading SATURATED via `all_active_frac=0.9872`. Pulling live `drive_pressures` caught
  `coherence`/`capability`/`predictive` pinned to byte-identical `0.45036942460343243`.
- Fixed: apply the leaky-integrator recurrence sequentially, once per tension event, instead of
  summing first and clamping once. A follow-up 8-angle review found this makes the result
  order-dependent (unlike the old commutative sum) and the caller's buffer isn't guaranteed
  order-identical across bus redelivery — fixed by sorting tensions by `(ts, artifact_id)` first.
- **Investigated further, not just assumed fixed**: re-deriving the old clamp math by hand shows
  it can only ever produce exactly `0.0` or `1.0` — never a mid-range number like
  `0.45036942460343243`. Traced the actual origin: a fold at `2026-07-17 08:39:42 UTC` snapped
  all six drives to exactly `1.0` in one tick (the fixed bug, caught in the act). Three of those
  six never received a differentiating impulse afterward, so the tie persisted under pure decay
  down to the observed value — same original `1.0`, just decayed.
- **New finding, not fixed here**: real, weight-differentiated tension kinds ARE being minted
  and logged as attributed for this trio (~600 times/9.6h) but never reach the fold buffer that
  actually updates `DriveEngine` — a separate, unexplained gap. Added to the agent board
  (`0498728c-21f1-4048-bb19-c5b0caa29350`) and retrospective §6 item 7. This is the actual
  blocker on the live tie ever resolving.

## Outcome moved

Future fold-batch collapse events are prevented. The specific collapse event that produced the
currently-observed `coherence`/`capability`/`predictive` tie already happened before this fix
shipped and is not retroactively corrected — see Concerns.

## Current architecture

`_update_drive_pressures()` (`orion/spark/concept_induction/bus_worker.py:805-873`, O2) folds a
buffered batch of tensions into `DriveEngine.update()` at most once per
`_DRIVE_FOLD_INTERVAL_SEC=900s`. `update()`'s live (`leaky_math_enabled=True`) path summed every
event's impact per drive, then clamped once. `_update_drive_pressures` feeds
`prior.get("pressures")` straight back in as `previous_pressures` on every fold, with no
tie-detection or reconciliation step.

## Architecture touched

`orion/spark/concept_induction/drives.py`'s `DriveEngine.update()` only. No changes to
`bus_worker.py`, fold cadence, tension-minting, or any persistence/reconciliation layer.

## Files changed

- `orion/spark/concept_induction/drives.py`: sequential per-tension update (live path only,
  legacy `soft_saturate` path untouched); tensions sorted by `(ts, artifact_id)` before
  applying, for delivery-order independence; explanatory comments for the fix and a
  defensive-no-op double-clamp found by review.
- `orion/spark/concept_induction/tests/test_drives_leaky.py`: 3 new regression tests (core
  collapse-fix proof with hand-verified math matching the actual live symptom's mechanism,
  saturation-still-works, cross-call determinism). All 11 pre-existing tests pass unmodified.
- `docs/superpowers/specs/2026-07-17-drive-engine-fold-batch-clamp-collapse-fix-design.md`:
  quick spec.
- `orion/autonomy/README.md`, `orion/autonomy/drives_and_autonomy_retrospective.md`: status
  updates (new §5d, §6 items 5/7 updated, §7 index entries).

## Schema / bus / API changes

None. `DriveEngine.update()`'s signature and return shape are unchanged.

## Env/config/settings changes

None.

## Tests run

```text
$ pytest orion/spark/concept_induction/tests/test_drives_leaky.py -q
14 passed

$ pytest orion/spark/concept_induction/tests -q
168 passed

$ PYTHONPATH=. python orion/autonomy/evals/run_homeostatic_drives_eval.py
RESULT: PASS (all 5 checks — flood->0 tensions, events mint tensions,
pressures differentiate, rest toward zero, dominant reflects events)
```

## Evals run

`orion/autonomy/evals/run_homeostatic_drives_eval.py` — PASS. Multi-tension-batch trajectory at
t=419s shows `continuity=0.6729`/`capability=0.6731` — genuinely differentiated (not identical),
confirming the fix works on a real multi-event batch, not just the hand-constructed unit tests.

## Docker/build/smoke checks

Not applicable — pure Python library code, no service restart required. Live verification was
done directly against the running `orion-spark-concept-induction` worker's already-deployed
behavior via Postgres queries (see the retrospective §5d for the exact evidence trail); this fix
itself has not yet been deployed as of this PR.

## Review findings fixed

- Finding: sequential application makes the result order-dependent; the caller's arrival-order
  buffer isn't guaranteed order-identical across bus redelivery/retries (removed-behavior angle).
  - Fix: sort `tensions` by `(ts, artifact_id)` before the sequential loop.
  - Evidence: `test_drives_leaky.py`'s determinism test; full suite re-run (168 passed) after
    the change.
- Finding: outer `clamp_signed()` in the impulse computation is a defensive no-op with no
  comment explaining it (simplification angle).
  - Fix: added a comment explaining it's kept as a cheap safety net, not load-bearing.
  - Evidence: manual review.
- Finding (not fixed, reported): duplicated event/drive-impacts iteration scaffold between the
  live and legacy branches (reuse angle).
  - Fix: none — the legacy branch is explicitly a rollback-only path; factoring it into a shared
    helper with the live path risks entangling the rollback path's guarantees with active
    development on the live path. Judgment call, not an oversight.
  - Evidence: n/a.
- Finding (not fixed, reported): full `orion/spark/concept_induction/tests` suite and the
  homeostatic drives eval weren't explicitly confirmed run in the first commit message
  (conventions angle).
  - Fix: both re-run and results included in this report and the final commit message.
  - Evidence: see Tests/Evals run above.
- Finding (the load-bearing one, altitude angle, 3 independent findings converging on the same
  point): this fix does not retroactively un-stick an already-persisted tie, and shipping it
  without being explicit about that risks the PR reading as closing the live incident when it
  only closes the forward-looking half.
  - Fix: not a code fix — addressed by making this explicit, prominently, in this PR's title,
    summary, status, and in the retrospective/README updates, plus filing the real blocker (§6
    item 7 / agent board `0498728c`) as its own tracked open question rather than letting it get
    lost.
  - Evidence: this report's own framing; retrospective §5d.

## Restart required

No restart required for this fix to be safe (pure aggregation-math change, no schema/config
change) — but the live `orion-spark-concept-induction` worker needs a redeploy to pick up the
fix at all, since the current running code still has the bug:

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-spark-concept-induction/.env \
  -f services/orion-spark-concept-induction/docker-compose.yml \
  up -d --build
```

## Risks / concerns

- Severity: high (framing risk, not correctness risk).
  Concern: this PR could be read as "the drive collapse bug is fixed" when the live symptom that
  prompted the investigation — three drives stuck at an identical value — is still stuck as of
  this writing and has no guaranteed resolution path.
  Mitigation: explicit status `DONE_WITH_CONCERNS`, explicit framing throughout this report and
  the docs, a tracked follow-up (§6 item 7, agent board `0498728c`) naming the actual blocker.
- Severity: medium.
  Concern: §6 item 7 (differentiated tensions minted but not reaching the fold buffer) is a real,
  unexplained gap discovered as a side effect of this investigation, not something this PR set
  out to find or fix. It's cognition-loop-adjacent (touches `bus_worker.py`'s tension-to-drive
  pipeline) and needs its own investigation and likely its own sign-off before a fix, per this
  repo's proposal-mode convention for invasive cognition changes.
  Mitigation: filed on the agent board and in the retrospective rather than silently deferred;
  not started in this PR.
- Severity: low.
  Concern: no mechanism exists at the persistence layer to detect or automatically break an
  already-stuck tie (raised by the altitude review as a question, not built here).
  Mitigation: deliberately not built — the root cause (§6 item 7) isn't understood yet, and a
  tie-breaking jitter or reconciliation pass would paper over the real gap rather than fix it.
  Worth reconsidering once §6 item 7 is resolved, if a tie can still form through some other path.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/drive-engine-fold-batch-collapse
