# Enable the randomized holdback at 5%

## Summary

- Flips `ORION_DISPATCH_HOLDBACK_FRACTION` 0.0 -> 0.05. Config only; the machinery shipped earlier and has been inert since.
- Motivated by a null and a non-null: **no individual action Orion has beats doing five other things**, but **acting at all** is associated with an ~8x larger movement in `execution_pressure` — and only a randomized arm can say whether that is real.
- Records the measurement, the power calculation, the cost, and an explicit exit criterion in `.env_example` so this cannot quietly become a permanent capability tax.
- Live-verified end to end: withheld ticks are firing and landing in `substrate_signal_control_cells` under `arm='randomized_holdback'`.

## Outcome moved

"Does acting help at all?" goes from unanswerable to answerable in ~2 days. Every arm available until now was observational.

## What the measurement actually found

`inspect capability:orchestration` was the last candidate for "an action that works" — raw delta -0.0365, n=909, t=-6.86. It does not survive.

Matched on the **exact** prior value of `execution_pressure` (a 6-valued step function, so exact matching beats decile bins), against ticks that acted *without* it:

```
before   treated n   treated    control    contrast   siblings (t/c)
0.0000        204    +0.1264    +0.1424     -0.0160     4.99 / 4.95
0.1518        129    -0.0265    -0.0396     +0.0131     5.00 / 5.00
0.2405       1201    -0.1039    -0.1250     +0.0210     4.99 / 5.00
0.3035         80    -0.1713    -0.1937     +0.0224     5.00 / 5.00
0.3923        330    -0.2839    -0.3007     +0.0167     5.00 / 5.00
```

Four of five bins positive, all tiny, sibling counts matched. The raw -0.036 was reversion from the starting value plus the effect of acting at all — the same two artifacts that inflated the docker family's raw -0.135 into a real-looking number.

Two incidental findings worth keeping:

- **`execution_pressure` takes six distinct values** across 82,998 frames (`0.3923 = 0.2405 + 0.1518`, `0.3035 = 2 x 0.1518`). It is a sum of a few near-constant contributors flipping on and off. Decile binning collapses 0.3035 / 0.3524 / 0.3923 into one bin, mixing three distinct states; exact-value matching is available and strictly better.
- **`inspect_execution_pressure` is named for a trigger it does not have.** Its copy says "execution pressure is elevated" (`orion/proposals/templates.py:50`), but it fires at mean baseline 0.1917 against a population mean of 0.2842 — i.e. when pressure is *below* average. Not fixed here; recorded.
- Two templates share `(inspect, capability:orchestration)` — `inspect_execution_pressure` and `inspect_attended_target` — which `substrate_action_outcomes`' `(dispatch_kind, target_id)` key cannot separate. Empirically only the first declares a signal, so today nothing is pooled, but the collision is real and `orion/execution_dispatch/builder.py:55-70` documents the same collision causing a live bug in the starvation counters.

## Why the holdback, and why now

Matched on exact prior value, acting ticks move `execution_pressure` further than idle ticks at every level, with the gap growing monotonically in the level:

```
before    act n   act delta   idle n   idle delta      gap
0.0000      584     +0.1368     8058      +0.1506   -0.0137
0.1518      223     -0.0320     5969      +0.0110   -0.0430
0.2405     1714     -0.1102    55095      -0.0555   -0.0547
0.3035      130     -0.1799     4118      -0.0943   -0.0855
0.3923      603     -0.2915    28634      -0.1267   -0.1648
```

Specific, not a generic "acting ticks differ" artifact: the same comparison gives -0.0008 for `resource_pressure` and +0.0085 for `reasoning_pressure`.

But acting-vs-idle is not randomized, and `execution_pressure` is fed by `cortex_exec_step_load` (`config/field/orion_field_topology.v1.yaml:182`), so a mechanical coupling to dispatch activity cannot be excluded from observational data. That is exactly the gap `arm='randomized_holdback'` exists to close.

## Power and cost

sigma ~ 0.13, ~1,200 acting ticks/day. At 0.05: ~60 withheld ticks/day, ~108 per arm (80% power, alpha 0.05, d=0.05) within ~2 days. The observed gap is 0.165, so a real effect of that magnitude resolves much sooner. Cost ~2% of Orion's daily actions.

## Files changed

- `services/orion-execution-dispatch-runtime/.env_example`: 0.0 -> 0.05, plus the measurement, power calculation and exit criterion inline.
- Local `.env` set to match (not committed, gitignored, verified with `git check-ignore`).

## Schema / bus / API changes

None.

## Env/config changes

- Added / removed / renamed keys: none
- Behaviour changed: `ORION_DISPATCH_HOLDBACK_FRACTION` 0.0 -> 0.05
- `.env_example` updated: yes
- local `.env` synced: by hand (`sync_local_env_from_example.py` reads `.env_example` from the primary checkout, so a worktree edit is invisible to it)

## Tests run

```
No code changed. The holdback path's tests already exist and pass:
pytest services/orion-execution-dispatch-runtime/tests/ tests/test_execution_dispatch_runtime_worker.py -q
-> 136 passed
```

## Evals run

The contrast eval (`orion/autonomy/evals/eval_action_value_contrast.py`) is hardcoded to `resource_pressure` and the three docker targets, so it cannot read this arm yet. The analysis above was done directly against `substrate_feedback_frames` with exact-value matching, which the eval does not implement. Generalizing the eval to an arbitrary `(signal, kind, target)` with exact-value bins for discrete signals is the follow-up; not claiming eval coverage for this change.

## Docker/build/smoke checks

```
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d   -> recreated, started
docker exec ... env | grep HOLDBACK -> ORION_DISPATCH_HOLDBACK_FRACTION=0.05
docker logs -> execution_dispatch_randomized_holdback withheld=5 fraction=0.050
substrate_signal_control_cells WHERE arm='randomized_holdback' -> populating,
  incl. execution_pressure bin 3
```

## Review findings fixed

No code changed, so no review gate. The consumer path was verified live rather than assumed, because enabling this costs real capability and a silently-dropped arm would spend that cost for nothing: `orion/feedback/outcome_resolution.py:211` sets `control_arm`, the writing branch at `:218-247` is reached because a holdback tick has `frame_dispatch_count == 0`, and the resulting rows were confirmed in Postgres.

## Restart required

Already applied:

```bash
cd /mnt/scripts/Orion-Sapienform-orchestration-contrast
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d
```

## Risks / concerns

- **Severity: medium.** This deliberately makes Orion do less — ~2% of daily actions. Bounded and reversible by one env value. The exit criterion is written into `.env_example`; if nobody reads the arm within a week, turn it off.
- **Severity: medium.** `execution_pressure` may be partly a readout of the dispatch pipeline's own state via `cortex_exec_step_load`. If so, the holdback will show a large "effect" that is mechanical rather than beneficial. The randomized arm measures the association honestly either way; interpreting it still requires tracing that channel's provenance, which this change does not do.
- **Severity: low.** A withheld tick's candidates are recorded as blocked with reason `randomized_holdback`; `_detect_blocked_review_loop` treats blocked candidates as obstruction. Known and deferred from the original holdback review, unchanged here, now actually reachable in production.

## PR link

<filled on push>
