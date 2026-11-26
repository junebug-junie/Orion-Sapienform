## Summary

Patch A of the approved shared-bet-primitives design (Refs #3989). It is a pure move: the same surprise
math that was copied into up to six files now lives in one place, and no live number changes.

- New `orion/prediction/surprisal.py` (stdlib only): `bet_terms`, `conformal_terms`, `fold`, `node_value`,
  `Z_FULL_SCALE`, `P_FLOOR`, `calm_excess_variance` (floored) and `calm_excess_variance_discrete` (joint
  pool), `independent_origin`, `shortest_horizon_per_target`, `bernoulli_entropy`, `log_loss`.
- `orion/prediction/grading.py` grows binary `calibration(probs, hits, bins)` and re-exports `log_loss` and
  `bernoulli_entropy` as part of the grader's API (defined once, in the stdlib module, because the bet terms
  need them without numpy).
- Every confirmed duplicate in the design's inventory is deleted; callers import the shared ones.
- Golden tests replay live stored rows through the pre-refactor code (a `git archive` of cf5889fdd) and pin
  every node value and bet term of the new code to 1e-9. Result: all 6 substrate nodes are bit-identical;
  the only inexact values are 10 reachable log losses at 1 ulp (6.9e-18, numpy log vs math log).
- street_rhythm stays shadow with no `calm_band` rule; a test pins it.
- Live anchor (Orion's review): replaying stored rows through the fold reproduces 95.6 % (room/body) and
  96.4 % (joint) of the node values the live substrate wrote in the last 24 h, the same on main and here.
  All 44 forecast series' conformal excess (84,500 scored rows, 3 days) are bit-identical old vs new.
  "No live change" on the running substrate stays UNVERIFIED until #4017 runs after the rebuild.

## Outcome moved

Failure mode removed: one surprise formula drifting from its copies. Before, a change to the fold, the calm
variance, or the yes/no terms had to be made in 4-6 files by hand. Now there is one definition, and a test
fails if a second `def bernoulli_entropy / log_loss / calm_excess_variance / independent_origin` or a second
`Z_FULL_SCALE =` appears anywhere in `orion/`, `services/` or `scripts/`. Patch B (act surprise as
-ln p(outcome)) imports this library.

## Current architecture

The four substrate surprise folds (`prediction_surprise`, `prediction_joint_surprise`,
`prediction_turn_surprise`, `prediction_street_rhythm`) each computed `z = sum(excess)/sqrt(sum calm_var)`
and `clip(z/3, 0, 1)` themselves, with their own `Z_FULL_SCALE`. orion-prediction's `turn_surprise.bet_terms`
and `street_rhythm.outcome_terms` held the same yes/no algebra. Binary entropy, yes/no log loss, Wilson and
binary calibration bins each existed in 2-4 places.

## Architecture touched

No service boundary moved. Shared library code under `orion/` (copied whole into every image that uses it:
orion-prediction, orion-substrate-runtime, orion-sql-writer, hub, thought, attention-runtime all `COPY orion`).
No bus, schema, env, Docker or migration change.

## Files changed

Duplicates removed (line numbers at cf5889fdd):

| Duplicate | Removed from | Count |
|---|---|---|
| Window fold `z = sum(excess)/sqrt(sum calm_var)`, `clip(z/3)` | `orion/substrate/prediction_surprise.py:178`, `prediction_joint_surprise.py:164`, `prediction_turn_surprise.py:112`, `prediction_street_rhythm.py:134` and its replay `:154`, `services/orion-prediction/app/turn_surprise.py:151` (`turn_surprise`) | 6 |
| One-outcome-per-target loop | `prediction_surprise.py:169`, `prediction_joint_surprise.py:155` | 2 |
| `Z_FULL_SCALE = 3.0` | `prediction_surprise.py:63`, `prediction_joint_surprise.py:47`, `prediction_turn_surprise.py:35`, `prediction_street_rhythm.py:47`, `calm_band.py:63`, `curiosity_seed_bar.py:84` | 6 |
| `node_value` (clip z/3) | `services/orion-prediction/app/turn_surprise.py:162` | 1 |
| `independent_origin` | `prediction_surprise.py:79`, `prediction_joint_surprise.py:67` | 2 |
| Floored calm variance | `prediction_surprise.py:66`, `calm_band.py:127` | 2 |
| Discrete conformal calm variance (moved, not duplicated) | `prediction_joint_surprise.py:51` | 1 |
| `P_FLOOR = 0.05` | `calm_band.py:65`, `services/orion-prediction/app/surprise.py:45` | 2 |
| Conformal surprisal/excess formula | `services/orion-prediction/app/surprise.py:82,86` (now call `conformal_terms`) | 1 |
| Yes/no bet terms | `turn_surprise.py:132` (`bet_terms`), `street_rhythm.py:72` (`outcome_terms`); both now build their own dataclass from `surprisal.bet_terms` | 2 |
| Binary entropy | `action_bets.py:66`, `turn_surprise.py:136`, `street_rhythm.py:75`, `orion/curiosity/value.py:122` | 4 |
| Yes/no log loss | `turn_surprise.py:83`, `reachable.py:314` | 2 |
| Wilson interval | `orion/harness/evals/promise_ledger_replay_eval.py:156` (now calls `orion/self_monitor/record.py:85`, which gained a `z` argument) | 1 |
| Binary calibration bins | `orion/prediction/grading.py:33` (inside `grade`), `reachable.py:359` | 2 |

Not merged, as the design says: exceedance Jeffreys, `act_chain` categorical score and bins, the `_gap`
wrappers, the Family C gate, `live.verdicts` Holm, `action_bets._paired_verdict`, `family_a.calm_point`,
reverie's SQL bins. Nothing in `orion/grammar/act_ledger.py` or `orion/prime/*`.

Other files:

- `orion/prediction/surprisal.py`: new, the library.
- `orion/prediction/grading.py`: `calibration`, re-exports.
- `orion/prediction/tests/surprisal_golden.py`: fixture dump (read-only psql) / build / compare.
- `orion/prediction/tests/fixtures/surprisal_golden_2026-11-26.json.gz`: inputs + pre-refactor outputs (362 KB, ids hashed).
- `orion/prediction/tests/test_surprisal_golden.py`, `services/orion-prediction/tests/test_surprisal_golden.py`: golden, delete check, shadow pin.
- `orion/substrate/tests/test_{prediction_surprise,prediction_joint_surprise,calm_band,curiosity_seed_bar}.py`: import the moved names from their new home.
- `services/orion-prediction/evals/run_reachable_replay_eval.py`: 3-arg `log_loss`.
- `.github/workflows/prediction-surprise-node-tests.yml`, `orion-prediction-tests.yml`: path triggers and the new test dir.

## Schema / bus / API changes

- Added: none (library functions only).
- Removed: none on the bus or HTTP. Python names removed: `prediction_surprise.calm_excess_variance` (its
  argument was the calm surprisal, not the floor; removed rather than re-exported under the same name so a
  stale caller fails loudly), `prediction_joint_surprise.calm_excess_variance` (now
  `surprisal.calm_excess_variance_discrete`).
- Renamed: none.
- Behavior changed: none. `log_loss` now takes the clip explicitly (`log_loss(p, y, P_CLIP)`).
- Compatibility notes: `reachable.calibration` keeps its output keys (`bin`, `n`, `mean_p`, `observed_rate`).

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed)
- skipped keys requiring operator action: none

## Tests run

```text
Golden (new):
  orion/prediction/tests                                  30 passed
  services/orion-prediction tests/test_surprisal_golden   23 passed (with turn/street/reachable/surprise/grading/action_bets: 145 passed)
  exactness: substrate 8,528 numbers, 0 differ at tolerance 0; service 11,532 numbers, 10 differ, max 6.94e-18
  fixture: 1,000 room/body scores, 1,000 joint rows, 766 chat bets (300 scored turns), 401 walkway
           expectations, 355 reachable hours; read from live tables 2026-11-26 03:46Z (read-only)

orion-prediction full suite (new vs a git archive of main, same venv):
  new 27 failed / 594 passed; main 28 failed / 593 passed. The 27 are the same tests on both
  (heat_balance, closet_heat_balance, causal, events, backtest: local venv library versions); main's extra
  one is an archive artifact. No failure is new.
orion-substrate-runtime tests: 653 passed, 13 failed; the same 13 fail on main (cf5889fdd).
orion-attention-runtime tests + evals: 66 passed, 8 skipped.
orion-thought tests: 628 passed, 31 skipped.
Workflow steps run locally: prediction-surprise-node-tests, orion-attention-runtime-tests,
  self-monitor-tests, promise-ledger-tests, self-readings-tests, street-attention-tests,
  retired-pe-guards: all pass.
orion-static-gates: all 43 run steps pass locally.
```

## Evals run

```text
evals/run_turn_surprise_replay_eval.py --fixture ...2026-11-13.json --check   CHECK PASS; output byte-identical to main's
evals/run_street_rhythm_calm_eval.py --fixture ...2026-11-25.json --check     GATE PASS; output byte-identical to main's
```

## Docker/build/smoke checks

```text
Not run: no Dockerfile, compose, requirement or env change. Import paths checked instead: every image that
imports these modules copies all of orion/ (orion-prediction, orion-substrate-runtime, orion-sql-writer,
hub, thought, attention-runtime). orion-retro copies only orion/retro, which imports none of them.
The orion-prediction evals that load substrate modules by file path still run (they import
orion.prediction.surprisal, present in /app/orion).
```

## Review findings fixed

A separate review agent rebuilt the fixture from a fresh `git archive cf5889fdd` and matched the committed
`expected` at tolerance 0, so the golden tests do not compare the new code to itself. No material findings.
Minor findings and what was done:

- Finding: reachable log loss moved np.log -> math.log (<= 1 ulp on 10 live rows), not stated.
  - Fix: stated here and in the `surprisal.py` docstring; the contract is 1e-9, node values are bit-identical.
  - Evidence: tolerance-0 compare, max |d| 6.94e-18.
- Finding: `calm_band.bet_window_atoms` keeps its own closed-form bet terms (listed under Merge in the design).
  - Fix: deliberately left; comment added. Its atoms are compared within `_EPS`, and `-(y-q) logit q` vs
    `s - H` differ by ulps. Pinned by the golden key `calm_band_bet_window_atoms`.
  - Evidence: `orion/substrate/calm_band.py` `bet_window_atoms`.
- Finding: workflows that run code now depending on `orion/prediction/surprisal.py` did not trigger on it.
  - Fix: `orion/prediction/**` added to `promise-ledger-tests.yml` and `orion-prime-tests.yml` paths.
  - Evidence: workflow diff. (Pre-existing gap, not fixed here: no workflow runs `orion/curiosity/tests/test_curiosity_value.py`; the golden test pins `entropy_nats`.)
- Finding: promise eval Wilson docstring said "stdlib only". Fix: corrected.
- Finding: delete check read only module-level defs. Fix: now `ast.walk` over all `def`/`async def`.
- Finding: `fold` signature differs from the design (`fold(terms, window, key, thin)`). Fix: noted here.
  Window, key and thinning stay with each node's state; the shared pieces are `independent_origin` and
  `shortest_horizon_per_target`. `fold(terms, *, min_var)` only sums.
- Finding: `conformal_terms` recomputes the calm variance per call. Not changed: a handful of flops per
  scored forecast (one every few seconds).
- Finding: PR report not committed. Fix: committed.

## Orion's review (thread 05720fc6ebaf) and what changed

1. Fixture anchored to the bus: new read-only eval `services/orion-prediction/evals/run_surprise_node_live_parity_eval.py`
   replays stored rows against the node values the live substrate wrote: room/body 525/549, joint 541/561
   (24 h, identical on main). Off the n_cal grid the inversion raises (loud); live n_cal is 950-1010.
2. Bit-identical was measured on fixtures; live is UNVERIFIED until the post-rebuild run (#4017, verify-live).
3. One-off compare of every series' conformal excess: 44 series, 84,500 rows, 0 differences at ==.
   The fixture is refreshed only by a patch allowed to move a number (`surprisal_golden.py dump` + `build`
   against the pre-change tree).
4. `fold` now returns None whenever summed calm variance is not > `min_var` (default 0), for every caller.
   No live path reached a zero sum before (room/body/joint variances are > 0, turn rejects var <= 0,
   street already passed 1e-12), so this guards, it does not move a number.
5. One form: `calm_band.bet_window_atoms` now uses `surprisal.bet_terms`. Atoms move by <= 3.6e-15 in z;
   the closest atom tolerance edge to a stored 6-decimal value is 1.1e-10 away. `test_calm_band.py` pins a
   1000x margin; z_tail golden is unchanged at tolerance 0. Patch B uses -ln q(y) (`surprisal`) only.
6. Wilson: golden now pins fractional Kish n_eff (0.37 .. 163.4) and n_eff <= 0; `record.Z == 1.96` pinned.
   Each home keeps its own empty reading: record (0, 1), promise eval None/None.
7. No 2-arg `log_loss` call remains (grep). The golden helper no longer sniffs the signature: `build`
   passes `pre_change=True` explicitly.
8. Street pin fails with an "update deliberately" message and notes the 0.59/0.74 miscalibration; the
   tautological Z_FULL_SCALE checks now pin 3.0 and the bars; `test_curiosity_value.py` runs in CI; the
   promise lane pins its Wilson as literals and imports no golden code (its numpy/jinja2 installs predate
   this PR); unmerged items listed in #4018.

## Restart required

No restart is needed for correctness: no live number changes. The new code goes live when the images
rebuild. Rebuild order, if done by hand (the post-pull rebuild does this on its own):

```bash
scripts/safe_docker_build.sh orion-prediction up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

Other images that copy orion/ (sql-writer, hub, thought, attention-runtime, durable-runs, mind) pick it up
on their next rebuild; none of them needs one for this patch.

## Risks / concerns

- Severity: low. Concern: a later patch moves a number by accident. Mitigation: the golden tests fail at
  1e-9; re-building the fixture is a deliberate, documented step (`surprisal_golden.py build`).
- Severity: low. Concern: the delete check scans `orion/`, `services/`, `scripts/` on every run (~1 s).
- Severity: medium. Concern: "no live change" is UNVERIFIED on the running substrate (windows are in memory).
  Mitigation: #4017 (verify-live), run `run_surprise_node_live_parity_eval.py --since <rebuild> --check`.
- Issues opened: #4017 (verify-live), #4018 (unmerged near-duplicates reference). Closed: none (Refs #3989).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/4013

🤖 Generated with [Claude Code](https://claude.com/claude-code)
