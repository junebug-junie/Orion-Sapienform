## Summary

- A sleep's dream replay now takes at most one Temporal Self arc of each kind: the heaviest one.
- Before, body arcs (`interoception`) filled the arc slots. They run for hours, and their "returns" count sensor flicker, so they outweighed every other kind.
- The per-kind step runs before the 20-candidate cap, so a flood of one kind cannot push the other kinds out.
- The arc cap stays at a third of the replay (4 of 12). The dream arm and the control pool both draw from the same one-per-kind list.
- New replay eval rebuilds the last 14 days of real sleeps, with the old rule and the new rule.

## Outcome moved

Arc kind mix in dream replay, last 14 days (28 completed sleeps):

| | arcs | interoception | other kinds |
|---|---|---|---|
| what really ran | 39 | 35 (90%) | conversation 2, attention 2 |
| rebuilt, one per kind | 39 | 25 (64%) | attention 6, concern 4, conversation 2, imagery 2 |

Before, 9 sleeps replayed 2 or 3 body arcs. After, no sleep replays more than one arc of a kind. Concern and imagery arcs reach the dream for the first time in this window. The old-rule rebuild matches what really ran in 26 of 28 sleeps. The 2 that differ (dc-8d2ddce7485a, dc-0a88d68c3898) have arc rows rewritten after the sleep ran (`updated_at` 10-31 06:07).

## Current architecture

`replay.arc_candidates` deduplicated arcs by thing (kind + subject_ref) and kept the top 20 by weight. `select_replay` capped arcs at a third of the replay. Four different substrate nodes counted as four different things, so four body arcs could take all four arc slots.

## Architecture touched

orion-dream only: `arc_candidates` in `app/replay.py`. No contract, schema, bus, or env change.

## Files changed

- `services/orion-dream/app/replay.py`: one arc per kind before the cap. `one_per_kind=False` keeps the old rule, only for the eval's rebuild of past sleeps.
- `services/orion-dream/app/cycle.py`: comment only (states that the replay and the control pool share the one-per-kind list; corrects ARC_WEIGHT_MAX to 0.75).
- `services/orion-dream/README.md`: the replay rule.
- `services/orion-dream/tests/test_replay_arc_candidates.py`: regression tests from the 11-11 04:57 window (function, a real cycle with the control pool, and the per-kind step before the cap). Older tests that used one kind for many arcs now use distinct kinds. The per-subject test runs with `one_per_kind=False`, so it still proves the per-subject rule.
- `services/orion-dream/evals/dream_arc_kind_mix_replay.py`: fixture lane (CI) and read-only `--live` lane.
- `services/orion-dream/evals/test_dream_arc_kind_mix_replay_eval.py`: fixture lane under pytest.
- `.github/workflows/orion-durable-runs-tests.yml`: runs the arc replay tests and the eval. Before this, no CI job ran `test_replay_arc_candidates.py`.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: fewer arc candidates per sleep (one per kind). The cycle note `arc candidates=N` now counts kinds.
- Compatibility notes: none. The stored replay shape is unchanged.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
cd services/orion-dream && pytest tests evals -q -p no:cacheprovider
378 passed, 9 skipped
pytest services/orion-dream/tests/test_replay_arc_candidates.py services/orion-dream/evals/test_dream_arc_kind_mix_replay_eval.py  (CI step, from repo root)
24 passed
With the old rule forced as the default: 5 tests fail (regression tests catch the bug).
```

## Evals run

```text
python services/orion-dream/evals/dream_arc_kind_mix_replay.py          -> fixture: PASS
python services/orion-dream/evals/dream_arc_kind_mix_replay.py --live --days 14
  real (what ran):       39 arcs: interoception 35 (90%), conversation 2 (5%), attention 2 (5%)
  rebuilt, old rule:     41 arcs: interoception 37 (90%), conversation 2, attention 2
  rebuilt, one per kind: 39 arcs: interoception 25 (64%), attention 6 (15%), concern 4 (10%), conversation 2 (5%), imagery 2 (5%)
  fidelity 26/28 (93%) >= 80%: ok
  non-interoception share 10% -> 36%: ok
  most arcs of one kind in one sleep, new rule: 1
  live: PASS
```

Limit: non-arc items that lost their slot to an arc are not stored, so the rebuild cannot put them back. Two arc slots go free in the rebuild; a real sleep would fill them with the next existing item.

## Docker/build/smoke checks

```text
Not run. No dependency, config, or compose change. Behavior is covered by the cycle-level tests.
Live path after deploy: UNVERIFIED until the next sleep.
```

## Review findings fixed

- Finding: the per-subject test passed for the wrong reason. One-per-kind would pick the same arc.
  - Fix: the test runs with `one_per_kind=False` and adds a same-kind, different-subject row.
  - Evidence: test passes. It would fail if the per-subject key were removed.
- Finding: the live eval gate could not fail (both checks were true by construction).
  - Fix: those checks are now asserts. The gates are old-rule fidelity >= 80% and a rise in the non-interoception share.
  - Evidence: live run prints both gates (93%, 10% -> 36%).
- Finding: "freed slots" mixed rebuild error into the rule effect.
  - Fix: compares the old-rule rebuild with the new-rule rebuild.
- Finding: README "at most 20" wording, docstring summary line, two different evidence numbers, duplicated fixture.
  - Fix: reworded; both windows labelled (5 days: 15 of 18, 14 days: 35 of 39); tests load the fixture from the eval module.
- Also fixed: the cycle.py comment said ARC_WEIGHT_MAX is 0.8. The code says 0.75.

## Restart required

```bash
scripts/safe_docker_build.sh orion-dream up -d --build
```

Run from the clean primary main checkout after merge. The post-pull rebuild does this on its own.

## Risks / concerns

- Severity: low
- Concern: one body arc still reaches most sleeps (25 of 28). That is by design: one per kind, not zero.
- Mitigation: if body arcs should be rarer still, the next step is the arc weight (returns that count sensor flicker), not this cap.

## PR link

(filled after creation)
