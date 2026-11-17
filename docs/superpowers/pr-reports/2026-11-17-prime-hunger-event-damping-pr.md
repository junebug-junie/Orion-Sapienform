# Prime hunger: kept concerns no longer shield an ignored event want from damping (#3552)

## Summary

- Prime woke about once an hour for reading wants it then ignored. Each time it kept its 3 concerns and did nothing
  else. The halving rule (guard 1) treated "kept a concern" as an answer to every `horizon` need. The reading wants
  are also `horizon` needs, so they were never halved.
- The halving is now kept per **damp key**, not per need name. The key is the need name, except a funded event
  want's horizon need. That need uses `horizon:event:<its family>`, for example `horizon:event:hub.world_pulse_read`.
- A kept concern still protects its own horizon need. It no longer protects the event family.
- An event need's drive stops at 0.5, which is also the wake threshold. So after one ignored reading, no other
  reading can wake Prime until that halving fully recovers (the reading's own 3 h window).
- The log line `prime_needs_damped` now names the keys that were really halved. It used to name every shown need,
  which is why the live log said `connection,horizon` while `horizon` was never halved.

Fixes #3552

## Outcome moved

The replay eval has a new planted case with the live shape: 3 concerns kept on every thought, and a new reading
want every 45 minutes that Prime never takes up.

- Before: 24 need wakes in any 24 h (the daily budget was the only thing that stopped it), 48 thoughts in 2 days.
- After: 8 need wakes in any 24 h (one per 3 h recovery window), 16 thoughts in 2 days.

The live replay on 11-06..11-16 does not change (it has no event wants): idle 2.9 thoughts a day, busy 3.5.

## Current architecture

`orion/prime/hunger.py` computes Prime's needs (horizon, surprise, connection). Guard 1 (`after_answer`) runs at the
end of a thought that changed nothing. It halved each shown need **name**, except a name with any answered need.
`answered_refs(kept=...)` marks each kept concern's `prime_concern:<id>` as answered. Live, the held concerns have
adapter deadlines counted from when they opened, so keeping them does not reset their horizon drive: they are shown
on every thought, kept on every thought, and so `horizon` was always "answered". Guard 5 (same line cannot wake
twice) could not help, because each reading has a new ref.

Live evidence (11-16 18:06Z .. 11-17 02:58Z): 12 thoughts, 11 woke by `need:horizon world_pulse_read_seed:*`, all
`concerns +0 -0 kept 3 | no move`. `orion:prime:state` `needs.damp` held only `connection`.

## Architecture touched

- `orion/prime/hunger.py` only for the rule. `services/orion-mind/app/prime_loop.py` only for the log line.
- No bus, schema registry, env, or Docker change.
- Persisted state `orion:prime:state`: `needs.damp` is `dict[str, NeedDampV1]`. New keys are new strings in the same
  dict. Old state (keyed by name) loads and means the same: the `horizon` key now applies to held concerns only.
  Rollback: old code looks up by name only, so it ignores a `horizon:event:*` key; `prune` drops it after it recovers.

## Design choice and trade-offs

Three ways to fix this were considered.

- **Per ref.** Does not fix it. Every reading is a new ref, so the next reading would not be halved.
- **Per name, halve when any shown need of the name is unanswered.** Fixes the flood, but every ignored reading would
  then also halve the held concerns' horizon needs. Live, those needs are deadline needs (a promise nearing its end).
  A promise near its deadline could then fail to wake Prime. Not chosen.
- **Per (name, event family)** (chosen). An ignored reading halves the next readings. It does not halve an office
  visit, a camera ask, or a held concern.

Trade-offs of the chosen rule:

- Each event family can wake Prime once per its own recovery window. With several families funded at once, the total
  can be higher than one family alone. The daily budget (24) still bounds the total. Live on 11-17 only
  `hub.world_pulse_read` was funded.
- A family with a long window (curiosity finding, 12 h) stays halved for 12 h after Prime ignores one.
- Each family is held separately. Readings (3 h) can wake Prime up to 8 times a day, office visits (2 h) up to 12,
  curiosity findings (12 h) up to 2. Together that is about 22 a day, so for event wakes the daily budget (24) is
  still the only shared bound. That is no worse than before, but not tighter. If it matters, the next step is one
  shared event key used beside the per-family keys.
- Some wants never get a wake of their own. After Prime ignores a want of a family, a want of the same family whose
  window ends before the halving recovers never wakes Prime. It is still shown in the frame of other thoughts.
- Any thought, whatever woke it, that shows a family and changes nothing halves it again and restarts its window.
  This is the same guard 1 rule every need already had, now per family. A family held down this way is not lost:
  Prime saw it and did nothing.
- The "one wake per window" holds at the default threshold (0.5 = the event cap). If `PRIME_HUNGER_THRESHOLD` is set
  lower, a halved family can wake Prime again partway through its window.
- Inside one key, the old rule stays: a key with any answered shown need is not halved. For held concerns this means
  keeping one concern still exempts the others' horizon. That is unchanged behavior, not part of this bug. Guard 5
  bounds it (a concern's ref never changes, so about 5 wakes from 0.5 to 1.0).

Overlap with open PR #3559 (`fix/prime-event-wake-salience`): it gates which funded events may wake Prime (only the
top tenth by salience). It fixes the same symptom from the other side, and the two fit together: #3559 decides which
events may wake Prime, this PR stops an ignored one from waking Prime again for its family. Both edit
`hunger.py`, the replay eval and its tests, so whichever merges second needs a rebase. The planted readings here
carry salience 0.95 so these tests still exercise the halving if #3559 merges first.

## Files changed

- `orion/prime/hunger.py`: `NeedV1.damp_as` + `damp_key`; `event_damp_key(family)`; event horizon needs carry their
  family key; `compute_needs`, `damp_factor`, `after_answer` use the damp key; guard 1 docstring explains why.
- `orion/prime/state.py`: comment on `PrimeNeedsStateV1.damp` (what the keys are).
- `services/orion-mind/app/prime_loop.py`: `prime_needs_damped` logs the keys really halved.
- `orion/prime/evals/run_prime_hunger_replay.py`: planted case `reading_flood` and check `reading_flood_bounded`.
- `orion/prime/tests/test_prime_hunger.py`: the exact live case; per-family isolation; old state still loads.
- `orion/prime/tests/test_prime_evals.py`: the eval's check list includes the new check.
- `services/orion-mind/tests/test_prime_hunger_loop.py`: the live case through the real runtime tick and log.

## Schema / bus / API changes

- Added: `NeedV1.damp_as` (optional, in-memory only; `NeedV1` is not persisted or published).
- Removed: none.
- Renamed: none.
- Behavior changed: guard 1 halves unheld event wants under `horizon:event:<family>`.
- Compatibility notes: `PrimeNeedsStateV1.damp` shape unchanged; old and new state load in both directions.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed)
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/prime orion/mindloop                    -> 292 passed
pytest services/orion-mind/tests                     -> 421 passed
New tests fail on the old hunger.py (checked by swapping the file back):
  test_guard1_kept_concerns_do_not_shield_an_unanswered_event_need_issue_3552 -> assert set() == {'horizon:event:hub.world_pulse_read'}
  test_kept_concerns_do_not_shield_an_ignored_reading_want_issue_3552 (loop)  -> same
git diff --check                                     -> clean
```

## Evals run

```text
run_prime_hunger_replay.py  10/10 PASS
  reading_flood_bounded: 16 thoughts in 2 days, most need wakes in any 24 h 8 (at most 8)
  (before the fix: FAIL, 48 thoughts in 2 days, 24 in any 24 h)
  live replay unchanged: idle 2.9/day, busy 3.5/day, budget_holds 4, concerns_get_thought 18.0 h
run_prime_frame_eval.py, run_prime_block_eval.py, run_intention_take_on_replay -> rc 0
```

## Docker/build/smoke checks

```text
Not run: no dependency, env, compose or boot change. Pure rule change inside orion-mind's Prime loop.
```

## Review findings fixed

Code review ran in a subagent on commit 839e38642. No blocker. The reviewer confirmed: the damp key is read and
written the same way everywhere; a held event and a taken-up event are handled correctly; old and new saved state
load in both directions; `NeedV1.damp_as` is never sent or saved.

- Finding (should): the runtime test's second tick checked nothing. `runtime()` copies `extra_pages`, so the new
  reading menu never reached the fake server, and the "no second thought" check passed on the old code too.
  - Fix: serve the second menu through `rt.get_json`; assert Prime saw reading 2 at the 0.5 cap, halved below the
    threshold.
  - Evidence: with the old `hunger.py` the test now fails at `assert (0.5 == 0.5 and 0.5 < 0.5)` (reading 2
    undamped); with the fix it passes.
- Finding (nit): the eval's limit allowed 9 wakes, but the true bound is 8 (the 24 h count is strict and a halving
  recovers exactly 3 h after its wake). A regression to 9 would have passed.
  - Fix: limit is now 8.
  - Evidence: `reading_flood_bounded: ... 8 (at most 8)`.
- Finding (nit): the docstring said "any halving holds the family until it recovers"; that is true only at the
  default threshold.
  - Fix: docstring says "at the default threshold (= EVENT_DRIVE_CAP)" and names the setting that can lower it.
  - Evidence: `orion/prime/hunger.py` module docstring, guard 1.
- Finding (trade-off): per-family limits add up across families; some same-family wants never get their own wake;
  a busy Prime can keep re-halving a family.
  - Fix: written into "Design choice and trade-offs" above. No code change.

## Restart required

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
```

(From the clean primary `main` checkout after merge, or by the post-pull rebuild.)

## Live check after deploy

1. Share of thoughts woken by a reading want (should fall well below 11 of 12; at most about 8 a day):

```sql
select count(*) filter (where outcome_note like 'woke: need:horizon world_pulse%') as by_reading,
       count(*) as thoughts
from grammar_acts where actor = 'prime' and verb = 'infer' and opened_at > '<deploy time>';
```

2. The halving is stored under the reading family after an ignored reading:

```bash
docker exec orion-orion-athena-bus-core redis-cli GET orion:prime:state | python3 -c "import json,sys; print(json.load(sys.stdin)['needs']['damp'].keys())"
```

Expect `horizon:event:hub.world_pulse_read` to appear, and the log line
`prime_needs_damped ... needs=horizon:event:hub.world_pulse_read`.

## Risks / concerns

- Severity: low. Concern: several event families funded at once can each wake Prime once per window. Mitigation:
  the daily budget bounds the total; live only one family was funded on 11-17.
- Severity: low. Concern: a 12 h family stays halved for 12 h after one is ignored. Mitigation: that is the
  designed recovery ("over the fired content's own horizon"); a pending change still wakes Prime.
- Live effect is UNVERIFIED until deploy; see the live check above.
- Issues opened: #3562 (keeping one held concern still exempts other ignored concerns; pre-existing, bounded by guard 5). Issues closed: #3552 (on merge; live proof to follow in a comment).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3563

🤖 Generated with [Claude Code](https://claude.com/claude-code)
