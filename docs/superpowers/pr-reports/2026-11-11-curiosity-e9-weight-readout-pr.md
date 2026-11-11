# Curiosity replica: Juniper's chat speed (E9) and the replica's budget weight, in the same readout

## Summary

- The 14-day review (about 11-18) now runs as one command. The three-arm readout from #3183 gains two sections the
  design asks for.
- **E9, Juniper first.** Did Juniper's chat get slower once the replica started? It compares her turns in the 14 days
  before against the days after. It shows two numbers: how long a whole turn took, and how long each of her
  chat-model calls waited for a seat.
- Only the wait per call can raise a flag. A whole turn also gets longer when she asks for more, and a per-turn wait
  total grows with the number of steps. Neither shows that the replica got in her way.
- It also shows how much of the chat model's time Prime and the replica used.
- **The replica's budget weight.** The body economy learns a weight for each family from how its acts turn out. The
  readout shows the replica family's weight at the first and the latest market decision, and whether it is still
  on its starting value. The design sets no threshold, so it raises no flag.
- `--e9-cut` moves the comparison point. For example, `--e9-cut 2026-10-30T00:00` compares around Prime's start
  instead.

## Outcome moved

E9 was on the design's patch 5 list, but it was never measured, for Prime or for the replica. It is now measured on
live data, and it answers the review question "has the replica's weight drifted?" without hand-written SQL.

## Current architecture

`scripts/analysis/curiosity_three_arm_readout.py` (#3183) covered the arms, the stop rule, harm, fallbacks and
answers.

- Juniper's turns are told apart by `orion/regulation/juniper_turns.py` (`JUNIPER_INGRESS_PREDICATE`).
- Turn timing is in `cockpit_turn_sighting`.
- Per-call seat wait is in `llm_call_usage.wait_ms`, from 2026-10-29 only.
- The market's per-decision family weight is in `want_ledger.decision`.

## Architecture touched

Analysis only. No service, contract, env or schema change.

## Files changed

- `scripts/analysis/curiosity_three_arm_readout.py`: E9 and weight sections, their queries, `--e9-cut`.
- `scripts/analysis/tests/test_curiosity_three_arm_readout.py`: 6 new tests (20 total).
- `.github/workflows/orion-prime-tests.yml`: trigger on `orion/regulation/juniper_turns.py` too.
- this report.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none. Behavior changed: none.
- Compatibility notes: reads existing tables only.

## Env/config changes

- None.

## Tests run

```text
pytest scripts/analysis/tests/test_curiosity_three_arm_readout.py
20 passed in 0.50s
```

New tests cover:
- only a wait-per-call slowdown raises the flag. A longer turn does not, and neither do more steps at the same wait;
- the window edges, unmeasured turns and unfinished turns;
- no comparison point yet is left out, not reported as unknown, while a failed read is reported as unknown;
- time offsets in `--e9-cut` are kept;
- the weight's change and its basis.

## Evals run

```text
Live, 2026-11-11, read-only. Whole run 1.3 s warm (the first, cold run took 36 s; every query is under 0.5 s warm).

Around the replica start (2026-11-05 07:04 UTC):
  before 92 turns: whole turn median 113 s; wait per chat call median 0.3 s, p90 7.8 s (52 turns measured, from 10-29)
  after  41 turns: whole turn median 153 s; wait per chat call median 0.2 s, p90 1.5 s (40 measured)
  Prime + replica: 1.4% of chat-lane model time before, 2.1% after. p (wait slower) 0.87 -> no slowdown flag.
Around Prime's start (--e9-cut 2026-10-30T00:00):
  whole turn median 95 s -> 145 s (p < 0.001); wait per call cannot be compared (2 turns measured before).
Replica weight: 1.098 -> 1.103, still on its starting value (fewer than 10 graded acts).
```

## Docker/build/smoke checks

```text
Not applicable: analysis script, no runtime change.
```

## Review findings fixed

The code review ran in a subagent. It checked the time zones, the window edges, the join, double counting,
correlation-id collisions and connection cleanup against live data, and found them correct.

- Finding: before the three-arm test starts, "no comparison point yet" read as "could not read" and exited 3.
  - Fix: it is left out (`build_e9` returns `NOT_READ`); only a failed read is unknown.
  - Evidence: `test_no_cut_yet_is_left_out_and_a_failed_read_is_unknown_and_offsets_are_kept`.
- Finding: the flag used the per-turn wait total, which grows with the number of steps (about 7 harness calls per
  turn).
  - Fix: the flag uses wait per chat-lane call; the total is display only.
  - Evidence: `test_more_steps_with_the_same_wait_per_call_is_not_slower`.
- Finding: `--e9-cut` dropped an explicit offset (an MDT time moved 6 h). Fix: `parse_cut` keeps it.
- Finding: a turn with calls on other lanes only counted as a 0 s wait. Fix: a turn is measured only when it has a
  chat-lane call.
- Finding: the "14 days before" label hid that seat wait and lane share start on 10-29. Fix: the render prints the
  date each one is measured from.
- Finding: the main wiring was untested. Fix: `build_e9` takes an injectable reader and is tested.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium (finding, not caused by this patch).
  - Concern: Juniper's whole turns got slower: median 95 s in the two weeks before 10-30, about 145 s since. The
    seat wait does not explain it. Since 10-29 her turns make the same number of model calls (median 9) and spend
    the same model time (about 100 s) before and after the replica start.
  - Most of the jump happened around 10-29/10-30, before per-call timing existed, so its cause is **UNVERIFIED**.
    Other changes landed at the same time, for example the Strata lane move (#3016).
  - Mitigation: reported, not flagged. A separate look at where the time goes inside a turn is the next step if
    Juniper wants it.
- Severity: low.
  - Concern: the "before" side of the seat-wait comparison for the replica has only about 6 measured days, not 14.
  - Mitigation: the measured-from date is printed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3192

🤖 Generated with [Claude Code](https://claude.com/claude-code)
