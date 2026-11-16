## Summary

- A camera ask that the market holds now opens anyway when its rule (face, sighting, vehicle, ...) has had under 2% of
  the family's camera asks opened over the last 7 days. This stops a rule whose expected answer is below the family
  mean from being held forever and never graded (issue #3381, option (c) of #3350, approved by Juniper).
- The floor lives in the opener (`services/orion-sql-writer/app/ask_bind.py`), not in the market. It is the market's
  own exploration floor (attention patch 3b) applied one level down. The rule and its history already live in the
  opener; the market does not know which rule an ask comes from.
- Only a hold on value (`sated` or `price`) can open through the floor. A consent block never does.
- Flag `ORION_CAMERA_ASK_RULE_FLOOR_ENABLED`, shipped ON.
- The replay (`event_wants_replay.py --camera-asks`) gains steady-state lanes that reproduce today's starvation, with
  and without the floor, replays bids in time order across asks, and checks each live bid under the rule live when it
  was made.

## Outcome moved

Live 30-day replay, 2026-11-16, steady state with today's expected answers held fixed (the regime of #3381):

| rule | asks | held at first bid, old | held at first bid, new | never opened in 7 d, old | never opened in 7 d, new | floor opens |
|---|---|---|---|---|---|---|
| confirm | 168 | 0% | 0% | 0% | 0% | 0 |
| same | 50 | 0% | 0% | 0% | 0% | 0 |
| office | 1 | 0% | 0% | 0% | 0% | 0 |
| face | 3 | 100% | 0% | 100% | 0% | 3 |
| sighting | 4 | 50% | 25% | 50% | 0% | 2 |
| vehicle | 48 | 95.8% | 87.5% | 95.8% | 83.3% | 6 |

Answered asks per day under 4b: 6.833 old, 6.933 new (+0.10 a day; real answered rate 7.033). Lost answers: old face 2,
sighting 1, vehicle 3; new vehicle 3 only. The 4b gate passes on both (vehicle at least 70% never opened, confirm and
same at most 10% held, answered loss at most 10%).

The family floor's own 5% fails the gate here: 68.75% of vehicle asks never opened (13 floor opens). That is why the
rule floor is 2%.

The replay's time-ordered rewrite does not move the old numbers: the gate lane without the floor gives vehicle
87.5% held and 87.5% never opened, 7.0 answered a day, 8.49 re-bids per held candidate-day (main: 87.5%, 87.5%,
7.0, 8.5).

The gate lane (as if 4b had bid every ask of the window, with the floor): vehicle held 81.25% at the first bid, 75%
never opened; answered 7.0 a day of 7.033 (1 vehicle answer lost), all checks pass.

## Current architecture

- A camera opener calls `open_camera_ask`, which calls `ask_bind`: it posts an event want (family `vision.ask`,
  outcome value = the rule's expected answer) to the market and opens the card only if the market funds it.
- Since #3406 the market ranks the want only against past events with the same grading. Hunger is saturated (tau
  60 s), so every reference event reads `1.0 x family mean`. A rule whose expected answer is at or below the family mean
  ranks 0 and is held (`sated`). It re-bids on a backoff (10 min doubling to 3 h), forever.
- A held ask opens no card, so it gets no grade, so its expected answer never moves.
- The market's exploration floor (`orion/physiology/choice.py`) funds a Tier 2 FAMILY under 5% of its tier's acts
  started over 7 days. It cannot act per rule: a want names only its family, and the market's act history does not
  read `test_before` (where the rule is stamped).

## Architecture touched

- `orion/grammar/camera_ask_acts.py`: pure `rule_open_share` and `under_rule_floor`, constants
  `RULE_EXPLORATION_FLOOR = 0.02`, `RULE_FLOOR_WINDOW = 7 d` (equal to the market's `FLOOR_WINDOW`, tested),
  `RULE_FLOOR_HOLD_REASONS`.
- `services/orion-sql-writer/app/ask_bind.py`: the floor after the market's verdict. Opens are counted from the same
  cached 30-day ask read (`ASKS_SQL`, no new SQL), plus this process's own floor opens since that read, so a daily
  batch of held vehicle candidates cannot all floor-open inside one 300 s cache window.
- No bus, schema, channel, or market change. The market's decision row is unchanged (it still says `deferred`).

### Why the opener, not the market

The rule identity and the per-rule history already live in the opener: it computes the rule (`ask_rule`), reads
per-rule grades (`GRADES_SQL`), and reads every Orion-made ask of 30 days (`ASKS_SQL`). Putting the floor in the
market needs a new field on `WantV1` (a forbid model, so a consumer-first migration) and a new `test_before` read in
the market's act history, for one family. The opener applies the same rule as the market's floor, with the same
window and the same "acts started" count (each Orion-made ask is one `vision.ask` act opened at its `created_at`).

## Files changed

- `orion/grammar/camera_ask_acts.py`: the pure floor functions and constants.
- `services/orion-sql-writer/app/ask_bind.py`: apply the floor after a value hold; count own floor opens; log lines.
- `services/orion-sql-writer/app/settings.py`, `.env_example`, `docker-compose.yml`: the flag, ON.
- `services/orion-sql-writer/README.md`, `config/physiology/families.yaml`: document the floor.
- `services/orion-sql-writer/tests/test_ask_bind.py`: starvation regression (28 days, face held every day on the
  old rule; opens about once a week with the floor), vehicle mostly held (4 of 84 open), stamp and log, the
  in-cache self-limit, an uncommitted floor open still counted after a refresh, price vs consent, flag off, flag ships on. The fake ask history now carries open times and
  rules.
- `services/orion-sql-writer/tests/test_ask_bind_integration.py`: real Postgres and the real opener: a vehicle with no
  open this week opens through the floor once, the second is held. Seed history now includes vehicle asks.
- `services/orion-sql-writer/tests/test_camera_ask_acts.py`: the pure functions; window equals the market's.
- `services/orion-durable-runs/evals/event_wants_replay.py`: `rule_floor`, `steady_state`, `pin_outcomes`; bids in
  time order (heap) so a re-bid sees later asks' opens; `parity_by_era`; new live checks.
- `services/orion-durable-runs/tests/test_camera_asks_replay_parity.py`: floor starvation and gate on the live
  fixture; `parity_by_era`.
- `services/orion-durable-runs/evals/event_rank_grading_mutations.py`: one mutant string follows the moved line.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: a held camera ask can now open (`BindVerdict.why = "rule_floor"`). The card is stamped
  `want:<id>` as a funded one (the market answered), so the shadow loop posts no second want.
- Compatibility notes: none. A floor open is visible in data as a stamped card whose want's `want_ledger` decision is
  `deferred`.

## Env/config changes

- Added keys: `ORION_CAMERA_ASK_RULE_FLOOR_ENABLED=true` (orion-sql-writer).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced: the key was added to the primary checkout's `services/orion-sql-writer/.env` (the sync script
  only writes the checkout it runs in, and this branch is a worktree).
- skipped keys requiring operator action: none.

## Metric quality gate (the new signal: a rule's share of the week's camera-ask opens)

1. Provenance: `ask_bind._history_and_outcomes` reads `ASKS_SQL` (`app/ask_event_wants.py`: every Orion-made camera
   ask in `orion_ask`, 30 days, Juniper's own story cards excluded), maps each row to its rule with `ask_rule`, and
   `under_rule_floor` counts those created in the 7 days before the bid.
2. Independence: the market's family floor counts acts started across all Tier 2 families (different denominator,
   different unit). The rule's expected answer comes from grades of the same asks; the link is causal and intended
   (a low expected answer leads to holds, which lower the share, which opens the floor). It is not a transform of any
   signal already in the rank.
3. Theory anchor: forced exploration in a bandit. A greedy policy that only plays arms above the current mean can lock
   out an arm forever, because an arm never played never updates. A minimum sampling share per arm (the market's own
   5% family floor is the same idea) keeps every arm's estimate moving.
4. Live data (`grammar_acts`, actor `vision.ask`, last 7 days, 2026-11-16): confirm 54, same 30, vehicle 14, face 1,
   sighting 0, office 0 (99 opens). Shares 54.5%, 30.3%, 14.1%, 1.0%, 0%, 0%. Not flat or saturated. The rest state is
   the rule's natural share; a rule with no candidates sits at 0% but has nothing to open.
5. Existing mechanism: the market's exploration floor (`choice.py`, patch 3b). Reused: its window, its "acts started"
   count, its self-limiting quota. It cannot be applied per rule inside the market (see above).
6. Reversibility: one flag, one constant, no schema, no stored field. Turning it off restores the old rule.

## Tests run

```text
services/orion-sql-writer: pytest tests (scratch Postgres for the integration lane)   1390 passed, 163 skipped
services/orion-durable-runs: pytest tests                                              995 passed, 147 skipped
orion/physiology/tests + orion/grammar                                                 382 passed
tests/test_no_personal_terms.py, test_check_settings_defaults.py, test_compose_logging_journald.py,
test_gitignore_no_bare_catchall.py                                                     all passed
scripts/check_env_template_parity.py                                                    PASS
git diff --check                                                                        clean
```

## Evals run

```text
event_wants_replay.py --camera-asks --live --days 30 (read-only, live DB)   rc=0, every check true:
  vehicle_held_at_least_70pct, vehicle_never_opened_in_horizon_at_least_70pct, confirm_and_same_held_at_most_10pct,
  answered_per_day_falls_at_most_10pct, replay_reproduces_live_bids (30 of 30), steady_state_floor_passes_gate,
  steady_state_floor_opens_every_rule, steady_state_floor_loses_no_more_answers, today_values_floor_passes_gate,
  today_values_floor_opens_every_rule, next_vehicle_bid_held, next_confirm_and_same_bids_funded
event_wants_replay.py --camera-asks (fixture lane)                           all fixture checks true
event_rank_grading_mutations.py                                              10 of 10 mutants killed
```

`replay_reproduces_live_bids` failed on main before this patch (28 of 30): the two 2026-11-15 07:00Z vehicle bids
were decided under #3406's rule but were checked against the rule before it. `parity_by_era` now checks each bid
under the rule live when it was made (switch at #3406's merge, 2026-11-15 02:59Z; no bid ran between merge and deploy).

## Docker/build/smoke checks

```text
No image change (no new dependency). Integration tests ran against a throwaway postgres:16-alpine container.
Live path after deploy: UNVERIFIED (see the verify-live issue).
```

## Review findings fixed

Code review ran in a subagent on the branch diff. No must-fix findings.

- Finding: a floor open was dropped from the in-process count at the next cache read by time alone. Openers run in
  separate threads and transactions, so a read could miss an uncommitted floor-opened card and a second floor open of
  the same rule could follow.
  - Fix: own floor opens are kept by `ask_id` until a read contains that ask, or 600 s pass (insert lost or rolled back).
  - Evidence: `test_a_floor_open_still_counts_when_a_later_read_does_not_see_its_uncommitted_row`.
- Finding: the floor check and the count were taken under two lock acquisitions (check-then-act race).
  - Fix: one `with _lock:` around both.
  - Evidence: `ask_bind.py`; tests pass.
- Finding: the heap rewrite changes the stand-ins a re-bid sees.
  - Fix: stated in the `camera_asks_gate` docstring; the live lane without the floor was re-run before and after.
  - Evidence: no `by_rule` number moved (vehicle 87.5%/87.5%, 7.0 answered a day); re-bids 8.5 -> 8.49.
- Finding: README said "about one vehicle ask a week". The floor stops at 2% of the week's asks.
  - Fix: README and docstring now say 2-3 a week at today's ~100 asks a week (the replay opened 6 in 30 days).
- Finding: this report was untracked but cited by code comments.
  - Fix: committed.
- Finding (nit): the floor open was counted at `now`, the card at `ask.created_at`.
  - Fix: counted at `ask.created_at or now`.
- Finding (nit): the replay did not count asks too early to rank (live opens them unbid).
  - Fix: they are now counted as opens; live numbers unchanged.

## Restart required

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

## Risks / concerns

- Severity: low. Concern: the floor opens a few vehicle asks a week (6 in the 30-day replay; at most about 2% of the
  week's asks) that Juniper will mostly dismiss (about 13% are answered). Mitigation: that is the price of keeping every rule graded; the replay keeps 83% of vehicle asks
  unopened, and the flag turns it off.
- Severity: low. Concern: the count lags up to 300 s (the cache) for opens by other processes. Mitigation: own floor
  opens are counted at once; a missed funded open of another rule only makes the floor stricter.
- Severity: low. Concern: the replay's steady state pins hunger at 1.0 and the reference at today's ledger. Mitigation:
  it is the live regime (tau 60 s); the historical lanes are still reported beside it.
- Issues: fixes #3381. Opened: VERIFY_LIVE_PLACEHOLDER.

## PR link

PR_LINK_PLACEHOLDER
