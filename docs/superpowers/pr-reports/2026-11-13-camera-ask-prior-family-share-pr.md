# Camera asks: a rare ask rule starts from the family's answer rate, not from 0.5 (#3350)

Fixes #3350. Follows attention patch 4b (#3351, merged), which made camera asks obey the market. Design: #3213. Market rule unchanged, as decided on #3286.

## Summary

Orion's guess of how likely Juniper is to answer a kind of camera question now starts from how often she answers camera questions in general, not from a coin flip.

- Each ask rule's expected answer is now `(answered + k*m) / (closed + k)`, with `k = 2`.
  - `m` is the answered share of all of Orion's camera asks over the same 30 days. Today it is 0.817.
  - The old value was Laplace, `(answered + 1) / (closed + 2)`. It pulled every rule toward 0.5.
- A rule with no answers yet reads exactly `m`. One answer puts it above `m`. One "don't know" puts it below.
- The change is in one shared function, `rule_outcome_values` (`orion/grammar/camera_ask_acts.py`). The want producer, the bid site, the back-fill report and the replay all read it.
- The market is not touched: no appetite or Choice change.

## Outcome moved

Before, under patch 4b, a rare rule was stuck.
- The market funds a camera ask only when its expected answer beats the family's.
- The office rule had 1 answer in 1 ask, so it read 0.667, below the family's 0.82.
- So it was held, never opened, never graded, and it never moved. Office, face and people asks were 3 of the 6 lost answers in the replay.

Live 30-day replay, no-deadline rule, before and after:

| Rule | Asks | Held (before) | Held (after) | Never opened in 7 d (after) | Expected answer (after, mean at bid) |
|---|---|---|---|---|---|
| confirm | 153 | 3.9% | 0% | 0% | 0.980 |
| same | 40 | 0% | 0% | 0% | 0.988 |
| face | 2 | 100% | 0% | 0% | 0.953 |
| office | 1 | 100% | 0% | 0% | 0.824 |
| sighting | 4 | 50% | 0% | 0% | 0.949 |
| vehicle | 44 | 95.5% | 95.5% | 95.5% | 0.308 |

- Answered asks per day: 6.6 -> 6.5 under 4b. The loss falls from 3.0% to 1.5%. The only lost answers left are 3 vehicle asks; before, there were also 1 face, 1 office and 1 sighting.
- All gate checks pass:
  - vehicle held >= 70%: 95.5%;
  - vehicle never opened in 7 days >= 70%: 95.5%;
  - confirm and same held <= 10%: 0% and 0%;
  - loss <= 10%: 1.5%.
- Re-bids per held candidate per day: 8.5 (unchanged).

Why `k = 2`:
- The prior weighs as much as two of the rule's own answers, the same weight Laplace had, so a well-sampled rule barely moves.
- Live values (262 closes, `m` = 0.817):

  | Rule | Answered / closed | Laplace | New |
  |---|---|---|---|
  | vehicle | 5 / 47 | 0.122 | 0.135 |
  | confirm | 147 / 152 | 0.961 | 0.965 |
  | office | 1 / 1 | 0.667 | 0.878 |
  | face | 5 / 5 | 0.857 | 0.948 |
  | sighting | 4 / 5 | 0.714 | 0.805 |

- The replay is not sensitive to `k`. With `k` = 1, 2, 4 and 8, exactly the same asks open (the loss is 1.5% for each). Only vehicle's value drifts up with `k`: 0.28 / 0.31 / 0.35 / 0.42 mean at bid.

## Current architecture

- Patch 4a put each rule's Laplace value on the `vision.ask` want as `outcome_value`.
- Patch 4b makes the card's opening obey the market's decision on that want.

## Architecture touched

- `orion/grammar/camera_ask_acts.py`: `rule_outcome_values` gets the family-share prior. New constants: `PRIOR_WEIGHT = 2`, and `EMPTY_FAMILY_PRIOR = 0.5` for when the family has no closes.
- No schema, bus, env or market change.

## Files changed

- `orion/grammar/camera_ask_acts.py`: the value.
- `services/orion-sql-writer/app/ask_event_wants.py`: `GRADES_SQL` selects the market's own acts (opened in 30 days, producer or resolver), with `now`; docstring.
- `services/orion-sql-writer/app/ask_bind.py`: passes `now` to `GRADES_SQL`.
- `services/orion-durable-runs/evals/event_wants_replay.py`: the replay's grades filter matches.
- `services/orion-sql-writer/scripts/backfill_camera_ask_acts.py`: summary key `expected_outcome_by_rule_targets_only`.
- `services/orion-sql-writer/evals/ask_bind_mutations.py`: two window-parity mutants.
- `docs/superpowers/specs/2026-11-12-attention-camera-asks-bind-design.md`, `docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md`: personal names redacted (privacy gate).
- `services/orion-sql-writer/tests/test_camera_ask_acts.py`: live-count test. New tests: "a rare rule answered once reads above the family share"; "no closes reads `m`, and an empty family reads 0.5".
- `services/orion-sql-writer/tests/test_ask_event_wants.py`, `test_ask_bind.py`, `test_ask_bind_integration.py`, `test_camera_ask_acts_integration.py`: pinned values recomputed from the formula.
- `services/orion-sql-writer/evals/camera_ask_acts_mutations.py`: the old Laplace mutant is replaced by "prior back to 0.5 (Laplace)" and "no prior".

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behaviour changed: the `outcome_value` on `vision.ask` wants.
- Compatibility notes: nothing parses the value except the market.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- Skipped keys requiring operator action: none.

## Metric gate (CLAUDE.md 0A)

1. **Provenance.** Unchanged from 4a: Juniper's answers on `orion_ask`, graded into `grammar_acts` closes, read by `GRADES_SQL` (resolver closes, 30 days).
2. **Independence.** `m` is the same answers, pooled across rules. It is a prior, not a new signal.
3. **Theory.** Shrinkage of a small-sample rate toward its group's rate, as in empirical Bayes. A rule with little data borrows the family's rate, not an arbitrary 0.5.
4. **Live data.** The values above are not flat (0.135 to 0.993). A rule's rest point is the family's rate: it reads `m` with no data, not 0.5.
5. **Existing mechanism.** The same function, the same reads.
6. **Reversible.** One function; `k` is a constant.

## Tests run

```text
services/orion-sql-writer: pytest tests (throwaway pg16) -> 1366 passed, 162 skipped
services/orion-durable-runs: pytest tests -> 931 passed, 144 skipped
orion/grammar/tests orion/physiology/tests tests/test_act_ledger_contract.py -> 396 passed
python scripts/check_body_economy_contract.py -> OK
```

## Mutation check

```text
camera_ask_acts_mutations.py -> 13/13 KILLED (incl. "prior back to 0.5 (Laplace)", "no prior")
ask_bind_mutations.py        -> 15/15 KILLED (incl. "grades windowed on close, not open", "grades resolver-only")
```

## Evals run

```text
event_wants_replay.py --camera-asks                     (fixture) -> pass
event_wants_replay.py --camera-asks --live --days 30              -> all 4 checks pass (table above)
```

## Docker/build/smoke checks

```text
No build: Python only, already-copied paths. Live path UNVERIFIED until deploy. After deploy:
  docker logs orion-athena-sql-writer 2>&1 | grep -E "camera_ask_bid .*rule=(office|face|sighting)"
```

## Review findings fixed

The code review ran in a subagent with an explicit target (`git diff origin/main...HEAD`). It found no blockers. It checked live, SELECT only, that both family means read today n = 262, m = 0.8168, and that the docstring numbers are right.

- Finding: the writer counted grades by close time (`closed_at >= since`, resolver only), but the market counts its family mean by open time (opened in the last 30 days, producer or resolver). They agree today only because the oldest ask act opened on 10-15. Within about a day an ask opened 30-37 days ago and closed inside the window would count on one side only. A rule with no closes reads the writer's `m`. If that `m` were below the market's mean, the rule would rank below every saturated past event: the same never-graded trap, recreated.
  - Fix: `GRADES_SQL` now selects exactly what `build_family_appetite` reads. Both callers pass `now`. The replay filters the same way.
  - Evidence: a new Postgres test, `test_the_writers_family_share_is_the_markets_family_mean`. It seeds an act opened 31 days ago and closed 29 days ago, a withdrawn close and a producer close, and asserts the writer's `m` equals `build_family_appetite(...).mean_outcome` (51 acts, 10/51). New mutants "grades windowed on close, not open" and "grades resolver-only" are killed.
- Finding: a rule reading exactly `m` ties every saturated past event.
  - Fix: the behaviour is stated in the docstring and pinned by a test that calls the market's `rank_value`:
    - all 20 past events saturated: rank 0, held;
    - one unsaturated event: rank 1/20, funded;
    - one answer: rank 1.0.
  - Live history has unsaturated events (min intensity 0.0035 in 3 days), so a rule at `m` is funded. The replay agrees: office at 0.824 was not held.
- Finding: no live decision has used a graded value yet. All 41 vision.ask decisions in 3 days are `event:ungraded`, from the shadow producer.
  - Fix: marked UNVERIFIED below. Filed #3352 (`verify-live`).
- Nits fixed:
  - The back-fill summary key is now `expected_outcome_by_rule_targets_only`. Its `m` is the targets' share, not the family's.
  - `ask_event_wants.py` now says "5 of 47".
  - The docstring cites this report for the `k` sweep.
  - Non-finite grades are skipped, as the market skips them. A test covers this.

- CI finding (not from this change): the "Static repo gates" privacy check (`tests/test_no_personal_terms.py`) was red on `main` itself. Two merged specs held personal names: `2026-11-12-attention-camera-asks-bind-design.md` (1 line) and `2026-11-13-core-memory-pages-and-concept-growth-design.md` (10 lines).
  - Fix: each name is replaced by a neutral placeholder (`<name>`, `<person A>`, `<person B>`, `<person C>`, `<town>`). One placeholder per name keeps the text readable. No open PR touched either file.
  - Evidence: `pytest tests/test_no_personal_terms.py` -> 7 passed.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull && scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

The post-pull rebuild does this automatically. durable-runs reads this function only in the replay eval, so it needs no restart.

## Risks / concerns

- Severity: low.
  - Concern: a rule with no answers yet reads exactly the family share. With saturated hunger it would tie with the family's history and could be held.
  - Mitigation: in the replay, the one office ask read `m` (0.824) and was not held, because past intensities vary. Once a rule has a single answer it reads above `m`.
- Severity: low.
  - Concern: vehicle's value rises slightly, from 0.122 to 0.135 live.
  - Mitigation: it is still held at 95.5% in the replay.
- Severity: medium (evidence).
  - Concern: UNVERIFIED live. No live decision has used a graded camera-ask value yet.
  - Mitigation: #3352 (`verify-live`) lists the three checks to run after deploy.
- Issues closed: #3350. Opened: #3352.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3353

🤖 Generated with [Claude Code](https://claude.com/claude-code)
