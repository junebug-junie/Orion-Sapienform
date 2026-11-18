# Dream obeys its bid, and Prime can ask Orion to sleep a little later (Orion Prime step 4)

## Summary

- Orion's sleep (the dream cycle) now asks the body economy before it starts, and obeys the answer. It uses the same rule curiosity has obeyed since #3010. That rule now lives in one shared file, so both organs obey exactly the same rule.
- The dream asks only after every one of its own gates has passed: enough pressure (or the overdue backstop), the 6-hour gap since the last sleep, the arousal check, and something to replay. A "yes" means it sleeps now. A "no" on price, on "no hunger yet", or on a value at or below zero holds the sleep. The dream then asks again at a later check.
- A held sleep is never dropped. It goes anyway after 6 hours from its first ask, or at the overdue line (48 hours since the last good sleep), whichever comes first. An overdue sleep never asks at all. No answer from the market means it sleeps, as before.
- The dream now accepts Prime's "slow" signal. A slow signal is a temporary charge on the sleep's bids, at most half of the sleep's value. A sleep that comes soon after the last one (about 6 hours) waits up to a few hours. A hungry sleep outbids the charge at once. The dream does not accept "quicken".
- Prime needed no new move. It reads its receivers from the families file, so it now lists the dream there and nowhere else.
- Flag `ORION_PHYSIOLOGY_BID_DREAM_CYCLE=on`. Set it to `shadow` and restart for an instant rollback to the old shadow bid.

## Outcome moved

### The live evidence first (7 days to 2026-10-30 21:01 UTC)

What the dream's shadow bids said, compared with what really happened. The shadow bids have been live since 10-27 23:45.

| shadow bids | wants | market said fund | market said wait | really slept | held by the dream's own gates |
|---|---|---|---|---|---|
| 56 | 56 | 56 | 0 | 6 | 50 |

- The market never told the dream to wait. Every disagreement went the other way: the market said "fund", but the dream's own arousal gate held the sleep (50 checks).
- The journald lines still kept for this window show those holds were all arousal (6 "engaged", 5 "strained"). Older lines have aged out.
- Each held check minted a new want, so 56 bids is 56 wants. Under this patch, one want spans the whole stretch between sleeps.

What binding would have done to every real sleep in the 7 days. This replays the 13 real sleeps with the live market code, against the body state that was in force at each one (`services/orion-dream/evals/dream_binding_replay.py --live`):

| | sleeps | held by binding alone | held to the 6 h limit |
|---|---|---|---|
| priced on the dream's own rhythm (6+ sleeps of history) | 7 | **0** | 0 |
| cold start (fewer than 6 sleeps in 30 days, 10-24 .. 10-27) | 6 | 4 (median 160 min, max 360 min) | 1 |

- **Stop check: passed.** Binding alone holds no sleep today. All 4 holds came from the act ledger's first days for the dream. Back then the market rated each sleep only against 2 or 3 earlier ones. A sleep with real hunger (intensity 0.95) then still ranked 0 and read "no hunger yet".
- That cold start is over: the dream has had 11+ sleeps since. It can come back only if the dream sleeps fewer than 6 times in 30 days, and the 6-hour window still bounds it.
- Binding alone changes nothing about when Orion sleeps today. It makes the sleep obey a market that Prime can now put a charge on.

### The receptor decision: declared, slow only

Can a slow charge actually change a dream decision? Yes. Of the 7 warm-history sleeps, 4 bid at a value under 0.5 (net 0.17, 0.18, 0.20, 0.43). Those are the sleeps that came about 6 hours after the last one. A 0.5 charge holds them. The hungry sleeps (0.56 .. 0.88) outbid it at once.

| scenario (charge 0.5) | warm sleeps held | median hold | held to the 6 h limit | bids that met the charge / outbid it |
|---|---|---|---|---|
| in force all week (not reachable) | 4 / 7 | 360 min | 2 | 256 / 8 |
| 6 h on / 6 h off, phase 0 (the most Prime can do) | 2 / 7 | 240 min | 0 | 127 / 3 |
| 6 h on / 6 h off, phase 1 | 2 / 7 | 200 min | 0 | 58 / 3 |

Is it ever sane for Prime to slow sleep? Yes, inside a fence:

- **Delay costs no memory.** Each sleep replays everything since the last good sleep, back as far as 48 hours. A later sleep replays the same material plus whatever arrived meanwhile. Delay changes when consolidation happens, not what gets consolidated, until the 48-hour line.
- **The fence:**
  - The charge is at most half the sleep's value.
  - One signal lasts at most 6 hours, then the dream rests from Prime as long as the signal lasted.
  - A held sleep goes at 6 hours from its first ask.
  - An overdue sleep (48 hours) never asks, so no charge can touch it. The replay shows this as `floor:overdue`.
  - In the 7 days, no sleep came near the overdue line.
- **What a slow signal means in practice:** "sleep less often, in bigger batches". It moves the short-gap extra sleeps (10-27 had four) later. It leaves the long-gap sleeps alone.

Why not quicken: the dream had zero price deferrals. Its only holds were the cold-start "no hunger yet" kind, and a quicken credit never applies to a want valued at zero (#3011's rule). So a credit could change nothing. Declaring it would be a label with no behaviour.

## Metric quality gate

New signal: Prime's regulate grade for the dream ("did the charge change at least one sleep decision while it was in force"). Also the dream's binding verdict.

1. **Provenance.**
   - The decision comes from `orion/physiology/choice.py` `decide`, called by `services/orion-durable-runs/app/physiology_market.py` `Market.bid` for the dream's HTTP bid. It is published to `want_ledger` with producer `orion-dream.dream_cycle.binding`.
   - The verdict comes from `orion/physiology/binding.py` `bind_verdict`, called in `services/orion-dream/app/physiology_bind.py` `DreamBinder._bind`.
   - The charge's effect is written by `Market._record_prime_effect` (unchanged from #3011).
2. **Independence.**
   - The market's dream hunger is the time since the dream's last close. It is not the dream's own pressure, which counts new unprocessed material.
   - The two are related: both grow between sleeps. But the market does not decide whether to sleep; the dream's pressure gate does. The market decides only whether to sleep now or a little later.
   - The grade is the mechanism itself (two real decisions, with and without the charge), not a second measure of sleep rate.
3. **Theory anchor.** Compliance, as in #3011: a regulation that never changes the receiver's decision did nothing.
4. **Live sanity.**
   - The shadow ledger is degenerate at "fund" (56 of 56). That is expected: a Tier 1 family on a lane with near-zero price. It is why binding alone moves nothing today.
   - The grade is not degenerate. Of 330 three-hour windows, 136 saw a dream bid, and 60% of those would grade 1. That is not stuck at 0 or 1.
   - Caveat: this comes from the always-on bid stream. A real signal sees bids only while the dream is past its own gates.
5. **Existing mechanism.** This reuses the binding rule, the receptor field, the market's per-want record and Prime's regulate move. There is no new table, verb, channel or schema.
6. **Reversibility.**
   - One flag, `shadow`, rolls back the obeying.
   - Deleting one line (`receptors: [slow]`) removes Prime's reach.
   - With the flag at `shadow`, the market no longer charges the dream's shadow bids, so Prime is never graded on a hold that did not happen (review finding 3).

## Current architecture

- The dream cycle (`services/orion-dream/app/cycle.py` `run_cycle_once`) slept when its own gates passed. A shadow bid (`app/physiology_shadow.py`, Patch C3b) was placed before the arousal check and never obeyed.
- Curiosity's binding rule lived inside durable-runs (`app/run_bids.py`), where no other service could import it.
- Only curiosity investigate declared a receptor. The market applies Prime's charge to every bid of a family that has a receptor, whatever the producer.

## Architecture touched

- **New `orion/physiology/binding.py`:** the pure binding rule (`bind_verdict`, `BindVerdict`, `decision_detail`, the re-bid clamps) and `BINDING_PRODUCERS`. `services/orion-durable-runs/app/run_bids.py` re-exports it, so nothing that imported it there changed.
- **New `services/orion-dream/app/physiology_bind.py`:** `DreamBinder`.
  - One want per stretch between sleeps.
  - The window starts at the first bid. A re-bid waits for the market's retry time, and the dream re-bids at its own check cadence.
  - It holds a lock so two racing checks cannot both sleep the same want.
  - It releases the reservation after the sleep, including a sleep that fails.
- **`cycle.py`:** in `on` mode the bid comes after arousal and after the empty check, just before the act opens. A hold sets `hold_reason = economy_<reason>` and logs why. A sleep that ran after a hold says so in its cycle note.
- **Market:** Prime's charge is read only for bids from `BINDING_PRODUCERS`.
- **`config/physiology/families.yaml`:** `dream.cycle: receptors: [slow]`.

## Files changed

- `orion/physiology/binding.py` (new): the shared rule, and the list of producers whose bids are obeyed.
- `services/orion-durable-runs/app/run_bids.py`: imports the rule instead of defining it.
- `services/orion-durable-runs/app/physiology_market.py`: Prime's charge applies only to obeyed bids.
- `services/orion-dream/app/physiology_bind.py` (new), `app/cycle.py`, `app/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`: the binding bid and its flag.
- `config/physiology/families.yaml`: the dream's slow receptor, with the evidence.
- `config/physiology/contract.yaml`:
  - Both bid sites are registered as `bidder`, because importing the shared rule makes the checker see them as readers.
  - The dream's bid site is added to `protected_paths`.
- `scripts/analysis/body_economy_shadow_vs_real.py`: keeps the dream's obeyed bids out of the shadow agreement table.
- `orion/physiology/report_text.py`, `services/orion-hub/static/body-economy.html`, `orion/prime/acts.py`, `services/orion-mind/app/prime_loop.py`: the text now names the dream as obeying and as a receiver.
- `services/orion-dream/evals/dream_binding_replay.py` (new) and `evals/test_dream_binding_replay_eval.py` (new): the fixture lane runs in CI, and a live lane runs against the live DB.
- Tests:
  - `services/orion-dream/tests/test_physiology_bind.py` (new, 21 tests)
  - `services/orion-durable-runs/tests/test_prime_charge_market.py` (+2)
  - `orion/prime/tests/test_prime_regulate.py` (+1)
  - `orion/physiology/tests/test_prime_charge.py` and `orion/prime/evals/run_prime_frame_eval.py` (now expect two receivers)
- CI: `.github/workflows/orion-durable-runs-tests.yml` runs the dream binding tests and eval.
- Docs: `services/orion-dream/README.md`, `services/orion-durable-runs/README.md`, `services/orion-mind/README.md`.

## Schema / bus / API changes

- Added: none. There is no new schema, channel or table. `WantV1` and `WantDecisionV1` are unchanged.
- Behavior changed:
  - The dream's bids carry producer `orion-dream.dream_cycle.binding` and are obeyed.
  - `POST /dreams/cycle/run` can answer `{"status": "held", "reason": "economy_<reason>"}`.
  - The market no longer applies Prime's charge to a shadow bid of a receiving family.
- Compatibility: `want_ledger.shadow` is still always true. The producer tells obeyed bids apart, as in #3010.

## Env/config changes

- Added keys: `ORION_PHYSIOLOGY_BID_DREAM_CYCLE=on` (orion-dream). The same default is set in `settings.py`, `.env_example` and compose.
- Removed / renamed: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. The primary checkout's `services/orion-dream/.env` now has `ORION_PHYSIOLOGY_BID_DREAM_CYCLE=on`. The existing `ORION_PHYSIOLOGY_` prefix covers it, and orion-dream is in `DEFAULT_SERVICES`.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-dream: pytest tests evals                     -> 345 passed, 9 skipped
services/orion-durable-runs: pytest tests                   -> 621 passed, 117 skipped (skips need a test Postgres DSN)
PYTHONPATH=. pytest orion/prime orion/physiology/tests scripts/analysis/tests tests/test_check_body_economy_contract.py
                                                             -> 833 passed
services/orion-mind: pytest tests                           -> 158 passed
python scripts/check_body_economy_contract.py               -> OK (96 state-reading modules, all registered)
.github/workflows/orion-static-gates.yml, every run step locally -> 40 of 40 OK
```

## Evals run

```text
PYTHONPATH=. python services/orion-dream/evals/dream_binding_replay.py          (fixture, CI)
  8 h after the last sleep, no charge: funded at once; slow 0.5: held 180 min (19 bids), then funded by its own
  hunger; 16 h after: outbid the charge at once; overdue (48 h): floor:overdue
  PASS
PYTHONPATH=. python services/orion-dream/evals/dream_binding_replay.py --live --days 7
  13 real sleeps; tables above; 3 h signals: 136/330 windows saw a bid, 60% of those would grade 1
  PASS (gates: binding alone holds no warm-history sleep > 60 min; under the most Prime can do <= 5% of warm sleeps
        run only because their window ran out (0 of 7); some bids outbid the charge; the grade is not stuck)
python services/orion-durable-runs/evals/curiosity_binding_replay.py  -> PASS (unchanged after the rule moved)
python services/orion-durable-runs/evals/prime_charge_replay.py       -> PASS
python scripts/analysis/eval_body_economy_shadow_vs_real.py           -> PASS
python orion/prime/evals/run_prime_frame_eval.py                       -> 9/9 PASS (regulate now expects the dream)
```

## Docker/build/smoke checks

```text
No image build. The change is Python and YAML only. orion-dream's Dockerfile already copies app/ and orion/.
durable-runs already copies orion/ and config/physiology. Not deployed (task rule).
```

## Review findings fixed

A code-review subagent read the first commit (all its tests green). Its findings:

- Finding (should-fix): the 6 h window started at the first check that wanted to sleep. Arousal often holds those checks for hours first (live 10-29 23:00 .. 10-30 05:00 and 09:00 .. 13:02). So most real sleeps would reach "window over" at their first bid, and the market and Prime would never decide them.
  - Fix: the window starts at the want's first bid (`HeldWant.first_bid_at`).
  - Evidence: `test_arousal_holds_do_not_use_up_the_economys_window`. The live eval's want start now matches this (the real sleep's start is the first check past arousal).
- Finding (should-fix): a funded sleep that raised kept its metacog reservation until it expired.
  - Fix: release in the exception path.
  - Evidence: `test_a_funded_sleep_that_raises_frees_its_reservation`.
- Finding (should-fix): in rollback (`shadow`), and on forced runs, the market still charged the dream's shadow bids and recorded "held". So Prime would be graded on holds nothing obeyed. The same gap existed for curiosity.
  - Fix: `BINDING_PRODUCERS` in `orion/physiology/binding.py`. The market reads Prime's signal only for those producers.
  - Evidence: `test_a_shadow_bid_of_a_receiving_family_is_never_charged_or_recorded`, `test_the_producer_is_one_the_market_charges_for_prime`.
- Finding (should-fix): the sleep loop and a manual run could both admit the same want and open two acts with one cycle id.
  - Fix: a per-process lock and an `admitted` flag. The second check is refused (`economy_duplicate`) and opens nothing.
  - Evidence: `test_two_checks_racing_on_one_want_open_one_sleep`.
- Finding (nit): two eval gates could not fail (a hold past 6 h is impossible by construction), and cold-start holds were silent.
  - Fix: the gate is now the warm window-end share (5%, #3011's number). Cold-start holds are printed as "NOT gated", with the reason.
- Finding (nit): "at most 6 h" ignored restarts.
  - Fix: README, families.yaml and the module docstring now say the 6 h is per process lifetime, and the 48 h overdue line is the bound across restarts.
- Finding (nit): no test of want reuse across held checks. Fixed by the arousal-window test above.
- Finding (nit, not fixed): Prime's frame lists receivers as bare names (`- dream.cycle: slow`), with no line on what slowing the dream means.
  - Status: listed under Risks. Adding a per-receiver description is a prompt and config change of its own.

## Restart required

After merge, from the clean primary `main` checkout synced with `origin/main`:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-dream up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
```

Order:

1. The market first, so the dream's binding producer is charged and shadow bids are not.
2. Then the dream.
3. Then orion-mind. It reads `families.yaml` from its repo mount, so a restart is not needed for the receiver list. Rebuild it only to pick up the doc-string change, or skip it.

If the dream goes first, its binding bids still work. The old market image still carries the old families file, so the dream gets no Prime charge until orion-durable-runs is rebuilt.

Rollback without a code change: set `ORION_PHYSIOLOGY_BID_DREAM_CYCLE=shadow` in `services/orion-dream/.env`, then run the orion-dream command. To also remove Prime's reach, delete `receptors: [slow]` under `dream.cycle` in `config/physiology/families.yaml`.

## Risks / concerns

- Severity: medium. Concern: the live path is UNVERIFIED. Nothing is deployed. No dream sleep has bid in binding mode, and no Prime signal to the dream has been written.
  - Mitigation: after the deploy, `docker logs orion-athena-dream | grep physiology_bind` should show one `verdict=` line per sleep attempt.
  - Also run: `select decided_at, outcome, reason, net, decision->>'why' from want_ledger where producer='orion-dream.dream_cycle.binding' order by decided_at desc limit 10;`.
- Severity: medium. Concern: Prime's slow charge can delay a short-gap sleep by up to about 4 hours (warm replay, the most Prime can do). One real consequence is not in the replay: a held sleep can then lose its idle window when Juniper comes back, and wait for the next idle stretch. That stretch is bounded by the overdue backstop, which also needs idle.
  - Mitigation:
    - The rest period means Prime can hold at most half of any stretch.
    - The dream's sleep count and holds show in `want_ledger` and the cycle notes.
    - `orion/prime/evals/run_prime_report.py` tallies the regulate acts.
- Severity: low. Concern: the old arousal hold in `cycle.py` is still in place.
  - Mitigation: as in #3010, an old gate is deleted only after its binding bid has run `on` for 7 days with the checks green. That is the next PR.
- Severity: low. Concern: a cold history (fewer than 6 dream sleeps in 30 days) makes the market read "no hunger yet" at high intensity. In the replay that held 4 sleeps (one for 6 h).
  - Mitigation: the window bounds it, and today's history is warm. The replay prints cold-start holds every run.
- Severity: low. Concern: Prime is not told what slowing the dream means (review nit, not fixed).
  - Mitigation: the frame shows each signal's effect in the market's own words ("value 0.18 ... did not beat Prime's slow charge 0.50"). The regulate act's note says "N of its runs bid" for the dream too ("runs" means sleeps).
- Severity: low. Concern: the dream's want lives in process memory, so a restart opens a new want and a new 6 h window.
  - Mitigation: the 48 h overdue line is the hard bound across restarts.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3014

🤖 Generated with [Claude Code](https://claude.com/claude-code)
