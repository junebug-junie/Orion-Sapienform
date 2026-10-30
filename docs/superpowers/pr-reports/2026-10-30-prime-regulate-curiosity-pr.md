# feat(prime): Prime can ask curiosity to slow down or quicken (Orion Prime step 3)

Stacked on #3009 and #3010; merge those first.

## Summary

- Prime can now send one organ a "slow down" or "quicken" signal. Today the only organ that listens is curiosity's investigate line, because it is the only organ whose bids the body economy obeys (#3010). Curiosity listens to "slow" only (see "Why curiosity does not take quicken").
- The signal is not a command. The body economy turns it into a temporary charge on curiosity's bids. The charge is at most half of what a curiosity run is worth. A hungry, valuable run still outbids it, and a run's hunger keeps growing while it waits.
- Each signal is an act on the ledger, with the existing verb `regulate`. It is graded on one thing: did the charge change at least one of curiosity's decisions while it was in force. The market writes down what the charge did to every bid; Prime only reads that record.
- Prime sees the result in its next thoughts: how many runs bid, how many its charge held, how many outbid it, and the market's own sentence for the latest one ("value 0.71 - lane charge 0.00 beat Prime's slow charge 0.54").
- One in five signals is held back at random. A held-back signal is written but not applied; the market records what it would have done. That gives a fair comparison.
- Both sides have a switch, shipped on: `PRIME_REGULATE_ENABLED` (Prime stops offering the move) and `ORION_PHYSIOLOGY_PRIME_REGULATION` (the market ignores every signal).

## Outcome moved

- **New capability, bounded.** Prime had one move: ask curiosity a question. It now has a second: change how often curiosity runs, a little, for a few hours. In the 7-day replay below, the most Prime can do holds about 1 run in 5 for a median of 2 hours, and never past a run's 6-hour window.
- **The design's proposed signal `target_moved` was not shipped.** It failed the metric quality gate on live data (next section). The act is graded on something honest instead.

## Metric quality gate

### The proposed signal: `target_moved` ("the organ's own rate moved in the stated direction by the horizon")

1. **Provenance.** Curiosity investigate starts: the first `run.resource_granted` event per run in `durable_resource_events` (joined to `durable_admission_runs.request`, line `investigate`, not urgent), which is also when `services/orion-durable-runs/app/run_acts.py` opens the run's act in `grammar_acts`. 171 starts over 30 days.
2. **Independence.** Curiosity's start rate is set mostly by Hub's own scheduler (daily cap, cooldown, the morning burst). A before/after rate comparison around a signal measures that schedule first and Prime second.
3. **Theory anchor.** A regulate act is a TOTE unit: the "test" after "operate" should check whether the receiving organ's behaviour changed because of the signal. A raw before/after rate is not that test: it has no counterfactual.
4. **Live sanity: FAILED.** 120 of the 171 starts fall between 06:00 and 10:00 UTC. For random 3-hour windows over 30 days, both the 3 hours before and the 3 hours after had zero starts 59% of the time (6 hours: 39%; 1 hour: 77%). A "slow" signal sent outside the morning burst would grade 0 almost always, whatever it did; one sent at the end of the burst would grade 1 almost always, whatever it did. The binary mostly reads the clock.
5. **Existing mechanism.** The market already decides every curiosity bid and already says why. The honest test is to ask the market what the charge changed.
6. **Reversibility.** `target_moved` was not added to `ActSignal` (`orion/schemas/grammar.py` is unchanged), so there is nothing to unwind.

### What the act is graded on instead: `artifact_produced` = "the charge changed at least one decision"

1. **Provenance.** `services/orion-durable-runs/app/physiology_market.py` `Market._record_prime_effect`, on every bid of the family while the signal is in force: the decision's net with and without the charge (`orion/physiology/prime_charge.py` `effect_of`). Written by the market into the signal's Redis hash. Prime never judges it; it reads the record after the signal ends.
2. **Independence.** It is the mechanism itself, not a second measure of curiosity's rate. It is not claimed as a rate signal. The effect on curiosity's real rate is measured separately, as treated vs held-back starts per signal hour (`orion/prime/evals/run_prime_report.py` `regulation`).
3. **Theory anchor.** Compliance: a regulation that never changes the receiver's decision did nothing, whatever the rate did. This is the receiver's own decision record, the same data the run event `run.economy_held` carries.
4. **Live sanity: PASSED.** Replay of 7 days of real curiosity bids (`services/orion-durable-runs/evals/prime_charge_replay.py --live`): 148 of 330 three-hour windows saw a curiosity bid; of those, 41% would grade 1. Not stuck at 0 or 1. The 182 windows with no bid grade 0 ("nothing came to slow"), which is true, and Prime is told so in those words.
5. **Existing mechanism.** Reuses Choice, the binding rule from #3010, the act ledger and the holdback pattern from Prime's questions. No new table, verb, channel or schema.
6. **Reversibility.** One Redis hash per family with a TTL; two flags. Turning either flag off stops it at once; nothing persists past the TTL.

## Current architecture

- Prime (orion-mind `app/prime_loop.py`, pure logic `orion/prime/`) had one move: a question for curiosity's next run, through Hub's focus slot.
- The body economy's Choice (`orion/physiology/choice.py`) decided bids on value minus lane prices. Curiosity investigate's bids bind (#3010): a price deferral or a net at or below zero holds the run and it bids again later; a full lane with a positive net queues at the GPU pool.
- `ActVerb` already had `regulate`, with no producer.

## Architecture touched

- **New `orion/physiology/prime_charge.py`**: the signal record, the bound, the effect classes. Pure.
- **`config/physiology/families.yaml`**: new optional `receptors` field; `durable.curiosity_investigate: receptors: [slow]`. Parsed and checked in `orion/physiology/families.py` (Tier 0 cannot have one).
- **Choice**: `decide(..., prime=signal)` adds the charge as `prime:<direction>` in the charge, names it in `why`, and lets it be the binding constraint of a price deferral. Retry for that case uses the default re-bid (no price to decay; hunger is re-tested).
- **Market (orion-durable-runs)**: reads `orion:prime:regulate:<family>` per bid when the flag is on and the family has the receptor; applies a treated signal; records what it did (or would have done, for a control signal) back into the hash. To say what the charge changed, the market decides the same bid both ways (the second time on a throwaway copy of its reservation book) and compares the two real decisions, so the exploration floor and the capacity line count the same in both arms.
- **Prime (orion-mind)**: the `regulate` move, receptors read from the repo mount, the Redis write, the act, grading after the signal ends, the frame section, and the guards.

### Why curiosity does not take quicken

A quicken credit must not be a mandate, the mirror of "slow is never a veto". So a credit is never applied to a want the market values at zero (the family just acted; nothing in its history is less hungry). Over the 7 days to 2026-10-30, curiosity's only price-side holds were exactly those ("no hunger yet", 2 of 46 runs in the #3010 replay) and lane prices were about 0. So on curiosity a quicken credit could change nothing. Declaring a receptor that cannot move anything would be a label with no behaviour. The market handles quicken and it is tested; an organ should declare it once it has real price deferrals.

### Why a Redis hash

The market already reads the posted prices from the bus Redis on every bid, and Prime already keeps its state there. A hash with a TTL needs no migration, expires by itself if Prime dies, and lets the market write its record next to the signal so Prime reads both with one call. A table would need a migration and a sweeper for no gain.

### Guards (design section 11)

- One signal per organ until it is graded; then the organ rests from Prime's signals for as long as the last signal lasted. So Prime's charge can hold an organ at most half of any stretch of time.
- 1 to 6 hours per signal (6 hours is curiosity's deferrable window: past it a held run goes anyway).
- A signal to a family without the receptor, or a direction it does not handle, is refused and shown to Prime with the reason. The market also ignores it.
- The charge is never above half the family's weight; the exploration floor still applies; a held run still goes at its window's end.

## How a slow signal plays out (fixture eval, real Choice and binding code)

A curiosity run admitted 2 minutes after the last one closed:

- no charge: held 600 s (no hunger yet), then funded;
- slow charge 0.5: held 3,600 s over 7 bids, then funded by its own hunger. First why: "value 0.00 - lane charge 0.00 did not beat Prime's slow charge 0.50"; last: "value 0.64 - lane charge 0.00 beat Prime's slow charge 0.50";
- quicken: held 600 s, the same as no charge. The credit does not apply to a want the market values at zero, and the why says so.

## Live replay: what a 0.5 slow charge would have done (7 days to 2026-10-30 20:02 UTC, 46 real curiosity runs)

| scenario | bids that met the charge | outbid it | runs held by it | held to the 6 h window end | started later than with no charge |
|---|---|---|---|---|---|
| in force the whole week (not reachable) | 113 | 43 | 19 / 46 | 2 | 19 runs, median 130 min, max 360 min |
| 6 h on / 6 h off, phase 0 (the most Prime can do) | 48 | 22 | 10 / 46 | 0 | 10 runs, median 120 min, max 240 min |
| 6 h on / 6 h off, phase 1 | 41 | 14 | 9 / 46 | 0 | 9 runs, median 120 min, max 360 min |

The 2 runs held to the window end in the unreachable case were both on 10-26, a day with 11 curiosity runs: each close reset curiosity's hunger, so later runs stayed below the charge. That is the charge doing its job on the busiest day, and those runs still ran at the window end. The rest period stops Prime from holding that pressure all day.

## Files changed

- `orion/physiology/prime_charge.py` (new): the signal contract, bound, effects.
- `orion/physiology/families.py`, `config/physiology/families.yaml`: `receptors`.
- `orion/dig/__main__.py`: merge resolution only (main's #3008 capacities help text and #3009's `arc|act` line), trimmed to the existing 2,500-character help cap.
- `orion/physiology/choice.py`: the charge in `decide`.
- `services/orion-durable-runs/app/physiology_market.py`, `app/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`: read, apply, record; the flag.
- `orion/prime/thought.py`: the move, its refusals, the prompt section (only when on), the frame section.
- `orion/prime/acts.py`: the `regulate` act, its grade, the plain-language effect line.
- `orion/prime/state.py`: open signals and per-organ rest in the persisted state (defaulted).
- `services/orion-mind/app/prime_loop.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`: write, grade, guards; two keys.
- `orion/prime/evals/run_prime_report.py`: `regulation` tally (changed share and real starts per signal hour, per arm). `orion/prime/evals/run_prime_frame_eval.py`: `regulate` check.
- `scripts/analysis/body_economy_binding_replay.py`: optional `prime` charge in the replay.
- `services/orion-durable-runs/evals/prime_charge_replay.py` (new): fixture and live eval.
- `config/physiology/contract.yaml`: the new modules registered as sources (they gate no organ).
- Tests: `orion/physiology/tests/test_prime_charge.py`, `orion/prime/tests/test_prime_regulate.py`, `orion/prime/tests/test_prime_evals.py`, `services/orion-durable-runs/tests/test_prime_charge_market.py`, `services/orion-mind/tests/test_prime_regulate_loop.py` (all new or extended).
- CI: `.github/workflows/orion-durable-runs-tests.yml` (the eval), `.github/workflows/orion-prime-tests.yml` (paths).
- Docs: `services/orion-mind/README.md`, `services/orion-durable-runs/README.md`.

## Schema / bus / API changes

- Added: Redis hash `orion:prime:regulate:<family>` (field `signal` = `PrimeRegulationV1` JSON written by Prime; fields `w:<want_id>` = `PrimeChargeEffectV1` written by the market). Not a bus channel; not in the schema registry (a private contract between two services, like `orion:prime:state` and `orion:physiology:latest`).
- Added: `receptors` in families.yaml. `PrimeStateV1.open_regulations` and `.regulation_rest_until` (defaulted; older persisted state loads, tested).
- New act kind on the existing grammar channel: `actor prime, verb regulate, signal artifact_produced`.
- Behaviour changed: a curiosity bid's `charge` and `net` include Prime's charge while one is in force, and `why` says so. `binding_constraint` can be `prime:slow`. `WantDecisionV1` is unchanged (no new field or reason).
- Removed / renamed: none. `ActSignal` unchanged.

## Env/config changes

- Added keys: `ORION_PHYSIOLOGY_PRIME_REGULATION=true` (orion-durable-runs); `PRIME_REGULATE_ENABLED=true`, `PRIME_FAMILIES_PATH=/repo/config/physiology/families.yaml` (orion-mind). Defaults match in `settings.py`, `.env_example` and compose.
- Removed / renamed: none.
- `.env_example` updated: yes, both services; settings and compose too.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (all three keys added; covered by the `ORION_PHYSIOLOGY_` and `PRIME_` prefixes).
- Skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. pytest orion/prime orion/physiology/tests orion/dig/tests scripts/analysis/tests tests/test_check_body_economy_contract.py
  -> 906 passed
services/orion-mind: pytest tests                         -> 158 passed (12 new regulate-loop tests, incl. one on
                                                             redis-py's own asyncio client via fakeredis: MULTI, TTL, bytes)
services/orion-durable-runs: pytest tests                 -> 619 passed, 117 skipped (the skips need a test Postgres DSN;
                                                             this change touches no SQL)
  new: tests/test_prime_charge_market.py (7: treated holds and records why, control records only, floor
       classed the same in both arms, off/expired/foreign/no-receptor never read as a charge, Redis down
       never fails a bid, fakeredis client semantics)
python scripts/check_body_economy_contract.py             -> OK (94 state-reading modules, all registered)
.github/workflows/orion-static-gates.yml, every run step locally -> 41 of 41 OK (incl. definition drift, body-economy contract, env parity, settings defaults)
.github/workflows/orion-prime-tests.yml, every step locally      -> pass (prime pure, refs contract, frame eval, orion-mind)
```

## Evals run

```text
PYTHONPATH=. python services/orion-durable-runs/evals/prime_charge_replay.py              (fixture, in CI)
  no charge held 600 s (2 bids); slow 0.5 held 3600 s (7 bids, funded); quicken held 600 s
  PASS
PYTHONPATH=. python services/orion-durable-runs/evals/prime_charge_replay.py --live --days 7
  46 real curiosity runs; table above; windowed 3 h grade: 148/330 windows saw a bid, 41% of those grade 1
  PASS (gate: at most 5% of runs held to their window's end under the most Prime can do; some bids outbid the charge;
        the grade is not stuck at 0 or 1)
PYTHONPATH=. python orion/prime/evals/run_prime_frame_eval.py       -> 9/9 PASS (new: regulate)
python services/orion-durable-runs/evals/curiosity_binding_replay.py -> PASS (Patch D unchanged)
orion/prime/evals/run_prime_report.py gains a `regulation` section (unit-tested; live run needs regulate acts first)
```

## Docker/build/smoke checks

```text
No image build: Python and YAML only; both Dockerfiles already copy orion/ and the services' app/; orion-durable-runs
copies config/physiology; orion-mind reads families.yaml from its read-only repo mount (/repo). Not deployed (task rule).
```

## Review findings fixed

A code-review subagent read the first two commits. All its tests were green. Its findings:

- Finding (must-fix): the committed tree failed the body-economy contract gate (three new modules unregistered); the PR report was untracked.
  - Fix: registered `prime_charge.py`, `orion/prime/acts.py`, `orion/prime/thought.py` as sources in `config/physiology/contract.yaml`; report committed.
  - Evidence: `check_body_economy_contract.py` OK.
- Finding: the held-back arm read the exploration floor off the uncharged decision, so a floor-funded want counted as "held" in control but "floor" in treated. The arms were not graded the same way.
  - Fix: the market decides each bid both ways (throwaway book copy) and `effect_between` compares the two real decisions. The replay does the same.
  - Evidence: `test_both_arms_class_an_exploration_floor_fund_the_same_way[treated|control]`.
- Finding: quicken funded any want, even one with no hunger, so it was a mandate and its grade was ~1 by construction.
  - Fix: a credit is never applied to a want ranked 0; curiosity declares `slow` only (reason above).
  - Evidence: `test_a_quicken_credit_lifts_a_price_deferred_want_but_never_funds_one_valued_at_zero`; fixture eval.
- Finding: the signal write was three Redis calls; a partial write could leave a charge in force while Prime recorded "not delivered", or a key with no TTL.
  - Fix: one MULTI pipeline (delete, hset, expire).
  - Evidence: the failed-write test asserts nothing is left in Redis; the fakeredis test runs the real client.
- Finding: market records were not tied to their signal. Fix: `signal_id` on each record; Prime counts only its own. Evidence: merge/parse tests and the loop test with a left-over record.
- Finding: a bad `hours` value (or an over-long family) lost the whole thought. Fix: lenient fields, clamped/truncated in the runtime check. Evidence: `test_a_malformed_hours_never_loses_the_thought`.
- Finding: with the market's switch off, Prime was told "no bid came". Fix: on an empty record Prime reads the market's `/health` and says the market has its signals switched off. Evidence: `test_with_the_markets_switch_off_prime_is_told_so_not_that_no_bid_came`.
- Finding: the report verdict could crash on closed acts with no usable window, and called an effect without a test. Fix: None guard; exact conditional binomial test at p < 0.05. Evidence: `test_regulation_verdict_needs_enough_signals_and_a_significant_rate_difference`.
- Finding: image rollback of orion-mind would drop all Prime state (new keys, `extra=forbid`). Fix: documented with a one-line strip command (README, Risks).
- Nits fixed: record cap off by one; flag-off test now asserts the hash is never read; the durable-runs workflow's push trigger now includes `orion/physiology/**` and `config/physiology/**`; the decision's `charge` including Prime's part is documented.
- Not fixed (nit): a failed publish of a regulate act's open is not retried, the same as the existing question acts. A close without an open is kept by the ledger as an open-less close; listed under Risks.

## Restart required

After #3009 and #3010 are merged and deployed, merge this, then from the clean primary `main` checkout:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
```

Order: the market first, so a signal Prime writes is read from the start. The other order is also safe: a signal written before the market knows it is simply not applied, and its act grades 0 with "no bid from it came".

Rollback without a code change: `PRIME_REGULATE_ENABLED=false` in `services/orion-mind/.env` and/or `ORION_PHYSIOLOGY_PRIME_REGULATION=false` in `services/orion-durable-runs/.env`, then the same command for that service.

## Risks / concerns

- Severity: medium. Concern: the live path is UNVERIFIED. Nothing has been deployed; no Prime signal has been written, read by the market, or graded on the live rail. Mitigation: after deploy, `docker logs orion-athena-durable-runs | grep prime_charge` shows one line per curiosity bid under a signal; `select verb, test_before, outcome_value, outcome_note from grammar_acts where actor='prime' and verb='regulate' order by opened_at desc limit 10;` shows the acts.
- Severity: medium. Concern: deploy order with #3009 and #3010. This branch contains both. Mitigation: merge and deploy those first, as their reports say; then this.
- Severity: low. Concern: Prime may rarely or never use the move. The prompt says to use it only when a concern is about how often an organ runs. Mitigation: `run_prime_report.py` counts signals per arm; if there are none after two weeks, the move is not earning its place.
- Severity: low. Concern: regulate closes join the `prime` family's act history, so they also feed Prime's own value and hunger clock (the same side effect #3009 noted for concern closes). Mitigation: measure with `run_prime_report.py`; the fix, if needed, is in `orion/physiology/appetite.py`.
- Severity: low. Concern: image rollback of orion-mind needs the two new state keys stripped first (command in the orion-mind README). The flag rollback needs nothing.
- Severity: low. Concern: a failed publish of a regulate act's open is not retried (same as question acts).
- Severity: low. Concern: the effect on curiosity's real rate (treated vs held-back starts per hour) needs at least 10 graded signals per arm before the report will call it; until then the verdict is `too_few`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3011

🤖 Generated with [Claude Code](https://claude.com/claude-code)
