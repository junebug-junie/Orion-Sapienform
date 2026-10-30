# feat(prime): Prime can ask curiosity to slow down or quicken (Orion Prime step 3)

Stacked on #3009 and #3010; merge those first.

## Summary

- Prime can now send one organ a "slow down" or "quicken" signal. Today the only organ that listens is curiosity's investigate line, because it is the only organ whose bids the body economy obeys (#3010).
- The signal is not a command. The body economy turns it into a temporary charge on curiosity's bids (or a credit, for quicken). The charge is at most half of what a curiosity run is worth. A hungry, valuable run still outbids it, and a run's hunger keeps growing while it waits.
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
- **`config/physiology/families.yaml`**: new optional `receptors` field; `durable.curiosity_investigate: receptors: [slow, quicken]`. Parsed and checked in `orion/physiology/families.py` (Tier 0 cannot have one).
- **Choice**: `decide(..., prime=signal)` adds the charge as `prime:<direction>` in the charge, names it in `why`, and lets it be the binding constraint of a price deferral. Retry for that case uses the default re-bid (no price to decay; hunger is re-tested).
- **Market (orion-durable-runs)**: reads `orion:prime:regulate:<family>` per bid when the flag is on and the family has the receptor; applies a treated signal; records what it did (or would have done, for a control signal) back into the hash.
- **Prime (orion-mind)**: the `regulate` move, receptors read from the repo mount, the Redis write, the act, grading after the signal ends, the frame section, and the guards.

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
- quicken: funded at the first bid.

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

- Added keys: `ORION_PHYSIOLOGY_PRIME_REGULATION=true` (orion-durable-runs); `PRIME_REGULATE_ENABLED=true`, `PRIME_FAMILIES_PATH=/repo/config/physiology/families.yaml` (orion-mind).
- Removed / renamed: none.
- `.env_example` updated: yes, both services; settings and compose too.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (all three keys added; covered by the `ORION_PHYSIOLOGY_` and `PRIME_` prefixes).
- Skipped keys requiring operator action: none.

## Tests run

```text
TESTS_PLACEHOLDER
```

## Evals run

```text
EVALS_PLACEHOLDER
```

## Docker/build/smoke checks

```text
No image build: Python and YAML only; both Dockerfiles already copy orion/ and the services' app/; orion-durable-runs
copies config/physiology; orion-mind reads families.yaml from its read-only repo mount (/repo). Not deployed (task rule).
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

After #3009 and #3010 are merged and deployed, merge this, then from the clean primary `main` checkout:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
```

Order: the market first, so a signal Prime writes is read from the start. The other order is also safe: a signal written before the market knows it is simply not applied, and its act grades 0 with "no bid from it came".

Rollback without a code change: `PRIME_REGULATE_ENABLED=false` in `services/orion-mind/.env` and/or `ORION_PHYSIOLOGY_PRIME_REGULATION=false` in `services/orion-durable-runs/.env`, then the same command for that service.

## Risks / concerns

RISKS_PLACEHOLDER

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
