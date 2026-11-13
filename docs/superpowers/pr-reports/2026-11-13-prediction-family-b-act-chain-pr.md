# Family B, first chain: what will Orion do in the next 15 minutes?

## Summary

- New `services/orion-prediction/app/act_chain.py`. It forecasts which act Orion does next, per 15-minute bucket. A bucket's state is the main verb of Orion's acts (`grammar_acts`), or `idle` when Orion did nothing.
- The served model is a first-order Markov chain, split by Juniper's clock blocks (asleep 03-08, weekend morning, day, evening, Denver time). Each level backs off to the coarser one. The backoff strength is chosen from training data by leave-one-out.
- It is graded per local day against three baselines: the overall mix, the mix for the clock hour, and the mix for the clock block. The grade is a loss gap, its standard error and `prob_better`, with no alpha line.
- Surprise = -log P(what happened). Excess surprise = surprise minus the forecast's own entropy. When the forecast is calibrated, its average is 0 by theory.
- This is a measurement and eval only: no table, endpoint, bus event, node or consumer. Not deployed.

## Outcome moved

Family B now has its first chain that predicts Orion's own behaviour, with a replay grade on real data. This is the honest replacement for the retired `route` signal and for the bus signal's "service rhythm". Silence where Orion usually acts is a scored outcome, so an outage of the act stream is visible.

## Why this chain first

| Candidate | Live data (checked 2026-11-13) | Verdict |
|---|---|---|
| Orion's acts (`grammar_acts`) | 4,629 acts since 10-23, 9 verbs used daily since ~10-27; 67 % of 15-min buckets busy | **picked** |
| Juniper presence | already a Family B forecaster: `app/reachable.py` (#3180) | reuse, not duplicate |
| AC on/off (`home_cooling_sample`) | healthy, but AC is off for fall by Juniper's choice (window cooling): ~10 switches in 49 days | waits for a cycling season, #3343 |
| Unified-turn lane | no lane field is persisted; `fcc_served_model` follows deploys, not world state | not a world state |
| Service health (`equilibrium_service_transitions`) | flapping; 3 services make most transitions | later |

## Results (replay, `evals/run_act_chain_replay_eval.py`, run 2026-11-13 20:15Z, after the review fixes)

Held-out: 1,423 buckets over 16 local days (10-29..11-13); each bucket is forecast only from rows landed before it. 7-day warm-up from 10-23.

| Model | log loss (nats/bucket) |
|---|---|
| marginal (baseline) | 1.859 |
| clock hour (baseline, best) | 1.800 |
| clock block (baseline) | 1.825 |
| Markov | 1.804 |
| **Markov + clock block (served)** | **1.793** |

Skill vs the best baseline: 0.4 % lower loss. Small.

| Served vs | gap/day | SE | prob_better |
|---|---|---|---|
| marginal | -0.102 | 0.060 | 0.95 |
| clock hour | -0.042 | 0.071 | 0.72 |
| clock block | -0.058 | 0.049 | 0.87 |
| Markov alone | -0.014 | 0.009 | 0.94 |

Plain reading: knowing the last act and the clock block beats knowing nothing (P 0.95). It probably beats the clock alone, but 16 days can't tell yet (P 0.72).

**Calibration (held-out days).** Excess surprise is +0.09 ± 0.08 nats (P below zero 0.13): within about one SE of calm. When the top state has a forecast p of 0.2-0.4, it happens 24 % of the time. At 0.4-0.6 it happens 41 %. At 0.6-0.8 (25 buckets) it happens 56 %. So the model is a little overconfident. The forecast share of each state matches the observed share within 0.06 (idle 0.25 vs 0.19 is the largest gap).

**Rest-state check (live data).** The rolling 1-hour excess has a median of +0.08. It sits at or below 0 in 44 % of hours, and its 5-95 % range is -0.76..+1.10. It returns to calm. It has no floor like `bus_synaptic`'s 0.27. One exception: the asleep block reads +0.19 ± 0.05, so it is overconfident at night (#3344).

**Spikes sit on real events.** The biggest surprise (10.1 nats) is the bucket of 2026-10-30 20:45Z. It holds the first `predict` act Orion ever opened (`min(opened_at)` = 20:58:15Z). That was a new kind of act, so a large surprise is correct.

**One design change, made before the final grade.** The first run used a fixed backoff of 8 pseudo-counts. The result: the served model lost to the clock hour, excess surprise was +0.21, and some rare acts got p ~ 1e-6. A context with hundreds of transitions trusted its zero counts, and the shrinkage multiplied down through the levels. Now each level picks its backoff strength by leave-one-out on its own training rows (empirical Bayes, MacKay & Peto 1995). No held-out data is used. Both runs are reported here. The chosen strengths are clock 32, Markov 64, Markov + block 64 (median). The grid goes to 4096, so it caps none of them.

## Metric quality gate

1. **Provenance.** Rows: `orion/grammar/act_ledger.py` `_INSERT_OPEN_SQL`, written by orion-sql-writer from every `act_opened` event. Verbs are the closed `ActVerb` list (`orion/schemas/grammar.py:37`). Surprise: `act_chain.score`.
2. **Independence.** The `say` state (8 % of buckets) includes `hub.chat_reply`, which Juniper's messages cause. So it overlaps partly with `reachable` and with `node:prediction.chat_turn` (same turns). The other 8 states come from Orion's own loops: world pulse, journal, Prime, curiosity, dreams. Family A (physical readings) is linked only indirectly, through heat vetoes that suppress acts.
3. **Theory anchor.** First-order Markov surprisal under a proper log score. Under a calibrated forecast, the expected surprisal equals the forecast entropy, so the mean excess is 0.
4. **Live sanity.** 10 states are seen and none dominates (top: look 26 %, make 25 %, idle 19 %). The score returns below 0 in 45 % of hours. It is not flat, not saturated and not floored. The asleep bias is recorded (#3344).
5. **Existing mechanism.** The only other Markov model is `reachable_state` (hours since Juniper's last message). Nothing forecasts acts. The module reuses `backtest.paired_gap` for the daily gap test.
6. **Reversibility.** No schema or table, and nothing reads it. Remove it by deleting one module, its eval and its tests.

## Current architecture

orion-prediction had Family A (continuous body/room readings) and one Family B forecaster (`reachable.py`, Juniper messages per hour). Nothing forecast Orion's own acts. The retired `route` and bus "service rhythm" signals had no successor.

## Architecture touched

orion-prediction only, read-only on `grammar_acts`. No contracts changed.

## Files changed

- `services/orion-prediction/app/act_chain.py`: states, models, replay, verdict.
- `services/orion-prediction/app/store.py`: `grammar_acts(start, end)` read.
- `services/orion-prediction/evals/run_act_chain_replay_eval.py`: live replay (read-only, ~3 s).
- `services/orion-prediction/tests/test_act_chain.py`: 11 tests (state rules and tie-break, late rows, clock blocks in Denver time, no leak into the condition, no leak into training conditions, live-equals-replay, the exact training edge (o-3 in, o-2 out), only settled buckets scored, deterministic chain, calm on iid data, silence scored, min-days gate).
- `services/orion-prediction/README.md`: section and eval command.
- `.github/workflows/orion-prediction-tests.yml`: run on `orion/schemas/grammar.py` changes (the verb list).

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none. Behaviour changed: none.

## Env/config changes

- None. `.env_example` unchanged; no sync needed.

## Tests run

```text
pytest services/orion-prediction/tests -q   (prediction image)  -> 459 passed
pytest tests/test_act_chain.py -q                                -> 11 passed in 2.73 s
mutation check (5 one-line bugs the review found surviving)      -> each now fails the suite
```

## Evals run

```text
python evals/run_act_chain_replay_eval.py --since 2026-10-23   (live DB, read-only) -> numbers above
```

## Docker/build/smoke checks

```text
Ran tests and the eval in the existing orion-prediction-prediction image with the worktree mounted read-only.
No service runtime change, so no build or up.
```

## Review findings fixed

The review ran in a subagent (orion-repo-agent; it did not have the Skill tool, so it reviewed by hand plus 8 one-line mutations). It found no leak in the code itself. The window math, the asof/final cutoffs and the LOO formula were checked and are correct. It found that the tests were weaker than the code.

- Finding: a forecast conditioned on a late row (`final[o-1]` instead of `asof[o-1]`) passed every test. The planted late row lost a tie to a real act.
  - Fix: the no-leak test now makes bucket o-1 idle as known, and asserts `prev_state == "idle"` with the late row present.
  - Evidence: that mutation now fails 1 test.
- Finding: the same leak on the training side (`final[t-1]` as the condition) passed every test.
  - Fix: new `test_training_conditions_use_what_was_known_not_the_final_outcome`: idle-as-known buckets are always followed by `look`, so only correct code learns idle → look.
  - Evidence: that mutation now fails 1 test.
- Finding: the tie-break test passed even with plain argmax ("say" comes before "make" in the verb list).
  - Fix: added the reverse order (say, then make → make).
  - Evidence: plain argmax now fails.
- Finding: LOO left the held-out point inside the parent (`P_a`), which inflates K in thin contexts.
  - Fix: the block level now gets a per-transition leave-one-out parent.
  - Evidence: the chosen K for the served level dropped from 256 to 64 (median); the replay numbers above are after this fix.
- Finding: replay's settled end bound and the training edge (o-3 in) were untested.
  - Fix: `test_replay_scores_only_settled_buckets`; the training test now asserts o-3 trains and o-2 does not.
  - Evidence: both mutations (no grace, `t_hi - 1`) now fail.
- Finding: float timestamps lose ~256 ns near a cutoff.
  - Fix: `_ns` uses exact integer microseconds.
  - Evidence: tests pass.
- Finding (UNVERIFIED in review): a close-before-open row keeps an early `created_at`, a possible small leak.
  - Fix: none needed. Checked live: 0 rows in `grammar_acts` have `created_at < opened_at`, which is the mark such a row would leave.
  - Evidence: `select count(*) filter (where created_at < opened_at) from grammar_acts` → 0.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium. Concern: only 16 held-out days, with the act producers still rolling out until ~10-30. Skill vs the clock is not settled (P 0.72). Mitigation: re-run the replay ~11-27, and add the live table (#3344).
- Severity: medium. Concern: overconfident in the asleep block (+0.19 ± 0.05). Mitigation: #3344. No consumer until it is explained.
- Severity: low. Concern: `forecast()` reads 28 days back, but the replay starts at `--since` (10-23). Until 11-20, a live forecast would also train on the sparse, mostly idle buckets before 10-23, which the replay leaves out. Mitigation: the gap closes on its own by 11-20; the live table (#3344) is graded on its own rows.
- Severity: low. Concern: backfilled acts (`vision.ask`, all days late) never count. That is correct for a live forecaster, but those camera asks are invisible to the chain. Mitigation: documented in the module docstring.
- Issues opened: #3343 (AC chain when the AC cycles; open window as a hidden state), #3344 (live table + asleep calibration). Closed: none.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
