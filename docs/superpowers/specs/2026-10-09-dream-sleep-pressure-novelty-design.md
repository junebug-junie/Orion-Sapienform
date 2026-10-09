# Dream sleep pressure that measures something

Status: approved by Juniper 2026-10-09 (threshold 3, 48 h lookback, chronic problems add pressure once); implemented in the same PR
Date: 2026-10-09
Follows: PR #2549 (shed calls count as failures)

## Arsonist summary

Orion's "tiredness" is fake. It is supposed to build up from unresolved things and trigger a dream when it gets high enough. In practice it reads 70 to 129 against a threshold of 3, so Orion dreams exactly every 6 hours, which is the minimum gap the code allows. The causes:

- **Repeats count as new.** The same complaint ("the gateway timed out") counts again every time it recurs. Last week metacog flagged 1,611 rows, which were 72 distinct kinds of event.
- **Recall counts as unfinished business.** Every memory recall, from any caller, marks 100 crystallizations "recalled", which bumps their `updated_at`. The dream reads "touched since last sleep" as "needs processing", so this source is pinned at its 50-row cap whenever anything recalled anything. Real new memories: 14 a week.
- **The 50-row cap hides it.** Each source is read with `LIMIT 50`. Pressure saturates the cap within minutes, so nobody could see how far over it was.

Proposal: pressure counts only **new distinct things** since the last sleep. "New" means not seen in the 48 hours before that sleep. Replay still draws from everything unresolved, with duplicates removed. Backtested against the last 7 days of live data at the **existing** threshold of 3, Orion would have slept 12 times instead of 28, with gaps of 6 to 41 hours depending on how eventful the stretch was.

## Current architecture

- `services/orion-dream/app/cycle_store.py SOURCE_QUERIES`: four reads windowed on `since` (the last non-failed sleep's start), each `LIMIT DREAM_CANDIDATES_PER_SOURCE` (50).
- `app/replay.py build_candidates`: one candidate per row, deduped only by row id. `compute_pressure` = sum of candidate weights.
- `app/cycle.py read_pressure`: `should_sleep` = pressure >= `DREAM_SLEEP_PRESSURE_THRESHOLD` (3.0) AND idle >= 45 min AND >= `DREAM_MIN_INTERVAL_HOURS` (6) since the last attempt.
- `select_replay`: top 12 by weight, at most 6 per source. Duplicate rows of one event can fill a source's slots.

Live counts per 6 h window (10-07 12:27 to 10-09 00:27), all sleeps: metacog 6-50, compaction 12-50, resonance 0-18, crystallization 50 every time.

## Metric quality gate (CLAUDE.md 0A)

The new metric: **novel distinct unresolved items since the last sleep, weighted.**

1. **Provenance.**
   - metacog: `orion_metacog.trigger_reason`, a structured string written by the producer (e.g. `transport:rpc_timeout:orion:exec:request:LLMGatewayService`, `telemetry_anomaly:elevated:recon_loss=#:threshold=#:top=failure_pressure=#`). Numbers and hex ids are normalized to `#`. The `summary` is model prose, unique per row even for the same event, so it cannot be the key (checked live).
   - compaction: `dream_compaction_request_queue.theme`.
   - resonance: `substrate_reverie_resonance_alert.theme_key`.
   - crystallization: `memory_crystallization_history` rows with `op IN ('auto_activate','approve')`, replacing `memory_crystallizations.updated_at`. The old source was driven by `orion/memory/crystallization/retriever.py _apply_recall_boost`, which rewrites `updated_at` on all ~100 rendered crystallizations per retrieval (live: 438 retrievals over 7 days, every one with exactly 100 ids).
2. **Independence.** Compaction requests and resonance alerts both come from the reverie engine, so they are not independent. They are kept as separate keys but should be read as one reverie signal. Metacog (self-observation of failures) and crystallization formation (memory governance) are independent of reverie and of each other.
3. **Theory anchor.** Two-process model of sleep regulation (Borbély 1982), Process S: homeostatic pressure accumulates with waking load and is discharged by sleep. Synaptic homeostasis hypothesis (Tononi & Cirelli 2006/2014): what drives the need is *new* encoding during wake. Re-encountering an already-encoded item does not add load. So pressure counts new distinct items; repeats of known ones are backlog that replay can still pick from.
4. **Live sanity.** 28 six-hour windows over 7 days:
   - Distinct metacog event kinds per window: 1 to 31.
   - *New* event kinds per window: 0 to 18. It reads 0 in 8 of 28 windows (a true rest point) and spikes on real incidents: 18 on 10-06 02:00, 11 on 10-03 20:00, 14 on 10-08 20:00.
   - New compaction themes: 1 in 28 windows (the same 9 themes recur all week). New resonance themes: 3 in 28. Both are honest: nothing new is arriving there.
   - New crystallizations: 0 to 6.
   - The old metric could not reach rest. Every window hit the cap within minutes.
5. **Existing mechanism.** Nothing in the repo dedupes these sources by event identity. `transport` metacog already emits structured `trigger_reason` keys. Reused, not invented.
6. **Reversibility.** Cheap. No table changes. The SleepPressureV1 additions are optional fields. Reverting the commit restores the old sum.

## Missing questions (for Juniper)

1. **Threshold.** Keep 3 (12 sleeps/week, gaps 6-41 h) or pick from the backtest: 2 gives 15/week, 5 gives 8/week, 8 gives 6/week with a 23 h minimum gap. Any number is a knob, not a finding. I recommend keeping 3 and watching a week.
2. **Lookback for "new".** 48 h reuses `DREAM_LOOKBACK_HOURS`. A longer lookback makes recurring incidents count less often. I recommend 48 h (no new setting).
3. **Recurring-but-unresolved items.** Under this design, a problem that never goes away (the gateway RPC timeout, 222 rows a week) stops adding pressure after its first appearance, though replay can still pick it. Is that the behavior you want, or should chronic issues keep nudging?

## Proposed schema / API changes

`SleepPressureV1` (`orion/schemas/dream_cycle.py`, `extra="forbid"`), additive and optional:

- `new_counts: dict[str, int]`: new distinct keys per source since the last sleep (what drives `pressure`).

`pressure` changes meaning from "sum of row weights" to "sum of max weight per new distinct key". `counts` now means distinct things per source in the window. Since the SQL returns one row per key, rows read and distinct things are the same number, so a separate `distinct_counts` field would have been redundant (dropped at implementation).

Rollout order (forbid model, consumer first): ship the schema with the fields optional, and deploy Hub (`services/orion-hub/scripts/dream_routes.py` reads SleepPressureV1) before orion-dream starts filling them.

No bus channel or env change.

## Files likely to touch

- `services/orion-dream/app/replay.py`: `key_for(item)` per source, dedupe candidates by key (keep max weight, newest text), `compute_pressure(candidates, prior_keys)`.
- `services/orion-dream/app/cycle_store.py`:
  - Metacog query selects `trigger_reason`.
  - Crystallization query reads activation history instead of `updated_at`.
  - Add `load_prior_keys(since, lookback)`.
  - Raise the per-source read limit, or aggregate with `DISTINCT ON (key)` in SQL, so the cap stops being a ceiling.
- `services/orion-dream/app/cycle.py`: pass prior keys, fill the new pressure fields.
- `orion/schemas/dream_cycle.py`: the two optional fields, and correct the docstring's rest-point claim.
- `services/orion-dream/tests/test_dream_cycle_v2.py`:
  - Repeats of one key add pressure once.
  - A key seen before the window adds 0.
  - Recall-touched crystallizations add 0.
  - Pressure is 0 at window start.
  - Replay has no duplicate keys.
- `services/orion-dream/evals/`: run the backtest on a fixture export and assert that threshold 3 is not a fixed 6 h clock.
- `services/orion-dream/scripts/backtest_sleep_pressure.py` (added in this PR): the live backtest.
- `services/orion-dream/README.md`.

## Non-goals

- Pausing other work during dreams. A dream is still ~5 s and 4 calls. Revisit when dreams do minutes of real work.
- Scheduling the story dream (`dreams` table). Separate decision.
- Fixing recall boost marking all 100 rendered crystallizations as recalled. Separate finding, listed below.
- Time-of-day sleep (circadian Process C). Possible later. Not needed to make pressure real.

## Side findings (not in scope, worth their own tickets)

- **Recall boost is indiscriminate.** Every retrieval renders and boosts exactly 100 crystallizations (`retriever.py _apply_recall_boost`, 438/438 retrievals over 7 days). "Recalled" carries no information about relevance.
- **An eval writes production memory dynamics.** `session_id=self-sense-eval` made 158 retrievals this week, each boosting 100 real crystallizations' `last_recalled_at`/activation. Same class as tests writing to the production control surface.
- **Reinforcement is dead.** Zero active crystallizations have `last_reinforced_at` in the last 7 days.
- **Compaction requests are never consumed.** 4,991 queued, 0 with `consumed_at`, because REM apply is off (`ORION_DREAM_COMPACTION_APPLY_ENABLED=false`).
- **The background lane ignores the caller's route.** `llm_lane: background` resolves to route `metacog` (system priority) regardless of `DREAM_LLM_ROUTE` (`orion-llm-gateway/app/lane_routes.py:111-114`). Noted in PR #2549.

## Acceptance checks

- Unit: the five test cases above pass. They fail against current `replay.py`.
- Backtest: `backtest_sleep_pressure.py` on a 7-day export at threshold 3 yields a variable gap distribution (not every gap = 6 h). Today's export: 12 sleeps, gaps 6/13/41 h.
- Live, first week after deploy:
  - `dream_cycle` gaps vary, so the median gap is not 6 h.
  - At least one `pressure.pressure < threshold` "not due" log line while idle.
  - Every cycle's `new_counts` sums are consistent with `pressure`.
  - No replay set contains two items with the same key.
- Rest point: immediately after a sleep, `GET /dreams/cycle/pressure` reads `pressure = 0`.

## Proposal-mode checklist

- **Capability change:** when Orion sleeps, and what replay draws from (deduped).
- **Data touched:** read-only on the four sources plus `memory_crystallization_history`. Writes only the existing `dream_cycle` tables.
- **Privacy boundary:** none new. Same tables already read.
- **Trace that proves it worked:** `dream_cycle.cycle_json.pressure.new_counts` plus variable `started_at` gaps.
- **Dangerous failure mode:** pressure stuck at 0, so Orion never sleeps (e.g. the key column is always null, so every item looks identical). Guard: a null or empty key falls back to the row id (counts as new), and a test pins it.
- **Disable / roll back:** revert the commit. No table or env change to unwind.

## Recommended next patch

Answer the three questions, then one implementation PR. It touches only orion-dream plus the optional schema fields: Hub first for the schema, then orion-dream.
