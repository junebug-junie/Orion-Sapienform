# Choice as expected free energy, Step 2a (built) and 2b (finding + proposal, not built)

Spec: `docs/superpowers/specs/2026-11-13-choice-as-expected-free-energy-design.md`, "Step 2: the attention schema
as a controller". Juniper said go on 2026-11-25. Tracks #3432.

## Summary

- Prime's thought frame now has one new section, YOUR ATTENTION SCHEMA. It says what Orion's attention schema
  finds most salient, whether that came top-down (a goal changed the winner) or bottom-up (salience alone), and
  the schema's confidence with what that number measures. No confidence recorded: the line says "unknown".
- The source is the newest `substrate_attention_self_model` row, read once per thought, read-only. A row older
  than 10 minutes, no row, a failed read or an empty DSN shows as an "absent" line that names why.
- Flag `PRIME_ATTENTION_SCHEMA_LINE_ENABLED` ships ON (code default, compose, `.env_example`, local `.env`).
  `false` is the rollback: no read, no section.
- **2b is not built, on purpose.** `predicted_shift` has been `None` on every row since
  2026-11-24 07:43:43Z, because `ACTIVE_INFERENCE_DOMAINS` is empty. A bet on it could never open. The metric gate
  below also shows the old reversion rule has no skill on the live forecast-surprise nodes. This report proposes a
  different bet (the attention winner holds) that passed the gate on held-out data. It waits for Juniper's yes.

## Outcome moved

Before this patch Prime never saw the attention schema. Broadcast, metacog and liveness read it; Prime did not. Now
every Prime thought sees one honest line about Orion's own attention. That is S2.1. S2.2 is replaced by a dated
finding and a proposal (see 2b). S2.3 holds for the frame line. No bets exist yet, so the flag has none to stop.

## Current architecture

What the schema computes today. I checked the code and the live rows on 2026-11-25:

- `reduce_attention_self_model` (`orion/substrate/attention_self_model.py`) runs in
  `orion-substrate-runtime`'s attention broadcast tick and writes `substrate_attention_self_model`. That was 307
  rows in the last day, a median of 224 s apart, with 7-day retention.
- `ACTIVE_INFERENCE_DOMAINS` is `frozenset()` (execution retired #3576 / #3809, biometrics #3575 / #3732). So:
  - `prediction_error_confidence` is null on every row. Its basis says "unknown: no Active-Inference
    prediction-error domain is live". Last non-null was 2026-11-24.
  - `predicted_shift` is null on every row since 2026-11-24 07:43:43Z (0 of 135 rows today; 101 of 285 rows on
    11-24, all before the retire).
- The only live confidence is the branch-gated `confidence`, which is
  `broadcast.coalition_stability_score`. That is a three-step dwell score (`attention_broadcast.py:863-877`):
  0.3 = nothing won or the winner is new, 0.6 = it has held 1-3 ticks, 0.9 = it has held longer. It is not a
  forecast. Over the last 7 days: 0.3 on 2,040 rows, 0.6 on 606, 0.9 on 186.
- `attention_reason` over the last 7 days: `bottom_up_salience` 2,771, `top_down_override` 62.
- The live prediction-error nodes are `node:prediction.room`, `body`, `room_joint`, `body_joint` and `chat_turn`
  (`substrate_node_prediction_error_history`). The perception, street, cabinet and strain nodes stopped writing
  on 11-24 / 11-25, which matches the alarm (#3829) and camera (#3874) moves.

## Architecture touched

- `orion/prime/attention_line.py` (new, pure): turns the row into the line.
- `orion/prime/thought.py`: adds the `FrameInputs.attention_schema` field and the section in `build_frame`.
- `services/orion-mind/app/prime_loop.py`: adds `_attention_schema_row` (psycopg2, read-only session, 3 s connect
  timeout, 2 s statement timeout) and `_read_attention_schema`, called in `_ask` (once per thought, in a thread).
- orion-mind settings, compose, `.env_example` and README.
- `orion/prime/state.py` and the other files out of scope were not touched.

## Files changed

- `orion/prime/attention_line.py`: the pure line builder and its staleness rule.
- `orion/prime/thought.py`: the frame field and the section.
- `services/orion-mind/app/prime_loop.py`: the read and the call site.
- `services/orion-mind/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: the flag and the DSN.
- `orion/prime/tests/test_prime_attention_line.py`: tests for the line (fresh, top-down, unknown, stale, absent).
- `services/orion-mind/tests/test_prime_attention_schema_loop.py`: tests for the loop (once per thought, stale,
  failed read, empty DSN, flag off, default on).
- `services/orion-mind/tests/prime_replay_scenarios.py`, `fixtures/prime_replay_golden.json`,
  `test_prime_replay.py`: one new scenario, recorded last; a re-record note in the docstring.
- This report.

## Schema / bus / API changes

- Added: none. The patch reads an existing table and adds no bus channel or schema.
- Removed: none.
- Renamed: none.
- Behavior changed: Prime's user prompt gains one section when the flag is on.
- Compatibility notes: an older orion-mind ignores the new env keys.

## Env/config changes

- Added keys: `PRIME_ATTENTION_SCHEMA_LINE_ENABLED=true`,
  `PRIME_ATTENTION_SCHEMA_PG_DSN=postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney` (the same
  read-only DSN the self-monitor and replica already use).
- Removed keys: none. Renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced: yes. Both keys were appended to `/mnt/scripts/Orion-Sapienform/services/orion-mind/.env`
  (the worktree has no `.env`, so the sync script had nothing to sync there).
- Skipped keys that need operator action: none.

## The frame line (S2.1)

The live row of 2026-11-25 07:27Z, rendered through the real read (`_attention_schema_row` against live Postgres,
read-only, from the host):

```text
YOUR ATTENTION SCHEMA (your own model of what you attend to and why; a reading, not an instruction):
read 2 min ago: most salient: "Orion is reading (1 accepted link in the last 15 min)"; bottom-up: salience alone chose it; no goal changed the winner; confidence 0.60 (how long the winner has held: 0.3 new or nothing won, 0.6 held 1-3 ticks, 0.9 longer)
```

A no-winner row (57% of live rows over 11-23..25; live render at 07:38Z). The reducer still tags these
`bottom_up_salience`, and their 0.3 is the broadcast's constant for an empty coalition, so no number is shown:

```text
read 1 min ago: nothing won attention this tick; no loop won, so nothing was chosen; confidence n/a (nothing won)
```

A stale row:

```text
YOUR ATTENTION SCHEMA (your own model of what you attend to and why; a reading, not an instruction):
(absent: the latest schema reading is 12 min old; older than 10 min reads as absent)
```

The line does not read `prediction_error_confidence`. That field is dead (null on every row), and the
dead-metric commit gate (`check_metric_dead_wiring`) refused the first version, which used it as a fallback.
Kill means kill: the fallback is gone, not overridden.

## 2b: predicted_shift as a graded bet (finding, then proposal)

### Key finding

`predicted_shift` is always `None` today. The argmax in `reduce_attention_self_model` skips every domain not in
`ACTIVE_INFERENCE_DOMAINS`, and that set is empty. A `predict` act on it would never open. So it was not built.

### Metric quality gate, candidate A: the reversion rule over the live forecast-surprise nodes

1. **Provenance.** The `substrate_node_prediction_error_history` rows for `node:prediction.*`, written by
   orion-prediction's surprise nodes. The rule is `compute_prediction_error_trend` (window 2): it predicts the next
   value moves back toward the previous one.
2. **Independence.** Each node is one forecast's own surprise. The rule would add no new sensor, only a guess
   about the next value.
3. **Theory anchor: it fails.** The 2026-07-23 finding scored reversion at 62% "against 50% chance". That is the
   wrong null. For any exchangeable (iid) series, "the next value is on the same side of the current one as the
   previous one" is true 2/3 of the time, by symmetry alone. So 62% was below the noise baseline.
4. **Live data (last 7 days, ties dropped):**

   | node | rows | exact zeros | reversion hit | shuffled-series null |
   | --- | --- | --- | --- | --- |
   | `prediction.room` | 1,828 | 1,051 | 433/809 = 0.535 | 0.755 |
   | `prediction.body` | 1,581 | 945 | 346/664 = 0.521 | 0.763 |
   | `prediction.room_joint` | 256 | 151 | 43/100 = 0.430 | 0.760 |
   | `prediction.body_joint` | 259 | 142 | 44/105 = 0.419 | 0.750 |
   | `prediction.chat_turn` | 80 | 52 | 19/29 = 0.655 | 0.780 |

   On every node the rule does worse than the same values shuffled. These series carry momentum, not reversion.
   The values do reach a true rest at 0 (more than half the rows). That was not tested here as a decay artifact:
   UNVERIFIED.

**Verdict: do not build a reversion bet on these nodes.** It would grade as wrong about half the time, and the
reason is the rule, not the data.

### Metric quality gate, candidate B: "the attention winner holds" (proposed)

The bet: when a schema row has a winner, predict the probability P that the same open loop still wins on the next
row. P is the schema's own confidence tier, calibrated on past rows. It closes on the next row (horizon = the next
schema row, at most 15 min). Outcome 1 = it held, 0 = it changed or nothing won.

1. **Provenance.** `substrate_attention_self_model`: `broadcast_selected_open_loop_id` and `confidence`
   (`coalition_stability_score`, `attention_broadcast.py:863-877`). Both are written by the same tick.
2. **Independence.** Not independent of the dwell score. It IS that score, graded. That is the point: it checks
   whether the schema's confidence means what it says about attention's next state. It adds no new input to any
   model.
3. **Theory anchor.** Attention schema theory (Graziano): the schema is a predictive model of attention, used to
   control it. A schema whose confidence predicts attention's next state is a working model. One that does not is a
   label.
4. **Live data (7 days, 1,172 row pairs with a winner; chronological 70/30 split):**
   - P(same winner next row) by tier: 0.3 = 127/379 (0.335), 0.6 = 339/607 (0.558), 0.9 = 107/186 (0.575). Base
     rate 0.489. The tiers are ordered, not flat or saturated.
   - Held-out test (n=352): Brier score 0.2537 at the base rate, 0.2422 for the tier-calibrated bet. Improvement
     0.0115, 95% bootstrap CI [0.0014, 0.0219]. Log loss 0.7006 vs 0.6776. Hit rate when calling P > 0.5: 221/352 =
     62.8%. "Always holds" scores 54.0%.
   - The skill is real but small, and most of it comes from the 0.3 tier. The 0.6 and 0.9 tiers barely differ.
5. **Existing mechanism.** Nothing grades the schema today. Prime's concern bets (`orion/prime/acts.py`, verb
   `predict`, actor `prime`) are the act pattern to copy. The self-monitor grades any actor/verb pair generically
   (`act_faculty`), so `act.attention_self_model.predict` needs no new grader.
6. **Reversibility.** The bet adds rows to `grammar_acts` only. No schema or manifest change, no training default.
   A flag stops new bets, and old rows stay as history.

**Proposal (not built):** in `orion-substrate-runtime`'s self-model tick, open one `predict` act per row that has a
winner. Use actor `attention_self_model`, object `attention_winner:<open_loop_id>`, `predicted` = the tier's
calibrated P, and horizon = 900 s. The next row closes it with outcome 1 or 0. That is about 170 acts a day, so
open at most one at a time. Put it behind `SUBSTRATE_ATTENTION_SCHEMA_BET_ENABLED` and make
`PRIME_ATTENTION_SCHEMA_LINE_ENABLED`'s rollback note name it. After 7 days, report the hit rate and Brier with n.

### S2.2 status

No `predicted_shift` acts exist. They cannot open, so the 7-day hit rate is n = 0. The successor bet needs Juniper's
yes. See the ticket under Risks.

## Proposal mode (approved cognition-loop change, stated anyway)

- **Capability:** Prime sees its own attention schema (what is salient, why, how steady) while it thinks. Prime
  does not regulate from it (that is 2c, after 2b shows skill).
- **Data touched:** one read-only `SELECT` of the newest `substrate_attention_self_model` row per thought. Nothing is
  written. The line goes into Prime's prompt and so into its existing cockpit sighting.
- **Privacy boundary:** none new. The row holds open-loop descriptions such as "Juniper is talking", which Prime
  already sees through the now and its event wants.
- **Trace that proves it worked:** the YOUR ATTENTION SCHEMA section in Prime's `user_prompt` (cockpit turn
  sighting / the `prime_thought` model call), and the `prime_attention_schema_unread` log line when a read fails.
- **Dangerous failure mode:** Prime treats a three-step dwell score as a forecast confidence and over-weights it.
  Mitigation: the line says what the number is ("how long the winner has held") and marks itself "a reading, not an
  instruction". A stale row presented as current would also mislead. The 10-minute rule and its tests stop that.
- **Disable / roll back:** set `PRIME_ATTENTION_SCHEMA_LINE_ENABLED=false` in `services/orion-mind/.env` and
  recreate orion-mind. No state migrates. No bets exist to stop.

## Tests run

All runs use `/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest -p no:cacheprovider` with
`PYTHONDONTWRITEBYTECODE=1`.

```text
orion/prime                         branch: 395 passed   main: 384 passed  (+11 new: test_prime_attention_line.py)
services/orion-mind/tests           branch: 461 passed   (+6 new: test_prime_attention_schema_loop.py)
orion/substrate/tests/test_attention_self_model.py   passed (unchanged module)
.github/workflows/orion-static-gates.yml, every run step, local: 42/42 OK
scripts/check_integration_seam.py --require-base:
  OK (16 organs; 49 families, 19 workflows, 1 intention adapters, 32 source kinds covered)
Pre-commit check_metric_dead_wiring: refused the first commit (prediction_error_confidence is never_produced);
  fixed by removing that read, not by the override.
```

## Evals run

```text
Prime replay golden (services/orion-mind/tests/test_prime_replay.py): one scenario added, recorded last
(attention_schema_fresh_then_stale). 0 parts of any older scenario changed (checked by script).
2b metric gate (offline, read-only, 7 days of live rows): see the 2b section. Scripts are not committed;
they were one-off SELECTs plus a 70/30 chronological split.
```

There is no separate eval harness for the frame line. The replay golden is the eval. Whether the line changes
Prime's choices is a 2c question: step 2c gates on 2b showing skill first.

## Docker/build/smoke checks

```text
Live read smoke (host, read-only, real _attention_schema_row + attention_schema_line on live Postgres):
  2026-11-25 07:27Z row -> the bottom-up line shown above
  2026-11-25 07:38Z row -> the no-winner line shown above
No image build: the change is Python only, psycopg2-binary is already in services/orion-mind/requirements.txt.
In-container Prime thought with the line: UNVERIFIED until deploy (ticket below).
```

## Review findings fixed

The code review ran in a separate subagent against `origin/main...HEAD`.

- Finding: a no-winner row (57% of live rows) rendered "bottom-up: salience alone chose it ... confidence 0.30".
  That claims a choice happened when none did, and it shows the empty-coalition constant as a reading.
  - Fix: with no winning loop, the line says "no loop won, so nothing was chosen; confidence n/a (nothing won)".
  - Evidence: `test_no_winner_and_field_only_rows` asserts the exact text with and without field targets. The live
    07:38Z render is above.
- Finding (nit): a row dated in the future read as "read 0 min ago".
  - Fix: a row more than 60 s in the future reads absent ("dated in the future").
  - Evidence: `test_a_row_dated_in_the_future_reads_absent`.
- Not changed (nit): the reviewer suggested renaming "confidence" to "hold-stability". The spec asks for the
  schema's confidence by name, and the parenthetical says what the number measures. Left as is.
- Not changed (nit, harmless): `_attention_schema_on` falls back to `False` when the attribute is missing. Real
  `Settings` always has it (default True, tested). The fallback keeps older replay scenarios byte-identical.

## Restart required

Deploy is not done here. After merge, from the clean primary `main` checkout:

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
```

Only orion-mind changes. No other service needs a restart, and there is no order to follow.

## Risks / concerns

- Severity: low. Concern: the frame line is UNVERIFIED live in a real Prime thought until orion-mind is
  redeployed. Mitigation: verify-live ticket (#3937), checked against the cockpit sighting's prompt.
- Severity: medium. Concern: 2b's acceptance (S2.2) is not met: no bet exists. Mitigation: the finding and the
  gated proposal above. The design ticket (#3936) waits on Juniper.
- Severity: low. Concern: the 2026-07-23 reversion finding (62% vs 50%) used the wrong null. Other readers of that
  doc may still trust it. Mitigation: recorded on #3802.

Issues opened: #3936 (design, needs-juniper: the 2b successor bet), #3937 (verify-live: the frame line after
deploy). Commented: #3802 (the reversion rule fails the gate on the live nodes). Progress on #3432 (not closed:
2b is open).

## PR link

<filled>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
