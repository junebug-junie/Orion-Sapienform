## Summary

- Orion's surprise readings for the route, chat, execution, biometrics, bus-synaptic and codebase domains could only go up in the shared field. A calm tick (surprise exactly 0.0) sent no update, so the field kept showing the last spike as if it were current.
- Fix: every tick that scores a domain now saves its prediction-error receipt, including 0.0. That receipt is what the field digester and the attention baseline read. (`services/orion-substrate-runtime/app/worker.py`, six call sites, `if error > 0.0:` removed)
- Vision and perception already worked this way. The FalkorDB graph node was already fixed in an earlier patch; that patch deliberately left the receipt gated, which is how this bug survived.
- A new structural test refuses any `_prediction_error_receipt()` call inside an `if <x> > 0:` block. A new digester test proves a 0.0 receipt replaces a stale value and re-stamps its time.

## Outcome moved

- `node:substrate.route`'s field `prediction_error` channel can read a real calm 0.0 again. Live before the fix it showed 0.0003, last updated 2026-09-24T12:11:48Z, 12h+ stale, while the route tick kept running.
- The chat (0.2804, last receipt 21:09 while chat ticked at 00:24) and codebase (0.442) field values were also stuck at old spikes, and so was Candidate A's per-domain `last_value`.

## Current architecture

Each prediction-error tick wrote two things: a FalkorDB node (every tick, since 2026-07-30 / later sibling fixes) and a `ReductionReceiptV1` (only when error > 0, framed as an "audit trail"). But the receipt is the only input to:
1. `orion-field-digester`'s field node vector `prediction_error` channel (`state_deltas.py`, `prediction_signal`, mode=replace, stamps `node_vector_updated_at`), and
2. `orion-attention-runtime`'s Candidate A precision baseline (`substrate_node_prediction_error_baseline`, folded per receipt; `last_value` is used as the target's current error).

`prediction_error` is not in `NODE_DECAY_CHANNELS`, so nothing corrected a stuck value.

## Architecture touched

- `orion-substrate-runtime` worker only (producer side). No schema, bus channel, or env change. The receipt shape is identical; it is simply sent on calm ticks too.

## Files changed

- `services/orion-substrate-runtime/app/worker.py`: ungate the receipt at biometrics, execution, chat, route, bus_synaptic, codebase; add a canonical explanation above `_prediction_error_receipt`; fix stale comments.
- `services/orion-substrate-runtime/app/store.py`: docstring no longer claims the receipt is gated.
- `services/orion-substrate-runtime/README.md`: new 2026-09-25 note; mark the 2026-07-30 "receipt stays gated" passage superseded.
- `services/orion-substrate-runtime/tests/test_prediction_error_receipt_not_gated.py` (new): AST gate, a check that all 8 domains still have a receipt call site, and a behavioural `_route_tick` calm-tick test.
- `services/orion-field-digester/tests/test_prediction_signal_zero_refresh.py` (new): 0.0003 then 0.0 gives a field value of 0.0 and a re-stamped `node_vector_updated_at`.
- `services/orion-substrate-runtime/tests/test_worker_{bus_synaptic_tick,chat_tick_baseline_persistence,execution_tick_baseline_persistence,codebase_delta_consumer}.py`: flipped from "no receipt on calm tick" to "0.0 receipt on calm tick".
- `services/orion-substrate-runtime/tests/test_prediction_error_node_write_not_gated.py`: docstring and message updated.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: prediction-error receipts (`receipt:prediction_error:*`) are now also emitted with `prediction_error=0.0` on calm ticks.
- Compatibility notes: every reader handles 0.0 (reviewed: attention-runtime store, digester `state_deltas.py`, Hub routes, receipt pruner, compaction).

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=.:services/orion-substrate-runtime:services/orion-sql-writer/tests \
  python -m pytest services/orion-substrate-runtime/tests -q \
  --ignore=services/orion-substrate-runtime/tests/test_grammar_consumer_integration.py
  -> 17 failed, 354 passed. The same 17 fail on clean origin/main (identical sorted
     list; DB/env-dependent: cursor_reset_auth, cursor_tail_seed, quarantine_truth, ...).
     test_grammar_consumer_integration needs a live Postgres on localhost:5432.
PYTHONPATH=.:services/orion-field-digester python -m pytest services/orion-field-digester/tests -q
  -> 230 passed
Regression proof: test_prediction_error_receipt_not_gated.py on origin/main -> 2 failed
  (names all six gated sites); on this branch -> passes.
```

## Evals run

```text
No eval harness exists for this seam in orion-substrate-runtime. The regression is
covered by the structural and behavioural tests above; the real proof is the
post-deploy live check below.
```

## Docker/build/smoke checks

```text
Not deployed (per task: no production deploys). Post-deploy live check:

docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "
  select generated_at,
         field_json->'node_vectors'->'node:substrate.route'->>'prediction_error',
         field_json->'node_vector_updated_at'->'node:substrate.route'->>'prediction_error'
  from substrate_field_state order by generated_at desc limit 1"

Expect: the updated_at stamp advances to within a few minutes of now whenever the route
reducer processes events (it only ticks when route grammar events exist), and the
value reads 0.0 on calm ticks instead of 0.0003. Also check:
  select target_id, last_value, variance, last_receipt_created_at
  from substrate_node_prediction_error_baseline;
Expect route/chat last_receipt_created_at to move and last_value to read 0.0 on calm ticks.
```

## Review findings fixed

- Finding: `worker.py` vision tick comment still said "unlike _bus_synaptic_tick's fault-gated one".
  - Fix: reworded to say that every domain now emits every tick.
  - Evidence: commit on this branch.
- Finding: README 2026-07-30 passage still said the receipt "stays gated".
  - Fix: marked superseded, pointing to the new note.
  - Evidence: `services/orion-substrate-runtime/README.md`.
- Finding (should-fix, disclosure): the Candidate A attention behaviour change was under-described.
  - Fix: disclosed under Risks below, with a post-deploy check.
  - Evidence: live, `node:substrate.chat` is the top substrate attention target at salience 1.0 off `last_value=0.2804` from a 21:09 receipt (3.5h old).
- Finding (nits, not changed): the AST gate only catches the `> 0` shape (behavioural tests cover the rest). The codebase node's three delta kinds share one replace-mode channel, so a zero now overwrites a git spike within about 15 minutes. That already matches the FalkorDB node's behaviour.

## Restart required

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

Run from an up-to-date worktree of main after merge (the wrapper refuses the shared checkout).

## Risks / concerns

- Severity: medium (a live attention-loop behaviour change, intended)
  - Concern: Candidate A salience is `precision * |last_value|`. Until now `last_value` only ever held a domain's last spike. After deploy, calm ticks write 0.0, so chat (currently the top substrate target off a 3.5h-old 0.2804) and route drop to salience 0 when calm. Execution and chat EWMA variances rise once zeros mix in, which shifts `cross_domain_variance_floor`. Goal-provenance winners and dominance streaks will change.
  - Mitigation: this removes a stale high-water mark rather than adding a new signal. Rollback is a revert of this commit. Watch the top node target and per-domain `last_value`/`variance` for a few hours after deploy.
- Severity: low
  - Concern: slightly more receipts (a handful per hour on about 25k/hour), and the digester's `recent_perturbations` count rises a little.
  - Mitigation: that count is self-baselined by an EWMA z-score (`perturbation.py:116`), which absorbs a step change.
- Severity: low (existing, not fixed here)
  - Concern: `orion/substrate/bus_synaptic_surprise.py` and `orion/substrate/metacog_trend_signals.py` guard staleness on the whole field row's `generated_at`, not the channel's own `node_vector_updated_at`. So they could not see that route/chat were stale. This patch removes the stuck values they were reading. A per-channel guard would be the durable defence.
  - Mitigation: follow-up.
- Severity: medium (metric definition, NOT changed; needs Juniper's approval)
  - Concern: `route_prediction_error()` (`orion/substrate/prediction_error.py`) averages the mismatch across every run held in the projection (live: 814), not just the runs this batch touched. Unchanged runs match themselves and score 0, so one run flipping one of four decision fields reads 0.25/814 ≈ 0.0003, and a full decision flip caps near 0.0012. That is the exact stuck value observed. The signal is structurally diluted by projection size.
  - Recommendation: average only over runs that are new or whose `last_updated_at` advanced in this batch (and keep 0.0 when none did). Ship it as an adjacent metric or with explicit approval, since it changes the definition and the attention baseline's scale for route.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2326

🤖 Generated with [Claude Code](https://claude.com/claude-code)
