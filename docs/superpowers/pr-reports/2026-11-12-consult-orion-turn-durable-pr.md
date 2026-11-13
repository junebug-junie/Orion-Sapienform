## Summary

- Each of Orion's consult turns is now an admitted durable run, `consult.orion_turn`, with run id `consult-<id>-t<n>`.
  - The GPU pool queues it when the shared lane is busy.
  - durable-runs retries a failed attempt (budget `HUB_CONSULT_ORION_TIMEOUT_SEC=900` per attempt).
  - Hub reads the finished text back from `GET /runs/{run_id}`.
- This replaces the in-process outreach compose call. That call held no GPU-pool lease and had no retry, and it killed the first live consult on 2026-11-11, on Orion's second turn (turn 3).
- durable-runs reuses its one-step harness graph. It is now a small builder shared by `harness.echo_smoke` and `consult.orion_turn`, so there is no new graph.
- `CuriosityTurnRequestV1` gains an optional `require_lookup` field, and Hub passes it through. Without it, Hub's lookup gate (3+ tool steps) would refuse a pure-writing reply to Claude.
- Removed: the consult-only `session_id` / `timeout_sec` overrides on `EndogenousOutreach.compose`, which were added in #3207.

## Outcome moved

- A busy lane now queues Orion's consult turn. It no longer ends the consult, and Claude's finished turns are kept.
- Live failure this fixes, consult `9f138a849520`:
  - The agent lane timed out at 600 s, then the chat fallback finished in about 18 min.
  - Claude replied with file:line proof.
  - Orion's next turn failed on both lanes (agent `no_final_frame`, then a chat timeout).

## Current architecture

- The consult loop called `EndogenousOutreach.compose` in-process: agent lane first, then chat fallback, with no lease and no retry.

## Architecture touched

- **Schemas:** `orion/schemas/consult_turn_run.py` (new), `durable_run.py` (Literal, brief union, validator, `require_lookup`), `registry.py`.
- **durable-runs:** `harness_smoke_graph.py` (generalized), `harness_step.py` (`retrieval_query` and `require_lookup` pass-through), `admission_runtime.py` (registration).
- **Hub:** `consult_loop.py` (submit and poll), `curiosity_investigation.py` (`require_lookup` pass-through; no curiosity reading framing on consult turns), `main.py`, `settings.py` / `.env_example` / compose, `endogenous_outreach.py` (overrides removed).

## Schema / bus / API changes

- Added: workflow `consult.orion_turn`, `ConsultOrionTurnBriefV1`, `CuriosityTurnRequestV1.require_lookup` (optional, omitted when None).
- Compatibility: additive on Literal/forbid models.
  - Deploy order: sql-writer, Hub, orion-actions and cortex-orch before durable-runs.
  - Only Hub submits this workflow, and only after Hub is deployed.
  - `require_lookup` is sent only by consult turns, so an old Hub never receives it.

## Env/config changes

- Added: `HUB_CONSULT_DURABLE_URL=http://127.0.0.1:8124`.
- Changed: `HUB_CONSULT_ORION_TIMEOUT_SEC` from 600 to 900 (now per durable attempt).
- Local `.env`: synced, and the 900 value was set by hand because the sync never overwrites a diverged value.

## Tests run

```text
services/orion-hub/tests full suite: 1 failed, 4685 passed, 147 skipped. The 1 failure is the same one main has (test_fcc_model_labels_disabled_by_default).
services/orion-durable-runs/tests: 867 passed, 132 skipped (incl. new test_consult_turn_graph.py; harness smoke unchanged)
orion/schemas + orion/consult + orion/body_atlas: 393 passed
orion-static-gates (every step, local): 0 failing
```

## Evals run

None new. The evidence is the live trace of consult `9f138a849520` (journald, 2026-11-11 20:58 to 21:35Z).

## Review findings fixed

- **Finding (medium):** a resubmit of the same turn got HTTP 409 and failed the consult.
  - Cause: durable-runs refuses a different request under an existing run id, and each submit carried a new uuid4 correlation id.
  - Fix: the correlation id is uuid5 over the run id, so a resubmit is the same request. On a 409 where the run exists, the loop adopts that run.
  - Evidence: `test_a_resubmit_of_the_same_turn_adopts_the_existing_run`.
- **Finding (medium):** a prompt over the 60,000-char brief limit made every tick raise and jammed all consults.
  - Fix: the prompt cuts the oldest turns to fit. The latest turn stays whole, and each cut is marked in the text. A brief that still fails fails the consult, not the loop.
  - Evidence: `test_orion_prompt_fits_the_brief_limit_by_cutting_the_oldest_turns`, `test_a_brief_that_cannot_be_built_fails_the_consult_not_the_loop`.
- **Finding (low):** consult turns ran with curiosity's reading framing, so a queued read would later say "a curiosity run of yours".
  - Fix: no reading context and no reading parent for `consult_orion_turn` turns.
- **Finding (low):** a crash between storing Orion's turn and inviting Claude could store the turn twice.
  - Fix: the turn number comes from the run id, and an existing turn is never stored again.
  - Evidence: `test_a_completed_run_is_stored_once_even_if_polled_twice`.
- **Finding (low):** the brief was not in the schema registry, and consults could be enabled while nothing served the turn.
  - Fix: the brief is registered. Consults are enabled only when curiosity investigation and cortex kickoff are both on.

## Restart required

The post-pull rebuild builds hub, orion-durable-runs and the schema consumers. Afterwards, retry consult `9f138a849520` (Juniper approved):

```sql
UPDATE orion_consult SET status='pending', status_reason=NULL, created_at=now(), updated_at=now() WHERE consult_id='9f138a849520';
```

## Risks / concerns

- Severity: low.
  - Concern: `consult.orion_turn` maps to no body-economy want family, the same as `harness.echo_smoke`. Orion's GPU time for consults is admitted by the pool, but no act or bid is charged.
  - Mitigation: the pool still admits it and pool contention applies. Charging it is a later body-economy call.
- Severity: low.
  - Concern: Hub reads results on its tick, every 120 s, so each Orion turn adds up to 2 minutes of latency.
  - Mitigation: none needed; it does not change any outcome.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
