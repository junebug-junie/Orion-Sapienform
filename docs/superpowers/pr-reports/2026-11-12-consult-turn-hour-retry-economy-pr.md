## Summary

- Orion's consult turn now gets **60 minutes** per attempt instead of 15. After Hub's finalize reserve, Orion digs for about 54 minutes, up from 9.
- When the FCC motor **cuts a consult turn short**, durable-runs retries the turn. The retry is a fresh session, because the motor records its own stop as a finished session. So the cut attempt's findings (its cut-short draft) go into the retry's prompt, and Orion is told to continue from them. The last attempt keeps the honest "cut short" text, so Claude still gets what Orion found.
- Each consult turn is now a **tier-2 want in Orion's body economy** (`durable.consult_orion_turn`, act `ask`). Its bid binds: when the agent lane is busy and expensive, a turn can wait up to 30 minutes. Before this patch, consult turns never bid and their GPU time was never recorded as Orion's cost.
- **Hub chat on page load** shows every brief still waiting for Juniper in full, plus every other consult touched in the last 48 hours as one collapsed line with its status. Failures say why. A click opens the status and the turns. None of it enters Orion's context.

## Outcome moved

Failure mode: first real consult (9f138a849520), turn 3. It was cut short at 83 steps after 540 s of FCC time (`fcc_timeout`). Orion's curiosity digs take a median of 1052 s (p90 4508 s, n=181, 14 days), so a 540 s dig was cut short most of the time.

## Current architecture

Hub's consult loop submits each Orion turn as an admitted durable run (`consult.orion_turn`, single-step harness graph) with `timeout_sec=900`. Hub keeps 330 s of that for finalize, so the FCC run got 540 s. A cut-short turn comes back `ok` with the cut-short draft, so the graph checkpointed it as the result. The run was in no want family, so it never bid and opened no act. On page load, Hub chat showed only briefs still waiting for an answer.

## Architecture touched

- orion-durable-runs: single-step harness graph (cut-short retry, consult only), retry helper, act families, binding families.
- Body economy config: `config/physiology/families.yaml`.
- orion-hub: consult settings, store query, consult route, `consult.js`.

## Files changed

- `services/orion-durable-runs/app/harness_smoke_graph.py`: `retry_cut_short` (consult graph only). A landed step with a `CUT_SHORT_CODES` grounding status is a failed attempt while `retry_would_run` (a pool take-back replays first, no attempt spent). The cut text accumulates in state `cut_short_text` (max 16,000 chars, newest kept). The retry prompt gets `CUT_SHORT_CONTINUE_NOTE` plus those findings. The last attempt keeps the text with `cut_short` set.
- `services/orion-durable-runs/app/admitted_graph.py`: `retry_would_run` and `_retry_at`. They share the backoff formula with `failed_attempt`.
- `services/orion-durable-runs/app/run_acts.py`: `consult_orion_turn` family (actor `durable.consult_orion_turn`, verb `ask`, horizon 43200 s, an estimate).
- `services/orion-durable-runs/app/run_bids.py`: the family is added to `BINDING_FAMILIES`.
- `config/physiology/families.yaml`: tier 2, window 1800 s, agent lanes.
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`, `scripts/consult_loop.py`: `HUB_CONSULT_ORION_TIMEOUT_SEC` 900 → 3600.
- `services/orion-hub/scripts/consult_loop.py`, `consult_routes.py`: `recent(recent_hours=)` and `GET /api/consults?recent_hours=48`.
- `services/orion-hub/static/js/consult.js`: `renderSummary`, `statusText`, `loadRecentConsults`.
- `services/orion-hub/README.md`, the design spec (Patch 1c section): docs.
- Tests: `test_consult_turn_graph.py`, `test_run_acts.py`, `test_run_bids_binding.py`, `test_consult_loop.py`, `test_consult_js.py`.

## Schema / bus / API changes

- Added: query param `recent_hours` on `GET /api/consults` (waiting briefs sort first). Optional field `cut_short` in a consult step record (durable-run checkpoint and state-row detail). Graph state key `cut_short_text`.
- Removed: none.
- Renamed: JS `OrionConsult.loadWaitingBriefs` → `loadRecentConsults`. It is used only by `consult.js` itself.
- Behavior changed: a cut-short consult turn is retried instead of being accepted at once. Consult runs now bid in the body economy (binding) and open/close acts.
- Compatibility notes: no bus or schema contract changes. The new family is additive in families.yaml. The market refuses unknown families, so durable-runs and the market config ship together, and both read the repo config.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: `HUB_CONSULT_ORION_TIMEOUT_SEC=3600`.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: ran it. The live hub `.env` had a hand-set diverged value (900), which sync does not overwrite. I set it to 3600 by hand.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-durable-runs/tests: 888 passed, 136 skipped
services/orion-hub/tests/test_consult_loop.py + test_consult_js.py: 31 passed
orion/consult + orion/physiology/tests: 372 passed (with the hub consult tests)
every test reading families.yaml (body_atlas, curiosity, physiology, prime, analysis, dream, hub, mind, sql-writer, tests/): all passed
check_env_template_parity: PASS
```

## Evals run

```text
No eval harness for consults. The live check is the next consult turn (see Risks).
```

## Docker/build/smoke checks

```text
Not run: Juniper deploys (merge + git pull triggers the post-pull rebuild of orion-hub and orion-durable-runs).
```

## Review findings fixed

Review ran in a subagent (orion-repo-agent) on the first commit.

- Finding (high): the retry does not resume the claude session. The motor records its own stop (timeout, stall, line limit) as `completed` and a context ceiling as `ceiling` (`orion/harness/fcc_motor.py` `_record_session_end`), so `session_resume` starts fresh. The continue note told Orion "your earlier tool calls are above", which was false. The cut attempt's findings were also thrown away.
  - Fix: the cut text is carried in declared state `cut_short_text` and put into the retry's prompt. The note now says it is a new session.
  - Evidence: `test_a_cut_short_turn_is_retried...` (findings in prompt, no "above"), `test_every_cut_attempts_findings_reach_the_next_attempt`, `test_carried_findings_are_capped_keeping_the_newest`, and a pin on the motor rule.
- Finding (medium): with 20+ consults in 48 h, an old waiting brief could fall past `LIMIT 20` and lose its buttons.
  - Fix: waiting briefs sort first in SQL, and the JS asks for `limit=100` and shows rows oldest first by `created_at`.
  - Evidence: `test_recent_window_keeps_every_waiting_brief_however_old` asserts the ORDER BY.
- Finding (low): a consult shown as a summary line at page load kept that line when its brief arrived live.
  - Fix: `renderBrief` removes the summary line.
  - Evidence: `test_a_live_brief_replaces_the_consults_summary_line`.
- Finding (low): the continue note was keyed on `last_error`, which a take-back or listener wait can overwrite.
  - Fix: it is keyed on `cut_short_text`.
- Finding (low): the cut-short path skipped `replay_if_requeued`.
  - Fix: it calls `replay_if_requeued` first, as the exception path does.
- Finding (low): a 21,600 s act horizon could expire a slow but live turn and grade it refuted.
  - Fix: 43,200 s.

## Restart required

```text
No manual restart. The post-pull rebuild rebuilds orion-hub and orion-durable-runs on merge.
```

## Risks / concerns

- Severity: medium. Concern: a consult turn can hold the shared agent lane for up to 60 minutes an attempt, with 3 attempts. Mitigation: the turn now bids in the body economy, so a busy, expensive lane holds it back for up to 30 minutes, and its cost is recorded as an act.
- Severity: low. Concern: the 43200 s act horizon and the 1800 s window are estimates, not measurements (one completed consult run). Mitigation: re-check after ~10 consult turns: first grant → terminal time and the economy's hold reasons for `durable.consult_orion_turn`.
- Severity: low. Concern: the retry gets the cut attempt's findings only as its cut-short draft lists them (capped excerpts of tool results), not the full tool output. UNVERIFIED live. Check: the next cut-short consult turn's attempt-2 prompt in `cockpit_turn_sighting` should start with the continue note and the attempt-1 findings.

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
