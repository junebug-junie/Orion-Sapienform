# Curiosity outreach: every decision gets a row; Juniper's reply gets a stamp

Patch 2 of `docs/superpowers/specs/2026-09-22-curiosity-tab-redesign-design.md`
("Write side (two thin seams)"). Patch 1 (the run story page) is being built
in a sibling worktree and reads what this patch writes.

## Summary

- When Orion decides a curiosity finding is worth telling Juniper about and
  the outreach loop says "not now", that decision now leaves a row in the
  outreach decision log. Before, the pre-check, `disabled`, `no_outreach_loop`
  and an empty composition only wrote a log line, which vanishes on restart.
  Checked live: six runs since 2026-09-19 set `reach_out=true` and none had a
  row anywhere. (`curiosity_investigation.py::_maybe_reach_out`,
  `endogenous_outreach.py::record_blocked`)
- Every curiosity decision row — sent or not — now carries the run id, the
  line (`investigate` / `self_inquiry`) and the run-derived correlation id in
  `result_json`, so the run story can join without recomputing the uuid5.
- When Juniper answers an unsolicited message, her inbound turn's history row
  gets `client_meta.in_reply_to` (that message's correlation id) and
  `in_reply_to_source` (which loop spoke). Text and voice, one lookup, hard
  1.5s ceiling, never blocks the turn. (`outreach_provenance.py::
  reply_stamp_for_session`, `websocket_handler.py`)
- The unsolicited chat row now records `client_meta.source = <tag>`; `tags`
  is not a `chat_history_log` column, so nothing in that table said which
  loop produced the message.
- Fixed a silent bug found while wiring: `_meta_unsolicited` compared
  `str(True)` (`"True"`) against `"true"`. Live rows store a jsonb boolean, so
  the existing provenance-injection selection matched nothing.

## Outcome moved

- Door-A decision rows: pre-check blocks went from **never recorded** to
  recorded, with `run_id`/`line`/`correlation_id` on every row.
- The run story's last two lines (outreach decision, Juniper's reply) can be
  facts instead of "not recorded" — once the hub is redeployed and the next
  reach-out happens.
- Provenance injection (`fetch_latest_outreach_provenance`) can now actually
  select a live row.

## Current architecture

`_maybe_reach_out` (Hub curiosity loop) is the only Door A. It calls
`outreach.blocked_reason()` and, on a block, logged and returned. Otherwise it
composed via a second unified turn and handed the text to
`EndogenousOutreach.offer_message`, which runs the shared gates and records
through `_record` → `endogenous_outreach_decisions.record_decision`. The
decision table had no `run_id`; the only key was the forward uuid5. Juniper's
reply was a fresh `uuid4` turn with no back-reference. `outreach_provenance.py`
already had the "latest unsolicited row, not cleared by a later solicited
reply, within 12h" rule, but only for rows carrying a provenance capsule, and
its unsolicited check never matched the live boolean.

## Architecture touched

- Service: `orion-hub` only.
- Writers: `EndogenousOutreach._record` stays the single decision writer
  when a loop exists; `record_decision` is called directly only when there is
  no loop object at all (`disabled` with no provider, `no_outreach_loop`).
- Read path added to the inbound chat turn: one bounded `chat_history_log`
  read per turn, on a worker thread, with a timeout.
- No bus, schema, registry or channel change. `client_meta` on
  `ChatHistoryMessageV1` is a free dict.

## Files changed

- `services/orion-hub/scripts/endogenous_outreach.py`: `record_blocked()`;
  `offer_message(meta=...)` merged into every row it records, every such row
  now carries `correlation_id`; history row `client_meta.source`.
- `services/orion-hub/scripts/curiosity_investigation.py`:
  `_record_outreach_skip()`; `_maybe_reach_out(line=...)` records on every
  exit; three call sites pass `line`.
- `services/orion-hub/scripts/outreach_provenance.py`:
  `select_active_unsolicited_row(require_capsule=)`, `select_reply_target`,
  `reply_stamp_from_target`, `_fetch_session_rows` (shared, bounded, selects
  `id`), `fetch_reply_target`, `reply_stamp_for_session`; `_meta_unsolicited`
  accepts the live boolean.
- `services/orion-hub/scripts/websocket_handler.py`: one call after
  `turn_client_meta = dict(client_meta)`, gated on `bus and not no_write`.
- `services/orion-hub/README.md`: section 4.1 addendum.
- Tests: `test_curiosity_investigation.py` (5 new + fake updated),
  `test_endogenous_outreach.py` (4 new), `test_outreach_provenance.py`
  (13 new), `test_websocket_agent_claude_routing.py` (1 static check).

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: `endogenous_outreach_decisions` rows with
  `result_json.source='curiosity_outreach'` now exist for pre-check blocks
  and carry `run_id`/`line`; all Door-A rows carry the `correlation_id`
  column. `chat_history_log.client_meta` may carry `source`, `in_reply_to`,
  `in_reply_to_source`. `offer_message` gains an optional `meta` kwarg.
- Compatibility notes: purely additive JSON keys; `chat_history_rehydrate.py`'s
  unsolicited exclusion is untouched.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-hub
.venv python -m pytest tests/test_outreach_provenance.py tests/test_websocket_agent_claude_routing.py \
  tests/test_endogenous_outreach.py tests/test_curiosity_investigation.py \
  tests/test_chat_history_no_raw_publish.py tests/test_turn_orchestrator_ws_frames.py \
  tests/test_turn_orchestrator_cockpit_hops.py tests/test_hub_agent_mode_fcc_routing.py -q
484 passed

.venv python -m pytest tests -q -k "outreach or curiosity or websocket or history or provenance or turn_orchestrator"
786 passed, 1 skipped, 1 failed
  FAILED tests/test_turn_orchestrator_utterance_origin.py::
    test_execute_unified_turn_uses_mind_appraisal_text_for_stance_not_harness
  -> fails identically on main (verified in the primary checkout, same
     assertion: the harness user_message now carries the "Mind work-shape"
     preamble). Pre-existing, not touched by this branch.

python scripts/check_env_template_parity.py   -> PASS (88 services compared)
python scripts/check_schema_registry.py       -> file does not exist in this repo (Makefile note, confirmed 2026-07-12)
python scripts/check_bus_channels.py          -> file does not exist in this repo (same note)
python scripts/check_bus_reply_channels.py    -> 12 resolved, 0 uncovered
python scripts/check_inner_state_registry.py  -> OK (16 entries)
```

## Evals run

```text
No eval harness covers the outreach decision log or the reply stamp.
services/orion-hub/evals exists but has nothing for this seam; follow-up
noted below rather than claimed.
```

## Docker/build/smoke checks

```text
Not run (instructed: do NOT docker compose up the hub). Deterministic checks
only. Live-path proof is UNVERIFIED until the hub is redeployed and the next
curiosity reach-out fires -- see Risks.
```

## Review findings fixed

Code-review subagent ran against `main...HEAD` (read-only). Findings and
what changed:

- Finding (must): the reply stamp was applied to `turn_client_meta`, but the
  unified lane (`orion`/`agent` modes, `ORION_UNIFIED_TURN_ENABLED=true`, the
  lane every real Juniper turn takes) `continue`s before that dict is built
  and published its history rows with no `client_meta` at all. Live check:
  `jsonb_typeof(client_meta) = 'null'` on every solicited row in 14 days. Zero
  turns would have been stamped.
  - Fix: stamp computed once before the lane split; threaded
    `run_unified_turn(client_meta=)` → `execute_unified_turn` →
    `_publish_unified_turn_chat_history` onto both envelopes; legacy lane
    merges the same dict.
  - Evidence: `test_chat_history_no_raw_publish.py::
    test_unified_turn_history_rows_carry_the_reply_stamp` (behavioural, both
    envelopes carry a copy; absent when unstamped) and the rewritten static
    ordering check in `test_websocket_agent_claude_routing.py`.
- Finding (must): README claimed the stamp covered every inbound turn.
  - Fix: reworded to name the lane split and the orchestrator threading.
  - Evidence: README section 4.1 addendum.
- Finding (should): `in_reply_to` used the row's `id`; the design's join key
  is `correlation_id` (sql-writer sets them equal, but that is its invariant).
  - Fix: `_fetch_session_rows` selects `correlation_id`; `select_reply_target`
    prefers it, falls back to `id`.
  - Evidence: `test_reply_target_prefers_the_correlation_id_column_over_id`.
- Finding (should): `_outreach_provider()` was consulted before the
  `outreach_enabled` check (a gate-order change the README denied).
  - Fix: original order restored; the `disabled` row goes straight to the
    decision-log module.
  - Evidence: `test_disabled_and_no_loop_still_leave_a_row_via_the_decision_module`.
- Finding (should): `record_blocked` overwrote the endogenous loop's
  `_last_result`, which `GET /api/debug/endogenous-outreach` serves.
  - Fix: save/restore around `_record`; documented in the docstring + README.
  - Evidence: `test_record_blocked_goes_through_the_single_writer` asserts
    `status()["last_result"]` is untouched.
- Finding (should): vacuous `assert X if hasattr(...) else True`.
  - Fix: replaced with the real `status()` assertion above.
- Finding (should, not fixed): provenance injection and the reply stamp each
  run the same bounded session scan per turn. Left as a follow-up: they live
  in different call frames (`execute_unified_turn` vs the handler) and the
  table is 472 rows.
- Noted (nit, confirmed fine): `wait_for(to_thread(...))` leaves the thread
  to finish; the engine is disposed in the thread's `finally`. `**extra`
  merge order is safe (core keys last). Inbound path never raises.
- Noted: the `_meta_unsolicited` fix means provenance injection
  (`turn_orchestrator._situation_with_outreach_provenance`) will start
  actually injecting after an endogenous message — intended, never live
  before. Called out in the README.

## Restart required

```bash
# From a worktree, not the shared checkout:
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: should
  - Concern: no index on `chat_history_log(session_id, created_at)`. The
    stamp read is one more seq scan per inbound turn (bounded, 12h window,
    LIMIT 500). Live table is 472 rows; the provenance injection already does
    the same scan per unified turn.
  - Mitigation: table is small; add an index migration if it grows.
- Severity: should
  - Concern: the reply stamp is a heuristic ("next message in that session
    within 12h with nothing solicited in between"). It will label a message
    that changes the subject as a reply.
  - Mitigation: the design says the UI labels it as such; nothing downstream
    treats it as ground truth.
- Severity: nit
  - Concern: on a timeout the worker thread finishes its query in the
    background; the engine it built is disposed in `finally`.
  - Mitigation: bounded query; the loop just proceeds without the stamp.
- Severity: info
  - Concern: `record_blocked` sets `EndogenousOutreach._last_result`, so the
    outreach status endpoint can show a curiosity pre-check block as the last
    decision. `offer_message`'s blocked paths already did this.
- UNVERIFIED (needs deploy): a live blocked run producing a row with
  `result_json.run_id`; a live Juniper reply carrying `in_reply_to`.

## Follow-ups

- Eval harness for the outreach decision log (none exists).
- The "6 wanted, 0 sent" curiosity outreach budget question (design doc,
  Missing question 2) — separate proposal.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2290
