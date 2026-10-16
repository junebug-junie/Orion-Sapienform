## Summary

- Paintings now open their `make` act once. Before, every retried or replayed `prepare` step published another `act_opened`.
- The "did Juniper answer" check now reads when her next turn **started**, not when Orion finished writing the reply to it.
- A Juniper turn that errored still counts as her answering.
- Outreach `say`/`ask` acts use the same check, so they get the same fix.
- Follow-up to #2710 (act ledger, rung 2 of the thinking architecture, spec PR 2669).

## Outcome moved

- **Duplicate opens.** Live since the #2710 deploy: `grammar_events` held 27 `act_opened` rows from `visual_steps.act` for only 7 distinct acts (13 for 3 when first counted). After this patch: one open per painting attempt.
- **Wrong zeros on `juniper_replied`.** Live, the chat-history row lands 1 to 41 minutes after Juniper sends a turn. Example: she sent a turn at 2026-10-15 23:45:52 and its row was written at 00:22:11. A say act with a horizon between those two times read 0 under the old rule. Run against live data, the new SQL returns 23:45:52 for a 23:44 to 00:14 window. The old SQL returns nothing.
- **Mirror bug, also fixed.** A turn Juniper sent *before* an act opened, but whose row was written *after*, used to count as her answering that act. Now it does not.

## Current architecture

- `services/orion-thought/app/visual_steps.py`: `painting_act_transition` returned `"open"` for every non-terminal `prepare` result. Durable runs retry `prepare` with backoff (live: `attempt_unresolved` while another painting is in flight, about 40 minutes of retries), and they also replay it. Each retry or replay published a new `act_opened`. The writer's `grammar_acts` upsert already kept only the first open (`WHERE opened_at IS NULL`), so the ledger rows were correct. `grammar_events` carried the duplicates.
- `orion/grammar/act_ledger.py` `_JUNIPER_REPLY_SQL` read `min(chat_history_log.created_at)` in (opened, horizon]. The hub writes that row only after the reply is finalized, and writes none for an errored turn.

## Architecture touched

- **orion-thought, painting act producer.** `prepare_step`'s freeze mutator sets `stage_json.act_opened_at` the first time it writes the plan. It publishes `act_opened` only when that write happened in this call. The marker lives in the same row write, under the store's claim lock, so a replayed or retried prepare never opens again. Close logic is unchanged, including #2710's rule that any result with outcome `produced` closes at 1.
- **orion-sql-writer, horizon resolver (shared module `orion/grammar/act_ledger.py`).** The resolver reads turn starts from two sources:
  1. A logged Juniper turn (the canonical `JUNIPER_TURN_PREDICATE`). Its start is its hub ingress hop's `ts` when one exists, else its `created_at`.
  2. A Juniper ingress hop on its own (new `JUNIPER_INGRESS_PREDICATE`). This covers an errored turn, or a turn whose reply is still being written.
- **Turn-start source, traced to code.** `orion/hub/turn_orchestrator.py` calls `begin_cockpit_timeline` (seq 0, stage `ingress`) at the top of the unified turn, before any step that can fail. It publishes the hop over the bus. `services/orion-sql-writer/app/cockpit_turn_sighting_persist.py` writes it to `cockpit_turn_sighting.ts`.
- **What marks a turn as Juniper's.** Only the hub browser client (`static/js/app.js` typed and spoken, `thought-process.js`) sets `surface_context`. The orchestrator copies it into the ingress raw only when it is present.
  - Live data, last 10 days: 37 ingress hops carried `surface_context`, and all 37 were Juniper's logged turns.
  - None of the 1,100+ internal-turn hops (outreach, reading, curiosity, self-sense-eval) carried it.
  - Coverage: 3,179 ingress hops since 2026-09-07, one per correlation.

## Files changed

- `services/orion-thought/app/visual_steps.py`: the open is published once, from `prepare_step`, behind the `act_opened_at` marker. The transition function now decides closes only.
- `services/orion-thought/tests/test_visual_steps.py`: three regression tests against the fake stage store. A replayed prepare opens once. A prepare retried behind another painting opens only on its own claim. A prepare that failed before its plan write opens on the replay that writes the plan.
- `services/orion-thought/tests/test_painting_act.py`: the transition table no longer expects `open`. Covers the open payload and the flag-off case.
- `orion/grammar/act_ledger.py`: rewrote `_JUNIPER_REPLY_SQL` to read turn starts. Updated the close notes.
- `orion/regulation/juniper_turns.py`: added `JUNIPER_INGRESS_PREDICATE`, with its live evidence.
- `services/orion-sql-writer/tests/test_act_ledger_integration.py`: six real-Postgres tests:
  - slow reply crossing the horizon reads 1
  - turn in flight with no logged row yet reads 1
  - errored turn reads 1
  - no turn reads 0 (also checks that an internal turn, the act's own turn, a turn sent before the act, and a turn sent after the horizon all do not count)
  - logged turn without an ingress hop reads 1
  - outreach `ask` reads from turn start too

## Schema / bus / API changes

- Added: none. `stage_json.act_opened_at` is a new key inside thought's own private jsonb, with no other reader.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - Paintings publish one `act_opened` per attempt. The open now happens at plan write, not at the first `prepare` call. A run queued behind another painting no longer opens while it waits.
  - If such a run is abandoned before it claims an attempt, its close-at-0 still arrives. The writer builds the row from the close (close-before-open) and grades it as before.
  - `juniper_replied` closes at the start time of Juniper's next turn. The close note now reads "juniper chat turn started before horizon".
- Compatibility notes: no migration. Existing `grammar_acts` rows are untouched.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
services/orion-thought: pytest tests/test_painting_act.py tests/test_visual_steps.py -> 70 passed
services/orion-thought: pytest tests -> 687 passed, 1 failed (test_settings_mind_enrichment::test_mind_enrichment_defaults_off; unrelated, local env)
sql-writer integration (throwaway postgres:16 container, ORION_SQL_WRITER_TEST_DATABASE_URI):
  pytest -m integration services/orion-sql-writer/tests/test_act_ledger_integration.py -> 15 passed
Regression proof (new tests against the pre-patch code):
  act_ledger.py at HEAD~ -> 5 of the 6 new integration tests fail
  visual_steps.py at HEAD~ -> all 3 new prepare-open tests fail
tests/test_regulation_arousal.py tests/test_act_ledger_contract.py -> passed
services/orion-dream tests/test_idle_counts_only_juniper.py -> passed
services/orion-hub tests/test_act_ledger_producers.py -> 13 passed
services/orion-sql-writer tests/test_act_ledger_wiring.py + evals/test_act_ledger_calibration_eval.py -> 7 passed
services/orion-sql-writer tests (full) -> 14 failures in unrelated files (biometrics, notify, chat_history identity, subscription catalog); none import the touched modules
```

## Evals run

```text
services/orion-sql-writer/evals/test_act_ledger_calibration_eval.py -> passed (unchanged)
Live-data replay of the new resolver SQL against conjourney (read-only):
  open act act_f65e23823db7b2a2e41b1aab (opened 07:47:38, horizon 08:17:38 UTC 2026-10-16) -> no turn start in window -> 0
    (matches its existing resolution: closed 0 at 08:17:38; Juniper's next turn started 16:55:07)
  window 2026-10-15 23:44 -> 00:14 -> new SQL 23:45:52 (turn logged 00:22:11); old SQL -> none
  cost: about 22 ms on the live table (the inline raw-jsonb version took 7.6 s; see the MATERIALIZED note in the code)
```

## Docker/build/smoke checks

```text
No image or compose change. Runtime path UNVERIFIED until deploy (not deployed by this PR).
```

## Review findings fixed

Code review ran in a subagent against `origin/main...HEAD`. It found no blockers.

- Finding: the chat-reply basis string, stamped on every new act, still said "reply time = when Orion's answer to it was stored".
  - Fix: re-measured on the new clock. The median from Orion's reply to Juniper's next turn **start** is 1785 s, P(<= 1800 s) = 0.50 (n=250). The basis string now states this. The horizon stays at 1800 s because it is still at the median.
  - Evidence: `orion/grammar/acts.py`. The hub producer test now asserts the new wording.
- Finding: the outreach horizon was measured on the old clock. The `act:say`/`act:ask` posteriors now mix outcomes graded under both definitions.
  - Fix: re-measured. The median from an outreach message to her next turn start is 36554 s, P(<= 36000 s) = 0.49 on both clocks. The horizon stands, and the comment and basis are updated (`services/orion-hub/scripts/act_emit.py`).
  - Known residue: the 3 `juniper_replied` acts closed before deploy were graded on the old clock. The live open act (`act_f65e23823db7b2a2e41b1aab`) reads 0 under both rules. Prod rows were not touched.
- Finding: three docs still said the signal resolves from `chat_history_log` (sql-writer README, `orion/schemas/grammar.py`, sql-writer `settings.py` comment).
  - Fix: all three updated.
- Finding: this PR report was untracked.
  - Fix: committed.
- Finding: the raw-jsonb test in the second source could be planned as a full scan.
  - Fix: it is now an `EXISTS` primary-key probe over the in-window ingress hops only. Live cost: 24 ms.
- Finding: the created_at fallback limit, the per-attempt vs per-run open marker, and the smoke script that sends `surface_context` were undocumented.
  - Fix: comments added in `act_ledger.py`, `visual_steps.py`, and `juniper_turns.py`.
- Finding: the flag-off test of `_publish_painting_open` passed only because of an earlier monkeypatch.
  - Fix: it now sets the flag explicitly.
- Not fixed: a partial index on `cockpit_turn_sighting (ts) WHERE stage='ingress'`.
  - Reason: the window scan reads a 17 MB heap in about 20 ms, once per open reply act per minute. That is cheap enough not to justify a migration here. Recorded as a risk below.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && scripts/safe_docker_build.sh orion-sql-writer up -d --build
cd /mnt/scripts/Orion-Sapienform && scripts/safe_docker_build.sh orion-thought up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: if the single open publish fails (bus error), it is not retried.
  - Mitigation: the close still lands. The writer builds the row from the close (close-before-open, covered by an existing test) and grades it. Only `opened_at`/`horizon_at` stay null for that act.
- Severity: low
  - Concern: a Juniper turn on a non-unified hub path has no ingress hop. It still counts only from its `created_at`, so if its reply crosses the horizon, it can read 0.
  - Mitigation: every live Juniper turn in the last 10 days went through the unified turn and had an ingress hop.
- Severity: low
  - Concern: the "Juniper" test on ingress is structural: the turn came from a hub client surface. A manual script that posts a turn with `surface_context` set (for example `scripts/trace_hub_skill_runner_e2e.py`) would count as her answering.
  - Mitigation: none needed today; noted.
- Severity: low
  - Concern: the ingress window scan has no `ts` index. Its cost grows linearly with `cockpit_turn_sighting` (about 1k hops/day).
  - Mitigation: about 20 ms today. Add a partial index on `(ts) WHERE stage='ingress' AND seq=0` if it reaches hundreds of ms.
- Severity: info
  - Concern: the world-pulse reading `look` act shows the same duplicate-open pattern live (180 `act_opened` for 6 acts from `world_pulse_read.act`). It is out of scope for this PR.
  - Mitigation: follow-up.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
