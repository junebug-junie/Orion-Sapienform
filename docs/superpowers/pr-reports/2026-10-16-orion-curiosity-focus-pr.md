## Summary

- Orion can now queue one question for its next curiosity run from a chat with Juniper, with the
  new `queue_curiosity_focus` tool. Before this, Orion had no way to do it and burned 197 steps
  looking for one.
- One slot. A new focus replaces the pending one and the reply names what it replaced. Orion's own
  choice cannot replace a focus Juniper asked for; Juniper's ask can replace anything. A pending
  focus expires after 48 hours.
- The next regular investigate run takes it only after every normal gate (cooldown, daily cap,
  energy, queue backlog), shows it first, and uses it as the recall query. The run is not forced:
  it writes `focus_disposition` ("explored" or "set_aside: <why>") on its own `:TurnOutcome`, and
  Hub records it. No disposition is recorded as unknown, never as explored.
- Orion's `curiosity` tool gains `kind=focus`: queued -> taken by run X -> disposition, or
  expired / replaced / cleared / failed. Run items that took a focus carry `extra.focus`.
- The claim gate now also marks any `focus_id` in a reply that no `queue_curiosity_focus` result
  returned, so Orion cannot tell Juniper a focus was queued when it was not.
- The Curiosity tab shows the pending focus with a Clear button. No input box for Juniper (her
  decision 3).

## Outcome moved

Failure mode removed: "Juniper asks Orion to choose and queue a curiosity topic; Orion cannot, and
the turn dies at the context ceiling hunting for a way." Orion now has one tool for it, and can
later say truthfully whether the run happened and what it did with the focus.

Live outcome: `UNVERIFIED` until deploy and acceptance check 8 (Juniper asks Orion to pick and
queue; evidence = the three log lines below with one chat correlation id, plus the Hub tile).

## Current architecture

- Orion's only curiosity tool (`curiosity`) was read-only. The one introspect write tool was
  `record_correction` (Orion's Day corrections), with an offer-once lifecycle in
  `orion/orion_day/corrections.py`.
- The investigate kickoff picks no subject; Orion chooses inside the run. The only "next topic"
  memory was the continuation note Orion writes on its own `:TurnOutcome`.
- Paths that exist but are wrong for this: `/api/urgent` (skips gates, emails), `/api/run-now`
  (operator-only, no topic), the self-question pool (self questions only).

## Architecture touched

- New Postgres table `curiosity_focus_request` (Hub is the only writer).
- New introspect bus RPC `orion:introspect:queue_curiosity_focus:request` (governor -> Hub).
- Hub: new responder; the investigate loop takes / releases / stamps the focus; the curiosity read
  tool and the Curiosity tab read it.
- Harness: claim gate extended to `focus_id`.
- Orion's graph (`orion_worldview`) is never written by Hub (PR #2199 boundary holds): the
  disposition is read back with a read-only query.

## Files changed

- `services/orion-sql-db/manual_migration_curiosity_focus_request_v1.sql`: new additive table +
  partial unique index (the one slot). `-- ORION-MIGRATION-REQUIRED-BY: orion-hub`; passes the
  post-pull auto-apply allowlist (`auto_apply_refusals` == []).
- `orion/curiosity/focus.py`: the SQL and lifecycle (queue with advisory lock, take, release, stamp,
  fail, clear, reads), disposition parsing and read-back Cypher, kickoff section, plain-words story.
- `orion/curiosity/kickoff_prompt.py`: focus section shown first (writable graph only);
  `focus_disposition` in the `:TurnOutcome` example when a focus was shown.
- `orion/schemas/introspect.py`, `orion/schemas/registry.py`, `orion/schema_skew_discovery.py`:
  `QueueCuriosityFocusArguments`, the operation, `CuriosityArguments.kind="focus"`.
- `orion/introspect/{transport,tools,brief}.py`: the tool, its description, the brief line
  ("say queued only if the tool returned a focus_id").
- `orion/harness/{claim_gate,cut_short,runner}.py`, `services/orion-harness-governor/app/bus_listener.py`:
  `focus_id` claim gate on every reply path.
- `services/orion-hub/scripts/curiosity_focus_listener.py`: the responder (chat only, kill switch,
  Juniper rule, next-run words from the tab's own schedule read).
- `services/orion-hub/scripts/curiosity_investigation.py`: `_take_focus` (only in `_investigate`,
  after every gate), `_stamp_focus` (both turn-end paths), `_settle_focus_after_run_end` (failed
  durable run: release if never started, else failed), release on cancel-before-seen, recall query.
- `services/orion-hub/scripts/curiosity_introspect_listener.py`: `kind=focus`, `extra.focus` on runs.
- `services/orion-hub/scripts/curiosity_routes.py`, `templates/curiosity_atlas.html`: pending focus
  tile, Clear (`DELETE /curiosity/api/focus/{id}`), run chip mark.
- `services/orion-hub/scripts/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`:
  wiring and `HUB_CURIOSITY_FOCUS_ENABLED`.
- `orion/bus/channels.yaml`, `config/metrics/metric_definitions.lock.json`: the new channel.
- READMEs: governor tool table row, Hub responder section.
- Tests/evals: see below. `.github/workflows/orion-memory-episode-tests.yml` runs the Postgres
  lifecycle (must not skip); `orion-reading-tests.yml` runs the Hub/harness tests and the eval.
- `docs/superpowers/specs/2026-10-16-orion-queue-curiosity-focus-design.md`: Juniper's decisions
  and the measured cadence recorded.

## Schema / bus / API changes

- Added: table `curiosity_focus_request`; channel `orion:introspect:queue_curiosity_focus:request`
  (schema `IntrospectRequestV1`, governor -> orion-hub, single consumer); args model
  `QueueCuriosityFocusArguments`; operation `queue_curiosity_focus`; `curiosity` `kind="focus"`;
  `HarnessMotorResult.focus_results`; route `DELETE /curiosity/api/focus/{focus_id}`; schedule
  payload `pending_focus`; `/api/runs` run field `focus`; `:TurnOutcome.focus_disposition`
  (written by Orion in its own graph, read by Hub).
- Removed / renamed: none.
- Behavior changed: the investigate kickoff may now include a focus section; its recall query is
  the focus question when one was taken.
- Compatibility: additive. An old governor never sends the new operation; an old Hub has no
  subscriber, so the tool reports "answer unknown" (no reply), never "queued". A Hub without the
  migration logs `curiosity_focus_table_missing` and answers "answer unknown"; the kickoff and run
  reads skip the focus silently.

## Env/config changes

- Added keys: `HUB_CURIOSITY_FOCUS_ENABLED=true` (orion-hub; settings default True; compose
  fallback `true`).
- Removed / renamed: none.
- `.env_example` updated: yes (`services/orion-hub/.env_example`).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes
  (`orion-hub: +HUB_CURIOSITY_FOCUS_ENABLED='true'`, written to the primary checkout's
  `services/orion-hub/.env`, which is the deploy checkout; verified with grep, git-ignored).
- skipped keys requiring operator action: none.

## Expiry: measured cadence

Live Postgres, `durable_admission_runs`, regular investigate runs (non-urgent), last 30 days:
206 runs, every day had runs (6-8). The line runs right after local midnight (America/Denver)
until the daily cap (7) is spent, 30 min apart: admissions sit at 06:10-09:20 UTC every day.
Gaps between runs: p50 0.5 h, p90 20.8 h, p99 21.0 h, max 21.1 h; none over 24 h. A focus queued
at a random moment waits 8.2 h on average and at most ~21 h. 48 h never expired a focus in that
window and covers one whole missed day. Kept at 48 h.

## Tests run

```text
# Postgres lifecycle on a throwaway postgres:16 (and the in-memory fake, same scenarios)
ORION_MEMORY_EPISODE_TEST_DATABASE_URL=... pytest orion/curiosity/tests/test_focus_lifecycle.py
  -> 54 passed (20 scenarios x {fake, postgres} + helpers)
pytest services/orion-hub/tests/test_curiosity_focus.py            -> 37 passed
pytest orion/introspect/tests orion/harness/tests/test_claim_gate.py -> all passed
pytest tests/test_curiosity_atlas_template.py tests/test_curiosity_atlas.py -> all passed (node render)
pytest services/orion-hub/tests/test_curiosity_*.py tests/test_orion_day_*.py + evals -> 600 passed, 3 skipped
pytest orion/harness orion/curiosity orion/introspect tests/test_curiosity_*.py (with Postgres) -> 1542 passed, 3 skipped
pytest services/orion-harness-governor/tests -> 85 passed
CI-shaped combined process (orion-reading-tests list incl. new files) -> 786 passed, 20 skipped
python scripts/check_definition_drift.py --gate      -> PASS (re-locked: +1 channel)
python scripts/check_metric_lineage.py --gate        -> PASS
python scripts/check_migration_required_by.py --base origin/main -> OK
python scripts/check_settings_defaults.py orion-hub --example-drift -> OK
python scripts/check_env_template_parity.py          -> PASS
auto_apply_refusals(migration, fresh DB)             -> []
git diff --check                                      -> clean
```

Mutation check (27 mutations, each must fail the focus tests; 16 before review, 11 for the review
fixes): Juniper protection off; take
ignores expiry (SQL); stamp may downgrade `ran` (SQL); release after stamp (SQL); replace leaves
the old row pending (SQL); missing disposition reads explored; cancel-before-seen not released;
take on an unreadable graph; loop ignores kill switch; recall ignores the focus; responder accepts
curiosity turns; responder ignores kill switch; governor skips the focus gate; claim gate passes
everything; failed durable run settles nothing; kickoff hides the focus; graph-unread retry erases
the disposition; no stale sweep; completed run not settled; raised turn not stamped; mid-turn
cancel leaves taken; curiosity reads do not verify; template pre-fills explored; dead retry
reported queued; clear hides expiry; run focus extra bloated; focus why unclipped.
Result: ALL KILLED.

## Evals run

```text
python services/orion-hub/evals/run_curiosity_focus_eval.py -> PASS
  queue (Juniper rule refused with nothing written; Juniper replace names the replaced focus;
  reply says the line is at today's cap) -> take by run A -> kickoff shows it first and asks for
  focus_disposition -> run B takes nothing -> stamp reads the run's set_aside back ->
  curiosity(kind=focus): "taken by run a1a1a1a1a1a1 ... disposition: set aside: ..." ->
  a run with no disposition reads "unknown (the run recorded none)".
python services/orion-hub/evals/run_orion_day_correction_eval.py -> PASS (unchanged neighbor)
```

## Docker/build/smoke checks

```text
Not run: no image build or live smoke from this worktree. The change is Python + SQL + template in
images that copy the repo; deploy is from the primary main checkout after merge (below).
Live path: UNVERIFIED.
```

## Review findings fixed

Code review ran in a subagent (orion-repo-agent, read-only). All MUST and SHOULD findings fixed
in `ea23f87a5`; each fix is covered by a test that a mutation proved can fail.

- Finding (MUST): `curiosity` results with a focus went over the 12000-char MCP budget (run list
  18,878 chars; focus list up to 29,964 with escape-heavy text).
  - Fix: run items carry only `{focus_id, status, disposition kind}`; focus items clip question /
    why / story on serialized length and say nothing twice.
  - Evidence: worst-case tests at the MCP server's own serialization (`ensure_ascii=False`,
    pinned to `orion/introspect/mcp_server.py`): focus list 10,633, run list with focus 11,878.
- Finding (MUST): the claim gate would mark a real `focus_id` that Orion read back with
  `curiosity(kind=focus)` as unverified and append "Nothing was queued" (false negative).
  - Fix: `curiosity` results also verify a focus id; tools are matched by name only (not by a
    string in some other tool's arguments); the notice now says "unknown", never "nothing queued".
  - Evidence: `test_a_focus_id_read_back_with_the_curiosity_tool_is_verified`,
    `test_a_tool_whose_arguments_merely_mention_the_name_does_not_count`.
- Finding (SHOULD): the `:TurnOutcome` template pre-filled `focus_disposition: "explored"`, so a
  run that copied it would read explored without deciding.
  - Fix: a placeholder (`"<explored | set_aside: why not>"`) that parses as unknown if copied.
  - Evidence: `test_the_kickoff_template_value_is_a_placeholder_that_reads_unknown`.
- Finding (SHOULD): a focus could read "taken" forever (Hub cancelled mid-turn, `_generate`
  raised, admission unconfirmed, an error before dispatch).
  - Fix: mid-turn cancel and a raised turn stamp `failed`; `take_focus` sweeps taken rows with no
    end after 48 h to `failed` with `turn_ok` NULL ("end never recorded"); a late stamp still wins.
  - Evidence: lifecycle tests on Postgres + fake; `test_a_raised_turn_fails_the_focus...`.
- Finding (SHOULD): a completed durable run whose stamp write failed was never settled.
  - Fix: on `completed`, a still-taken focus is stamped (disposition re-read).
  - Evidence: `test_a_completed_durable_run_stamps_a_focus_its_turn_end_missed`.
- Finding (SHOULD): a retry with an unreadable graph erased a known disposition.
  - Fix: `STAMP_SQL` keeps the stored disposition when `graph_read` is false.
  - Evidence: `test_an_unreadable_graph_on_a_retry_keeps_the_known_disposition` (Postgres + fake).
- Finding (SHOULD): turn-end stamping was only pinned by counting source lines.
  - Fix: behavioral tests through `_turn_result_for`: empty attempt -> failed, retry -> ran, cache
    hit does not re-stamp, another line's tag never stamps, a raised turn -> failed.
- Finding (SHOULD): the run chip said "unknown" for failed / unfinished runs.
  - Fix: chip wording follows the focus status. Evidence: node render test.
- NITs fixed: repeating a replaced/expired/cleared focus in the same chat turn is refused (was
  ok=true with nothing queued); Clear never hides an expiry.
- NITs left (see Risks): model-supplied `origin`; `<RUN_ID>` token inside a focus question gets
  substituted; `ORDER BY written_at` mixes types if a run writes two dispositions.

## Restart required

Deploy order (from the clean primary `main` checkout after merge):

1. Migration. The post-pull rebuild applies it automatically (additive, REQUIRED-BY orion-hub).
   By hand if needed:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_curiosity_focus_request_v1.sql
```

2. orion-hub (responder, loop, read tool, tab), then 3. orion-harness-governor (tool list, brief,
   claim gate). Hub first, so the tool never exists without its responder:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-hub up -d --build && scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

4. Acceptance check 8: in chat, Juniper asks Orion to pick something and queue it. Evidence, one
   chat correlation id: `curiosity_focus_queued focus_id=... chat_corr=...` (Hub log) ->
   `curiosity_focus_offered run=... focus_id=...` (next investigate kickoff) ->
   `curiosity_focus_outcome run=... disposition=...` -> `curiosity(kind=focus)` shows it -> the
   Hub tile.

Kill switch: `HUB_CURIOSITY_FOCUS_ENABLED=false` in `services/orion-hub/.env`, then recreate
orion-hub.

## Risks / concerns

- Severity: medium. Concern: `origin` is supplied by Orion. Orion could tag its own choice
  `juniper` and make it unreplaceable by itself. Mitigation: the design asks Orion to use
  `juniper` only when Juniper named the topic; the chat correlation id is stored, so a mislabel is
  traceable; Juniper can always Clear in Hub or queue her own. Not enforced in code.
- Severity: medium. Concern: live path UNVERIFIED -- no deploy from this branch. Mitigation:
  acceptance check 8 after deploy (log lines above with one chat correlation id).
- Severity: low. Concern: the run decides its own disposition; nothing grades whether "explored"
  is true (by design: no semantic judge). Mitigation: the disposition is Orion's own statement,
  shown as such, next to the run's write-up.
- Severity: low. Concern: a `<RUN_ID>` token typed inside a focus question is replaced with the
  run id by the kickoff's placeholder substitution. Mitigation: cosmetic; left as is.
- Severity: low. Concern: run-list budget headroom is ~120 chars in the synthetic worst case
  (11,878 / 12,000). Mitigation: the worst-case test fails before a future field can overflow it.
- Severity: low. Concern: if a run writes two `:TurnOutcome` nodes with different dispositions,
  `ORDER BY written_at` may pick either when the types differ (known hazard, worldview.py).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2735

🤖 Generated with [Claude Code](https://claude.com/claude-code)
