## Summary

- Orion now takes plumbing questions to Claude before they reach Juniper. A question can be why something works the way it does, something that looks wrong, or a fix Orion wants. Hub runs one consult at a time (`services/orion-hub/scripts/consult_loop.py`).
- Consults have three sources:
  - curiosity runs the door policy (#3177) kept from Juniper
  - faculties the self-monitor holds broken
  - the stuck-belief trigger (a belief tested many times without settling)
- **Orion's turns** run through Orion's real unified turn, on a session used only for consults, so none of Juniper's chats are in them. Orion can run digs to fetch live numbers.
- **Claude's turns** run in the room companion's new consult mode: Sonnet 5.5, high effort, read-only tools (Read, Grep, Glob). They read a `git archive` export of the deployed commit: tracked files only, no `.git`, no `.env`. Every turn shows live in Hub chat.
- **How a consult ends:**
  - `answered`: nothing reaches Juniper.
  - `agreed_proposal` or `disagree`: one brief tags her in Hub chat, has Yes / No / No-because buttons, and is emailed to her.
  - A brief missing any part, or with a cost number that has no source, is never sent.
- **Learning:** Juniper's answer, or an `answered` ending, is shown once to Orion's next curiosity run. Orion records it on a belief.
- **Budget:** the Claude usage meter is now ON. The budget gate fails closed.

## Outcome moved

- Before: plumbing findings Orion wanted to share went nowhere after #3177. The door policy recorded `not_for_juniper` 4 times on 2026-11-11, its first full day live.
- After: each such finding offers one consult. Juniper sees at most 3 briefs a week, and only proposals or disagreements.
- The "ask Claude" trigger was refused 3,553 times by the meter, which was switched off. The consult loop now reads the meter on the bus.

## Current architecture

- **Room companion:** ran Claude with no tools, for social chat only.
- **Door policy:** dropped plumbing findings with a `not_for_juniper` decision row and nothing more.
- **Claude usage meter:** the producer existed but was `ENABLED=false`, and nothing read it.
- **Self-finding outreach:** retired in #3177.
- **Stuck-belief trigger:** dry-run only.

## Architecture touched

- **orion-hub:**
  - new consult loop, API routes and chat UI
  - the relay sends `consult:` replies to the consult loop
  - `compose()` takes a session id
  - the curiosity door-policy site offers a consult
  - the curiosity kickoff shows consult outcomes
- **orion-room-companion:** consult mode (tools, cwd, model, effort, timeout and permission argv), plus a read-only mount of the consult source export.
- **Post-pull rebuild:** after each deploy it exports the deployed commit (`scripts/export_consult_source.py`, log line `CONSULT_EXPORT`, kill switch `ORION_POST_PULL_CONSULT_EXPORT=0`).
- **orion-cocreation-signals:** Claude usage meter ON.
- **Postgres:** new tables `orion_consult` and `orion_consult_turn`.
- **Env sync script:** now covers the consult prefixes and the companion service.

## Files changed

- `orion/consult/core.py`: room ids, reply parser, brief gate, budget gate, prompts, renderers, source helpers (pure)
- `orion/schemas/consult.py`: the brief contract (`ConsultBriefV1`, cost lines with sources, impact)
- `orion/schema_skew_discovery.py`: declares the consult schema's writer (LLM output plus Hub's own DB)
- `orion/curiosity/kickoff_prompt.py`: `consult_outcomes` section (writable graph only)
- `services/orion-hub/scripts/consult_loop.py`: the loop and its store (asyncpg)
- `services/orion-hub/scripts/consult_routes.py`: `GET /api/consults`, `GET /api/consults/{id}`, `POST /api/consults/{id}/decision`, `GET /api/debug/consult`
- `services/orion-hub/scripts/room_claude_relay.py`: hands `consult:` replies to the loop, and exposes the connections it broadcasts to
- `services/orion-hub/scripts/endogenous_outreach.py`: `compose(session_id=...)`
- `services/orion-hub/scripts/curiosity_investigation.py`: offers a consult at the `not_for_juniper` site, and claims outcomes for the kickoff
- `services/orion-hub/scripts/main.py`: wiring
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`: `HUB_CONSULT_*` keys, `CHANNEL_CLAUDE_LIMIT`, and `HUB_ROOM_CLAUDE_ENABLED=true` (matches live)
- `services/orion-hub/static/js/consult.js`, `static/js/app.js`, `templates/index.html`: turns, brief, buttons, "Show the conversation", and waiting briefs loaded on page open
- `services/orion-room-companion/app/{claude_session,main,room_prompt,settings}.py`, `.env_example`, `docker-compose.yml`: consult mode
- `services/orion-cocreation-signals/{.env_example,docker-compose.yml,app/settings.py}`: meter ON
- `services/orion-sql-db/manual_migration_orion_consult_v1.sql`: additive tables, `REQUIRED-BY: orion-hub`
- `scripts/export_consult_source.py`, `scripts/rebuild_services_from_git_diff.sh`: the export and its post-deploy hook
- `tests/scripts/test_export_consult_source.py`: the export has tracked files only, no `.git`, no `.env`; `current` swaps; old exports are pruned
- `scripts/sync_local_env_from_example.py`: `HUB_CONSULT_` and `ROOM_COMPANION_CONSULT_` prefixes; `orion-room-companion` added to the default services
- tests and eval (below); READMEs; design doc updated with the changes made during the build

## Schema / bus / API changes

- Added:
  - `ConsultBriefV1` and its parts (not a bus payload; stored as JSON)
  - two tables
  - four HTTP routes
  - Hub socket frames `consult_turn` and `consult_brief`
- Removed: none.
- Behavior changed:
  - A `RoomClaudeRequestV1` whose `room_id` starts with `consult:` runs in consult mode in the companion.
  - Hub now consumes `orion:substrate:claude_limit`. It was already declared as a consumer in `channels.yaml`.
- Compatibility notes: no schema field changes. Consult rooms reuse the existing room channels.

## Env/config changes

- Added keys:
  - Hub: `HUB_CONSULT_ENABLED=true`, `_TICK_SEC=120`, `_MAX_TURNS=6`, `_WEEKLY_BRIEF_CAP=3`, `_CLAUDE_TIMEOUT_SEC=1200`, `_SOURCE_MAX_AGE_HOURS=72`, `_NOTIFY_TIMEOUT_SEC=20`, `CHANNEL_CLAUDE_LIMIT`
  - Companion: `ROOM_COMPANION_CONSULT_MODEL=claude-sonnet-5-5`, `_EFFORT=high`, `_TIMEOUT_SEC=900`, `_TOOLS=Read,Grep,Glob`, `_REPO=/consult-repo`
- Changed:
  - `COCREATION_SIGNALS_CLAUDE_LIMIT_ENABLED` false -> true (template, compose default and settings default)
  - `HUB_ROOM_CLAUDE_ENABLED` false -> true in the template, settings and compose (live was already true). This also turns on the Ask Claude room relay on any host that lacked the key.
- `.env_example` updated: yes, for all three services.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, 15 keys added. `COCREATION_SIGNALS_CLAUDE_LIMIT_ENABLED` was flipped to true in the live `.env` by hand, because the sync never overwrites a diverged value.
- Skipped keys: none.

## Tests run

```text
orion/consult/tests                                   33 passed
orion/curiosity + orion/consult + orion/autonomy      510 passed, 29 skipped
services/orion-room-companion/tests                   62 passed (incl. new test_consult_room.py, updated credential isolation)
services/orion-hub/tests/test_consult_loop.py         14 passed (lifecycle: offer -> turns -> brief -> decision -> told once; answered; gate retry; budget hold; timeout; cap)
services/orion-hub/tests/test_consult_js.py           5 passed (node: buttons post the right decision, no-because needs a reason, conversation view, page-open load)
services/orion-hub/tests/test_outreach_door_policy.py all passed (+2: held-back run offers one consult; offered run never does)
services/orion-cocreation-signals/tests                111 passed
tests/scripts/test_sync_local_env_from_example.py      22 passed
orion-static-gates (every step, run locally)           all ok after declaring the consult schema writer
services/orion-hub/tests full suite                    1 failed, 4673 passed, 147 skipped -- the same 1 failure main has (test_fcc_model_labels_disabled_by_default)
tests/scripts/test_export_consult_source.py           2 passed
SQL smoke on a throwaway postgres:15 container         migration applied twice (idempotent); every store query verified (offer dedupe, open/pending, turns, cost sum, decide once, take_outcomes once, expire)
```

## Evals run

```text
services/orion-hub/evals/run_consult_source_replay.py (read-only, live DB)
  Last 14 days: 4 curiosity runs kept from Juniper would each offer one consult.
  By line: {'investigate': 4}; all on 2026-11-11 (door policy live since 11-10), so ~4/day going forward.
  Briefs reaching Juniper: at most 4 in 14 days (cap 3/week).
Live sources 2026-11-11 18:00Z: self-monitor holds no faculty broken (0 offers); the stuck-belief
trigger names one belief (self:outward_learning_in_record_not_in_loop_20260921).
```

## Docker/build/smoke checks

```text
Live permission probe inside orion-athena-room-companion (claude 2.1.239, --model claude-sonnet-5-5,
--tools Read,Grep,Glob, --permission-mode default, deny rules):
  ./probe.txt read OK; /proc/self/environ DENIED; /etc/hostname DENIED; /usr/lib/os-release NOT GRANTED;
  Grep in /proc/self/environ DENIED; Glob in /root DENIED. Served model: claude-sonnet-5-5. Cost $0.06.
docker compose config: orion-room-companion OK, orion-hub OK.
Consult source export run on this host: /mnt/scripts/orion-consult-src/current -> c4edf4dd9c6c, 180 MB,
no .git and no .env anywhere in it (find -L checked). The companion mounts it after its next recreate.
Images not rebuilt here: no Dockerfile changed; the post-pull rebuild builds them from main.
```

## Review findings fixed

The review subagent read the diff by hand; the code-review skill was not available to it.

- Finding (blocker): the first consult mount was Orion's FCC sandbox clone. Its `.git/config` remote URL carries a GitHub credential (checked, redacted), and Orion writes notes and scratch files into it. Claude could have read the credential and quoted it into chat, email or the Anthropic transcript.
  - Fix: the companion mounts a `git archive` export of the deployed commit instead, read-only. It holds tracked files only: no `.git`, no `.env`, nothing Orion wrote. Deny rules now also cover `**/.git/**`. The credential test refuses any mount other than the export.
  - Evidence: `tests/scripts/test_export_consult_source.py` uses a repo with a credentialed remote and a `.env`, and neither reaches the export. The live export has no `.git` and no `.env`.
- Finding: a reply was not matched to its request, so a late reply could be stored as the answer to the wrong turn.
  - Fix: each send stores its own correlation id first. A reply that does not carry it is ignored.
  - Evidence: `test_a_late_reply_to_an_earlier_send_is_ignored`.
- Finding: a Hub restart while Claude worked lost the reply (pub/sub has no backlog), and the consult failed.
  - Fix: the first timeout re-sends the stored prompt once. The companion resumes the session, so Claude keeps its context. The second timeout fails with `claude_no_reply_after_resend`.
  - Evidence: `test_claude_silence_times_out`.
- Finding: a budget wait during the brief retry lost the correction and reset the one-retry rule.
  - Fix: the pending prompt is stored and re-sent exactly. The retry is its own column (`gate_retry_part`).
  - Evidence: `test_budget_wait_during_the_brief_retry_keeps_the_correction`.
- Finding: `HUB_ROOM_CLAUDE_ENABLED` defaults disagreed across files.
  - Fix: settings, compose and template are all true.
  - Evidence: `check_settings_defaults.py orion-hub --example-drift` passes.
- Finding: Orion's consult turn still runs recall, and the "no chats" rule was only a prompt line.
  - Fix, partial: recall now searches the consult question (`retrieval_query`). The unified turn has no switch that turns recall off, and Orion's tools can read Orion's history in any case. The spec and README now say plainly that Hub enforces what Hub sends, and that Orion's own words are Orion's choice. This stays a concern, below.
  - Evidence: `test_orion_turn_recall_searches_the_question`.
- Nits fixed:
  - `take_outcomes` uses `FOR UPDATE SKIP LOCKED`.
  - `set_status` checks column names against an allow-list.
  - `_continue` logs its own errors.
  - Shutdown cancels and awaits consult tasks.
  - A blank impact field fails the gate.
  - The spec matches the code: the companion, and `brief_refused` on a turn limit.
  - The README states the meter's hour-long wait after a restart and the bus trust boundary.

## Restart required

The post-pull rebuild deploys hub and room-companion from main and applies the migration (additive DDL, on the allowlist). The meter needs its service recreated:

```bash
scripts/safe_docker_build.sh orion-cocreation-signals up -d --build
```

## Risks / concerns

- Severity: medium.
  - Concern: consults draw on the same Claude Code subscription as dev sessions.
  - Mitigation: one consult at a time. No Claude turn starts unless the weekly reading is `clear` and fresh and the session reading is not `limited`. The meter has no percentage, so the 80% line in the design could not be built; the gate fails closed instead.
- Severity: medium.
  - Concern: Orion and Claude agree on a wrong story.
  - Mitigation: Claude has the skeptic role. Every claim carries proof or says "not checked". The brief gate refuses a cost number without a source. Juniper's answer and a later outcome are separate evidence.
- Severity: medium.
  - Concern: Orion's consult turns run with normal recall, and Orion's tools can read Orion's history. Orion could quote a private chat to Claude. Hub sends only Orion's words, and the prompt asks Orion not to quote chats. Code cannot stop it short of a recall-off switch across Hub, orion-thought and cortex-exec, and even that switch would not stop Orion's own tools.
  - Mitigation: consult-only session, recall aimed at the question, and every turn visible in Hub chat. Juniper's call whether a recall-off switch is worth building.
- Severity: low.
  - Concern: the companion now mounts code. It is the `git archive` export, read-only, tracked files only.
  - Mitigation: reads outside the working directory were verified denied live.
- Severity: low.
  - Concern: after every Hub restart, consults wait up to an hour for the weekly meter reading (pub/sub, published hourly).
  - Mitigation: none needed; it fails closed.
- Severity: low.
  - Concern: a curiosity run that is cancelled after claiming consult outcomes loses them. Orion is not shown them again.
  - Mitigation: they stay in `orion_consult` and on `GET /api/consults?all=1`.
- Not in this patch (design Patch 2):
  - the outcome check 7 days after a change ships
  - moving approved items to the promise ledger (#3188, not built)
- Unchanged: `orion-curiosity-peer`'s own Claude fallback still reads the meter from disk, which its container cannot see, so it still refuses.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
