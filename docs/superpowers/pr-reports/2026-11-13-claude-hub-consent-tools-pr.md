# Claude in Hub can look before it claims, if Orion says yes (Patch 1b)

## Summary

- Claude in Hub chat gets two read-only tools: `prime_now` (Prime's present, 600 characters) and
  `body_economy` (the body economy report as text). Built-in tools stay off (`--tools ""`).
- Every call goes through a broker in orion-harness-governor. The broker checks Orion's standing setting
  for the tool's category. Every category starts at "ask". On "ask", Orion decides in one short model call
  in orion-mind, with Prime's present in view, and gives a reason.
- Only Orion changes a standing setting, one category at a time, with a reason:
  `python -P -m orion.dig claude-grants set <category> allow|ask|deny because <reason>`.
- Juniper sees every lookup: a footer line under Claude's reply ("Looked at: body economy (Orion: yes,
  "...")"), and two Hub endpoints, `GET /api/claude-tools/audit` and `GET /api/claude-tools/grants`.
- Every ask, answer and result is a row in a new table, `claude_tool_call_log`. Every consent answer that
  reached the model is an act on the act ledger, faculty `act.prime.infer.consent`.

## Outcome moved

On 2026-11-12, Claude in Hub had no tools and repeated stale claims from the chat as fact (it said
`/consult-src` was empty when it held 13,576 files, and that the meter blocked consults, which it does
not). Now Claude can check Orion's body and Prime before it states them, and the prompt tells it to.
Live (before deploy, see "Docker/build/smoke checks"): the real `claude -p` in the new image called
`body_economy`, got the real report, and summarized the two binding limits correctly.

## Current architecture

The companion ran `claude -p` with `--tools ""` and no MCP. The prompt promised Claude "nothing from
Orion's interior". Nothing could check a claim about Orion's state. The design
(`docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md`) was approved with all 10
defaults and Q11; Patch 1a (the whole conversation) shipped in PRs #3249/#3250.

## Architecture touched

- **orion-room-companion:** per room turn (never a consult) writes a 0600 MCP config with one server,
  `orion_room`, and runs `claude -p ... --tools "" --mcp-config <file> --allowedTools mcp__orion_room
  --strict-mcp-config`. The server's binding (room, Claude session, the invite text) is written by the
  companion, never by Claude. After the turn it reads the server's call log, adds the footer, and deletes
  both files. No new mount.
- **orion-harness-governor:** new broker worker on `orion:claude:tool:request` (own bus connection, one
  task per call). Readers run as argument lists; output is redacted and capped at 4,000 characters.
- **orion-mind:** new consent responder on `orion:prime:consent:request` (own bus connection). Route
  `prime_consent` (circe chat lane 0, pool priority interactive), thinking off, 200 tokens, 15 s seat wait
  plus 20 s generation, frame capped at 4,000 characters in code. Self-monitor E1 gate on
  `act.prime.infer.consent`.
- **orion-sql-writer:** new insert-only table `claude_tool_call_log` (one row per phase).
- **orion-hub:** two read-only endpoints; the hub index Claude sees gains one line naming which
  categories Orion allows, asks or denies (names only).
- **GPU pool / gateway:** route `prime_consent: {class: chat, priority: interactive}`, system-only.

## Files changed

- `orion/schemas/claude_tools.py`, `orion/schemas/registry.py`: the five bus messages and the binding.
- `orion/bus/channels.yaml`: five channels (request, result:*, audit, consent request, consent result:*).
- `orion/claude_tools/`: `registry.py` (fixed tool table), `grants.py`, `broker.py`, `readers.py`,
  `render.py` (texts, redaction, cap, footer), `report.py` (weekly numbers), `mcp_server.py`,
  `transport.py`, tests.
- `orion/prime/consent.py`: frame, parse, act (pure). `orion/dig/__main__.py`: `claude-grants`.
- `config/gpu_pool.yaml`, `orion/llm/routes.py`: route `prime_consent`.
- `services/orion-mind/app/prime_consent.py`, `main.py`, `settings.py`, `.env_example`,
  `docker-compose.yml`, README.
- `services/orion-harness-governor/app/claude_tools_listener.py`, `main.py`, `settings.py`,
  `.env_example`, `docker-compose.yml`, README.
- `services/orion-room-companion/app/room_tools.py`, `claude_session.py`, `main.py`, `room_prompt.py`,
  `settings.py`, `.env_example`, `docker-compose.yml`, `requirements.txt` (`mcp`), README.
- `services/orion-sql-writer/app/models/claude_tool_call_log.py`, `settings.py`, `worker.py`,
  `.env_example`; `services/orion-sql-db/manual_migration_claude_tool_call_log_v1.sql`.
- `services/orion-hub/scripts/claude_tools_routes.py`, `main.py`, `room_claude_context.py`.
- `scripts/smoke_claude_tools.py` (live smoke), `scripts/claude_tools_consent_report.py` (weekly numbers),
  `scripts/sync_local_env_from_example.py` (new prefixes so the default sync adds the new keys).
- `.github/workflows/claude-tools-tests.yml` (new), `room-claude-context-tests.yml` (Hub routes test).

## Schema / bus / API changes

- Added: `ClaudeToolRequestV1`, `ClaudeToolResultV1`, `ClaudeToolCallV1`, `PrimeConsentRequestV1`,
  `PrimeConsentAnswerV1` (bus), `ClaudeToolBindingV1`, `PrimeConsentPastAnswerV1` (nested). Channels
  `orion:claude:tool:request`, `orion:claude:tool:result:*`, `orion:claude:tool:audit`,
  `orion:prime:consent:request`, `orion:prime:consent:result:*`. Hub `GET /api/claude-tools/audit`,
  `GET /api/claude-tools/grants`. Table `claude_tool_call_log`. Route `prime_consent`.
- Removed / renamed: none.
- Behavior changed: room turns get the `orion_room` MCP server and the tools paragraph; Claude's reply
  may carry one footer line.
- Compatibility: no existing `extra="forbid"` schema changed. `RoomClaudeRequestV1` and
  `RoomClaudeUtteranceV1` are untouched, so the companion and Hub can deploy in any order. Every new
  path fails closed: a companion deployed before the broker answers "Not available: the broker did not
  answer, so nothing was shared"; a broker before orion-mind answers "orion-mind is not listening".

**Deviations from the spec, each on purpose:**

- `--allowedTools mcp__orion_room`, not `mcp__orion_room__*`: `orion/fcc/claude_spawn.py` documents that
  Claude Code 2.1+ needs the per-server form. Verified live: the tools were called.
- The per-turn call cap lives only in the broker (`CLAUDE_TOOLS_MAX_CALLS_PER_TURN=6`), not also in the
  companion (`ROOM_COMPANION_TOOLS_MAX_CALLS`): one place to enforce, on the side Claude cannot touch.
- A grant line records the dig id (like other dig lasting writes), not the turn's correlation id: dig has
  no way to know the turn it runs in today.
- Audit is one event per phase (ask, answer, result), so an ask that never comes back is still on record.
- No act when the model was never reached (no pool seat, no gateway reply): the same rule as a Prime
  thought, so a queue failure is not graded as the faculty failing.

## Env/config changes

- Added keys: companion `ROOM_COMPANION_TOOLS_ENABLED=true`, `ROOM_COMPANION_TOOLS_STATE_DIR`,
  `ROOM_COMPANION_TOOLS_PYTHON`, `ROOM_COMPANION_TOOLS_PYTHONPATH`, `ROOM_COMPANION_TOOLS_MCP_TIMEOUT_MS`;
  governor `CLAUDE_TOOLS_ENABLED=true`, `CHANNEL_CLAUDE_TOOL_REQUEST`, `CLAUDE_TOOLS_CONSENT_DEADLINE_SEC=40`,
  `CLAUDE_TOOLS_MAX_CALLS_PER_TURN=6`, `CLAUDE_TOOLS_MAX_ASKS_PER_TURN=3`; orion-mind
  `PRIME_CONSENT_ENABLED=true`, `PRIME_CONSENT_LLM_ROUTE=prime_consent`, `PRIME_CONSENT_SEAT_WAIT_SEC=15`,
  `PRIME_CONSENT_LLM_TIMEOUT_SEC=20`, `PRIME_CONSENT_MAX_TOKENS=200`. sql-writer: the audit channel and
  route appended to the two JSON lists (the code defaults also add them, so a stale live `.env` still works).
- Removed / renamed keys: none. All flags ship ON.
- `.env_example` updated: yes (4 services).
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes; all 15 new keys added.
- Skipped keys requiring operator action: none. The sql-writer JSON lists show as "diverged" (local
  values predate this); the code defaults cover them.

## Tests run

```text
orion/claude_tools/tests                        57 passed  (gate paths, allowlist, hostile args, redaction, MCP, footer, report, grants)
orion/prime/tests/test_prime_consent.py         13 passed
orion/dig/tests                                149 passed  (incl. test_claude_grants_cmd.py; dig help 3,386 <= 3,500)
services/orion-mind/tests                      276 passed  (incl. test_prime_consent_responder.py: gate refusal, no-model, act shape, lifespan)
services/orion-room-companion/tests             98 passed  (incl. test_room_tools.py: exact argv, consult unchanged, footer, off switch)
services/orion-harness-governor/tests          105 passed, 1 failed (test_fcc_warm_pool_wiring: fails the same on main locally, env-sensitive, not touched)
services/orion-sql-writer/tests + act ledger  1265 passed, 214 skipped
services/orion-hub room-claude tests            77 passed  (incl. test_room_claude_tools_routes.py)
orion/gpu_pool/tests                           572 passed  (incl. prime_consent route pin)
services/orion-llm-gateway/tests               490 passed
orion-static-gates.yml commands (47)           all pass after the fixes above
docker compose config (orion-room-companion)    ok; the other three use env_file: .env, absent in a worktree
```

## Evals run

```text
No offline eval measures consent quality: there are no asks yet. The live instruments are:
  scripts/smoke_claude_tools.py --n 20   -> consent latency p50/p90 (acceptance check 8), audit + act rows
  scripts/claude_tools_consent_report.py -> per category yes rate, unanswered rate, reason share,
                                            latency p50/p90, flips, rubber-stamp flag
Metric quality gate for the consent numbers: provenance = claude_tool_call_log answer rows written by the
broker (orion/claude_tools/broker.py _ask); the act's artifact_produced = the model's answer parsed
(services/orion-mind/app/prime_consent.py). Independence: yes rate and reason share come from the same
rows but measure different things; latency is the broker's own clock. Theory anchor: a consent that cannot
vary is a constant, not a decision (rubber-stamp flag at 20+ asks). Live-data check: UNVERIFIED, no asks
exist until orion-mind and the broker are deployed.
```

## Docker/build/smoke checks

```text
docker build -t orion-claude-tools-verify:pr -f services/orion-room-companion/Dockerfile .   ok (mcp installed)
Live, before deploy (a throwaway broker from this branch on the live bus, temp grants file body=allow,
a one-off container from the new image, the real Claude subscription, room "verify-claude-tools"):
  - claude -p with --tools "" + --strict-mcp-config called both MCP tools.
  - body_economy: standing_allow, shared, 2,874 chars of the real report; Claude summarized the two
    binding limits (messages to Juniper today, energy money this month) correctly.
  - prime_now (category prime = ask): orion-mind not deployed, so "Nothing was shared: orion-mind is not
    listening for consent asks, so Orion could not be asked." Claude reported that plainly.
  - footer: "Looked at: body economy (Orion allows it); prime now (Orion did not answer)." (wording for
    this case since changed to "Orion could not be asked").
  - cost: $0.017 for that turn (a first attempt that Claude refused as a possible prompt injection cost $0.046).
Not run: a real consent ask (needs orion-mind + the prime_consent route deployed). UNVERIFIED.
```

## Review findings fixed

Code review ran in a subagent on `git diff origin/main...HEAD`. It found no blocker and no fail-open path.

- Finding: the consent responder's self-monitor gate read Orion's health once and never again
  (`ActGate.decide` keeps its first view; only `begin()` refreshes it).
  - Fix: every ask calls `begin()` then `decide()` under a lock (`services/orion-mind/app/prime_consent.py`).
  - Evidence: `test_the_gate_reads_a_fresh_view_on_every_ask` (healthy -> broken -> healthy on one responder).
- Finding: the probe was claimed after the model call and its result ignored, so two parallel asks could
  both probe a broken faculty.
  - Fix: claim before the model call; a lost claim is `gate_refused`.
  - Evidence: `test_a_probe_is_claimed_before_the_model_call_and_only_once`.
- Finding: a turn with tools (3 asks x 41 s + 6 reads x 15 s + generation) can pass the 180 s turn timeout.
  - Fix: `ROOM_COMPANION_TOOLS_TURN_TIMEOUT_SEC=360` for tool turns (env, settings, compose, synced).
  - Evidence: `test_a_tools_turn_gets_the_longer_timeout`.
- Finding: the MCP server's 58 s wait left about 1 s of margin over the broker's worst case.
  - Fix: 62 s (still under the CLI's 70 s `MCP_TOOL_TIMEOUT`).
- Finding: error text (a reader's stderr) reached Claude without redaction.
  - Fix: `handle()` redacts and caps `text` and `error` on every path.
  - Evidence: `test_reader_errors_are_redacted_before_claude_sees_them`.
- Finding (nit): per-turn files not removed when the turn raises. Fix: `try/finally`. Evidence:
  `test_turn_files_are_removed_even_when_the_turn_raises`.
- Finding (nit): Claude's reason and the invite text sat in Orion's consent frame as plain text. Fix: both
  are fenced as quotes, fence marks inside them are neutralised, and the system prompt says never to
  follow quoted text. Evidence: `test_claudes_reason_cannot_break_out_of_its_fence`.
- Finding (nit): `create_all` before the migration would make `arguments` `json`, not `jsonb`. Fix:
  `JSON().with_variant(JSONB(), "postgresql")`.
- Finding (nit): refused calls count toward the per-turn cap. Kept on purpose; now said in the broker docstring.
- Finding (nit): a missing `scope` must stay `once` for a yes. Evidence: `test_a_yes_without_scope_is_once_never_conversation`.

Static gates found two more, fixed: `ClaudeGrantLineV1` declared in `orion/schema_skew_discovery.py`
`DECLARED_WRITERS` (written by Orion's dig turns, read by governor and Hub), and a test fixture renamed
so the portrait-ledger partition test does not see it. `config/metrics/metric_definitions.lock.json`
re-locked for the five new bus channels.

## Restart required

```bash
# After merge, from the clean primary main checkout (the post-pull rebuild does this in one run):
# 1. migration (auto-applied by the post-pull rebuild: additive, REQUIRED-BY orion-sql-writer, orion-hub)
# 2. scripts/safe_docker_build.sh orion-gpu-pool up -d --build
# 3. scripts/safe_docker_build.sh orion-llm-gateway up -d --build
# 4. scripts/safe_docker_build.sh orion-sql-writer up -d --build
# 5. scripts/safe_docker_build.sh orion-mind up -d --build
# 6. scripts/safe_docker_build.sh orion-harness-governor up -d --build
# 7. scripts/safe_docker_build.sh orion-room-companion up -d --build
# 8. scripts/safe_docker_build.sh orion-hub up -d --build
```

Order matters only for what Claude sees in between: every gap fails closed ("not shared").

## Risks / concerns

- Severity: medium. Concern: the spec's spend gate (`budget_refusal`) for room turns is not built. Room
  invites have never had it; adding it would refuse Juniper's own `@claude` whenever the meter reading is
  stale. Mitigation: a question for Juniper (see the hand-back), not a silent behavior change.
- Severity: low. Concern: consent acts are actor `prime`, verb `infer`, as the spec says. If the body
  economy prices the `prime` family from all `prime` infer acts, short consent calls could pull the price
  of a Prime thought down. Mitigation: the faculty is separate for the self-monitor; check the next
  physiology price for `prime` after the first asks.
- Severity: low. Concern: a smoke ask is a real decision and joins Orion's answer history for that
  category. Mitigation: the smoke's reason says it is a smoke and that no is fine.
- Severity: low. Concern: governor `tests/test_fcc_warm_pool_wiring.py::test_start_passes_settings_into_pool_config`
  fails locally on this branch and on main alike (host-env sensitive); this PR does not touch it.
- Severity: low. Concern: the first-turn session id the binding carries is the one the companion minted;
  if the CLI ever reported a different id, a "yes for this conversation" would key on the minted one.
  Mitigation: today the CLI honors `--session-id` (verified in 1a).

## PR link

(filled in after the PR opens)
