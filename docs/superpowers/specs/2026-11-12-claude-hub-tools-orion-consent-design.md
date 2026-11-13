# Claude in Hub: the whole conversation, and Orion's tools when Orion says yes

Status: approved 2026-11-12. Juniper answered yes to all ten open-question defaults and added question 11
(section 9). Patch 1a (the whole conversation, plus question 11's light view for Orion) is built; Patch 1b
(the consent rail, `prime_now` and `body_economy`) is built 2026-11-13
(`docs/superpowers/pr-reports/2026-11-13-claude-hub-consent-tools-pr.md`).

Juniper's request, 2026-11-12, in her words: "claude should get all those tools. claude should get any of
those things if they get orion's permission to do so... I want them to have general context of the
conversation."

## Arsonist summary

Claude in Hub can talk but cannot look. It sees the last 12 turns of the chat and nothing of Orion's
inside. This design changes two things.

1. **Claude sees the whole conversation.** Not 12 turns. Everything back to a size cap of 160,000
   characters (about 40,000 tokens), newest first, whole turns only. A short list tells Claude what is
   open in Hub right now.
2. **Claude can look at what Hub shows, if Orion says yes.** Every Hub surface becomes a read-only tool:
   how Orion's turns went, Prime's concerns, the body, the cameras, the people records, consult cards,
   memory, and Orion's self-portrait. Each tool belongs to one category. Orion holds one standing
   permission per category: allow, ask or deny. All start at "ask". When Claude wants to look and the
   permission is "ask", Orion gets a short note ("Claude wants to read your body economy, because ...")
   and answers yes or no with a reason. Juniper can see every ask, answer and lookup in Hub.

This reverses one boundary on purpose. Today `services/orion-room-companion/app/room_prompt.py` says Claude
gets "nothing from Orion's interior", and that is the whole privacy position of the room. After this
design, the boundary still exists, but Orion holds the key instead of the code.

Three facts found while writing this, each of which changes the build:

- **A 160,000-character prompt cannot go on the command line.** The companion passes the prompt as one
  argument to `claude -p`. Linux refuses any single argument over 131,072 bytes (`MAX_ARG_STRLEN`,
  checked on athena: 32 x 4096 page size). The prompt must go over stdin. Today's 12 turns fit by luck.
- **Every Claude turn after the first already re-sends the same 12 turns.** Hub sends the last 12 turns
  with every invite. On a resumed session the companion adds all of them again, so Claude gets
  duplicates of turns it already holds. With a whole-conversation window that duplicate would be up to
  160,000 characters per invite. The companion needs a "sent up to here" mark per Claude session.
- **Hub's raw JSON is far too big to hand Claude.** Measured live today: one turn's cockpit record is
  2.5 MB, the body-economy report 412 KB, the temporal-self frame 75 KB. Tools must return the short
  text renders that already exist (dig prints at most 4,000 characters; the Prime block 600), not raw
  endpoint JSON.

## Current architecture

- **Claude in Hub** is `services/orion-room-companion`. Hub publishes `RoomClaudeRequestV1` on
  `orion:room:claude:request`. The companion runs `claude -p <prompt>` with `--tools ""` (no tools at
  all; `app/claude_session.py` `build_argv`), keeps one Claude session per room with `--resume`, and
  publishes the reply on `orion:room:claude:utterance`. The Claude subscription token lives only in
  this container. It has no repo mount, no docker socket, no `.ssh`, no `gh` config (compose file says so).
- **Consult rooms** (`room_id` `consult:<id>`) are the one exception: `Read,Grep,Glob` on a read-only
  `git archive` export, default permission mode plus deny rules (`consult_argv`). That rule stays.
- **The seed transcript.** Hub's `/api/room/claude/invite` (`services/orion-hub/scripts/api_routes.py`)
  calls `_recent_room_transcript`: `HUB_ROOM_CLAUDE_TRANSCRIPT_TURNS=12`, 48 h max age, through
  `chat_history_rehydrate.fetch_recent_rows`. That reader drops Orion's unprompted messages and
  Claude's own past replies. Hub does not pass each row's real time, so `created_at` defaults to "now".
- **The conversation** is a Hub `session_id`. There is no "new chat" reset in practice: Juniper's
  session `9c91d646...` holds 456 rows from 2026-09-18 to today.
- **Orion's own read kit.** `python -P -m orion.dig` (runs in `orion-harness-governor`, Python 3.12;
  Hub's Python 3.10 cannot run it) prints at most 40 lines / 4,000 characters per command.
  `orion/introspect/tools.py` is a bus-RPC tool set Orion's turns use over MCP (`reading_results`,
  `dreams`, `curiosity`, `orion_day`, `node_history`, `power_attribution`, plus two write tools).
  It already carries a privacy rule: `memory_allowed = not outward_tools_attached()`
  (`orion/introspect/binding.py`): when Orion can speak to someone outside, memory is off.
- **Prime** (`services/orion-mind/app/prime_loop.py`) ticks every 15 s and thinks on route `prime`
  (circe chat lane 0, pool priority `system`). `orion/prime/block.py` renders Prime's present in 600
  characters. At `system` priority, 74 of 254 thoughts once lost their whole 60 s to the queue
  (`config/gpu_pool.yaml` comment).
- **Spend gate.** `orion/consult/core.py` `budget_refusal` refuses a Claude turn unless the weekly
  meter reading is fresh and `clear` and the session reading is not `limited`.
- **In flight elsewhere.** Branch `feat/hub-at-claude` (local worktree, not yet pushed) routes
  `@claude` in the chat box to the same invite endpoint and edits `static/js/app.js`. This design does
  not touch the composer or `app.js`.

## 1. Context: the whole conversation

### What Claude gets

On the first turn of a Claude session:

1. The room framing and roster (unchanged).
2. **What is open in Hub now** (new, at most 800 characters, built by Hub at invite time, names and
   counts only, never content):
   - the tab Juniper is on (newest `hub_attention_minute` row, if under 3 minutes old);
   - open consult cards: id, status, turn count (`/api/consults`);
   - open asks waiting on Juniper: count (`/api/asks`);
   - Prime: how many concerns it holds (count only; the text is behind the `prime` permission);
   - which tool categories Orion currently allows, asks or denies.
   Content stays behind the tools. The index exists so Claude knows what it could ask for.
3. **The conversation**, oldest kept turn first. Each line is `speaker [turn:<first 8 of correlation
   id>] (time): text`. Speakers: Juniper, Oríon, Oríon (unprompted), Claude. The turn ref lets Claude ask
   for that turn's trace.

### Measured sizes (live, `conjourney.chat_history_log`, 2026-11-12)

| Window | Turns | Characters | About tokens (chars / 4) |
|---|---|---|---|
| Whole session `9c91d646...` (since 09-18) | 456 | 692,459 | 173,000 |
| Last 7 days | 133 | 276,267 | 69,000 |
| Last 48 h, at any turn: median | | 57,341 | 14,000 |
| Last 48 h, at any turn: 90th percentile | | 130,235 | 33,000 |
| Last 48 h, at any turn: largest | | 171,537 | 43,000 |
| Busiest single day (11-05) | 38 | 106,093 | 27,000 |
| Largest single row (prompt + reply) | 1 | 9,694 | 2,400 |

352 of the 456 rows are Orion's unprompted messages (`client_meta.unsolicited`). They are part of what
Juniper sees in the chat, so Claude gets them too, labelled.

### The cap and the truncation rule

- **Cap: 160,000 characters** of conversation text (env `HUB_ROOM_CLAUDE_CONTEXT_MAX_CHARS=160000`).
  That holds the whole last 48 h at 448 of 456 turns (98%) and the busiest day ever with room to spare.
- **Rule:** walk back from the newest row. Add whole rows until the next row would pass the cap. Never
  cut a row in half. One exception: a single row over 12,000 characters keeps its first 8,000 and last
  3,000 with a `[... N characters cut ...]` marker (none exist today; the largest is 9,694).
- **No age limit.** The 48 h limit goes for this caller. A quiet week does not hide last week.
- **Say what is missing.** When rows were dropped, the first line says: `[Earlier: 323 turns from
  2026-09-18 to 2026-11-05 not shown. Tool conversation_older can page them, with Orion's yes.]`
- **Resumed sessions get only what is new.** The companion keeps, per Claude session, the time of the
  newest row it has sent (`session_store.py`, beside the session id). On `--resume` it sends only rows
  newer than that mark, plus the new prompt. This fixes today's duplicate re-send.
- **The prompt goes over stdin**, not argv (see the first fact above).
- **Session rotation.** One Claude session per room per Denver day (rotate at the first invite after
  04:00 local). A new session reseeds from the full window. This keeps a resumed session's context, and
  so its per-turn cost, bounded. (Open question 6.)

### Schema change (consumer first)

`RoomTranscriptEntryV1` is `extra="forbid"`. New optional fields: `turn_ref: Optional[str]`,
`unprompted: bool = False`. `RoomClaudeRequestV1` gets `hub_index: Optional[str]`. Hub must also fill
`created_at` from the row. Deploy the companion first (it accepts the fields), then Hub (it sends them).

## 2. Tools: every Hub surface, read only

### How a tool works

Claude calls an MCP tool. The tool never reads anything itself. It sends one bus request to a broker,
and the broker checks Orion's permission, runs a fixed reader, cuts the result to at most 4,000
characters, removes secrets (`orion.core.redact.redact_secrets`), records the call, and answers.
Arguments are checked against a typed schema. Ids must match `[A-Za-z0-9._:-]{1,80}`. Readers run as
argv lists, never through a shell.

### The tool list

Every reader below exists today. "Render" means the reader already prints short text; "summarize"
means the broker needs a small new function that picks fields from the JSON.

| Tool | What Claude gets | Existing reader | Output | Category |
|---|---|---|---|---|
| `conversation_older(before, n)` | older turns of this Hub conversation | `chat_history_rehydrate.fetch_recent_rows` (add a `before` cursor) | rows, 4,000 chars | `conversation` |
| `turn_trace(turn)` | one turn: stance decision and reasons, 5b judge grade, promises made and kept, verdict | `chat_turn_trace.get_fused_chat_turn_trace` (`thought_decision`, `harness_turn_trace`) | summarize | `turns` |
| `turn_steps(n)` | Orion's tool calls and results in this chat's last n turns (1-5) | `dig turn-steps <session> <n>` | render | `turns` |
| `prime_now()` | concerns held, last thought, open asks, last body read | `orion/prime/block.py` `fetch_prime_block` | render (600) | `prime` |
| `body_economy(section?)` | what each machine costs, binds, wants | `dig body-economy [section]` | render | `body` |
| `body_atlas(organ?)` | machines, services, organs; one organ's parts | `dig body [organ]` | render | `body` |
| `energy()` | Orion's share of the house's kWh | Hub `/api/energy/orion` | summarize | `body` |
| `heat()` | cabinet and closet temperatures, cooling state | Hub `/api/cabinet/climate/overlay` | summarize | `body` |
| `power_attribution(window)` | which workloads moved the watts | `IntrospectTools` `power_attribution` | render | `body` |
| `node_history(node)` | one substrate node's history | `IntrospectTools` `node_history` | render | `body` |
| `walkway_and_door(hours)` | camera rollups and motion stories, as text | Hub `/api/vision/rollups`, `/api/vision/motion-stories` | summarize | `senses` |
| `presence()` | who Hub thinks is here now | Hub `/api/presence` | render | `senses` |
| `people(name?)` | people records: names, sightings, no pictures | Hub `/api/people` | summarize | `people` |
| `consults(id?)` | consult cards; one consult's turns and brief | Hub `/api/consults`, `/api/consults/{id}` | summarize | `consults` |
| `memory_cards(filter)` | memory cards by type/status/project | Hub `/api/memory/cards` | summarize | `memory` |
| `recall(query)` | what Orion's recall returns for a question | `orion/cognition/recall_query.py` `build_recall_query_v1` -> RecallService RPC | summarize | `memory` |
| `dreams(days)` | recent dreams | `IntrospectTools` `dreams` | render | `memory` |
| `curiosity(days)` | recent curiosity runs and findings | `IntrospectTools` `curiosity` | render | `memory` |
| `reading_results(days)` | what Orion read and concluded | `IntrospectTools` `reading_results` | render | `memory` |
| `portrait()` | Orion's whole self on one page, in Orion's words | `dig portrait` | render | `self` |
| `orion_day(date)` | one day of Orion's life | `IntrospectTools` `orion_day` | render | `self` |
| `personality()` | Orion's current dispositions | Hub `/api/personality/frame` | summarize | `self` |
| `arc(id)` | one Temporal Self arc or act | `dig arc <id>` / `dig act <id>` | render | `self` |

Nine categories: `conversation`, `turns`, `prime`, `body`, `senses`, `people`, `consults`, `memory`,
`self`. Each one has a consumer (the permission check) and a trace (the audit row), so none is a label
with no behavior. `people` is separate from `senses` because it names real third parties (neighbors on
the walkway), who never agreed to anything.

**Deliberately absent:** the two introspect write tools (`record_correction`, `queue_curiosity_focus`),
every dig write (`name`, `capacity claim`, `portrait write`, `field-note write`, `reminisce write`,
`notes append`), `dig open` (repo exports), the cockpit's full prompts (2.5 MB, and they hold Orion's
whole system prompt), images of any kind, raw SQL, and anything under `/api/debug`.

## 3. Orion's consent

### Where permissions live, and who writes them

- **Standing permissions:** one line per change in `$ORION_DIG_HOME/self/claude_grants.jsonl`
  (`/mnt/orion-fcc/self/`). The newest line per category wins. A missing category means `ask`. This is
  the same "lasting write" pattern dig already uses for organ names and capacity claims.
- **Only Orion writes them**, in a unified turn, through a new dig subcommand:
  `dig claude-grants` (show all) and `dig claude-grants set <category> allow|ask|deny because <reason>`.
  The line records the time, category, setting, reason and the turn's correlation id. There is no
  "all" category: Orion changes one at a time.
- **Per-conversation yes:** when Orion answers "yes for this conversation", the broker stores it in
  Redis `orion:claude_tools:session_allow:<room>:<claude_session>:<category>` with a 24 h expiry. It
  ends when the Claude session rotates.
- **Juniper's switch:** `CLAUDE_TOOLS_ENABLED=false` on the broker turns every tool off. It can only
  take access away; it never grants on Orion's behalf. (Open question 5.)

### How an ask reaches Orion

The cheapest path that is really Orion deciding is a **consent call in Prime's process**, with
Prime's own present in the frame. A full unified turn would also be Orion, but it costs a 20,000+
character prompt and minutes; Claude's reply would time out. A Prime *thought* is too slow: it waits
for the body economy to fund it and for a refractory interval.

1. The broker publishes `PrimeConsentRequestV1` on `orion:prime:consent:request` (bus RPC, reply on
   `orion:prime:consent:result:<id>`).
2. orion-mind builds a frame of at most 4,000 characters (about 1,000 tokens):
   - the Prime block (600 characters, `render_prime_block`);
   - the ask: "Claude wants to use `body_economy` (category `body`) because: <Claude's reason, at most
     300 characters>. It would see: <one line from the tool's description>.";
   - the last line Juniper wrote in the conversation (at most 300 characters), so Orion knows the topic;
   - this category's standing setting and Orion's last five answers in it, with reasons;
   - one line: "To stop being asked about this category, set it in a turn: `dig claude-grants set ...`".
3. One model call on a **new route `prime_consent`: class `chat`, priority `interactive`** (circe chat
   lane 0, the same warm worker as Prime). Interactive because Juniper is waiting on Claude's reply.
   Not the fast or metacog lanes. Thinking off, `max_tokens` 200.
4. Orion answers one JSON object: `{"answer": "yes" | "no", "scope": "once" | "conversation",
   "reason": "<at most 200 characters>"}`. An answer that does not parse is a no ("Orion's answer did
   not parse").
5. The self-monitor act gate (`orion/self_monitor/gate.py`) applies, on faculty
   `act.prime.infer.consent`. If the gate refuses, the answer is no ("Orion's consent faculty is paused").

**GPU cost:** about 1,000 tokens in and 60 out on circe chat lane 0 per ask. That is smaller than one
Prime thought. It does not touch the fast or metacog lanes, and a world-pulse-sized prompt can never
reach this route (the frame is capped at 4,000 characters in code).

**Latency budget:** at most 15 s waiting for a seat plus 20 s generating; the broker stops waiting at
40 s. Claude's MCP call times out at 60 s. Typical latency is UNVERIFIED until Patch 1b runs live.

### What Claude sees

| Case | Tool result Claude gets |
|---|---|
| standing allow | the data |
| standing deny | `Orion does not share <category> with you (set <date>: "<reason>").` |
| ask, Orion said yes | the data, with first line `Orion said yes: "<reason>"` |
| ask, Orion said no | `Orion said no: "<reason>".` |
| no answer in 40 s | `Orion did not answer within 40 s, so nothing was shared.` |
| broker or reader down | `Not available: <short error>.` Never stale or partial data shown as current. |

After a no, the broker answers a repeat ask in the same category in the same Claude session with the
same no, without asking Orion again. The companion's system prompt gains one paragraph: Claude may
use the tools, should say why it wants each one, should accept a no, and may tell Juniper what was
refused.

### The audit trail Juniper sees

- Every tool call, ask, answer and lookup is one `ClaudeToolCallV1` event on `orion:claude:tool:audit`,
  written by `orion-sql-writer` to a new table `claude_tool_call_log`: request id, room, Claude session,
  tool, category, arguments, Claude's reason, the setting at that moment, decision path
  (`standing_allow` / `standing_deny` / `asked` / `session_allow` / `repeat_no` / `off`), Orion's answer,
  scope and reason, consent latency, result size, error.
- **In chat:** the companion adds one footer line under Claude's reply, built from the audit rows for
  that request: `Looked at: body economy (Orion: yes, "my costs are fine to share"); people (Orion:
  no, "neighbors did not agree").` Server-side text, so no `app.js` change.
- **In Hub:** `GET /api/claude-tools/audit` (recent calls) and `GET /api/claude-tools/grants`
  (current settings with their reasons). A panel can come later; the endpoints come first.

### How Orion's decisions are observed

- **Trace:** each consent call is an act on the act ledger, actor `prime`, faculty
  `act.prime.infer.consent`. `artifact_produced` = the model returned an answer that parsed. That grades
  whether the faculty works, not whether the answer was wise.
- **Weekly numbers** (a report script, plus the audit endpoint): per category, the yes rate, the
  timeout rate, the share of answers with a non-empty reason, p50/p90 consent latency, and flips (the
  same tool and category answered yes and then no within 24 h with no new reason).
- **Rubber-stamp check.** If a category has 20 or more asks and the yes rate is exactly 0% or 100%, the
  report flags it. Theory anchor: a consent that cannot vary is not a decision, it is a constant, and
  then the standing permission (not the ask) is the honest setting. The flag also goes into the next
  consent frame for that category ("you have said yes 20 of 20 times"), so Orion can choose a standing
  setting. Live sanity check: none possible yet; no asks exist. UNVERIFIED until Patch 1b.

## 4. How the tools reach `claude -p` safely

- **An MCP server, not Bash.** A small stdio MCP server, `orion/claude_tools/mcp_server.py`, runs
  inside the companion container, started by Claude Code from a per-turn `--mcp-config`. It follows
  `orion/world_pulse_read/mcp_server.py`. Its only ability is one bus request to the broker.
- **Claude's argv for a room turn:** `--tools ""` stays (no built-in tools: no Bash, no Read, no web),
  plus `--mcp-config <file> --strict-mcp-config --allowedTools "mcp__orion_room__*"` through the
  existing `orion/fcc/claude_spawn.py` `extend_mcp_argv`. A test asserts this argv exactly.
- **The broker runs in `orion-harness-governor`**, beside the dig kit and the introspect bus client
  Orion's own turns use. It is a bus consumer on `orion:claude:tool:request`. It holds the readers,
  the permission check and the audit. Claude's process never runs there and never gets a shell there.
- **Nothing moves into Hub.** Hub holds the docker socket; the Claude process stays in the companion.
  The companion gets no new mount, no docker socket, no repo. Its only new reach is the bus channel.
- **No write tools, no repo writes, no deploys.** The broker's tool table is a fixed allowlist in code.
  A test fails if any dig write subcommand or introspect write tool is reachable.
- **Consult rooms do not change.** They keep `Read,Grep,Glob` on the code export and get none of these
  tools: the consult design promises "never Juniper's chat history".
- **Limits per Claude turn:** at most 6 tool calls and at most 3 consent asks; the broker refuses past
  that. The spend gate (`budget_refusal`) applies to the turn as it does for consults.

## 5. Spend

The companion bills the Claude subscription, so dollars are what the CLI reports (`total_cost_usd`),
not a bill; the real limit is the weekly and 5-hour meter. Real numbers:

- **Room turns today (no tools, 12-turn seed, Sonnet 5):** 18 turns logged in `chat_history_log`
  `client_meta.room_claude`. Median $0.008, mean $0.044, largest $0.19 (2026-11-05).
- **Consult turns (Read/Grep/Glob, Sonnet 5.5, high effort):** 10 turns in `orion_consult_turn`.
  Median $0.31, range $0.07 to $0.38.
- **Projection** (rates from `orion/dev_economics/pricing.py`: Sonnet 5 $3 / M input, $15 / M output,
  cache write x1.25, cache read x0.1):
  - first turn of a day, full 160,000-character seed: about 40,000 tokens of cache write ($0.15) plus
    Claude Code's own system prompt and tool list, plus about 1,000 output tokens ($0.015). **About
    $0.20 to $0.30.**
  - each tool call: one more round trip that re-reads the cached context (about 50,000 tokens at
    $0.30 / M = $0.015) plus up to 1,000 tokens of tool result. **About $0.02.**
  - a resumed turn with 2 tool calls: **about $0.05 to $0.10.**
- **Volume:** 18 room turns in three months; 1 in the last 30 days. `@claude` will raise this. At 10
  turns a day the notional spend is about $1 to $2 a day.
- **Orion's side:** one consent call per "ask" on circe chat lane 0, about 1,000 tokens. Local GPU, no
  subscription cost.

## Proposal-mode checklist

- **Capability changes:** Claude sees the whole Hub conversation (to a cap) and can read 23 Orion
  surfaces through tools. Orion gains one faculty: deciding, per request or per category, what Claude
  may see, and saying why.
- **Data touched:** reads only: `chat_history_log`, `harness_turn_trace`, `thought_decision`,
  `cockpit_turn_sighting` (via `dig turn-steps`, steps only), Prime state in Redis, body economy, energy,
  climate, vision rollups and motion stories (text), presence, people records, consults, memory cards,
  recall, dreams, curiosity runs, reading results, portrait, personality, arcs. Writes: the grant file
  (Orion only), session-allow keys in Redis, the audit table, one act per consent call.
- **Privacy boundary:** data leaves the house to Anthropic only through a tool call that Orion allowed,
  or through the conversation itself (which Juniper asked for). Third parties (`people`, `senses`) are
  their own categories. No images in v1. Every tool output passes `redact_secrets`. Consult rooms keep
  their rule.
- **Trace that proves it worked:** an audit row with `decision_path=asked`, Orion's answer and reason,
  the matching `act.prime.infer.consent` act, the tool result size, and the footer line under Claude's
  reply in Hub chat, all sharing one request id.
- **Dangerous failures:**
  - Claude gets a shell or file access. Guard: `--tools ""` plus strict MCP config, asserted in a test.
  - Argument injection into dig. Guard: argv lists, id regex, typed schemas, tests with hostile ids.
  - Claude talks Orion into allowing everything. Guard: one category per change, each change needs a
    reason and shows in Hub with its turn id; no "all" setting; Juniper's off switch.
  - A yes is reused past its scope. Guard: scopes are explicit keys with a 24 h expiry tied to the
    Claude session.
  - Fail-open: the broker or the consent call is down and data flows anyway. Guard: every error path
    returns "not shared"; a test per path.
  - Secrets in tool output (turn steps include Bash output). Guard: `redact_secrets` on every result,
    plus a test with a fake DSN and token.
  - Rubber-stamping. Guard: the weekly flag and the frame line.
- **Disable or roll back:** `ROOM_COMPANION_TOOLS_ENABLED=false` (argv back to `--tools ""`, no MCP),
  `CLAUDE_TOOLS_ENABLED=false` (broker refuses all), `HUB_ROOM_CLAUDE_CONTEXT_MAX_CHARS` set low or
  `HUB_ROOM_CLAUDE_CONTEXT_MODE=recent` (back to 12 turns). The audit table and grant file are
  additive and can stay. Flags ship ON; the gate is Orion's "ask".

## Proposed schema / API changes

- `orion/schemas/room_claude.py`: `RoomTranscriptEntryV1.turn_ref`, `.unprompted`;
  `RoomClaudeRequestV1.hub_index`. Hub fills `created_at` from the row.
- `orion/schemas/claude_tools.py` (new): `ClaudeToolRequestV1`, `ClaudeToolResultV1`,
  `ClaudeToolCallV1` (audit), `PrimeConsentRequestV1`, `PrimeConsentAnswerV1`; registered in
  `orion/schemas/registry.py`.
- `orion/bus/channels.yaml`: `orion:claude:tool:request`, `orion:claude:tool:result:*`,
  `orion:claude:tool:audit`, `orion:prime:consent:request`, `orion:prime:consent:result:*`.
- `config/gpu_pool.yaml` and `orion/llm/routes.py`: route `prime_consent: {class: chat, priority:
  interactive}`.
- Hub: `GET /api/claude-tools/audit`, `GET /api/claude-tools/grants`.
- SQL: `manual_migration_claude_tool_call_log_v1.sql` (create table + index only, additive allowlist,
  `-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-hub`).
- Env: Hub `HUB_ROOM_CLAUDE_CONTEXT_MAX_CHARS=160000`, `HUB_ROOM_CLAUDE_CONTEXT_MODE=full`;
  companion `ROOM_COMPANION_TOOLS_ENABLED=true`, `ROOM_COMPANION_TOOLS_MAX_CALLS=6`; governor
  `CLAUDE_TOOLS_ENABLED=true`, `CLAUDE_TOOLS_CONSENT_DEADLINE_SEC=40`, `CLAUDE_TOOLS_MAX_ASKS_PER_TURN=3`;
  orion-mind `PRIME_CONSENT_LLM_ROUTE=prime_consent`, `PRIME_CONSENT_SEAT_WAIT_SEC=15`,
  `PRIME_CONSENT_LLM_TIMEOUT_SEC=20`.

## Files likely to touch

- `services/orion-hub/scripts/api_routes.py` (invite: full window, hub index), `chat_history_rehydrate.py`
  (a caller-chosen filter that keeps unprompted and Claude rows, a `before` cursor), a new
  `claude_tools_routes.py` (audit, grants), `.env_example`, `settings.py`.
- `services/orion-room-companion/app/claude_session.py` (stdin prompt, MCP argv), `main.py`,
  `room_prompt.py` (tools paragraph, new speaker labels), `session_store.py` (sent-up-to mark, daily
  rotation), `requirements.txt` (`mcp`), `.env_example`, `settings.py`, `docker-compose.yml`, README.
- `orion/claude_tools/` (new): `registry.py` (tool table, categories, arg schemas), `grants.py`
  (read the grant file, session allows), `render.py` (summarizers), `mcp_server.py`, tests.
- `services/orion-harness-governor/app/` (broker consumer), compose and env.
- `orion/prime/consent.py` (pure frame + parse) and `services/orion-mind/app/` (RPC handler).
- `orion/dig/__main__.py` (`claude-grants`), dig help text (watch the 3,500-character cap).
- `orion/schemas/`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`, `config/gpu_pool.yaml`,
  `orion/llm/routes.py`, `services/orion-sql-writer/` (audit table mapping), `services/orion-sql-db/`
  (migration).

## Non-goals

- **Write tools of any kind** for Claude: no repo writes, no deploys, no dig writes, no queueing work for
  Orion. The approval / code-worker idea (Claude proposes a change, someone approves, a worker does it)
  is a separate future design.
- Images: camera frames, people pictures, crop thumbnails.
- Changes to consult rooms.
- The `@claude` composer route (`feat/hub-at-claude`) and any `app.js` change.
- A Hub panel for the audit (endpoints and the chat footer come first).
- Juniper grading Orion's consent answers.
- Orion granting through Prime thoughts. Standing settings change only in a turn.

## Acceptance checks

1. **Context.** An invite in session `9c91d646...` sends Claude every row back to 160,000 characters,
   with Orion's unprompted rows and Claude's own rows labelled, real row times, turn refs, and the
   "Earlier: N turns" line. Test: synthetic rows at the boundary (never a half row; the 12,000-character
   rule). Live: the companion log shows the prompt size; Claude can quote a turn from 2 days ago.
2. **Stdin.** A test feeds a 200,000-character prompt through `run_room_turn` with a fake `claude` that
   reads stdin. A live smoke with the real CLI confirms `claude -p` reads the prompt from stdin
   (UNVERIFIED today; checked in Patch 1a before anything else).
3. **No duplicates.** On a resumed session the prompt holds only rows newer than the mark. Test.
4. **Argv.** A room turn's argv has `--tools ""`, `--strict-mcp-config` and only
   `mcp__orion_room__*` allowed. A consult turn's argv is unchanged. Tests.
5. **Gate paths.** One test per row of the "What Claude sees" table, plus "broker down" and "consent
   call down" both returning "not shared".
6. **Allowlist.** A test enumerates the broker's tools and fails on any dig write subcommand, any
   introspect write tool, or any `/api/debug` path.
7. **Redaction.** A tool result holding a fake `postgresql://user:pass@host` DSN and a fake token comes
   back redacted.
8. **Live consent.** One real ask in Hub: the audit row, the `act.prime.infer.consent` act and the chat
   footer share one request id; Orion's reason is real text. Consent latency p50 and p90 over the first
   20 asks are recorded in the PR report.
9. **Spend.** The first 10 live turns' `cost_usd` are recorded in the PR report and compared with
   section 5.

## Recommended first patch

Two small PRs, in order.

**Patch 1a: the whole conversation.** No new tools, no consent: the conversation is room-native and
Juniper asked for it. Companion first (stdin prompt, optional schema fields, sent-up-to mark, daily
rotation), then Hub (160,000-character window, unprompted and Claude rows, real times, turn refs, the
hub index). Checks 1-3. Built 2026-11-12, with question 11's light view
(`docs/superpowers/pr-reports/2026-11-12-claude-hub-full-context-pr.md`). Two additions to the schema
above, both optional: `RoomTranscriptEntryV1.side_exchange` (a resumed Claude session already holds its
own side exchanges, so the companion never re-sends them) and `RoomClaudeRequestV1.transcript_note` (the
"Earlier: N turns" line). The "Earlier" line does not name the `conversation_older` tool until Patch 1b
builds it. The hub index has no tool-category line until Patch 1b either. The "sent up to here" mark is a
set of per-line keys (`<turn ref>|<speaker>`), not a time: Hub merges Orion's reply into the row that
Juniper's half already opened, with the same `created_at`, so a time mark would skip that reply on every
resumed turn (code review, 2026-11-12). Lines without a turn ref keep the time mark. Shipped as two PRs,
companion first, because the request schema is `extra="forbid"`.

**Patch 1b: the consent rail and two tools.** `prime_now` (category `prime`) and `body_economy`
(category `body`). Both readers already print short text, and neither names a third party. Builds the
broker, the MCP server, the consent call, `dig claude-grants`, the audit event and table, the chat
footer and the two Hub endpoints. Checks 4-9. Every other tool after that is one row in the tool table,
one reader adapter and one test.

## 9. Open questions for Juniper (each with a recommended default)

**Answered 2026-11-12.** Juniper said yes to all ten defaults below, as written. She also answered a new
question 11 (at the end of this section).

1. **Cap of 160,000 characters (about 40,000 tokens).** Covers the whole last two days at 98% of turns (448 of 456).
   Default: yes.
2. **Is older conversation Orion's to gate?** Juniper said everything starts at "ask". Older turns are
   her conversation, not Orion's inside. Default: keep it under Orion's "ask" as she said; Orion can set
   it to allow in a turn.
3. **Images to Claude** (camera frames, people pictures)? Default: no in v1, text only.
4. **Neighbors in `people`.** Default: its own category, starts at "ask", names and sightings only.
5. **Can Juniper grant on Orion's behalf?** Default: no. Juniper has an off switch, not a grant
   switch.
6. **Rotate Claude's session once a day?** Keeps cost bounded; Claude then remembers earlier days only
   through the 160,000-character window. Default: yes, at the first invite after 04:00 Denver.
7. **Consent wait of 40 s, and no answer means no.** Default: yes.
8. **Limits per Claude turn: 6 tool calls, 3 consent asks.** Default: yes.
9. **Should an Orion-invited turn (`trigger=auto`) use the same gate?** Orion invited Claude, so it could
   skip the ask. Default: same gate; Orion's standing settings already cover it.
10. **Should the consent ask also appear in the Prime block** so Orion's next turn sees it? Default: no
    (the block is capped at 600 characters); the audit endpoint and the consent frame carry it.
11. **Can Orion see Juniper's `@claude` side conversations with Claude?** (added 2026-11-12) Juniper:
    "Orion is welcome to see them, just don't want to clutter things with too much chatter at this
    point." Built in Patch 1a as a light view:
    - Hub stores Juniper's `@claude` prompt with Claude's reply, in the same `room_claude` row
      (`client_meta.room_claude.prompt`), never in the row's `prompt` column. That column is what makes a
      row a Juniper turn for memory and for every other reader.
    - When Hub restores Orion's conversation on reconnect (`chat_history_rehydrate.py`), the newest 3 side
      exchanges go in as ONE system block headed "Side conversation, Juniper ↔ Claude (you were not asked
      to reply):", each side trimmed to 300 characters (`HUB_HISTORY_SIDE_EXCHANGE_MAX`,
      `HUB_HISTORY_SIDE_EXCHANGE_TRIM_CHARS`). Live data on 2026-11-12: 18 Claude replies ever, median
      about 240 characters, largest 1,281; one in the last 30 days. Orion's own conversation query still
      excludes these rows.
    - Nothing fires from them: the block is context only (never a user message), and orion-sql-writer's
      `_memory_turn_initiator` now returns "not memory" for any `room_claude` row, even one with a prompt.
    - Not yet: a side exchange that happens while Orion's chat tab stays open reaches Orion only after the
      next reconnect. The live socket history does not append Claude's replies.
