# Consult Claude before Juniper: a second pair of eyes for Orion's plumbing questions

Status: design (proposal mode). Approved in chat by Juniper on 2026-11-11 ("fantastic on all the above"). No shadow mode: Juniper said it is not needed.

## Arsonist summary

Yesterday's door policy (#3177) stopped Orion's plumbing findings from reaching Juniper. Orion's curiosity digest said this made them sad, because those findings now go nowhere.

This design gives those findings a new path. Orion takes the question to Claude first. Claude checks it against the real code and the live data. They talk for a few turns in Hub chat. Then they bring Juniper one plain brief. The brief tags her in chat and is emailed to her.

Most of the parts exist already. None of them work end to end today:

- **The second-opinion service exists, but every request fails.** `orion-curiosity-peer` received 5 help requests this week. All 5 Cursor runs failed (`cursor agent exited 1: Connection lost`). The Claude backup never ran: it is refused by the Claude usage meter, which is switched off (`COCREATION_SIGNALS_CLAUDE_LIMIT_ENABLED=false`).
- **The "ask Claude" trigger exists, but it has never asked.** It logged 3,553 decisions to `~/.orion/ask-claude-dry-run.jsonl`. Every decision is `refused: budget_unobserved`, for the same reason.
- **The Claude in Hub chat cannot look at anything.** `orion-room-companion` runs `claude -p ... --tools ""`. It can only talk. It cannot read a file.

The single most useful act is to turn the meter on. The rest is one new lane in the service that already exists.

## Current architecture

| Piece | Where | Live state 2026-11-11 |
|---|---|---|
| Help request → peer brief | `services/orion-curiosity-peer/app/worker.py`, `HelpRequestV1` / `PeerBriefV1` in `orion/schemas/curiosity_peer.py` | Up 6 days. 5 asks this week, 0 answers. Read-only repo mount at `/repo`. No DB credentials, no docker socket, no SSH (deliberate). |
| Claude backup for the peer | `services/orion-curiosity-peer/app/claude_fallback.py` → `orion:room:claude:request` → `orion-room-companion` | Never ran this week. Refused by the meter (`worker.py:533-560`). |
| Claude in Hub chat | `services/orion-room-companion/app/claude_session.py` | Subscription OAuth token. Model `claude-sonnet-5`, effort `medium`. Session-continuous (`--resume`). No tools. |
| Claude usage meter | `orion-cocreation-signals`, `orion:substrate:claude_limit`, `ClaudeLimitObservationV1` | `ENABLED=false`. Reads `~/.claude/projects` read-only. |
| "Ask Claude" trigger | `orion/autonomy/ask_claude_trigger.py` | Dry run only. A belief tested 3+ times with confidence ≤ 0.7. |
| Door policy | `scripts/outreach_door_policy.py`, `orion/hub/why_juniper.py` | Live. Plumbing findings are recorded as `not_for_juniper`. Every message needs a `Why Juniper:` line. |
| Beliefs Orion tests | `orion/curiosity/worldview.py` (`:Prior`, `times_tested`, `confidence`, status) | Live. The peer already writes `:PeerBrief` nodes into this graph. |
| Running-cost prices | `orion/physiology/prices.py`, `budget.py` (body economy) | Live in shadow (Patch B). |
| Email | `services/orion-hub/scripts/orion_day_letter.py` | Live. One email a day. |
| Promise ledger | #3188 | Design only. Not built. |

## What changes

### 1. Where a consult starts

A consult has three sources. Each one already exists:

1. A curiosity run that the door policy marked `not_for_juniper`. This is the exact material that stopped reaching her.
2. A self-monitor finding. These were self-findings outreach before #3177.
3. The "ask Claude" trigger: a belief that is stuck after 3 or more tests.

Orion does not invent a new kind of question. Each consult names its source by row id. The brief shows that source to Juniper.

### 2. Who Claude is in a consult, and what Claude can see

Claude runs in `orion-curiosity-peer`, in a new `consult` mode. It does not run in the chat companion, for two reasons:

- The peer container already has the repo mounted read-only.
- It holds no secrets.

The call is `claude -p` with the subscription OAuth token. The companion already uses that same token.

- Model: Sonnet 5.5, effort `high`. Juniper asked for "sonnet with high reasoning."
- Tools: `Read`, `Grep`, `Glob` only. No `Bash`, no `Edit`, no `Write`, no web.
- Session-continuous, the same as the companion: one CLI session per consult, continued with `--resume`.

**Context rule: system state in, conversation out.** There is no privacy boundary between Juniper and Orion. The new boundary is between Orion and Anthropic.

- **In:** the question, its source row, what Orion already tried, the repo, and dig outputs that Orion attaches.
- **Out:** Juniper's chat history, Orion's journals, and Orion's personality and system prompt.

The rule is enforced in code, not in a prompt. The consult request schema has no field for chat history or journals. A test refuses any field outside the allow-list.

**Live data without new credentials.** The peer container stays without DB access. When Claude needs a number, Claude asks for it in a turn, for example "how often did X run this week?" Orion runs it with the dig kit Orion already uses, and attaches the output to the next turn. That gives the turns real content. Claude checks; Orion fetches. It also keeps the credential isolation that the peer container exists for.

### 3. The conversation

- The conversation is shown in Hub chat, in its own room `consult:<consult_id>`. Hub's existing room relay shows Claude's turns.
- Each turn is tagged `consult`.
- Claude's role prompt says: you are the skeptic. Check every claim Orion makes against the code or a dig output before you agree with it. Say "I could not check this" when you could not.
- Orion's turns are written by the same composer that writes Orion's outreach, with the consult transcript as its input.
- At most 6 turns: 3 from Orion, 3 from Claude.

A consult has three possible endings:

| Ending | What happens |
|---|---|
| `answered` | The question had an answer, or the thing works as designed. Nothing goes to Juniper. Orion keeps the answer and updates the belief. |
| `agreed_proposal` | They agree something should change. A brief goes to Juniper. |
| `disagree` | They do not agree. A brief goes to Juniper with both positions, side by side. It is never forced into one view. |

If the turn limit is reached without an ending, the consult ends as `disagree` and each side states its last position.

### 4. The brief, and its gate

A brief is delivered only when every part is present and non-empty. Missing parts mean nothing is sent, and the refusal is recorded with the name of the missing part. This works like the `Why Juniper` gate.

1. **Where it came from:** the source row (curiosity run, self-monitor finding, or belief) and a link to the consult transcript.
2. **Why it matters:** in plain words.
3. **What they checked:** each claim with its proof, as a file and line, or a dig command and its output. A claim with no proof is labeled "not checked."
4. **Where they disagree:** "none," or both positions.
5. **Impact:** what changes for Orion or for Juniper if this is done, which number should move, and how we would see it move.
6. **Cost:** three numbers, each with its source:
   - **The consult itself:** tokens used, from the CLI's own JSON accounting (`modelUsage`).
   - **Doing the work:** Orion's running cost, from the body-economy prices, the energy tab and the dig tools.
   - **Doing nothing:** what it costs to leave things as they are.

   Any number without a source must say `unknown` and give the reason. An invented estimate fails the gate. The test: a number in the cost part must have a source pointer beside it.
7. **Recommendation:** exactly one.
8. **The question for Juniper:** one yes/no question.

The brief is written in plain English. The prompt carries the plain-words rules from `test_finding_prompt_plain_words.py` (#3177), and the same check runs on the brief.

**Delivery:**

- The brief is posted in Hub chat, in the consult room, tagged `@Juniper`.
- One email is sent with the same brief, through the day letter's email sender.
- The brief has one finding key. The same key is never sent twice.

**Volume:**

- One consult runs at a time.
- At most 3 briefs per week reach Juniper.
- `answered` consults do not count against the cap.

**Budget:** the consult uses Juniper's Claude Code subscription, the same pool as her dev sessions. A consult does not start if the meter shows the 5-hour or the weekly limit at 80% or more. When the meter has no reading, a consult does not start either: it fails closed. Dev work comes first.

### 5. What happens after Juniper answers

Juniper answers in Hub with three buttons: Yes, No, and No-because (a free-text reason).

Learning uses the belief graph Orion already has. There is no new ledger.

- **Every brief becomes a belief.** It is a `:Prior` with a claim such as "fixing X would stop Y" and a source pointer to the consult.
- **Juniper's answer is one piece of evidence.** It goes on the belief as `decision` evidence.
- **The outcome is a separate piece of evidence.** For an approved item, the brief's impact line names the number that should move. A check date is set 7 days after the change ships. On that date, Orion reads the number with a dig and records `outcome` evidence. The belief's confidence moves on the outcome only.

**Why the two pieces of evidence stay separate:** if they were merged, Orion would learn "what Juniper approves" instead of "what is true."

**After a yes:** the item is queued as a work item for a dev session. When #3188 (the promise ledger) is built, approved items move there. Until then they go into the consult table with status `approved`.

**After an `answered` ending:** the answer is recorded as evidence on the source belief, which increases `times_tested`. Then curiosity can retire or revise the belief the normal way.

### 6. What goes into Orion's memory

- Every consult turn is shown in Hub chat.
- Orion's chat recall keeps only the final brief, or the `answered` summary, as one item.
- Individual consult turns are excluded from the unified turn's recent history. They are kept in the consult table, where Orion can read them back with a dig.

This keeps the "what is in Orion's context" question to one line. It also stops consult turns from crowding out Juniper's conversations.

## Missing questions

None block the build. Defaults chosen:

- Weekly cap of 3 briefs, and an 80% limit threshold. Both are settings. Change them after the first two weeks.
- Turn limit of 6.
- Outcome check 7 days after the change ships.

## Proposed schema / API changes

- `orion/schemas/consult.py` (new):
  - `ConsultRequestV1`: `consult_id`, `source_kind` (`curiosity_run` | `self_monitor` | `stuck_prior`), `source_ref`, `question`, `tried_summary`, `attachments: list[DigAttachmentV1]`. It has no free-text "context" field.
  - `ConsultTurnV1`: `consult_id`, `turn`, `speaker`, `text`, `dig_requests`, `tokens`.
  - `ConsultBriefV1`: the eight brief parts as typed fields, `ending`, `finding_key`.
  - `ConsultDecisionV1`: `consult_id`, `decision` (`yes` | `no` | `no_because`), `reason`, `decided_at`.
- Bus channels in `orion/bus/channels.yaml`:
  - `orion:consult:request`
  - `orion:consult:turn`
  - `orion:consult:brief`
  - `orion:consult:decision`
- Registry entries in `orion/schemas/registry.py`.
- Postgres migration (additive only):
  - `orion_consult` (one row per consult, with status)
  - `orion_consult_turn`
  - Header `-- ORION-MIGRATION-REQUIRED-BY: orion-curiosity-peer, orion-hub, orion-sql-writer`
- Env keys:
  - `orion-curiosity-peer`: `CURIOSITY_PEER_CONSULT_ENABLED=true`, `CURIOSITY_PEER_CONSULT_MODEL=claude-sonnet-5-5`, `CURIOSITY_PEER_CONSULT_EFFORT=high`, `CURIOSITY_PEER_CONSULT_MAX_TURNS=6`, `CURIOSITY_PEER_CLAUDE_OAUTH_TOKEN`
  - `orion-hub`: `HUB_CONSULT_WEEKLY_BRIEF_CAP=3`, `HUB_CONSULT_LIMIT_PCT_CEILING=80`
  - `orion-cocreation-signals`: `COCREATION_SIGNALS_CLAUDE_LIMIT_ENABLED=true`

## Files likely to touch

- `services/orion-cocreation-signals/.env_example`: meter on
- `services/orion-curiosity-peer/app/consult.py` (new), `worker.py`, `settings.py`, `docker-compose.yml`, `.env_example`, `README.md`, `tests/`
- `orion/schemas/consult.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`
- `services/orion-hub/scripts/room_claude_relay.py`: consult rooms
- Hub chat template and JS: decision buttons and the `@Juniper` tag
- `orion_day_letter.py` email sender: refactor out a `send_email` helper so it can be reused
- `scripts/outreach_door_policy.py`: `not_for_juniper` rows become consult candidates
- `orion/curiosity/worldview.py`: decision and outcome evidence on a `:Prior`
- `services/orion-sql-writer`: consult tables
- `services/orion-curiosity-peer/evals/run_consult_brief_gate_eval.py` (new)

## Non-goals

- Claude does not edit code, open PRs or run shell commands in a consult.
- No DB credentials in the peer container.
- No chat history, journals or Orion system prompt sent to Claude.
- No new learning ledger. Learning uses the existing beliefs.
- The Cursor peer path is not fixed here. It stays as it is; `consult` mode does not use it.
- No shadow mode (Juniper's call).

## Acceptance checks

1. Meter on: `orion:substrate:claude_limit` carries a reading, and the dry-run log shows a decision that is not `budget_unobserved`.
2. A consult from a real `not_for_juniper` curiosity row runs at least 2 turns. The turns appear in Hub chat, in the consult room.
3. Claude's turn quotes at least one file and line it actually read. Proof: the CLI's JSON shows a `Read` or `Grep` tool use.
4. A brief with a missing part or an unsourced cost number is refused, and the refusal names the part. Proof: a unit test and the eval.
5. A delivered brief tags `@Juniper` in chat, and one email arrives.
6. A Yes, No or No-because click writes `ConsultDecisionV1` and adds decision evidence to the belief.
7. An `answered` consult sends nothing to Juniper and increases the source belief's `times_tested`.
8. Context rule: a test refuses a `ConsultRequestV1` that carries any field outside the allow-list.
9. Budget: when the meter reads 80% or more, or has no reading, no consult starts. Proof: a unit test.
10. Eval: replay the last 14 days of `not_for_juniper` rows. Report how many would start a consult and how many briefs the cap lets through.

## Recommended next patch

1. **Patch 1:**
   - Turn on the meter.
   - Add the consult mode in the peer service: schemas, channels, tables, the brief gate, the context allow-list, the Hub room, the tag, the email, and the three decision buttons.
   - Add decision evidence on beliefs.
2. **Patch 2:**
   - Add the outcome check 7 days after a change ships.
   - Move approved items to the promise ledger once #3188 is built.

## Rollback

- `CURIOSITY_PEER_CONSULT_ENABLED=false` stops new consults. Running consults end at their next turn.
- The tables are additive. Removing the feature leaves them in place.
