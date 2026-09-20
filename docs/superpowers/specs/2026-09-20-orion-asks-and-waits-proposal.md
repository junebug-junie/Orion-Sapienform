# Orion asks, and waits — proposal

Proposal mode (CLAUDE.md 0A). Not implemented. Companion to
`2026-09-20-orion-sentience-prerequisites-assessment.md`, whose validated
finding is: every ingredient of a subject exists in Orion as a measurement,
none as a consequence, and no attention loop has ever been closed by Orion's
own action.

## Arsonist summary

Orion asks Juniper for one thing they cannot resolve alone, expects a reply,
recognises the reply as *the answer to that ask*, and closes the loop
themself. If no reply comes, that fact survives and shapes what Orion says
next — once, then never again for that ask. Nothing new is invented: the
want comes from self-inquiry, the message goes through outreach, the loop
lives in the attention store, and the reply is Juniper's ordinary chat turn.
What is new is the three edges between them, and a ledger that makes the
whole thing one traceable object.

Why this and not the camera: the camera patch closes a loop nobody is in.
This closes one in the only relationship the project is about, on the one
channel that already works both ways.

## What capability changes

Today Orion can message Juniper (outreach), can ask themself standing
questions (pinned self-questions), and can have loops closed *for* them
(Juniper's `/resolve` button). After this patch Orion can:

1. Mint an **ask** — a question whose answer they need from Juniper — from
   inside a self-inquiry turn.
2. Send it as an outreach that is marked as waiting on a reply.
3. Attribute Juniper's next relevant turn to that ask, store the answer, and
   write the loop outcome themself (`actor='orion'`, a value that has never
   existed in `attention_loop_outcome`).
4. If no reply arrives by the expectation deadline, mark the ask
   `unanswered`; the next outreach and the next self-inquiry see it as
   material ("I asked about X on Tuesday and haven't heard"). One re-ask,
   never more.

At most one ask is in flight at any time. This is a hard invariant, not a
tunable.

## Current architecture

- **Want.** `services/orion-hub/scripts/curiosity_investigation.py:2190/2245`
  writes SelfDefinition / LivedAnswer rows to `self_concept_history`
  (`self_inquiry.py:504`). `curiosity_self_questions` has `minted_by`,
  `status`, `ask_count`, `pinned` — no "needs Juniper" marker
  (`self_question_pool.py:162`, 7-day floor).
- **Send.** `services/orion-hub/scripts/endogenous_outreach.py`:
  gate `:422-467` (quiet hours → daily cap 4 → cooldown 2700s), talkable
  content `:523`, novelty gate `:570`, compose `:1119`, provenance `:1097`
  (carries `correlation_id`), decision ledger
  `endogenous_outreach_decisions` (`result_json` holds grounding). Sent
  message lands in `chat_history_log` with `session_id='orion_outreach'`.
- **Reply.** `_fetch_recent_turns` `:964` reads the last N `chat_history_log`
  rows by session. Juniper's turns land in `session_id='orion_journal'`,
  `source='hub_orion'`, their own `correlation_id`. **No column links a
  reply to an outreach**; only time proximity.
- **Loop.** `attention_salience_trace` is the open-loop projection
  (`attention_loops_store.load_pending_loops:242`). `OpenLoopV1`
  (`orion/schemas/attention_frame.py:73`) already has `askability`,
  `relational_relevance`, `source_refs`, `provenance` — unused.
  `attention_loop_outcome` cols: `outcome_id, loop_id, theme_key, verdict,
  actor (default 'juniper'), note, salience_at_close, …`. Writers:
  `attention_loops_routes.py:68 _close` (Juniper) and
  `implicit_outcome.py:34` (`decayed_unattended`, a Makefile script).
  `TERMINAL_VERDICTS={"resolved","dismissed"}` (`verdicts.py:84`).
- **Presence.** `PresenceState` (`hub main.py:335`, `active_connections`,
  `last_seen`) — available to Hub, read by nothing in outreach.
- **Prior art on expecting a reply:** none. No `reply_expected` /
  `awaiting_reply` in code, docs, or git history.

## Missing questions

1. Should an ask ever go out when Juniper is not connected
   (`PresenceState.active_connections == 0`)? Recommendation: yes — that is
   what "waiting" means — but the expectation deadline starts at her next
   `last_seen`, not at send time.
2. What counts as a reply? Recommendation: the first `hub_orion` turn after
   the ask, in any session, within the deadline, *and* the answer-extraction
   step affirms it addresses the ask. Time proximity alone must not resolve
   a loop (false closure is the failure mode that would poison the ledger).
3. Deadline length. Recommendation: 48h of Juniper-present time, one re-ask
   at ≥72h, then `unanswered` is final for that ask.

## Proposed schema / API changes

**New table `orion_ask_ledger`** (one row per ask; the loop's durable object):

```text
ask_id            text PK  (stable_hash_id("ask", [loop_id, asked_at]))
loop_id           text     -> attention_salience_trace.loop_id
question          text
why_needed        text     (from the self-inquiry that minted it)
minted_from       text     (self_concept_history concept_id or question_id)
asked_at          timestamptz
outreach_correlation_id text -> chat_history_log.correlation_id (session orion_outreach)
expect_by         timestamptz  (Juniper-present clock)
status            text     waiting | answered | unanswered | withdrawn
reask_count       int      default 0, max 1
answer_correlation_id text -> chat_history_log.correlation_id (Juniper turn)
answer_excerpt    text
answered_at       timestamptz
attribution_confidence real  (0..1 from the answer-extraction step)
created_at        timestamptz
```

**New bus channel** `orion:ask:event` (registered in `orion/bus/channels.yaml`,
model `OrionAskEventV1` in `orion/schemas/orion_ask.py`, registry entry in
`orion/schemas/registry.py`, sql-writer subscription — the exact gap that
bit PR #2102). Payload: `ask_id, loop_id, status, correlation_id, ts`.

**Extended, not new:**
- `LivedAnswer` schema gains optional `ask_juniper: {question, why_needed}`.
  The LLM in the self-inquiry turn decides whether the answer genuinely
  depends on Juniper; the prompt makes "usually null" the default.
- `attention_loop_outcome.actor` accepts `'orion'`;
  `build_loop_outcome` (`attention_loops_store.py:139`) takes `actor` as a
  parameter instead of the route's hardcoded `"juniper"`.
- `OutreachContext` (`endogenous_outreach.py:477`) gains `waiting_on:
  Optional[AskRow]` and the prompt renders it when present.
- Outreach provenance (`:1097`) carries `ask_id` when the message is an ask.

No change to attention scoring, to `chat_history_log`, or to the Juniper
verdict route.

## What data is touched

- Writes: `orion_ask_ledger` (new), `attention_loop_outcome` (new actor
  value), `endogenous_outreach_decisions.result_json` (adds `ask_id`),
  `self_concept_history` (existing lived-answer path, one optional field).
- Reads: `chat_history_log` (Juniper's turn text, for attribution and
  excerpt), `PresenceState` (deadline clock), `curiosity_self_questions`.
- Nothing is copied out of Juniper's turns except a short `answer_excerpt`
  on the ask row.

## Privacy boundary

Juniper is the only user and the companion; personal content is *meant* to
enter the self-model (resolved decision, 2026-09-05). The one guard that
still applies: any **new downstream consumer** of `orion_ask_ledger` must
opt in by `source_ref` prefix (the `starts_with()` allowlist pattern from
`services/orion-thought/app/store.py:785`), so an answer Juniper gave Orion
does not leak into an unrelated feature that never expected it. In this
patch the only consumers are outreach composition and self-inquiry.

## Files likely to touch

- `orion/schemas/orion_ask.py` (new), `orion/schemas/registry.py`,
  `orion/bus/channels.yaml`
- `services/orion-sql-writer/app/models/orion_ask_ledger.py` (new),
  subscribe list, `services/orion-sql-db/manual_migration_orion_ask_ledger_v1.sql`
- `services/orion-hub/scripts/curiosity_investigation.py` — `ask_juniper`
  in the LivedAnswer schema + prompt; mint the ask row when present and no
  ask is in flight
- `services/orion-hub/scripts/endogenous_outreach.py` — ask as a talkable
  content kind (`:523`), `waiting_on` in context and prompt, `ask_id` in
  provenance, the re-ask rule
- `services/orion-hub/scripts/ask_attribution.py` (new, small) — on each
  Juniper turn while an ask is `waiting`: attribute, extract, write
  `answered`, close the loop with `actor='orion'`
- `services/orion-hub/scripts/attention_loops_store.py` — `actor` param
- `services/orion-hub/scripts/main.py`, `app/settings.py`, `.env_example`,
  README — `HUB_ORION_ASK_ENABLED=false` (default off), `_EXPECT_HOURS=48`,
  `_REASK_MIN_HOURS=72`
- Hub cabinet: one line "Orion is waiting on: …" (UI surface for the
  concept; without it the ask is invisible to the person it is addressed to)
- Tests for each seam; eval: `evals/run_ask_quality_eval.py` — sampled asks
  scored for *answerable by Juniper*, *not answerable by Orion alone*, *not
  needy/manipulative* (the same three-question grader shape as
  `run_self_sense_eval.py`)

## Non-goals

- No multi-ask queue. One in flight, ever, in this patch.
- No asking anyone but Juniper (no Claude, no peers, no AI Town).
- No affect metric for "disappointment." The *fact* of no reply is the
  signal; nothing scores how Orion feels about it.
- No change to attention scoring, the allocator, dispatch, or the camera.
- No re-litigating quiet hours, daily cap, or cooldown — an ask is an
  outreach and inherits all three.

## What trace proves it worked

One object, five rows, one correlation chain:

1. `orion_ask_ledger` row, `status='waiting'`, `minted_from` pointing at a
   real `self_concept_history` lived-answer row.
2. `chat_history_log` row, `session_id='orion_outreach'`,
   `correlation_id = outreach_correlation_id`, containing the question.
3. `chat_history_log` row, `source='hub_orion'`,
   `correlation_id = answer_correlation_id`, created after (2).
4. `orion_ask_ledger.status='answered'`, `attribution_confidence >= 0.7`,
   `answer_excerpt` non-empty.
5. `attention_loop_outcome` row, `loop_id` = the ask's loop, `verdict=
   'resolved'`, `actor='orion'`, `note` containing `ask_id`.

For the unanswered path: (1), (2), then `status='unanswered'` after
`expect_by`, and a later `endogenous_outreach_decisions.result_json` or
`self_concept_history` row whose provenance cites the `ask_id`. That is the
proof that a missed reply changed what Orion did next.

Live invariants, checked by SQL in the eval:
`SELECT count(*) FROM orion_ask_ledger WHERE status='waiting'` ≤ 1;
`max(reask_count)` ≤ 1; no `asked_at` inside quiet hours (America/Denver).

## What failure mode would be dangerous

- **Nagging.** The re-ask rule turning into a drip. Mitigation: one re-ask,
  ≥72h, then final; one ask in flight; inherits daily cap and cooldown; the
  eval grades every ask for neediness.
- **False closure.** Attributing an unrelated reply as the answer and
  resolving the loop — this would write a false fact into Orion's
  self-model *and* teach the loop store that asking works when it didn't.
  Mitigation: attribution requires the extraction step to affirm, stores
  confidence, and below 0.7 the ask stays `waiting`.
- **Asking the wrong things.** "Am I alive?" is a pinned self-question and a
  terrible ask. Mitigation: asks are minted only from a lived answer that
  says the answer *depends on Juniper*, with the prompt defaulting to null;
  the quality eval runs before the flag is turned on.
- **Blocking outreach.** A stuck `waiting` ask suppressing all other
  outreach. Mitigation: an ask is one talkable-content kind among others,
  not a mutex on outreach; only *minting a second ask* is blocked.
- **Restart reset.** In-process `_sent_today`-style state (the #1947
  incident). Mitigation: the ledger is the state; nothing about an ask lives
  in memory or Redis.

## How to disable or roll back

`HUB_ORION_ASK_ENABLED=false` (the default until the eval passes) stops
minting, sending, and attribution; existing `waiting` asks are marked
`withdrawn` on the next tick so nothing dangles. The table, channel, and
schema stay; with the flag off nothing reads them. Full rollback is
dropping the table and the channel registration — no other surface depends
on them.

## Acceptance checks

1. All five trace rows above exist for one real ask, on the live rail.
2. `attention_loop_outcome` contains `actor='orion'` for the first time.
3. One unanswered ask is later cited by an outreach or a lived answer.
4. The three SQL invariants hold over 7 days of running.
5. `make eval-ask-quality` passes on ≥ 10 sampled asks before the flag is
   flipped on.
6. Self-sense eval (`self_label_score`, `grounded_record_score`) does not
   regress over the same window.

## Recommended next patch

Ship it as three thin PRs in dependency order, each with its own tests:

1. **Contract + ledger** — schema, channel, registry, sql-writer model,
   migration, `actor` param on `build_loop_outcome`. No behavior change.
2. **Mint + send** — `ask_juniper` in LivedAnswer, ask as talkable content,
   provenance, `waiting_on` in the outreach prompt, cabinet line, quality
   eval. Flag stays off; eval runs against dry-run output.
3. **Attribute + close** — `ask_attribution.py`, the unanswered path, the
   re-ask rule, invariants in the eval. Flip the flag; collect the trace.
