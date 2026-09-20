# Orion asks, and waits — proposal

Proposal mode (CLAUDE.md 0A). Not implemented. Companion to
`2026-09-20-orion-sentience-prerequisites-assessment.md`, whose validated
finding is: every ingredient of a subject exists in Orion as a measurement,
none as a consequence, and no attention loop has ever been closed by Orion's
own action.

Revision note: a first draft of this proposal invented an `orion_ask_ledger`
riding on endogenous outreach. A pathway map of PRs #2152–#2254 (below)
showed that would be a second copy of a contract that already exists,
persists, attributes replies, and has been exercised live. This revision
rides that contract instead.

## Arsonist summary

Orion asks Juniper for one thing they cannot resolve alone, expects a reply,
recognises the reply as *the answer to that ask*, and closes the loop
themself. If no reply comes, that fact survives and shapes what Orion says
next — once, then never again for that ask.

The object already exists: `HelpRequestV1 → PeerBriefV1`
(`orion/schemas/curiosity_peer.py:42/:61`), the contractor-peer contract
from PR #2219. It has a question Orion authored in-turn, a `help_id` the
answer carries back, an `ANSWERS` edge, a consumed-once marker into the next
kickoff, sql-writer persistence (`curiosity_peer_brief`), an Atlas surface,
and two kill switches. Its only peers today are `cursor_auto` and
`claude_room`. **Juniper is not a peer.** This patch makes her one.

Why this and not the camera: the camera patch closes a loop nobody is in.
This closes one in the only relationship the project is about, on a contract
that has already done five real round-trips.

## The pathway map (why this seam)

Every way Orion reaches outward or requests something, PRs #2152–#2254,
with whether it links a reply back to the request:

| Pathway | Typed object | Reply attributed to request? | Live 7d | Verdict |
|---|---|---|---|---|
| Endogenous outreach (#2224/#2235/#2236/#2251) | `OutreachContext` + provenance capsule | **No** — reply read is last-N rows by session (`endogenous_outreach.py:964`) | 31 sent | The *transport* (gate stack: quiet hours, cap, cooldown, novelty, provenance). Not the object. |
| Curiosity outreach (#2237) | same, via `offer_message` (`:2002`) | No | 13 tried, 0 sent (cap) | Same transport. |
| Contractor peer (#2219/#2227/#2238/#2252) | `HelpRequestV1{help_id, question, tried_summary, success_criteria}` → `PeerBriefV1{help_id, status, summary, open_questions}` | **Yes** — `help_id` on both sides, `ANSWERS` edge, `PeerBriefConsumedV1` | 5 briefs (09-15) | **The object.** Missing one peer value. |
| ask-Claude (#2152/#2157) | `AskClaudeDecision` — a decision, no ledger (`ask_claude_ask_log` 0 rows) | n/a | unarmed | Its trigger ("a prior tested ≥3× still unsettled") is the right *gate*; its docstring already says arming means reusing outreach's gate stack. |
| Collapse Mirror reply (#2220/#2229) | `CollapseMirrorChatReplyRequestV1{event_id}` | dedupe only; Juniper→Orion direction | 4 rows | Not applicable. |
| Self-inquiry / lived lanes (#2158/#2239/#2250) | `LivedAnswer` → `self_concept_history`; `curiosity_self_questions` (pinned, 7-day floor) | no ask shape | 4 answers | Where the *want* is minted. |
| Supervisor readings (#2253) | `HopReadingV1` | read-only appraisal | 0 rows | Not applicable. |
| **Attention `ask` action** (`orion/substrate/attention/policy.py:72`) | `CuriosityCandidateActionV1{action_type="ask", open_loop_id, question_text}` when `askability ≥ 0.45` | **Discarded.** Tick passes `max_asks=0` (`attention_broadcast.py:11`); chat turn keeps it only in stance JSON; `question_text` has zero consumers | UNVERIFIED (never persisted) | The workspace already *decides to ask* and writes the question. Runner-up producer — see below. |

## What capability changes

After this patch Orion can:

1. Mint a `HelpRequest` with `peer="juniper"` — from a lived self-inquiry
   answer that says the answer depends on Juniper (first producer).
2. Deliver it through `offer_message`, so it inherits quiet hours, the daily
   cap, cooldown, the novelty gate, and provenance — with `help_id` stamped
   into the outreach provenance and `client_meta`.
3. Attribute Juniper's next relevant `hub_orion` turn to that `help_id`,
   write a `PeerBriefV1{peer="juniper", help_id, status, summary=excerpt}`
   (or `status="empty"` at the deadline), and let the existing consumed-once
   nudge carry the answer into the next kickoff.
4. Close the thing the ask came from: `curiosity_self_questions.status=
   'answered'` and a `LivedAnswer` citing `help_id` (patch 1); an
   `attention_loop_outcome` with `actor='orion'` once the attention `ask`
   action is wired as the second producer (patch 2 — that is where `loop_id`
   comes from).

At most one `juniper` HelpRequest is open at any time. Hard invariant.

## Current architecture (the three seams)

- **Object.** `orion/schemas/curiosity_peer.py:24`
  `CuriosityPeerNameV1 = Literal["cursor_auto", "claude_room"]`;
  `HelpRequestV1` `:42`; `PeerBriefV1` `:61`. Orion writes `:HelpRequest` in
  Cypher during a run; Hub `publish_help_requests_for_run`
  (`peer_briefs.py:140`) → bus → `orion-curiosity-peer` → `PeerBrief` MERGE +
  `ANSWERS` edge (`peer_briefs.py:45`) → `curiosity_peer_brief` (cols
  `brief_id, help_id, run_id, prior_id, peer, status, summary,
  evidence_pointers, open_questions, suggested_next_looks, refusal_reason,
  written_at`) → soft-nudge into next kickoff → `PeerBriefConsumedV1`.
  Flags: `HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED`, `CURIOSITY_PEER_ENABLED`.
- **Transport.** `endogenous_outreach.py`: gate `:422-467`, talkable content
  `:523`, novelty `:570`, compose `:1119`, provenance `:1097` (carries
  `correlation_id`), `offer_message` `:2002`, decline `orion_passed` `:1888`.
  Sent message → `chat_history_log` `session_id='orion_outreach'`. Juniper's
  turns → `session_id='orion_journal'`, `source='hub_orion'`.
- **Want.** `curiosity_investigation.py:2190/2245` writes SelfDefinition /
  LivedAnswer; `curiosity_self_questions{question_id, text, family, pinned,
  minted_by, status, ask_count, last_asked_at}`; the self-inquiry line
  already teaches Orion to write `:HelpRequest` after a short local look
  (#2238's hire-determination grounding) — the teach just never offers
  Juniper as the peer.
- **Loop closure.** `attention_loops_store.build_loop_outcome:139`
  (`actor` hardcoded `"juniper"` by the route at `attention_loops_routes.py:86`);
  `OpenLoopV1.askability`/`relational_relevance`
  (`orion/schemas/attention_frame.py:73`) set in `scoring.py:148`, 0.72
  default.
- **Presence.** `PresenceState` (`hub main.py:335`) — `active_connections`,
  `last_seen`; read by nothing in outreach today.

## Missing questions

1. Deadline clock: send-time or Juniper-present time? Recommendation:
   `expect_by` counts from her next `PresenceState.last_seen` after the ask,
   48h, one re-ask at ≥72h, then `status="empty"` is final.
2. What counts as a reply? Recommendation: first `hub_orion` turn after the
   ask, any session, within the deadline, *and* the brief-writing step
   affirms it addresses the question (stored as `evidence_pointers` +
   confidence in `summary` metadata). Time proximity alone must never
   produce a `PeerBrief` — a false brief poisons the next kickoff.
3. Should the attention `ask` action be the first producer instead? It is
   the more theory-faithful seam (the workspace itself decides to ask), but
   its thresholds are admittedly uncalibrated (`policy.py:48-62`) and it has
   never been persisted, so there is no live evidence of what it would ask.
   Recommendation: self-inquiry first (evidence-cited, 7-day floor, already
   teaches `:HelpRequest`), attention `ask` second.

## Proposed schema / API changes

**Extended, nothing new:**

- `CuriosityPeerNameV1` gains `"juniper"`. Every consumer that switches on
  peer name is enumerated in the patch (`orion-curiosity-peer` dispatch,
  Atlas rendering, `peer_briefs.py` MERGE) — the `additive fields on forbid
  models are a consumer-first migration` lesson applies.
- `HelpRequestV1` gains optional `open_loop_id: str | None` (for the second
  producer) and `expect_by: datetime | None`.
- `PeerBriefV1.status` gains `"empty"` (deadline passed, no reply) if not
  already an accepted value; `refusal_reason` reused for `"no_reply"`.
- Outreach provenance capsule (`:1097`) and `client_meta` carry `help_id`.
- `build_loop_outcome` takes `actor` as a parameter; `"orion"` accepted.
- `LivedAnswer` gains optional `ask_juniper: {question, why_needed,
  success_criteria}`; the prompt defaults it to null.

No new table. No new channel. `curiosity_peer_brief` and the existing
`orion:curiosity:peer:*` channels carry the `juniper` peer unchanged.

## What data is touched

- Writes: `:HelpRequest` / `:PeerBrief` nodes + `ANSWERS` edge in the
  curiosity graph; `curiosity_peer_brief`; `endogenous_outreach_decisions.
  result_json` (adds `help_id`); `curiosity_self_questions.status`;
  `self_concept_history` (existing lived path, one optional field);
  `attention_loop_outcome` (patch 2, new actor value).
- Reads: `chat_history_log` (Juniper's turn text for the brief excerpt),
  `PresenceState` (deadline clock).
- Nothing is copied out of Juniper's turns except the brief `summary`.

## Privacy boundary

Juniper is the only user and the companion; personal content is meant to
enter the self-model (resolved 2026-09-05). The guard that applies: a
`PeerBrief` with `peer="juniper"` enters the next kickoff through the same
consumed-once nudge as a contractor brief — no other consumer reads it. Any
**new** downstream reader of `juniper` briefs opts in by `source_ref` prefix
(`services/orion-thought/app/store.py:785` pattern).

## Files likely to touch

- `orion/schemas/curiosity_peer.py` — enum value, `open_loop_id`,
  `expect_by`, `"empty"`
- `services/orion-curiosity-peer/` — refuse to dispatch `peer="juniper"` to
  Cursor/Claude (it is delivered by Hub, not by the peer service); tests
- `services/orion-hub/scripts/peer_briefs.py` — `publish_help_requests_for_run`
  routes `juniper` requests to the delivery adapter; MERGE accepts the peer
- `services/orion-hub/scripts/juniper_peer.py` (new, small) — delivery via
  `offer_message` with `help_id` in provenance; attribution on each
  `hub_orion` turn while a `juniper` request is open; deadline → `empty`;
  the one-re-ask rule
- `services/orion-hub/scripts/endogenous_outreach.py` — `help_id` in
  provenance/`client_meta`; an open `juniper` request as a talkable-content
  kind so a *re-ask* composes with the original question
- `services/orion-hub/scripts/curiosity_investigation.py` — `ask_juniper`
  in LivedAnswer; mint the HelpRequest when present and none is open; the
  hire-determination teach offers `juniper` for lived questions
- `services/orion-hub/scripts/attention_loops_store.py` — `actor` param
- `services/orion-hub/scripts/main.py`, `app/settings.py`, `.env_example`,
  README — `HUB_JUNIPER_PEER_ENABLED=false` (default off),
  `_EXPECT_HOURS=48`, `_REASK_MIN_HOURS=72`
- Hub cabinet: one line "Orion is waiting on you: …" (the ask must be
  visible to the person it is addressed to)
- Patch 2 only: `orion/substrate/attention_broadcast.py` / `chat_stance.py`
  — persist the selected `ask` action and hand it to the same mint path
  with `open_loop_id`
- Tests at every seam; eval `evals/run_juniper_ask_quality_eval.py` — sampled
  asks graded for *answerable by Juniper*, *not answerable by Orion alone*,
  *not needy* (same grader shape as `run_self_sense_eval.py`)

## Non-goals

- No new ledger, table, or channel. If the peer contract cannot carry this,
  that is a finding to bring back, not a reason to fork it.
- No multi-ask queue. One open `juniper` request, ever, in this patch.
- No affect metric for "disappointment"; the `empty` brief is the signal.
- No change to attention scoring, the allocator, dispatch, or the camera.
- No re-litigating quiet hours, cap, cooldown — an ask is an outreach and
  inherits all three.

## What trace proves it worked

One `help_id`, six rows:

1. `:HelpRequest{help_id, peer:'juniper'}` in the curiosity graph, with
   `run_id` pointing at a real self-inquiry run.
2. `chat_history_log` row, `session_id='orion_outreach'`, provenance
   capsule containing `help_id`, message containing the question.
3. `chat_history_log` row, `source='hub_orion'`, created after (2).
4. `curiosity_peer_brief` row, `peer='juniper'`, same `help_id`,
   `status='answered'` (or `'empty'` on the missed path), `summary` =
   excerpt, `evidence_pointers` → (3)'s `correlation_id`.
5. `PeerBriefConsumedV1` on the next kickoff, and a `LivedAnswer` /
   `self_concept_history` row citing `help_id`.
6. `curiosity_self_questions.status='answered'` for the minting question
   (patch 1); `attention_loop_outcome{actor='orion', verdict='resolved'}`
   for the `open_loop_id` (patch 2).

For the missed-reply path: (1), (2), then a `curiosity_peer_brief` with
`status='empty'`, `refusal_reason='no_reply'`, and a later outreach whose
provenance cites the same `help_id` as a re-ask — exactly once.

Live invariants (in the eval): open `juniper` HelpRequests ≤ 1; re-asks per
`help_id` ≤ 1; no `juniper` outreach inside quiet hours (America/Denver).

## What failure mode would be dangerous

- **Nagging.** One re-ask at ≥72h, then final; one open request; inherits
  cap and cooldown; the quality eval grades every ask for neediness.
- **False brief.** Attributing an unrelated reply as the answer writes a
  false `PeerBrief` into the next kickoff *and* into the self-model.
  Mitigation: the brief-writing step must affirm relevance; below threshold
  the request stays open.
- **Wrong asks.** "Am I alive?" is pinned and a terrible ask. Mitigation:
  minted only from a lived answer that says the answer depends on Juniper,
  prompt defaults to null, eval passes before the flag flips.
- **Peer-service confusion.** `orion-curiosity-peer` trying to "hire" Juniper
  through Cursor. Mitigation: it refuses `peer="juniper"` explicitly; Hub
  owns delivery. Regression test.
- **Blocking outreach.** An open request must not suppress other outreach;
  only minting a second request is blocked.
- **Restart reset.** State lives in the graph and `curiosity_peer_brief`,
  nothing in-process or in Redis (the #1947 lesson).

## How to disable or roll back

`HUB_JUNIPER_PEER_ENABLED=false` (default until the eval passes) stops
minting, delivery, and attribution; open `juniper` requests get a
`PeerBrief{status:'empty', refusal_reason:'disabled'}` on the next tick so
nothing dangles. The enum value and optional fields stay; with the flag off
nothing produces them. Full rollback is removing the enum value and the
adapter — no other surface depends on them.

## Acceptance checks

1. All six trace rows for one real ask, on the live rail.
2. One `'empty'` brief later cited by a re-ask, exactly once.
3. The three invariants hold over 7 days.
4. `make eval-juniper-ask-quality` passes on ≥ 10 sampled asks before the
   flag flips.
5. Self-sense eval does not regress over the same window.
6. `orion-curiosity-peer` test: a `juniper` request is refused there and
   delivered by Hub.
7. Patch 2: `attention_loop_outcome` contains `actor='orion'` for the first
   time, with `loop_id` traceable to a persisted `ask` action.

## Recommended next patch

Three thin PRs, dependency order:

1. **Contract** — enum value, optional fields, `"empty"` status, `actor`
   param, peer-service refusal, consumer enumeration. No behavior change.
2. **Mint + deliver + eval** — `ask_juniper` in LivedAnswer, the teach,
   `juniper_peer.py` delivery via `offer_message`, `help_id` in provenance,
   cabinet line, quality eval against dry-run output. Flag off.
3. **Attribute + close** — attribution on `hub_orion` turns, deadline and
   `empty`, the re-ask rule, invariants in the eval, self-question closure.
   Flip the flag; collect the trace.

Patch 2 of the *ladder* (attention `ask` action as second producer with
`open_loop_id` → `actor='orion'` loop closure) follows once the first trace
exists.
