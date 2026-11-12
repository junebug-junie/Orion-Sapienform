# Intention ledger (step D): Orion keeps track of what Orion promised

Status: design (proposal mode), 2026-11-11. **Approved by Juniper 2026-11-12, every default
accepted** (see "Juniper's answers"). Next patch: D0, the replay eval.
Arc: unified-turn self-prediction, PR #2869
(`docs/superpowers/specs/2026-10-23-unified-turn-self-prediction-design.md`), step D:
"a reducer over turn outcomes keeps what Orion meant, did, left open and promised, and the next
turn reads it. Tier 1." Step F later needs D to keep "wait" as a promise.

## In one paragraph

Three of the four things step D names are already recorded for every turn. What Orion **meant**
is the stance plan. What Orion **did** is the turn record. What Orion **left open** has two
records, but neither one is about the conversation. Only **promises** have no record at all.
Orion makes a forward promise to Juniper in about 1 turn in 19. The promise falls out of the
8-turn chat window, and nothing brings it back. This design records each promise as an act
on the existing act ledger. The act's test is "kept by the horizon". The finalize judge
already reads every reply, so the judge finds the promise and later marks it kept. Stance
sees the open promises before each turn. The act ledger and the self-monitor organ already
grade acts, so Orion gets a promise-keeping track record with no new grader. **No new table.**

## Arsonist summary

- Do not build a fourth store. "Meant" and "did" are already one join away
  (`thought_decision` ⋈ `harness_turn_trace` on `correlation_id`). A ledger that copies them
  is a second place for the same facts to drift.
- "Open loops" sound like the right home for "left open", but they are not. All five
  loops that stance cites most are body-signal themes, for example "Execution prediction
  error" and "Chat prediction error". None is a conversational thread.
- Promises are real but rare: 5 of Juniper's 95 turns in 14 days held one (6 promise
  statements). At most 1 of them shows a later follow-up in chat. The count is small, so this is a
  measurement problem as much as a design problem. The design must say how it will know it
  helped (see Acceptance checks).
- Social-memory already has a promise detector. It is a word list
  (`services/orion-social-memory/app/synthesizer.py:108-127`). It has made **1 commitment
  in 67 rooms** across 18k room turns. Do not reuse it, and do not copy its method.
- The dangerous failure is a false promise. If the judge invents a promise, stance pushes
  Orion to follow up on something Orion never said. Every promise and every "kept" mark
  must quote text that really appears in the reply. Plain code checks the quote.

## Current architecture (traced 2026-11-11)

| step D asks for | where it lives today | who reads it | gap |
|---|---|---|---|
| what Orion **meant** | `thought_decision.imperative`, `stance_harness_slice`, `self_prediction` (stance, via sql-writer). 3,022 of 3,031 rows in 14 days have an imperative. | the motor (`orion/harness/prefix.py`), the 5b judge, `prediction_turn_self_bet` | none |
| what Orion **did** | `harness_turn_trace.run_artifact` (`step_count`, `source_fetches`, `final_text`, `exit_code`, `stop_reason`, `reflection.alignment_verdict`) | orion-prediction settles bets (`services/orion-prediction/app/turn_self_bets.py`); the self-monitor organ grades them (`orion/self_monitor/adapters.py:130` `turn_self_bet`, `:166` `turn_run`) | none |
| what Orion **left open** (1) | attention open loops: `OpenLoopV1` (`orion/schemas/attention_frame.py:112`), id = hash of target text (`orion/substrate/attention/scoring.py:115`). They close only by Juniper's Resolve/Dismiss, by implicit decay, or by Orion's non-final `acted` (`attention_loop_outcome`; reader `orion/substrate/attention/verdicts.py`). The temporal-self frame shows them as `open_threads` (`orion/temporal_self/frame.py:111`). The stance cue renders them (`orion/temporal_self/stance_cue.py:341`). | stance (`strain_refs`), reverie, temporal self | they are body-signal themes, not conversation |
| what Orion **left open** (2) | `harness_turn_trace.closure` (`HarnessPostTurnClosureV1`, `orion/schemas/harness_finalize.py:163`; producer `orion/harness/finalize.py:1904`). `surprise_unresolved` plus the imperative excerpt. | reverie (`orion/reverie/referent_loader.py`) | the next turn never sees it |
| what Orion **promised** | nothing on the unified turn. Social rooms only: `SocialCommitmentV1` (`orion/schemas/social_commitment.py`), word-list producer `extract_commitments` (`synthesizer.py:2313`). | social-memory | missing |

The two mechanisms this design reuses:

- **The act ledger** (thinking-arch rung 2, `orion/grammar/act_ledger.py`). An act opens with
  an expectation (`ActExpectV1`: a signal, a horizon and a basis). sql-writer stamps the
  current posterior mean as `predicted` at open (`open_act`, `:347`). The act closes by its
  producer (`close_act`, `:478`) or, at the horizon, as a real 0 (`resolve_due_acts`,
  `:826`). Every close is graded. The Hub already opens one act per reply to Juniper:
  `hub.chat_reply say juniper_replied`, 30-minute horizon (`_publish_reply_say_act`,
  `orion/hub/turn_orchestrator.py:1997`). There were 101 such acts in 14 days, with a mean
  outcome of 0.61.
- **The self-monitor organ** turns every closed act into a `SettledBetV1` with no extra code:
  `grammar_act` (`orion/self_monitor/adapters.py:100`), faculty
  `act.<actor>.<verb>[.<name>]`. Its "MY TRACK RECORD" stance cue
  (`services/orion-thought/app/bus_listener.py:626`) already shows notable faculties to stance.

## What live data showed (Postgres, 14 days to 2026-11-11 04:00 UTC)

**Who the "chat" turns are.** Stance placed bets on 446 chat-caller turns. Only 95 are
Juniper's (a `chat_history_log` row with a prompt). 152 are messages that Orion started
itself (outreach, reading findings). 200 never reached chat history. Every one of Juniper's
95 turns has a turn record. None of them died. 86 of the 95 ran the 5b judge with a model
pass. The other 9 went through the quick deterministic gate.

**Meant vs did.** Every turn has an imperative. Of the 389 chat-caller turns with a record,
69 were killed (`exit_code -9`, 38 of them `outreach_agent_lane_timeout`), and the judge
marked 31 `misaligned`. All of the killed turns were Orion-started. "Did it finish?" is
already graded as the `turn.run` faculty. The ledger does not need to repeat it.

**Left open.** Stance cited 38 different open loops in 14 days. The most-cited loop is
"Execution prediction error". Stance cited it in 198 rows, and attention traced it 240 times
in the same 14 days. Its last verdict was an implicit decay in September. None of the 38
loops got a resolve or dismiss from Juniper. 15 decayed and 8 got Orion's own `acted`.
These loops are body-signal pressure, not things Orion said Orion would do. 852 of 2,676
turns (32%) closed with `surprise_unresolved`. Only reverie reads that.

**Promised.** A rough scan of the replies to Juniper's 95 turns found 6 forward promise
statements in 5 turns (this scan was for measurement only, and it is not a design proposal).
In paraphrase:
- "It gets first look at tomorrow's run" (11-03, a promise about a scheduled run);
- "I'll hold the deep read until the pipeline digests it, then pull it back" (11-04);
- "Ping me after it finishes, or I'll bring what surprised me next time we're in a technical thread" (11-04);
- "I'll have a considered position on the open questions once the read lands" (11-05);
- "If you want, I'll re-read it the next time I'm cut" (11-06);
- "I'll go look properly if you want, but I'm not chasing it tonight" (11-08).

The same scan found one more in a turn that never reached chat history (10-28: "I'll have
more to say when the model cards get read").

All of Juniper's promise turns were in one long session. That session had 53 to 165 more
messages after each promise, so each promise left the 8-turn window
(`orion/schemas/pre_turn_appraisal.py:40`) within hours. In the later replies, a text search
found a later reference for 1 promise (11-05, 12 hours later). It found no reference for
the two 11-04 promises. Follow-through outside chat (for example a read that ran) is
`UNVERIFIED` here.

## Design

### D1. Producer: the 5b judge names promises

Add one optional field to `FinalizeReflectionV1` (`orion/schemas/harness_finalize.py:73`). The
model has no `extra="forbid"`, so an old reader drops the new field without error:

```python
class PromiseV1(BaseModel):
    quote: str = Field(max_length=240)   # exact span of the reply that makes the promise

class FinalizeReflectionV1(...):
    promises_made: list[PromiseV1] = Field(default_factory=list, max_length=3)
    promises_kept: list[str] = Field(default_factory=list, max_length=3)  # act_ids
```

- The prompt (`harness_finalize_reflect.j2`) gets one short section. "Promises made" means
  Orion commits to do or say something **after this reply**. An offer that waits for
  Juniper ("if you want, I'll...") is not a promise until Juniper accepts it. The judge
  lists a promise only by quoting it.
- The prompt also gets the open promises (up to 3, each with its act_id and quote). The
  judge returns `promises_kept` only when this reply really delivers what the promise said.
- **Deterministic gate (code, not model):** a `promises_made` quote must be a substring of
  `final_text` after whitespace normalization. A `promises_kept` entry is accepted only when
  it names an open act. Anything else is dropped and counted in the log.
- The field rides the judge call that already runs. It adds no new model call. It does not
  run on the quick deterministic gate. The 9 quick-gate turns are a known blind spot, and
  the log records them.

### D2. Store: one act per promise on the existing act ledger

The Hub's `_publish_reply_say_act` already runs after delivery. It opens one more act per
accepted promise:

| ActV1 field | value |
|---|---|
| `actor` / `verb` | `hub.chat_reply` / `say` |
| `parent_act_id` | the reply's own `juniper_replied` act |
| `object_ref` | `promise:<n>` (faculty becomes `act.hub.chat_reply.say.promise`) |
| `test_before` | `harness_turn_trace:<correlation_id>#promises_made[<i>]` (the quote stays in the turn record, not copied) |
| `expect.signal_id` | **new** `promise_kept` |
| `expect.horizon_sec` | 72 h (default, see Missing questions) |

A kept promise closes as `producer`, value 1, with the note `kept_in:<correlation_id>`. A
promise that passes its horizon closes as `expired`, value 0, through the existing resolver.
That is the same rule as `artifact_produced`. This needs only one contract change: add
`promise_kept` to `ActSignal` (`orion/schemas/grammar.py:65`), consumer-first. This is how
each existing store fails as the home for promises:

- `attention_loop_outcome` / open loops: ids are hashes of attention target text. They are
  closed by humans or decay. They have no horizon and no producer close.
- `harness_turn_trace.closure`: one row per turn. It has no lifecycle across turns.
- `SocialCommitmentV1`: it is keyed by social room. Its producer is a word list, and it lives
  in another service.
- temporal self `expectations_pending`: it already folds `action_outcome` rows when an act
  closes, so a closed promise reaches the day story with no change. It is a projection, not
  a place that opens records.

### D3. Consumer: stance sees open promises

orion-thought already builds the track-record cue next to the stance call (`bus_listener.py:626`).
Add one more small read, `open_promises_cue`, on the turns where Hub opens a reply act
(Juniper's turns). It shows up to 3 open `promise_kept` acts, oldest first:

```
OPEN PROMISES (mine, not yet kept):
- [act:act_1a2b] 2 days ago: "I'll bring what surprised me next time we're in a technical thread"
```

Stance may carry a promise into its imperative. Stance is not told to. The 5b judge gets
the same list (D1). Nothing else reads the list: no outreach trigger, and no motor prefix.

### D4. Grading comes for free

- The self-monitor `grammar_act` adapter turns each close into `act.hub.chat_reply.say.promise`.
  "MY TRACK RECORD" then shows the keep rate when it is notable ("I kept 1 of 6 promises").
- `predicted` (the posterior keep rate, stamped at open) vs the outcome is a calibrated
  self-prediction. It joins the arc's grading through the same organ, not a side ledger.
- Temporal self folds each close as an `action_outcome` event, so the day story can say it.

### Tiers

- Extraction and the "kept" mark ride the 5b judge. On Juniper's turns the judge is Tier 0,
  and the change adds about 150 prompt tokens and at most 3 short output items.
- Expiry is the act-ledger resolver tick in sql-writer: Tier 1 upkeep. There is no GPU.
- The stance read is one indexed SQL query, and it rides the turn's own tier.

## Proposal-mode checklist

- **Capability that changes:** Orion can remember a promise after it falls out of the chat
  window, can see it before the next turn, and learns how often Orion keeps promises.
- **Data touched:** reads `harness_turn_trace.run_artifact.final_text` (already stored).
  Writes `grammar_acts` rows (new signal) and one new reflection field. It deletes nothing.
- **Privacy boundary:** there is none between Juniper and Orion. What leaves the turn: a
  quote of at most 240 characters of **Orion's own** reply. The quote stays inside
  `harness_turn_trace`. The act row holds only a pointer to it. The quote reaches only the
  stance prompt and the 5b judge prompt of later turns with Juniper. It never reaches
  outreach, reverie, or Orion-started turns.
- **Trace that proves it worked:** a `grammar_acts` row with `signal_id='promise_kept'`
  whose `test_before` quote is in that turn's `final_text`. A later turn's stance prompt shows
  the promise (`cockpit_turn_sighting`). The act closes either `producer` with `kept_in:<corr>`
  or `expired`. The self-monitor faculty `act.hub.chat_reply.say.promise` appears.
- **Dangerous failure modes and their guards:**
  1. *False promise* (the judge invents one): the substring gate drops it. The eval
     measures precision before the cue may steer.
  2. *False "kept"* (a self-grade that saturates the keep rate at 100%): an entry must name an
     open act. The eval hand-checks 20 closes. If precision is below 0.8, the "kept" mark
     turns off and only expiry closes promises.
  3. *Stale nagging:* at most 3 shown. Horizon 72 h. Expired acts disappear from the cue.
  4. *Self-reinforcing loop* (the cue makes Orion promise more, which shows more promises):
     guard metric = promises per Juniper turn. If the 14-day rate is more than 2× the
     baseline (5 of 95 turns), that is a red result, and the flag goes off.
  5. *Promises steering outreach:* a non-goal. No code path from the ledger to outreach.
- **Disable / roll back:** `ORION_PROMISE_LEDGER_ENABLED` (orion-hub, orion-thought), shipped
  **ON**. Off: the judge field is ignored, the cue is not rendered, and no acts open. The
  resolver expires open acts as usual. No migration: `grammar_acts.signal_id` is text.

## Missing questions (defaults in brackets)

1. Count "if you want, I'll..." offers as promises? [No, not until Juniper accepts one. In
   live data 2 of the 6 statements were offers.]
2. Horizon? [72 h for every promise. Most promises in the data are "next time we talk" or
   "when X lands". A per-promise deadline from the text is not built: it has no consumer yet.]
3. Include Orion-started messages (outreach, reading findings)? [No for v1. They have no
   reply act to hang on. The scan found one promise in such a turn (10-28). Revisit after the eval.]
4. Can Juniper close or drop a promise by hand in Hub? [Not in v1. Expiry is the honest
   default. Add a Hub button only if false promises show up.]
5. Retire the social-memory word-list commitment detector (1 commitment ever)? [Yes, in a
   separate small PR, because a word list must not grow. The decision belongs to Juniper,
   because social rooms are a different surface.]

### Juniper's answers (2026-11-12)

Juniper accepted every bracketed default above:

1. Offers count as promises only after Juniper accepts them. The judge decides acceptance from
   the reply plus Juniper's next message, not from a word list.
2. One fixed 72 h horizon for every promise.
3. Orion-started messages are out of v1.
4. No Hub close/drop button in v1.
5. Retire the social-memory word-list commitment detector, in a separate small PR.

## Files likely to touch

- `orion/schemas/harness_finalize.py`: `PromiseV1`, two fields on `FinalizeReflectionV1`
- `orion/cognition/prompts/harness_finalize_reflect.j2`: one section plus the open-promise list
- `orion/harness/finalize.py`: substring gate, pass open promises in
- `orion/schemas/grammar.py`: `ActSignal += "promise_kept"` (consumer-first: sql-writer first)
- `orion/grammar/acts.py`: promise act builder and basis text
- `orion/hub/turn_orchestrator.py`: open promise acts next to `_publish_reply_say_act`; close kept ones
- `services/orion-thought/app/open_promises_cue.py` (new, about 80 lines) plus `bus_listener.py` wiring and `stance_react.j2` block
- `services/orion-hub/.env_example`, `services/orion-thought/.env_example`, `settings.py`: flag (ON)
- tests: the gate, the act open/close, the cue render, "never in the motor prompt"
- `orion/self_monitor/evals/` or `services/orion-hub/evals/`: the replay eval below

## Non-goals

- No new table, no new reducer service, and no copy of meant/did facts.
- No use of attention open loops for conversation threads (a separate problem, owned by the PE
  suite rebuild, PR #3168).
- No outreach trigger: a promise never makes Orion message Juniper by itself. Step F owns
  "wait".
- No word lists and no regex promise detector in the runtime path.
- No deadline parsing from the promise text.

## Acceptance checks

1. **Replay eval before the cue ships (precision):** run the new judge section offline over the
   last 30 days of Juniper's turns. Hand-label every extracted promise. Pass: precision ≥ 0.8
   and the substring gate drops 100% of non-quotes. Report recall against the 6 hand-found statements.
2. **Fixture:** a reply whose "promise" quote is not in `final_text` opens no act.
3. **Fixture:** a `promises_kept` id that is not an open act closes nothing.
4. **Fixture:** an open promise past 72 h closes `expired`, value 0, and leaves the cue.
5. **Fixture:** the open-promise text never appears in the motor prompt (`prefix.py`).
6. **Live:** within 7 days of deploy, at least one `promise_kept` act is open, and its quote
   appears in a later stance prompt (`cockpit_turn_sighting`). Otherwise `UNVERIFIED`.
7. **Does it help (the settled outcome):** keep rate = kept / (kept + expired) for promises
   made in the 28 days after deploy, compared with the replay baseline (eval 1 also grades
   follow-through on old turns with the same judge). The read is pre-registered for
   2026-12-16. The likely n is 10 to 25 promises. Report the change as an effect size with
   its uncertainty, not as a significance test: a Beta(1, 1) prior on each keep rate, the
   posterior difference (after minus before), its 90% interval, and P(keep rate rose). There
   is no 0.05 line and no pass/fail significance verdict. A wide interval from thin data is
   reported as it is; it is never used as a veto. The guard metric still applies: promises
   per Juniper turn must stay ≤ 2× the baseline (5 of 95 turns). If it does not, the flag goes
   off (failure mode 4).
8. **Calibration:** the self-monitor shows `act.hub.chat_reply.say.promise` with a Brier score
   for `predicted`, and its status is not `ungradeable`.

## Recommended next patch

**D0, the replay eval only:** add the `PromiseV1` field and prompt section behind the flag. Run
acceptance check 1 against 30 days of stored turns, and report precision, recall and the
baseline keep rate. Do not open acts and do not show the cue yet. If precision is below 0.8,
the design stops and comes back. If it passes, D1 to D3 follow in one PR (consumer-first order:
sql-writer, then thought, then hub).
