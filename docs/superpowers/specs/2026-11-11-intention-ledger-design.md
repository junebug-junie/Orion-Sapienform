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
- **Doer (added 2026-11-14, after D0 measured precision 0.72):** each `promises_made` item
  also carries `doer` (`orion`, `juniper`, `someone_else`, `a_system`): who carries out the
  future act. The gate keeps only `orion`, before the cap, and drops a missing doer. This
  removes forecasts about a system and asks for Juniper to act, the two biggest false
  classes in the D0 replay, without a word list.
- **Completion (added 2026-11-16, after the doer rerun measured precision 0.60):** each item
  also carries `act` (what Orion will do, in plain words) and `done_when` (what a later
  reply or run would show once the act is done, written so it could also show it was not
  done). The judge writes `done_when` null when nothing could break the line: a presence or
  readiness line ("I'll be here when you wake") names a state Orion is in anyway, not an
  act. This is Searle's preparatory condition for a promise: the act is not one the speaker
  would do in the normal course of events. The gate drops a null, missing or
  placeholder-length (`< 12` chars) `done_when`, before the cap. D1 can read `done_when` to
  know what to check when it settles a promise.
- **Force (added 2026-11-16, same rerun):** each item also carries `force`, the line's speech
  act (Searle's illocutionary point): `commits` (binds Orion to a future act; an offer's
  speech act is also labelled `commits`, so the gate keeps it as a stored item with
  `offer: true`, but it is not a promise and opens no ledger act until Juniper accepts it:
  Juniper's answer 1 and amendment 4 are authoritative), `reports` (an act already done this
  reply), `predicts`, `suggests`, `wishes`.
  The gate keeps only `commits` (rule 0a), before the cap, and drops a missing force. On
  the dev window the completion rule alone left recommendations, wishes and reports of done
  acts, which the doer rule cannot see because Orion is their subject.
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

## Amendments (2026-11-16): D1 + D2 in shadow, and Orion's critique

Juniper approved building D1 and D2 in **shadow** ("okay hit it", 2026-11-16). Promises become acts on
the act ledger and close as kept, expired or void, but nothing learns from them and nothing steers
Orion yet: no OPEN PROMISES cue in stance, no settled bet, no posterior update. Issue #3531 decides
when that changes. Orion read the PR #3532 thread and raised eleven points. Each answer below is what
this patch builds, or a default that waits on Juniper (marked **needs-juniper**).

### What shadow means, in code

- Every promise is one act: actor `hub.promise` (its own actor, not `hub.chat_reply`), verb `say`,
  object `juniper`, signal `promise_kept`, horizon 72 h. `test_before` is
  `harness_turn_trace:<corr>#promises_made[<i>]` (`orion/grammar/promise_acts.py`). Its own actor keeps
  the reply's per-actor readers clean: workspace binding, the body-economy family `hub.chat_reply` and
  downstream-use linking would otherwise count a promise as a second reply.
- `SHADOW_ACT_SIGNALS = {"promise_kept"}` in the same module. sql-writer persists and closes these
  acts but writes no `substrate_action_outcomes` row and no posterior update (`close_act`, the same
  path a withdrawn camera ask uses). The self-monitor builds no bet from them (SQL filter in
  `collect.py`, and the adapter). The act-chain forecaster in orion-prediction leaves them out of
  Orion's act rhythm. Going live is one reviewed code change: empty that set.
- A promise act never gets a cost. Its GPU time and energy belong to the reply that made it, already
  on the reply's act with the same correlation id; its wall time is days of waiting.
- Hub reads the open promises before each of Juniper's turns (`orion/hub/promise_ledger.py`, the read
  in `orion/grammar/promise_view.py`) and passes them on `HarnessRunRequestV1` to the 5b judge only.
  The motor prompt never sees them (test: `test_motor_prompt_never_shows_open_promises` drives the
  motor runner with them on the request). The same single read supplies the offers the ledger maps an
  acceptance to, and only a turn that got that read (Juniper's own) opens acts. The acts publish in a
  background task, so the reply never waits on them. The read is bounded at the database
  (statement_timeout 1.5 s, connect 2 s, pool wait 1 s) and fail-open; live it takes about 3 ms.
- D3 (the stance cue) is built and wired behind `THOUGHT_OPEN_PROMISES_CUE_ENABLED`, which ships
  **false**: the one deliberate exception to "flags ship on". #3531 turns it on.

### Answers to Orion's critique

1. **Precision numbers (from #3532).** The rules were frozen on the dev window before any held-out
   run. Dev (10-13..11-12, 185 turns): 26/31 = 0.84, 95% Wilson 0.67-0.93. Held-out (111 turns):
   10/12 = 0.83, 0.55-0.95. All 296 turns: 36/43 = 0.84, 0.70-0.92. Pooled recall 35/44 = 0.80,
   0.65-0.89. The held-out interval alone cannot tell 0.8 from 0.6. That is why the acts are shadow.
2. **Held-out blinding.** No. The labeller saw `force` and `done_when` on the after items while
   labelling (#3532 report, "Labelling"), and 5 borderline calls were all labelled true. Worst case,
   if all 5 were false: held-out 8/12 = 0.67. For #3531 the report writes a blinded sheet
   (`scripts/report_promise_ledger.py --sample 60`): each item shows only the quote and the reply
   around it. It never shows doer, force, act, done_when or the offer flag. Keys are hashes. The
   labeller fills true/false and `--score` prints precision with a Wilson interval.
3. **Recall floor for the shadow-to-live gate.** Proposed default: live goes ahead only when pooled
   recall is at least 0.7 on at least 40 hand-found promises, alongside precision at least 0.8 on at
   least 60 labelled items. Recall needs its own hand-found set: the blinded sheet samples only what
   the finder stored, so it cannot see misses. #3531 builds that set by reading judged Juniper turns
   blind and marking every promise, then matching by key as the replay eval already does.
   **needs-juniper** (the floor and the n).
   *Arithmetic corrected 2026-11-17 (Orion's second review):* the first draft said 150 turns. At the
   replay's rate of real promises (44 over 296 turns, offers included, 0.15 per turn) 150 turns hold
   about 22, so 40 needs about 270 turns read blind. Live there are about 5.8 judged Juniper turns a
   day (28 days to 2026-11-17), so live turns alone reach 270 in about 47 days; the blind read may
   include older turns (re-read blind, the finder replayed on them). The same arithmetic applies to
   precision: at about 0.11 stored items per judged turn (firm 0.08 plus offers), 60 live items take
   about 3 months, not 2 weeks. The 296-turn "pooled recall" (35/44) is not blind recall: its
   denominator holds only promises some finder run found, so a promise every run missed is invisible.
   #3531 carries the corrected plan.
4. **Offers.** Restored, and enforced in the ledger, not the prompt. `force: commits` still labels an
   offer's speech act (an offer does bind Orion once accepted), but an item with `offer: true` opens
   no act (`plan_promise_events`, `offers_not_opened`). The next judged turn of the same session sees
   that reply's firm offers as `pending_offers`; only an entry in `offers_accepted` that matches one of
   them opens an act, on the earlier turn's ref, from the moment of acceptance. An accepted quote
   that matches none is counted (`accepted_unmatched`) and opens nothing. Offers in an Orion-started
   message are out of v1 (Juniper's answer 3).
5. **Kept marks need a quote.** Done. The judge returns `promises_kept` as `{"id", "quote"}`. The gate
   keeps an id only when it was shown and its quote is literally in the reply (the same normalized
   substring test as a made promise), and checks it again against the final text after response
   repair. A quote that only repeats the promise's own words is refused: promising again is not
   keeping. The quotes are stored in `FinalizeReflectionV1.promises_kept_evidence`; `promises_kept`
   stays a list of ids for its existing reader (Prime's hand-back). A bare id fails closed.
6. **One record.** The ledger act is the record of the promise's life (open, kept, expired, void). The
   quote has one home, the promising turn's `harness_turn_trace` row. The act holds only the pointer,
   and Prime's hand-back already links promises by the identical string, so the act id is derivable
   from Prime's ref (`promise_act_id(ref)`). Prime's concern still holds `promise_text`, a frame-only
   cache that `promise_words_private` guards (no promise sentence in a choice label). This patch does
   not change Prime, because changing what Prime holds or when its concern closes would steer Orion
   during shadow. At go-live (#3531) the proposed change is: Prime's turn concern closes when its act
   closes, and Prime reads the quote through the ref instead of caching it. `promise_words_private`
   then checks the same thing against the ref read. **needs-juniper** (whether Prime's concern
   should follow the act at go-live).
7. **Censoring.** A promise can be marked kept only by the judge reading a reply to Juniper. When
   Juniper starts no chat turn inside the 72 h horizon, there was no reply to keep it in, so the
   outcome is unseen: the resolver closes it `void` (note `void: ...`), never graded, never a bet.
   This is broader than Orion's proposal (only promises whose done_when needs her input) and needs no
   new judge field: it censors exactly the cases where the measuring instrument never ran. When she
   did write and no reply delivered it, the close is `expired`, a real 0. **needs-juniper** (this
   default; the alternative is Orion's narrower rule, which needs a judge field).
8. **Cap of 3.** The judge's view is lifted to 8 open promises (`PROMISE_JUDGE_VIEW_MAX`), oldest
   first; the cap on promises listed per reply stays 3. At the replay rate (about 1 firm promise per
   12 judged turns, 72 h horizon) more than 8 open at once should not happen. The report still names
   any expired promise that was never in the view at any Juniper turn in its window as
   `un-closeable`, kept out of the keep rate.
9. **Guard baseline.** Re-based on the replay of the deployed finder, judged turns only: 23 firm
   promises over 289 judged turns = 0.080 per judged turn (dev 15/179, held-out 0/86 and 8/24). The
   guard counts firm non-offer items per judged Juniper turn (judge model pass; quick-gate turns are
   not judged and are reported apart). Red above 2x, as before. *Frozen 2026-11-17 (Orion's second
   review):* this baseline was set from replay data before any live promise act existed, and it stays
   23/289 for the whole shadow period (pinned by `tests/test_report_promise_ledger.py`); it is never
   re-based on shadow data. Because the slices are uneven (0/86 to 8/24), the report prints a 90%
   Poisson score interval on the live rate beside the ratio (a turn can hold more than one promise, so
   the rate is a count rate, not a proportion). Whether red should need the interval, not the point
   ratio, to clear 2x is a rule change for Juniper (#3531).
10. **Keep evidence.** Every keep records its kind in the close note (`kept_in:<corr> evidence=reply`),
    and the report splits keeps by kind. In v1 the only producer is the judge reading a reply to
    Juniper. A Prime carry run (`intention.carry`) that delivered something for the promise does NOT
    close it: delivery is not keeping (the carry may say "I could not"), and the carry reply is not
    judged against the promise. The report names each expired promise whose turn got a carry reply
    inside its window (`expired_with_carry_reply`), so #3531 can see whether that gap is real before
    a second evidence kind is built.
11. **Old rows and priors.** Stored items without `force`/`done_when` (written before #3532 deploys:
    12 such items on Juniper's turns in the last 14 days, live 2026-11-16) are skipped explicitly
    and counted (`skipped_no_force` in Hub's `promise_ledger` log line, `skipped_pre_force` in the
    report); nothing guesses their force. No promise act opens until #3532 is live in the governor.
    The 12-16 read's "before" arm is the replay (old turns graded by the same judge), not live data.
    The prior for `promise_kept` with no history is the act ledger's cold binary prior (mean 0.5,
    `cold_start` true), and every promise shares one posterior key and one faculty bucket
    (`act.hub.promise.say`), not one per promise.

### Acceptance checks added by this patch

- Postgres lifecycle (`services/orion-sql-writer/tests/test_promise_acts_integration.py`): a keep
  closes `producer` 1 with `evidence=reply`; a horizon with Juniper writing closes `expired` 0; a
  horizon with no Juniper turn closes `void`; none of the three writes an outcome row, a posterior or a
  cost; a keep after the resolver closed is ignored.
- Live (UNVERIFIED until deploy after #3532): a `grammar_acts` row with `actor='hub.promise'`; Hub's
  `promise_ledger corr=... opened=1` log line; a later `promise_view corr=... open_shown=1`; a close
  with `evidence=reply` or a `void`/`expired` close at 72 h; `scripts/report_promise_ledger.py`
  shows them.

## Amendments (2026-11-17): Orion's second review (PR #3561 and #3532 threads)

Built in this patch (fix/promise-ledger-orion-review):

- **One Hub line per Juniper turn.** `promise_ledger corr=... outcome=<found_nothing|nothing_opened|acted|skipped>`
  with every count (`items`, `opened`, `offers_not_opened`, `skipped_no_force`, `repeats_open`, ...);
  a skipped turn names its reason (`no_reflection`, `not_hub_orion`, `unsolicited`; `flag_off`,
  `no_bus` and `no_view` are defensive, since Hub only calls the ledger on a Juniper turn with the
  flags on and a view). The `promise_view` line also logs on every read, empty, failed
  (`outcome=unread`) or not. A Juniper turn with neither line means the ledger never ran for it.
- **A promise said again word for word, while it is open, opens no second act** (`repeats_open`;
  normalized quote equal to an open promise the judge was shown, and at least 40 characters, so a short
  generic line on a new topic still opens; the log names the matched act ids). A reworded repeat is
  not caught.
- **Report:** the keep-rate line names its kept and expired counts, and void and un-closeable beside
  them; the act ids of expired promises with a carry reply are listed; the guard prints a 90% Poisson
  interval and the baseline is frozen (amendment 9). The guard counts only judged turns from
  2026-11-16 18:05Z, when the governor went live with the #3532 rules (`FINDER_LIVE_AT`): before that
  no turn could store a firm item, and the 28-day window held 172 such turns that pulled the rate to 0.
- **Offer wording** in D1 corrected: Juniper's answer 1 and amendment 4 are authoritative; an
  unaccepted offer never opens an act (`test_unaccepted_offer_never_opens_and_accepted_offer_opens_on_the_earlier_turn`).

Not built (tickets): the not-due state for conditional promises and promises a chat reply cannot show
kept; the go-live gate criteria (recall/precision n, lower bound, side posterior, Prime close path);
the promise index drifting if a turn's run artifact is ever re-published. See #3531 and the issues
named in the PR report.

## Amendments (2026-11-17, round 2): Orion's third set of questions (threads 0c1e, e14555, 883a)

Built in fix/promise-ledger-orion-review-2:

- **The guard arms at 60 judged turns** (`GUARD_MIN_JUDGED_TURNS`). Before that it prints the ratio
  and "not armed", never red. At the frozen rate, 60 turns expect 4.8 firm promises; red (ratio > 2)
  then needs 10 or more, which noise alone gives about 2.4% of the time (at 50 turns: 8, about 5%).
  Judged Juniper turns ran 26-56 a week from 2026-10-05 to 11-15 (`chat_history_log` joined to
  `harness_turn_trace`, read 2026-11-17), so it arms in roughly 1.5-2.5 weeks.
- **The guard counts only turns sent from a Hub browser surface** (`JUNIPER_INGRESS_PREDICATE` on the
  turn's ingress hop; `turns_not_from_browser` counts the rest). A coding session's message posted over
  the Hub API is not Juniper's turn (#3602). The frozen baseline holds 2 such turns (11-15, 6cf88d2b and
  9f846e8e) and 2 of its 23 firm promises; without them it is 21/287 = 0.073, not 0.080. The baseline
  stays frozen at 23/289; re-freezing it is Juniper's call (#3531).
- **The keep-rate line names its ceiling:** the chat-showable share, 9 of the 23 firm replay promises
  (one reader, not blind; #3595). Until #3595 lands, the reply-only close cannot show more than that.
- **The report lists duplicate candidates** (same session, within 72 h, same `act` or `done_when`, or a
  24-character shared span) and **accepted offers**, with `accepted_unmatched` (an accepted quote that
  matches no firm offer in the reply the Hub showed: the session's previous chat row, Hub's own rule).
  Both leave out turns not sent from a browser surface. On the 23 replay promises: 0 candidates.
- **The blinded sheet shows Juniper's message** that the reply answered (`juniper_said`), still not the
  judge's reading.
- **Quote match** also folds the modifier-letter and fullwidth apostrophes, and removes zero-width
  characters (as a space) and markdown asterisks (replay turn 868ed31f lost a real line to `*...*`). Three
  fixture label hashes were re-keyed for the asterisk change; a blinded sheet already written keeps its
  old `key` for a quote with `*` (scoring does not recompute keys, so it is unaffected).
- **Prime never reads the promise acts in shadow** (`test_prime_never_reads_the_promise_acts_while_they_are_shadow`
  scans `orion/prime` and `services/orion-mind/app`).

## Amendments (2026-11-17, round 3): Orion's fourth set of questions (threads 0c1e, e14555)

Built in fix/promise-ledger-orion-review-3:

- **Act-opening uses the guard's rule.** Hub reads the pre-turn view, and so can open promise acts,
  only for Juniper's own turn sent from a Hub browser surface (`_promise_ledger_applies`;
  `is_browser_surface_turn` is `JUNIPER_INGRESS_PREDICATE` on the payload). Before, any
  `utterance_origin="juniper"` turn qualified, which includes every `/api/chat` call: live 11-15, two
  coding-session notes were posted that way into Juniper's own session and persisted as `hub_orion`.
  Round-ups, `intention.carry` and curiosity turns were already outside it (`utterance_origin="orion"`).
- **Every viewed turn logs one `promise_ledger` line**, also when the governor returns no run
  (`reason=no_run`: a timeout or an error reply), a harness error (`reason=harness_error`), a raise or
  cancel anywhere after the view read (`reason=interrupted`, from the `execute_unified_turn` wrapper), and
  a failed background publish (`outcome=failed`). Live 11-17 before the 17:58Z restart: 29 view lines, 1 ledger line; 28 were
  `memory-scoreboard-eval` `/api/chat` turns (`no_write`, the class #3619 fixed), one of them still in
  flight at the restart. These eval turns no longer read the view at all.
- **The report prints both bases:** baseline 23/289 and clean 21/287; ceiling 9/23 and clean 9/21 (both
  coding-session promises fall in "delivered outside chat"). Red stays on the frozen basis.
- **Abstentions:** 2 of the 23 ("I won't float anything back up while you're asleep", "I'll stop muttering
  about bus error"). They pass the gate (commits, orion, a `done_when`) and would open an act that no
  reply can show kept. The ceiling already counts them outside the 9. Whether they are promises: #3595.
- **Duplicate window:** the report prints how many same-session pairs within 72 h the duplicate rule
  compares. On the 24 replay items reconstructed from `/tmp/promise-ledger-replay-v3`: 44 pairs, 0
  candidates. The float-up / won't-float pair is not flagged (longest shared run 11 characters).

## Amendments (2026-11-18): Orion's findings fixed (#3595, #3596, #3530, #3560, #3531 defaults)

Juniper, 2026-11-18: "we need to fix the promises Orion found." Each rule below is the recommendation
already written on its ticket, **applied as a default on that instruction and reversible**. Shadow stays on;
go-live is not flipped; `THOUGHT_OPEN_PROMISES_CUE_ENABLED` stays off.

### Settle classes (#3595)

- The judge writes two more fields per promise: `settles_in` (`chat` | `run` | `absence`) and `condition`
  (the trigger a conditional promise waits on, or null). Rows written before them read as `chat`, unconditional.
- Shown an open promise with a condition, the judge may list it in `promise_conditions_met` when the turn shows
  the condition arose. Only ids it was shown survive the gate.
- At the horizon, an UNKEPT act closes, by the item its ref points at:
  - conditional, and no judged reply inside the horizon listed its condition met: `void:not_due`;
  - `absence` (an abstention): `void:absence_unchecked`, until a breach check exists (#3595 rec 1);
  - `run`: `void:not_chat` (a chat reply's silence is not a break);
  - otherwise the v1 rule: expired 0 if Juniper wrote, `void:` if she did not.
  Every settle-class void is censored (`void:` prefix): never graded, never a bet, never a posterior update.
- Known bias toward `not_due`: only promises the judge is shown (oldest first, `PROMISE_JUDGE_VIEW_MAX` = 8) can be
  marked condition-met, so with more than 8 open promises a newer conditional one closes `not_due`. Read errors fall
  back the other way (an unread item = the v1 rule; an unread condition scan = met, graded) and are counted in the
  resolver summary (`promise_item_unread`, `promise_condition_unread`). The report prints the settle-class void share
  of closed acts so a judge drift toward ungraded labels is visible.
  A chat reply that delivers a run or absence promise still keeps it (`evidence=reply`).
- Later delivery (#3595 rec 2): the report lists void acts that a carry run for the same turn delivered later.
  It never reopens the act; the act is the record of truth.
- Prime vs the act (#3595 rec 3, #3531 point 6): in shadow the two records may disagree; the report counts
  concerns with no act, acts with no concern, and concerns still open after their act closed. At go-live,
  Prime's promise concern should close when its act closes (keyed by the shared ref); that wiring is the go-live
  PR's job (the resolver closes expired/void acts in Postgres, not on the bus).
- One clock: the act horizon, Prime's promise-concern life and the carry horizon read one constant,
  `orion/schemas/promise_horizon.py` (Prime-side PR). Prime's concern places no bet, so one promise is at most
  one settled bet (the act, after go-live).

### Offer acceptance window (#3595)

An offer is shown to the judge only from Orion's immediately previous reply in the session, when that reply
answered a Juniper turn and was stored at most `OFFER_ACCEPT_TTL_SEC` (= the 72 h promise horizon) before the
view is read. A miss shows as the gate's `dropped_unknown_offer` (the judge quoted an offer it was not shown:
too old, or another row landed in between) or the ledger's `accepted_unmatched` (a quote survived the gate but
matched no offer the view read; should stay 0).

### Stable promise refs (#3596)

The ref format `harness_turn_trace:<corr>#promises_made[<i>]` does not change (Prime's hand-back uses it).
Instead, sql-writer's upsert keeps the first non-empty `reflection.promises_made` stored for a corr; a
re-published run artifact updates everything else and logs `harness_turn_trace_promises_kept_first`.
Limit: this pins the FIRST run's refs only. If a re-run's Hub opens acts from its own reflection, a promise at an
index past the stored list reads as `unreadable` (and the v1 rule at the horizon), and one at an index the first
list had is merged into the first run's act. Whether a Juniper turn is ever re-run is UNVERIFIED; the log line
counts it if it happens.

### Long quotes (#3530)

A judge quote over `PROMISE_QUOTE_MAX_CHARS` that is literal in the reply is kept as its leading part, cut at a
word boundary under the cap (`trimmed_long`). A prefix of a literal span is literal, so the substring gate still
holds; a non-literal long quote is still `dropped_too_long`. The replay case d05f08e3 (243 chars, hold-deep-read)
now passes the gate as a 235-char leading part.

### Go-live criteria defaults (#3531; recorded, NOT applied: go-live stays off)

- **Shadow end date: 2027-01-12.** Decide on whatever n exists then, intervals printed; extend only by decision.
- **Only Juniper-browser promises count toward the gate.** Eval, curiosity and session-lane items are a side
  precision number by lane, never in the gate set.
- **Headline keep rate: the graded settle class** (chat-showable promises, and conditional ones whose condition
  arose), with each void kind and the replay ceiling (9/23; clean 9/21) printed beside it. This replaces the
  2026-11-17 default "over all firm acts" now that settle classes exist.
- **Duplicates footnoted:** n stands; duplicate candidates print beside it; voiding duplicates only if real
  candidates appear in shadow data.
