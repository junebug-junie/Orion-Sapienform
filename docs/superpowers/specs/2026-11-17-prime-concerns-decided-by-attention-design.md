# Prime's concern slots are decided by attention

Date: 2026-11-17. Status: design (docs only). Refs #3574 (the jam), #3434 (epic), #3430 (letter threads as
intentions), #3089 (attention as the want competition).

Juniper's direction, 2026-11-17, verbatim: "so what if it doesnt exist, why not make it exist. we have attention
because it helps to develop attention--why not use it".

This changes how Prime keeps and loses what it is concerned about. That touches cognition and autonomy, so the
proposal-mode items are in this document (CLAUDE.md 0A).

## In one paragraph

Prime can hold 3 concerns. Today a concern stays as long as Prime's model says "keep" on each thought, and the
model says "keep" to all 3 every time. So nothing ever leaves on its own, and new letter threads find no room.
This design takes that decision away from the model and gives it to attention's own rules: a concern stays
while something real keeps happening about it, judged against its own source's history, and fades when it goes
quiet. When the slots are full, a fresh candidate takes the place of a held concern that has gone quiet, and the
trace says why. The live data shows one hard fact: **today nothing in Orion produces evidence about a concern
after it opens.** So patch 1 makes the clock honest, patch 2 runs the contest, and patch 3 makes the missing
evidence exist.

## Arsonist summary

- **The time-to-live (TTL) has never fired.** Since concern acts began on 10-30 there are 54 concern acts:
  46 dropped by Prime, 5 settled, 3 open, **0 faded, 0 lapsed** (`grammar_acts`, actor `prime`, verb `predict`).
  The 24 h fade exists in code (`orion/prime/state.py:88`, `observe()` at `state.py:418-437`) and has done
  nothing in 18 days. The reason is one line: a bare keep sets `last_evidence_at = now`
  (`orion/prime/thought.py:792`).
- **The field name already says the right thing.** `last_evidence_at` is "when evidence last touched this". The
  code fills it with "when the model last said keep". This design makes the value match the name.
- **Attention cannot rank concerns today, because there is nothing to rank them by.** 0 of 53 letter threads
  (letters 11-08..11-15) got any later event in `temporal_self_event` that references them within 72 h. Their
  refs point backward, to the curiosity runs and chats the thread was written from. The 3 held concerns had 0
  later events too. A salience signal built on this is always 0. It fails the metric gate as a ranking signal.
- **"So what if it doesn't exist, make it exist" is the right answer to that.** The evidence is missing because
  no organ can see a concern or point at one. Patch 3 makes a concern an experience other organs can see and
  cite. That is how attention to concerns can grow, instead of being declared.
- **Do not put concerns in the 2.1 s field frame.** That frame picks one winner for "what is Orion attending to
  now", and its consumers (reverie, curiosity, the self-model) act on the winner. A concern is an hours-scale
  thing. Mixing it in would make a held concern compete with the world for the one winner. Use attention's
  rules, in their own contest.
- **Churn is the known way this goes wrong.** 10-30..11-03, concerns followed attention around: 22 of 38 drops
  said "attention has moved", most within an hour (`state.py:29-32`). The contest below cannot displace a
  concern until it has been quiet for half its TTL, so that pattern cannot come back.

## Current architecture (verified live 2026-11-17)

### How a concern lives and dies

| Step | Where | What happens |
|---|---|---|
| Open from a letter | `orion/prime/day_letter.py:263-288` `thread_concerns` | Up to 2 threads (`LETTER_CONCERNS_MAX`), never past 3 held (`MAX_CONCERNS`, `state.py:76`). The rest are counted as "without room" and then forgotten: nothing stores them. |
| Open from a funded event want | `thought.py:798-812` | Event mode is on (`PRIME_CONCERNS_FROM_EVENT_WANTS=true`, orion-mind `.env`). The model must name an `event_id` from `GET /physiology/event-wants`. |
| Open from a promise | `orion/prime/handback.py` | `event_ref = turn:<correlation_id>`, TTL 72 h (`state.py:96`). |
| Keep | `thought.py:789-792` | Any id under `concerns.keep` gets `last_evidence_at = now`. |
| Fade | `state.py:429-437` `observe` | `now - last_evidence_at > concern_ttl`. Never reached, because of the keep. |
| Horizon need | `orion/prime/hunger.py:209-232` `horizon_needs` | Drive rises from half the TTL. A keep resets it, so held concerns never reach it. Since 11-16 every need wake came from funded reading wants instead. |
| Drop | `thought.py:778-788` | Only the model can drop. 46 of 54 concern ends were drops. |

### Live state

`redis-cli -u redis://100.92.216.81:6379/0 get orion:prime:state`, read at 13:01Z on 11-17:

- `197c7a88cacc4b76`: letter 11-13 thread 0 (walkway render break). Opened 11-14 10:06Z, kept 11-17 09:17Z.
- `68bb10d17d8df205`: letter 11-13 thread 1 (chat memory never promoted). Opened 11-14 10:06Z, kept 11-17 09:17Z.
- `3d5f2e70b45d3db9`: letter 11-16 thread 0. Opened 11-17 09:13Z.

This moved since #3574 was filed. At 09:12Z a day-letter turn settled the chat promise (`438e69ed6bb3d798`) and
Prime dropped it. The 11-16 letter then got 1 thread into the freed slot: "5 thread(s), 1 now held as
concerns, 4 without room". The jam is not fixed. The two 11-13 threads are 75 h old and still held. A slot opened
only because a promise ended by another path.

Prime thoughts since Patch 5 (11-16 00:00Z to 11-17 09:17Z): 18 thoughts. 17 ended `concerns +0 -0 kept 3 | no
move`. 1 ended `-1` (the settled promise). 0 moves were sent. Two recent moves were refused, not skipped: one
because Prime could not read whether curiosity had a focus waiting, and one because Prime put the letter thread's
`event_ref` in `refs` as an act id ("act id must look like act_ + 24 hex characters"). Prime tried to act on
thread 0 and had no way to point at it except `concern_id`.

### What attention is today

orion-attention-runtime builds a `FieldAttentionFrameV1` (`orion/schemas/field_attention_frame.py:40`) about
every 2.1 s with the world-first rules (`orion/attention/world_first.py`): every candidate is judged by its
percentile against its own 7-day history; event-written sources are news for a 300 s orienting window and then
fade (`event_fade`, `world_first.py:231-239`); an empty eligible set is an explicit no-winner. Target kinds are
only `node | capability | channel | edge | field | system` (`field_attention_frame.py:13-20`, `extra="forbid"`).
Chat and reading enter as rate candidates against their own 7 days (`chat_candidate` `world_first.py:1075`,
`reading_candidate` `world_first.py:1120`). From the 11-17 relative-rate analysis (not re-pulled here): world
targets win 91-100% of the frames where they are eligible, body targets 47-68%, and about 70% of frames rest.

The body economy is the other half of attention (#3089, `2026-11-04-attention-as-want-competition-design.md`):
funded event wants carry `salience`, the share of the source's own last 30 days strictly below this event
(`orion/physiology/event_wants.py:34-39` `source_salience`). Prime already ranks event-want concerns by their
place in that list (`orion/prime/choice.py:89` `market_ranks`), and hunger lets only the top tenth of a source wake
Prime (`hunger.py:97` `EVENT_WAKE_SALIENCE = 0.9`).

`temporal_self_event.salience` is the same idea for experiences, set at insert by each organ's magnitude rule
(`config/integration/organs.yaml`, `orion/temporal_self/now.py` `MAGNITUDE_RULES`). Many kinds have no rule:
`chat_turn`, `prime_thought`, `reading_turn` and others are always NULL. `curiosity_run` is scored only since
11-16 07:02Z (22 of 85 rows in 7 days).

## The contest, in plain words

1. **A concern lives while something real touches it.** Real means: the world or another organ produced an
   event that points at the concern, or Prime acted on the concern (sent a question or took it on). Saying
   "keep" is not real. A concern nothing touches fades after its TTL, as the code already intends: 24 h, or
   72 h for a promise.
2. **Every candidate gets a score on one scale.** The score is how unusual its freshest evidence was for that
   evidence's own source, faded by how long ago it happened. This is world-first's rule: own-history percentile,
   then a fade, never a model's opinion and never raw counts across sources.
3. **A held concern can be displaced only after it goes quiet.** It must have had no evidence for half its TTL
   (12 h, or 36 h for a promise), and no act about it may be in flight. That is the same moment hunger's
   horizon need starts to pull, so Prime always gets a chance to act first.
4. **When the slots are full, the best waiting candidate takes the weakest displaceable slot**, if it beats it.
   The concern act closes with end `displaced` and a note with both scores and the reason.

### Score

For candidate `c` with freshest evidence `e`:

```text
score(c) = salience(e) x event_fade(age(e), ttl(c), grace = ttl(c) / 2)
```

- `salience(e)`: for a funded event want, the want's own `salience` (30-day own-source percentile). For any other
  evidence, `temporal_self_event.salience` of the evidence event. NULL means **unscored**, never 0 (silence is
  not calm, `world_first.py` docstring).
- `event_fade`: the existing function, full strength until half the TTL, then linear to 0 at the TTL. The
  orienting window is the concern's TTL instead of 300 s, because a concern is an hours-scale thing.
- Ranking: scored candidates by score, highest first. Unscored candidates rank after scored ones, newest
  evidence first. Ties go to the incumbent.

**Plainly: with today's data the contest is almost all recency.** Only 2 of 58 letter threads (11-08..11-16) cite
an event with a salience value. So in patch 2 a fresh letter thread will usually take the slot of a thread
that has been quiet 12 h or more. That is still a big change from today, where it never does. It becomes a
salience contest as scored evidence grows (curiosity runs since 11-16, and patch 3).

### Displacement rule

The weakest held concern `h` is displaced by the best waiting candidate `w` when all of these are true:

- `h` is displaceable: no evidence for at least half its TTL, no open nomination about it, not taken on, not a
  promise (see question Q3);
- `w` beats `h`: both scored and `score(w) >= score(h) + 0.1`; or `h` unscored and `w` scored; or both unscored
  and `w`'s evidence is newer than `h`'s;
- at most `LETTER_CONCERNS_MAX` (2) displacements per tick.

The margin 0.1 is a bracketed default, a knob to grade on live data, not a finding.

### Interactions

- **Hunger's horizon need** (`hunger.py:209`): it now reads a real evidence clock, so it rises for a quiet
  concern instead of being reset by every keep. Guard 1 (`hunger.py:410-416`, `touched`) must stop counting a
  bare keep as an answer to a horizon need; a sent move, a take-on, or a drop still is. Expected effect: a
  quiet concern wakes Prime about once at half its TTL, perhaps once more after `MIN_RISE`, then fades or is
  displaced. That costs fewer thoughts than today, where 11 of 18 thoughts since 11-16 were woken by reading
  wants and all of them ended "kept 3, no move".
- **The 72 h promise rule** stays. A promise's clock also counts only evidence. A promise is never displaced
  by default; it fades at 72 h unless taken on or touched (Q3).
- **#3430 (letter threads as intentions):** a take-on is evidence and protects the concern while its run is open,
  the same as `horizon_needs` already skips taken-on concerns. Without patch 1, #3430 cannot get a new thread
  into a slot, so patch 1 unblocks it.
- **Event mode** (`thought.py:798`): unchanged for opening. A funded event want that Prime does not hold is a
  waiting candidate with a real score, so the contest can bring it in without the model naming it. That is Q4,
  off by default.

## Where it lives

**Recommendation: attention's rule, Prime's tick.** The contest is written once, as a pure function in
`orion/attention/concern_contest.py`, next to `world_first.py`, reusing `event_fade` and `source_salience`. It
runs on Prime's tick in orion-mind, which applies the verdict to `orion:prime:state`.

Why not a new `concern` target kind in orion-attention-runtime (option A):

- **Cadence.** The runtime ticks every 2.1 s. A concern's evidence changes over hours.
- **One winner.** The field frame names one dominant target that other organs act on. A concern competing there
  would take that winner away from the world.
- **Contracts.** `FieldAttentionTargetV1.target_kind` is a closed `Literal` with `extra="forbid"`. A new value is
  a consumer-first migration across every reader of `substrate_attention_frames`.
- **Ownership.** The waiting list and the slots are Prime's private state. Option A needs two new contracts:
  attention reads Prime's state, and Prime reads attention's verdict back. Two places for state to drift.
- **Reversibility.** A pure function behind one flag on Prime's tick is cheap to remove.

What this keeps from Juniper's direction: the decider is attention's rule (own-history percentile, fade,
explicit rest), not the model; and patch 3 makes concerns visible to the rest of the mind, which is how the
missing evidence starts to exist. If she wants the runtime itself to own it, that is Q1.

## Metric quality gate (each new salience signal)

| Signal | 1 Provenance | 2 Independence | 3 Theory anchor | 4 Live check | Verdict |
|---|---|---|---|---|---|
| **S1 forward hard-ref evidence** (an event after opening whose `source_ref`, `subject_ref`, `correlation_id` or `related_refs` matches a concern's subject refs, from an organ other than Prime; plus Prime's own sent moves and take-ons naming the concern) | `temporal_self_event` columns; `grammar_acts` (actor `prime`); `thought.py:792` is what it replaces | Independent of the keep (the keep is the thing it removes). Not a transform of any other contest input. | World-first's event-written rule: an event is news for a window, then rest. An unfinished thing is live while something touches it. | 0 of 53 letter threads got a later event within 72 h. The 3 held concerns: 0. Refs resolve correctly to the past events (checked: `curiosity:520a7ab3ce87` -> `curiosity_run` 11-12). | **Pass as a clock, fail as a ranking signal.** As a clock it is a timestamp that returns to rest by definition. As a score it is always 0: degenerate. Use it only for patch 1. |
| **S2 birth salience** (max `temporal_self_event.salience` over the events a letter thread cites) | Thread text refs `[kind:id]`; `temporal_self_event.salience` set at insert per organ rule | Derived from curiosity-run salience; not used anywhere else in Prime's contest | Same as world-first: how unusual the evidence was for its own source | 2 of 58 threads scored (11-08..11-16). 12 threads have no ref left (the thread text is cut with "..." and the refs fall off). `chat_turn` has no magnitude rule; `curiosity_run` is scored only since 11-16 07:02Z. | **Fail today (coverage).** Wire it into the score as "unscored when NULL" only. Re-check about 12-01 with the query in the appendix. Do not build on it before that. |
| **S3 event-want salience** | `orion/physiology/event_wants.py:34` `source_salience`, carried in `want_ledger.decision->'want'->>'salience'` | Already the input to `market_ranks` and hunger's event wake; reused, not new | 30-day own-source percentile, the house rule | 7 days, funded: `hub.world_pulse_read` n=245, median 0.000, p90 0.775; `vision.ask` n=85, median 0.423, p90 0.851; `self_regulate.shed` n=9, median 0.457. Varies, reaches 0 (rest) and high values. | **Pass (existing signal).** Note: reading wants have a median of exactly 0 (strict-below percentile on a mostly-zero history), so most reading wants rank as rest, which is correct. |
| **S4 other-organ cites of a concern** (patch 3) | `experience_use` rows citing a `prime_concern` experience | Risk: a loop. Shown in the now, then cited, then kept, then shown again. `organs.yaml` says salience is never from use counts, for this reason. | Weak until measured. A cite says "I used this", which is closer to a model's report than to an event. | Cannot check: does not exist yet. | **Not wired.** Patch 3 records it in shadow for 14 days. Juniper decides (Q5). |

Existing-mechanism check (gate step 5): `rg "event_wants|world_first|concern|salience"` found the pieces reused
above (`event_fade`, `source_salience`, `market_ranks`, `horizon_needs`). There is no existing per-concern evidence
clock, no store for threads without room, and no `concern` target anywhere in attention.

Reversibility (gate step 6): patch 1 and 2 are each one flag in orion-mind. `last_evidence_at` keeps its name and
type. The new state fields are omitted while empty, the same rollback rule the state already uses.

## Missing questions

All answered from the code or live data except the ones listed for Juniper at the end.

- Does the full letter thread (before the 300-char cut) still carry its refs? The `note_md` of 11-14 has 16 refs
  and its `carry_forward_md` threads have 1. **UNVERIFIED** where the cut happens (letter writer or Prime).
  Patch 2 must take subject refs from the uncut thread.
- Which services read `concern_record` and would see a new `displaced` end? The state comments say Hub reads
  `orion:prime:state` ("rebuild Hub before orion-mind"). **UNVERIFIED** for this field; patch 2 checks it.

## Proposed schema / API changes

All in Prime's own persisted state. No bus channel, no new table, no endpoint in patches 1 and 2.

- `PrimeConcernV1` (`orion/prime/state.py:102`): add `subject_refs: list[str]` (max 8, each max 200 chars),
  omitted when empty. Filled at open: the `event_ref`; for a promise, the turn correlation id; for a letter thread,
  the `[kind:id]` refs from the uncut thread text.
- `PrimeConcernV1`: add `evidence_salience: Optional[float]` (0..1), the salience of the freshest evidence,
  omitted when None.
- `last_evidence_at`: no change in type or name. Its meaning becomes what the name says.
- `PrimeStateV1`: add `waiting: list[PrimeConcernV1]` (max 10, newest kept), omitted when empty. Threads and
  candidates without room live here instead of being forgotten.
- `ConcernEnd` (`state.py:165`): add `"displaced"`. `track_record` gets a "displaced" count.
- Consumer-first rollout for all of the above (`extra="forbid"`): rebuild Hub before orion-mind, and drop the new
  keys from `orion:prime:state` before rolling orion-mind back. Same rule as `event_family` and `needs`.
- New module `orion/attention/concern_contest.py`: `score(candidate, now)`, `standings(held, waiting, now)`,
  `displacements(standings) -> list[(held_id, waiting_id, reason)]`. Pure, no I/O.
- Patch 3 only: new `TemporalSourceKind` `prime_concern` (open and close of a concern act) under the `prime`
  organ in `config/integration/organs.yaml`, with a reader in `temporal_self_sources.py`. That is the
  integration gate's "emit" half; Prime already reads the now (`organs.yaml` `prime.reads_now`).

Env (orion-mind, ship on):

- `PRIME_CONCERN_EVIDENCE_CLOCK_ENABLED=true` (patch 1). `false` restores "a keep refreshes".
- `PRIME_CONCERN_CONTEST_ENABLED=true` (patch 2). `false`: no displacement; threads without room still wait.
- `PRIME_CONCERN_CONTEST_MARGIN=0.1` (patch 2).

## Proposal mode

- **Capability change.** Prime loses the power to hold a concern forever by saying "keep". A concern stays only
  while evidence or Prime's own acts touch it. When full, a fresh candidate can take a quiet concern's slot.
- **Data touched.** `orion:prime:state` (concerns, a new waiting list, the concern record). Read-only:
  `temporal_self_event`, `GET /physiology/event-wants`. Patch 3 writes `prime_concern` rows to
  `temporal_self_event` through the existing experience contract.
- **Privacy boundary.** No new boundary. Promise text stays out of `text` and out of the published view
  (`state.py:118-125`, choice-as-EFE S-1.8). Patch 3 publishes the concern's `text` (as the published
  `orion:prime:concerns` view already does), never `promise_text`.
- **Trace that proves it worked.** (1) A `prime` `predict` act closes with a note starting `faded:` or
  `displaced:` (none exist today in 54 concern acts). (2) A letter outcome line says "N now held" with N > 0 while
  an older thread was still held before it. (3) Prime's frame shows a standings line per held concern: evidence
  age, score or "unscored", and whether it is displaceable.
- **Dangerous failure mode.** Churn: concerns replaced as soon as anything newer appears, so nothing is held
  long enough to act on (the 10-30..11-03 pattern). Guarded by the half-TTL quiet rule and the in-flight act
  rule. Second danger: a self-feeding loop in patch 3 (shown, cited, kept, shown); guarded by shadow mode and Q5.
  Third: losing a commitment to Juniper; guarded by never displacing a promise by default.
- **Disable or roll back.** Set `PRIME_CONCERN_CONTEST_ENABLED=false`, then
  `PRIME_CONCERN_EVIDENCE_CLOCK_ENABLED=false`, and restart orion-mind. To roll the image back, first remove the
  new keys from `orion:prime:state` (`subject_refs`, `evidence_salience`, `waiting`, and any record with end
  `displaced`).

## Files likely to touch

Patch 1 (evidence clock):

- `orion/prime/thought.py` (line 792: a keep no longer sets `last_evidence_at`; a sent move or take-on naming the
  concern does)
- `orion/prime/hunger.py` (guard 1: a bare keep is not an answer to a horizon need)
- `orion/prime/state.py` (docstring; `observe` unchanged)
- `orion/prime/tests/`, `orion/prime/evals/` (replay of 11-14..11-17)
- `services/orion-mind/app/settings.py`, `.env_example`, `README.md`, prompt text in `thought.py:163-166`
  ("A concern you do not keep fades" becomes "a concern nothing touches fades; keeping does not renew it")

Patch 2 (contest):

- `orion/attention/concern_contest.py` (new), `orion/attention/tests/test_concern_contest.py`
- `orion/prime/state.py` (`subject_refs`, `evidence_salience`, `waiting`, `displaced`)
- `orion/prime/day_letter.py` (`thread_concerns`: put threads without room in `waiting` with refs)
- `services/orion-mind/app/prime_loop.py` (apply standings on the tick; read evidence)
- `services/orion-hub/` reader of Prime state, if it validates the record (rollout order)

Patch 3 (make concern evidence exist):

- `orion/schemas/temporal_self.py` (`TemporalSourceKind` `prime_concern`), `temporal_self_sources.py`,
  `config/integration/organs.yaml`, `scripts/check_integration_seam.py` must pass

## Non-goals

- No `concern` target in `FieldAttentionFrameV1` and no change to orion-attention-runtime.
- No embedding or text similarity as evidence. That is the 10-30 failure in a new form: the energy concern was
  renewed by a conversation that was only about the same topic (`state.py:13-15`).
- No change to `MAX_CONCERNS = 3` or to the TTL lengths.
- No model-written score, rank or "importance". The model still adds, drops, picks and moves; it no longer
  decides how long a concern lives.
- No magnitude rule for `chat_turn` here. It is a real gap for S2 and gets its own ticket if Juniper wants it.
- Not fixed here: Prime's move refs cannot point at a letter thread (the 11-16 refusal above). The answer for
  Prime is the move's `concern_id`; the refusal text should say so. Filed as #3609.

## Acceptance checks

1. Replay eval (patch 1): replay `orion:prime:state` history 11-14..11-17 with the evidence clock on. The two
   11-13 threads fade at 11-15 10:06Z (24 h after opening, no evidence). The 11-14 letter (written 11-15
   07:25Z) still finds 3 full slots, because it came before the fade, and with patch 1 alone its threads are
   lost (no waiting list yet). The 11-15 letter (written 11-16 07:47Z) gets 2 threads held. With patch 2, 2 of the
   11-14 threads enter at 07:25Z on 11-15 by displacing the two 11-13 threads, which had been quiet for 21 h
   (past half their TTL); the 11-15 05:42Z promise is not displaceable and stays.
2. Unit tests (patch 1): a keep leaves `last_evidence_at` unchanged; a sent move or take-on with the concern's id
   moves it; a refused move does not; a promise still lasts 72 h from its last evidence.
3. Unit tests (patch 2): a concern quiet for less than half its TTL is never displaced, whatever the candidate's
   score; an open nomination or take-on protects it; a promise is never displaced; at most 2 displacements per
   tick; unscored ranks below scored; a tie keeps the incumbent; every displacement writes a note with both
   scores and the reason.
4. Churn check (patch 2, live, 7 days): concerns ended `displaced` within 12 h of their last evidence: 0.
5. Live (patch 1+2, 7 days after deploy): at least 1 `faded` or `displaced` concern end, and every letter in that
   week shows "N now held" with N > 0, or says why not (no displaceable concern).
6. Gate re-check (about 12-01): re-run the S2 query below. S2 becomes part of the score only if at least half of
   that fortnight's threads have a scored cited event.

## Recommended next patch

Patch 1, the evidence clock: a bare keep stops renewing a concern (`thought.py:792`), a sent move or take-on
renews it, and hunger stops counting a keep as an answer to the horizon need. It is about 20 lines plus tests and
a replay eval, it needs no new signal, and it alone ends the jam in #3574. Patches 2 and 3 follow in order.

## Questions for Juniper (yes/no)

- Q1. Run the concern contest on Prime's tick with attention's rules, not inside orion-attention-runtime as a new
  target kind? (Recommended: yes.)
- Q2. Ship patch 1 (a bare keep no longer renews a concern) before the contest exists? (Recommended: yes.)
- Q3. Never let the contest displace a promise, so a promise ends only by settling, take-on, drop, or its 72 h
  fade? (Recommended: yes.)
- Q4. Let the contest bring a funded event want into a slot without the model naming it? (Recommended: no for now;
  the model still opens event-want concerns.)
- Q5. In patch 3, count another organ's cite of a concern as evidence, after 14 days in shadow? (Recommended: no
  until the shadow data shows it is not a loop.)

## Appendix: live queries (read-only)

Concern act ends, all time:

```sql
select split_part(coalesce(outcome_note,'(open)'),':',1), count(*)
from grammar_acts where actor='prime' and verb='predict' group by 1;
-- 2026-11-17: dropped unresolved 46, settled 5, (open) 3; faded 0, lapsed 0
```

Event-want salience by family, 7 days:

```sql
select family, outcome, count(*),
  percentile_cont(0.5) within group (order by (decision->'want'->>'salience')::float),
  percentile_cont(0.9) within group (order by (decision->'want'->>'salience')::float)
from want_ledger
where decided_at > now() - interval '7 days' and decision->'want'->>'event_ref' is not null
group by 1, 2;
```

S1 and S2 sizing: for each letter in `orion_day_letter` (11-08..11-16), split `carry_forward_md` into threads,
extract `[kind:id]` refs with `\[([a-z_]+):([0-9a-zA-Z\-]{6,})`, then count `temporal_self_event` rows with
`source_ref`, `subject_ref` or `correlation_id` starting with the id (or any `related_refs` element containing it),
excluding `consolidation_window_close`. S1: rows in the 72 h after the letter's `created_at` (result 0 of 53
threads). S2: max `salience` over rows before it (result 2 of 58 threads scored).
