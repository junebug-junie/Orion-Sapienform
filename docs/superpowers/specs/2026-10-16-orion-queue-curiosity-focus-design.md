# Orion queues its own next curiosity focus — design (proposal mode)

Status: APPROVED by Juniper 2026-10-16 and IMPLEMENTED (branch `feat/orion-curiosity-focus`,
report `docs/superpowers/pr-reports/2026-10-16-orion-curiosity-focus-pr.md`). Juniper's decisions
are recorded under "Decisions" below and override the proposal text where they differ; the
sections after it describe what was built.

## Arsonist summary

Juniper told Orion in chat: "It's your choice on what you want to focus on. Decide then
queue it up in your Curiosity run." Orion could not do it. Its only curiosity tool reads
past runs; nothing lets a chat turn put a topic in front of the next run. Orion spent 197
steps looking for a way and the turn died at the context ceiling.

The fix is small. Add one write tool, `queue_curiosity_focus`, that Orion can call from a
chat with Juniper. It stores ONE pending focus in Postgres. The next normal investigate run
takes it and shows it at the top of the kickoff, the same way it already shows Orion's Day
corrections. The run still goes through every normal gate, cooldown and daily cap. It is
never urgent. Afterward, Orion's existing `curiosity` read tool says "queued -> ran as run X",
so Orion can tell Juniper what really happened.

This copies a shape that already works in this repo (`record_correction` + the corrections
offer). It is not a new subsystem.

## Current architecture

How the investigate line picks a topic today:

- `CuriosityInvestigation.tick()` (`services/orion-hub/scripts/curiosity_investigation.py:2001`)
  runs the gates (`scheduling_block_reason` :650, queue backlog :2102, energy hold :2107,
  PG role, graph ACL, material) and then `_investigate()` (:2204).
- Code does NOT pick the subject. The claim is unset at kickoff (`_investigation_subject_from_view`
  :387); Orion chooses inside the run from a menu of priors and material.
- The only "next topic" memory is the **continuation note**: at the end of a run Orion writes
  `continue_line`/`continue_note` on a `:TurnOutcome` node in its own graph
  (`orion/curiosity/worldview.py:698`). The next kickoff shows it as "WHERE YOU LEFT OFF"
  (`orion/curiosity/kickoff_prompt.py:85`) and uses it as the recall query
  (`_standing_question_from_view`, curiosity_investigation.py:377).
- Several "offer once to the next run" queues already exist and are taken in `_investigate`:
  dream hypotheses, carry-forward, corrections (:2281-2283). They share one lifecycle:
  TAKE `FOR UPDATE SKIP LOCKED` and stamp the run, RELEASE if the run is cancelled before Orion
  saw it (`_release_offers` :1552, called :2363/:2417), STAMP an outcome at turn end.
- The self-inquiry line gets first refusal on every scheduled tick (:2027). So "next tick" can
  mean a self-inquiry run first; the focus waits for the next *investigate* run.

What Orion can call from chat (`orion/introspect/tools.py:164`): `reading_results`, `dreams`,
`curiosity`, `orion_day`, `record_correction`, `node_history`. `curiosity`
(`CuriosityArguments`, `orion/schemas/introspect.py:247`) is read-only: runs and open
self-questions.

The one write tool, `record_correction`:
- tool -> bus RPC on `orion:introspect:record_correction:request`
  (`orion/introspect/transport.py:7`), `refused="not recorded"` (tools.py:202-209).
- Hub responder `services/orion-hub/scripts/orion_day_correction_listener.py` refuses unless
  `binding.invocation_context == "unified_chat"` (:119), checks, then INSERTs one row as the
  last step (so ok=false always means "not recorded").
- Row lifecycle in `orion/orion_day/corrections.py` (TAKE :65, RELEASE :78, STAMP :90),
  table `services/orion-sql-db/manual_migration_orion_day_correction_v1.sql`.
- `orion/harness/claim_gate.py` marks a reply `[unverified]` if it names a `correction_id` no
  tool result returned (the 2026-10-15 fabricated-id incident).

Paths that exist but are wrong for this:
- **Urgent** (`/api/urgent`, curiosity_routes.py:636 -> `start_urgent` curiosity_investigation.py:2532,
  schema `orion/schemas/curiosity_urgent.py:33`): skips gates, cooldown and cap, runs at
  `priority="urgent"`, emails a report. That is for incidents and Juniper's button, not Orion's
  self-chosen interest.
- **`/api/run-now`** (curiosity_routes.py:512): forced tick, no topic, operator-only.
- **Self-question pool** (`orion/curiosity/self_question_pool.py`, `curiosity_self_questions`):
  questions about Orion, minted only inside self-inquiry runs (`_upsert_orion_minted_questions`
  :2982), drawn by rotation. No "next run uses this" guarantee, and not world questions.

## Existing-mechanism check (result)

- No focus / queue tool exists (`rg "curiosity_focus|queue_curiosity|next_focus"`: only unrelated
  reverie hits).
- Closest seams: continuation note (right meaning, wrong writer — it lives in `orion_worldview`,
  which Hub must never write, the PR #2199 boundary) and corrections (right writer, right
  lifecycle). **Recommendation: corrections' lifecycle + continuation's prompt slot.** Do not
  write a fake `:TurnOutcome` into the graph from Hub.

## Decisions (Juniper, 2026-10-16)

1. **One pending slot, replace** -- a new call replaces the pending focus and the reply returns
   what was replaced. EXCEPT a focus with `origin="juniper"` cannot be replaced by an
   `origin="self"` call: refused with "Juniper asked for this focus; it stays until a run takes
   it or it expires". A `juniper` call may replace anything.
2. **Expiry 48 hours**, checked against live cadence (below).
3. **No Hub input box** for Juniper in this patch. The tab shows the pending focus and a Clear
   button only.
4. **Shown, not forced -- but the run owes a disposition.** At turn end the run records whether it
   explored the focus or set it aside and why (`"explored"` | `"set_aside:<reason>"`), as
   `focus_disposition` on its own `:TurnOutcome`. Missing reads as unknown, never as explored.
   `curiosity` shows queued -> taken by run X -> disposition / expired / failed.

### Expiry measured (2026-10-16)

`durable_admission_runs`, regular (non-urgent) `curiosity.investigate` runs on the investigate
line, last 30 days: 206 runs, one every day (6-8 per day). The line runs right after local
midnight (America/Denver) until the daily cap (7) is used, 30 min apart, then waits for the next
local day: admissions cluster 06:10-09:20 UTC every day. Gaps between consecutive runs: p50 0.5 h,
p90 20.8 h, p99 21.0 h, max 21.1 h; 0 gaps over 24 h. Expected wait for a focus queued at a random
time: 8.2 h. So 48 h never expired a focus in that window; it covers the worst real wait plus one
whole missed day. Kept at 48 h (`orion.curiosity.focus.FOCUS_TTL`).

## Missing questions (proposal text, answered above)

1. One slot, replace or refuse? Proposed: one pending slot, **replace**, and the reply returns
   the text it replaced. Juniper asked Orion to decide; the latest decision should win, and the
   replaced text stays visible in the table and in Hub. (Refuse makes Orion fight its own
   older choice; a queue invites a backlog nobody drains.)
2. Expiry: proposed 36 h, a code constant. Covers an overnight quiet window plus a day at cap.
   Expired rows stay with `status='expired'`.
3. Juniper-origin asks: when Juniper names the topic ("look into X next"), Orion calls the same
   tool with `origin="juniper"`. Should Hub also get a non-urgent "queue for next run" box next
   to the urgent button? Proposed: not in the first patch.
4. Should a run be required to address the focus, or only shown it? Proposed: shown, first in the
   kickoff and as the recall query, with plain words that Orion chose it. Not enforced — the arc
   is about Orion choosing, and "did it address it" would need a semantic judge.

## Proposed schema / API changes

**Tool args** (`orion/schemas/introspect.py`), `extra="forbid"`:

```python
class QueueCuriosityFocusArguments(BaseModel):
    question: str  # 1..500 chars, stripped, non-blank
    why: str       # 1..500 chars: why Orion wants this now
    origin: Literal["self", "juniper"] = "self"
```

Add `"queue_curiosity_focus"` to `IntrospectBusOperation` (introspect.py:44) and its tuple (:46).

**Bus**: `QUEUE_CURIOSITY_FOCUS_REQUEST_CHANNEL = "orion:introspect:queue_curiosity_focus:request"`
in `orion/introspect/transport.py`, plus `orion/bus/channels.yaml`. Same envelope kinds and
`orion:introspect:result:<corr>` reply as the other introspect tools.

**Reply item** (`IntrospectItemV1`, kind `curiosity_focus`): `focus_id`, `status="pending"`,
`replaced` (previous question text or null), `expires_at`, and `next_eligible_at` from the same
schedule read the Curiosity tab uses (`_read_schedule`, curiosity_routes.py:130), so Orion can
say "queued; the next run can start no earlier than 14:30" instead of "running now".

**Table** `curiosity_focus_request` (new additive migration,
`-- ORION-MIGRATION-REQUIRED-BY: orion-hub`, CREATE TABLE/INDEX only so the post-pull
auto-apply allowlist accepts it):

| column | meaning |
|---|---|
| `focus_id uuid pk` | uuid5(question, why, origin, chat_correlation_id): a retried call returns the existing row, nothing new written (as built) |
| `question`, `why`, `origin` | Orion's words, append-only |
| `chat_correlation_id` | `binding.parent_trace_id` of the chat turn |
| `created_at`, `expires_at` | |
| `status` | `pending` / `replaced` / `cleared` / `expired` / `taken` / `ran` / `failed` (as built; `status_at`, `replaced_by`) |
| `offered_at`, `offered_run_id` | stamped by TAKE, cleared by RELEASE |
| `outcome_stamped_at`, `turn_ok` | stamped at turn end |
| `disposition`, `disposition_raw` | as built: what the run wrote on its `:TurnOutcome`; NULL = unknown; raw kept when malformed |

Partial unique index `WHERE status = 'pending'` enforces the one slot. Replace = one transaction:
mark the old pending row `replaced`, insert the new one.

**Read side**: `curiosity` run items gain `extra.focus = {focus_id, question, origin}` when that run
took a focus; new `kind="focus"` lists recent focus rows with status and `offered_run_id`. This is
how Orion answers "did my queued focus run?" truthfully.

**Claim gate**: extend `orion/harness/claim_gate.py` to `focus_id` exactly as for `correction_id`.

**Hub**: `_read_schedule` payload gains `pending_focus` (question, origin, queued_at, expires_at);
the Curiosity tab renders it as one line under the investigate schedule tile, and the run list
marks runs that took a focus. `DELETE /api/focus/{focus_id}` lets Juniper clear it.

**Env**: `HUB_CURIOSITY_FOCUS_ENABLED=true` (default ON, per repo convention) in
`services/orion-hub/.env_example` and `app/settings.py`, next to
`HUB_CURIOSITY_ORION_DAY_CORRECTIONS_ENABLED` (settings.py:1038).

## Proposal-mode items

- **Capability that changes**: Orion can, from a conversation, choose what its next normal
  investigate run looks at, and later read back whether that run happened.
- **Data touched**: one new Postgres table written only by Hub. Reads of the Redis schedule keys
  already read by the Curiosity tab. `orion_worldview` is not written (PR #2199 boundary holds).
- **Privacy boundary**: the focus text comes from a Juniper<->Orion chat and is shown to Orion's
  own curiosity run and to Juniper in Hub. No outward channel. Not sent to the FCC sandbox's
  read-only role (`orion_readonly` gets no grant, as with corrections). Chat-only: the responder
  refuses `invocation_context != "unified_chat"` (curiosity runs already have `continue_line`).
- **Trace that proves it worked**: in order, with one chat correlation id:
  `curiosity_focus_queued focus_id=… corr=<chat>` (Hub log + row) ->
  `curiosity_focus_offered run=<id> focus_id=…` (kickoff log, row `offered_run_id`) ->
  `curiosity_focus_outcome run=<id> turn_ok=…` (row `status='ran'`) ->
  `curiosity(kind="focus")` shows `ran as run <id>` -> Hub tab shows the same.
- **Dangerous failure modes**:
  1. Orion says "queued" with no tool result (the 2026-10-15 correction pattern). Guard: claim gate
     on `focus_id`; brief says no id = not queued.
  2. A focus becomes a gate bypass. Guard: TAKE happens inside `_investigate`, after every gate;
     a test asserts `tick()` with a pending focus still returns the same block reasons.
  3. A failing run re-takes the focus every tick forever (the trap named at :2256). Guard: a taken
     focus is never re-queued after Orion saw it; failure is stamped `ran, turn_ok=false`. Only a
     cancel-before-seen releases it.
  4. A focus silently expires and Orion tells Juniper it ran. Guard: `expired` is a visible status
     in both the read tool and Hub.
  5. Injection: chat text steering a self-directed run. Same trust as the continuation note:
     shown as an offer, 500-char cap, no tool or prompt authority.
- **Disable / roll back**: `HUB_CURIOSITY_FOCUS_ENABLED=false` -> the responder refuses with
  "not recorded: focus queue is off" and the kickoff never takes. Full rollback: remove the tool
  spec; the table is inert and can stay.

## Files likely to touch

- `orion/schemas/introspect.py` — args model, operation literal.
- `orion/introspect/transport.py`, `orion/introspect/tools.py`, `orion/introspect/brief.py` — tool.
- `orion/bus/channels.yaml`, `orion/schemas/registry.py` (if introspect args are registered).
- `orion/curiosity/focus.py` (new, mirrors `orion/orion_day/corrections.py`): SQL, take/release/stamp,
  kickoff section formatter.
- `orion/curiosity/kickoff_prompt.py` — focus section above "WHERE YOU LEFT OFF".
- `services/orion-hub/scripts/curiosity_focus_listener.py` (new, mirrors the correction listener).
- `services/orion-hub/scripts/curiosity_investigation.py` — take in `_investigate`, release in
  `_release_offers`, stamp at turn end, recall query prefers the focus.
- `services/orion-hub/scripts/curiosity_introspect_listener.py`, `scripts/curiosity_introspect.py` — read side.
- `services/orion-hub/scripts/curiosity_routes.py` + Curiosity tab template/JS — pending tile, delete.
- `orion/harness/claim_gate.py` — `focus_id`.
- `services/orion-sql-db/manual_migration_curiosity_focus_request_v1.sql` (new).
- `services/orion-hub/.env_example`, `services/orion-hub/app/settings.py`, README.
- Tests next to each; eval: extend `services/orion-hub/evals/run_orion_day_correction_eval.py`'s
  shape into a focus round-trip eval (queue -> tick -> kickoff contains it -> read tool says ran).

## Non-goals

- No queue of many focuses, no ranking, no topic taxonomy.
- No urgent priority, no gate/cooldown/cap exemption, no email.
- No write into `orion_worldview` from Hub.
- No judge of whether the run "addressed" the focus. (As built: the run states its own
  disposition; Hub records it verbatim, it does not grade it.)
- Not fixing the 197-step / context-ceiling behavior itself (separate issue: a turn with no
  matching tool should say so early). This patch removes the reason for that hunt only.
- No Hub "queue for next run" input for Juniper in the first patch (open question 3).

## Acceptance checks

1. Unit: chat binding + valid args -> one pending row; curiosity binding -> ok=false "not recorded",
   no row; flag off -> refused, no row; retried identical call -> same `focus_id`, one row.
2. Unit: second queue replaces the first; reply `replaced` holds the old text; old row `replaced`.
3. Unit: `tick()` at cooldown / cap / outside window with a pending focus -> same block reason,
   focus still `pending`.
4. Unit: `_investigate` takes the focus; kickoff text contains the question; cancel-before-seen
   releases it; turn failure stamps `ran, turn_ok=false` and does not re-pend.
5. Unit: expired rows are not taken; status reads `expired`.
6. Unit: claim gate marks a `focus_id` no tool returned.
7. Eval: round trip on the bus with a fake tick, ending in `curiosity(kind="focus")` showing
   `ran as run <id>`.
8. Live smoke after deploy: Juniper asks Orion to pick and queue; evidence = the four log lines
   above with one chat corr id, plus the Hub tile. Until then: `UNVERIFIED`.

## Recommended next patch

One PR: migration + `orion/curiosity/focus.py` + Hub responder + tool + kickoff section + take/
release/stamp + read-side `kind="focus"` + claim gate + flag. Hub tile and DELETE route in the same
PR if small; otherwise the immediate follow-up. Apply the migration, deploy orion-hub and the
governor (introspect tool list), then run acceptance check 8.
