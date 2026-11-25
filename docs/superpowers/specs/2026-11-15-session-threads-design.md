# Session threads: coding sessions write to Orion, Orion asks questions at each round-up (every 15 min since 2026-11-25)

Date: 2026-11-15. Status: built in the same PR (Juniper approved the build on 2026-11-15).

## Arsonist summary

Coding sessions write design specs about Orion all day. Orion never sees them until they are merged,
and Juniper cannot see what Orion thinks of them. This adds one path: a session sends its spec to Hub,
Hub gives a receipt at once, and at each round-up (hourly until 2026-11-25, now every 15 min) Orion reads every held message in ONE turn and comes back
with questions. The questions go to Hub chat, back to the session that sent the spec, and onto the
pull request.

Juniper's two decisions:

1. The acknowledgement is an immediate Hub RECEIPT. It is labelled as Hub's. It is not Orion's words.
   No GPU turn runs at intake.
2. Orion's questions also go on the PR as a GitHub comment (user `orion-sapienform`).

## Current architecture

- The only two-sided thread in Hub chat is the consult display (`static/js/consult.js`,
  `scripts/consult_loop.py`). Consults invite Claude, run one at a time, and have a weekly cap of 3.
- Orion's consult turn is a durable run (`consult.orion_turn`) on the agent lane.
- Orion's turns read `/mnt/orion-fcc/repo` at `origin/main`. An unmerged spec is not there.
- No hook sends anything from a coding session to Orion. No PostToolUse or UserPromptSubmit hook exists.
- Hub has Orion's GitHub token (`GH_TOKEN` from `ORION_GITHUB_TOKEN`) and uses it read-only.

## Missing questions

Answered in this build; Juniper may change any of them in review.

- **Which tier is the round-up's act family?** Tier 2 (discretionary), shadow bids. The hourly cadence
  is Juniper's rule, so the body economy does not hold the run. A tier is Juniper's decision.
- **What happens to a message that arrives while Orion is reading?** It waits for the next round. The
  thread goes back to `held` after Orion's answer.
- **What does a second round see?** The thread's earlier turns (clipped to 3,000 characters each) and
  the new message in full. Each round is a fresh Orion session. If Orion already asked on the thread,
  the second round is the ANSWER PHASE (below): no spec text again, and no new questions.
- **How many rounds of questions does Orion get?** ONE per thread (added 2026-11-17, Juniper: "just
  add one round of questions Orion can ask"). The promise-ledger threads reached 4-6 rounds of
  question, reply, question.
- **Cursor sessions?** Not wired (`.cursor/hooks.json` is unchanged). The CLI works from any session.

## Proposed schema / API changes

- **Tables** (`manual_migration_orion_session_thread_v1.sql`, additive, required by orion-hub):
  `orion_session_thread` (one per PR or note; `source_ref` unique; status
  `held | reading | asked | no_questions | closed | failed`; `closed` added 2026-11-17, a free-text
  column, no migration) and `orion_session_turn` (speaker
  `session | hub | orion`; `roundup_id`; `delivered_at`; `pr_comment_url`; `meta`).
- **Durable workflow** `session_roundup.orion_turn` (`orion/schemas/session_roundup_run.py`,
  `SessionRoundupBriefV1`, registered). Run id `session-roundup-<utc bucket start, YYYYmmddTHHMM>`. Act family
  `durable.session_roundup` (`config/physiology/families.yaml`). Additive literal on forbid models:
  consumers deploy first, Hub last.
- **Hub routes**: `POST /api/session-threads/message`, `GET /api/session-threads?hours=48`,
  `GET /api/session-threads/{id}`, `GET /api/session-threads/pending?session_id=&branch=`,
  `POST /api/session-threads/delivered`, `GET /api/debug/session-threads`.
- **Websocket frame** kind `session_thread_turn`.
- **Hooks**: PostToolUse on Bash (`gh pr create` with a spec file sends it); UserPromptSubmit and
  SessionStart (Orion's undelivered questions arrive as `additionalContext`).
- No bus channel. No new service.

Rules that carry weight:

- A message that arrives in bucket B is read in the round-up of the next bucket (at most one interval, 15 min, later; a round still running delays it further, since one round runs at a time). A bucket with nothing held submits nothing. The clock is taken after
  the read of held threads.
- One round per hour. The threads are marked `reading` before the run is submitted. A round id already
  on a thread is never submitted again.
- Session text is quoted data inside a labelled fence with a per-round nonce. It cannot close the fence.
- Code cuts Orion's reply at `### thread:<id>` lines. No usable section for a thread: the whole reply
  is stored on it, marked `unparsed`. A repeated or out-of-order heading makes the whole round
  `unparsed`. Empty text is a failed round, never a stored turn. Session text Orion quotes back does
  not cut a section short: a closed data fence in the reply is replaced by one Hub line before the cut,
  and a heading inside a closed markdown code block is not a heading. Both apply only to headings for
  threads NOT in the round: a round thread's heading inside a quote makes the round `unparsed` (Hub
  cannot tell the copy from the real one). A section with no words besides Hub's cut note is empty.
  Hub does not strip heading lines that also appear in the round's session data: a session knows its
  own thread id, so its text can hold Orion's real heading byte for byte, and stripping it would move
  one thread's questions onto another thread's pull request.
- An `unparsed` thread (outside its answer phase) does NOT use its one round of questions (added
  2026-11-18, Orion's question on thread b3053f6df376: a parse accident is not Orion's silence). The
  whole reply is kept as an Orion turn with `meta.unparsed` and `meta.requeued`, never posted, never
  handed to a session; the thread's messages are unread again and the thread is read again next hour.
  It counts one failure toward the cap below, so a thread unparsed three times in a row ends `failed`.
  In a round of more than one thread the whole reply is NOT stored on the unparsed thread (added
  2026-11-18, thread 455a41af5b44 Q4): it holds the other threads' sections and any session text
  Orion quoted in them. Only Hub's line is stored (`meta.reply_not_stored`); the reply stays on the
  round's durable run. This holds in the answer phase too: such a thread is requeued, not closed.
- A failed round returns its threads to `held`. Three failures in a row leave a thread `failed`; a new
  session message on the thread puts it back to `held` with a zero count. A failed round stores no
  Orion turn, so it never uses the thread's one round of questions.
- A round the BODY never ran is not a failure (`core.is_body_refusal`): the run failed with
  `workflow_deadline` or `gpu_pool_unavailable:deadline|backlog_max_age` before any turn was attempted
  (the GPU pool kept it waiting past its admission deadline: cabinet heat, a busy lane), or it was
  still waiting for the pool (`queued`/`admitted`/`waiting_resource`) 30 minutes past its deadline.
  The threads are held again with their count unchanged, so a hot week never ends a thread. Hub
  writes one line per streak (not one per hour). Still strikes: a deadline hit after Orion's turn was
  attempted (`workflow_deadline_after_turn`), an overdue run that is running or paused, a refusal
  that is a property of the request (an unknown class, `replay_payload_too_large`). Backstop: once a
  thread's NEWEST unread message is 14 days old (`core.BODY_REFUSAL_MAX_AGE_SEC`), a body refusal
  counts too, so no thread is retried for ever. A new session message restarts that clock and zeroes
  the strike count (2026-11-18, thread 455a41af5b44 Q3; it was the oldest message before).
- A prompt the GPU pool refuses as bigger than the lane (`min_ctx_exceeds_class`,
  `core.is_sizing_refusal`) is Hub's sizing, not the thread's (2026-11-18, Q6): no strike, Hub halves
  its prompt budget, and caps it at 2 characters per token of the lane's largest window when the
  refusal names it (`core.shrunk_budget`; `SessionRoundupLoop.budget_chars`, floor 8,000 characters).
  The budget returns to `HUB_SESSION_ROUNDUP_PROMPT_MAX_CHARS` 24 h after the last such refusal, or on
  restart. At the floor it is a strike.
- Orion's questions reach a session by thread, not by the `gh pr create` event: the prompt hook asks
  Hub for undelivered Orion turns on any thread this session wrote on, or on this session's branch.
  A later send for the same PR (`orion_session_thread.py send --pr N`, from any session) joins the same
  thread, so its sender still gets the open questions. A plain `git push` sends nothing.
- Delivery is once per SESSION, not once per thread (2026-11-18, Q5): `POST .../delivered` carries the
  claiming session id (`meta.delivered_to`). A question turn stays pending for a new session on the
  thread or branch until the thread moves on: a session writes on it after the questions, or Orion
  writes a later turn there (a closing note). At most 20 sessions, at most 14 days old, and only turns
  claimed under this rule (a turn delivered before it shipped stays delivered). "No questions" and
  closing notes ask nothing: handed back once in total.
- The prompt tells Orion when a thread's pull request is closed or merged (a `[Hub]` line above its
  fence, from GitHub's `GET /pulls/<n>`): Hub posts a PR comment only on an open PR, so there the
  questions reach only the session hook and Hub chat (2026-11-18, Q2).
- A run that has not ended 30 minutes after its deadline fails the round, so one stuck run cannot block
  later rounds.
- Code posts the PR comment. Only on the configured repo, and only on an open pull request whose head
  branch is the thread's branch. `@name` and `#123` are broken. `NO QUESTIONS` and unparsed replies are
  never posted.
- **One round of questions per thread** (`MAX_QUESTION_ROUNDS_PER_THREAD = 1` in
  `orion/session_threads/core.py`, a constant, no env key). A thread whose Orion turns include one with
  questions (not `NO QUESTIONS`, not a closing note, not a re-queued `unparsed` reply) is in its ANSWER
  PHASE:
  - A session reply is still stored, held and read at the next round-up: Orion sees the answers.
  - The prompt puts a Hub line above that thread's data fence: the round is used, read the answers,
    write `NO QUESTIONS`, optionally one short line. The spec text is not shown again.
  - Whatever Orion writes for that thread is a CLOSING NOTE: stored as the Orion turn with
    `meta.closing_note` and `meta.dropped_question_lines`, thread status `closed`. Never a PR comment.
    The session sees it as information ("nothing here needs an answer"), with no reply command and no
    circle-back steps. Hub logs `session_thread_questions_dropped` with the count of question-like lines.
  - The intake receipt and the CLI's receipt line say the round is used. The waiting hook does not wait
    on such a thread.
  - A new thread (a new PR, a new note) gets its own round. A later push of the same PR's spec joins the
    same thread (`source_ref pr:<n>`), so it is in the answer phase too.
  - `NO QUESTIONS` uses no round: a later message on that thread may still be asked about.
- Intake has no login (Hub is host-local). It is bounded: 50 waiting threads, 20 unread messages a
  thread, the same message stored once.

## Change 2026-11-25: Orion reviews every PR

Juniper: "all submitted prs get orion's review. orion can still only write up to 3 times per PR.
orion now checks every 15 minutes from 1 hour where it was. claude pushes prs in draft form, and moves
it out of draft form after they respond to Orion's first review. this is mandatory."

- **Every PR is sent.** The `gh pr create` hook and `send --pr N` attach the spec files (as before) and
  the PR's code diff: `gh pr diff N` without spec files (attached whole) and without generated, lock or
  binary files (named, not shown). It travels as an attachment with `kind: "diff"` in the same `spec`
  list; Hub cuts it at 40,000 characters (`core.DIFF_MAX_CHARS`) and the cut line names `gh pr diff N`.
  The prompt shows it as `--- pull request code diff (...) ---`.
- **Code review template.** `core.CODE_REVIEW_TEMPLATE`, in every round-up prompt: eight questions from
  CLAUDE.md 0A (what it changes for Orion, live evidence or UNVERIFIED, metric gate, integration seam,
  empty shells, tests passing for the wrong reason, contracts and config, what to check live). Not a bug
  hunt: the code review skill already runs on every PR. Same answer format, so the parser is unchanged.
- **3 rounds of questions per thread** (`MAX_QUESTION_ROUNDS_PER_THREAD = 3`, was 1 since 2026-11-17).
  One thread per PR, so Orion writes on a PR at most 3 times. Rounds 2 and 3 get a `[Hub]` line saying
  how many rounds are left (`core.rounds_left_note`, from `question_rounds_used` set by the loop); the
  spec and diff are still shown. The 4th Orion turn is the answer phase, as before.
- **Every 15 minutes**: `HUB_SESSION_ROUNDUP_INTERVAL_SEC=900`. Buckets and round ids already work for
  any interval. One round runs at a time, so a slow round delays the next one; it never overlaps it.
- **Draft gate** (`scripts/hooks/pr_draft_gate.py`, Claude Code PreToolUse on Bash). `gh pr create`
  without `--draft` is refused. `gh pr ready` is refused until the PR's thread
  (`GET /api/session-threads?pr=N`, new) shows a session turn after Orion's first question turn, or
  Orion's first answer was `NO QUESTIONS`, or the thread is `failed`, or it is older than
  `ORION_PR_READY_WAIT_H` (24 h) with no Orion answer. Unanswered questions are never excused by the
  wait. Hub down: refused. Override per command: `ORION_ALLOW_PR_READY=1`. Kill switch:
  `ORION_PR_DRAFT_GATE=0`. A hook bug never blocks (exit 0, logged).
- Orion's review on PR #3998 (thread ba37694599a2), answered in the same PR: the gate sends a PR that
  has no thread (the create hook never reached Hub) to Orion itself and refuses, so the review and the
  24 h clock start then; the 24 h clock runs from the thread's creation, not from Orion's first read; an
  override is reported on the thread as a session turn, so Orion can count waived reviews; rounds 2 and
  3 are not gated (Juniper's rule names the first review only).
- Not covered: a PR opened outside Claude Code (Cursor, a human's terminal) gets no diff sent and no
  gate. `send --pr N` covers the first by hand.

## Files likely to touch

- `orion/session_threads/core.py` (pure rules), `orion/schemas/session_roundup_run.py`,
  `orion/schemas/durable_run.py`, `orion/schemas/registry.py`
- `services/orion-hub/scripts/session_roundup_loop.py`, `session_thread_routes.py`, `main.py`,
  `static/js/session-threads.js`, `static/js/app.js`, `templates/index.html`, settings, env, compose
- `services/orion-durable-runs/app/harness_smoke_graph.py`, `admission_runtime.py`, `run_acts.py`
- `config/physiology/families.yaml`
- `scripts/orion_session_thread.py`, `scripts/orion_session_thread_lib.py`, `scripts/hooks/*`,
  `.claude/settings.json`, `scripts/smoke_session_threads.py`

## Non-goals

- Orion does not approve, block, or merge a PR. Orion asks questions.
- No change to the consult queue, its cap, or its act family.
- No write to `chat_history_log`, recall, or memory from a session thread.
- No reading of GitHub by Orion's turn for this (the spec text is in the prompt).
- No per-message GPU turn. One turn an hour at most, none when nothing is held.

## Acceptance checks

1. `python3 scripts/smoke_session_threads.py` against the deployed Hub: every line `PASS`.
2. The note thread appears in Hub chat as one collapsed line; a click shows Session and Hub receipt.
3. After the next hour starts: Hub log has `session_roundup_started` then `session_roundup_applied`;
   `python3 scripts/smoke_session_threads.py --check <thread>` passes; the thread shows Orion's turn.
4. `durable_run_workflow` has one `session_roundup.orion_turn` run for that hour, and none for an hour
   with no held thread.
5. For a real PR with a spec: one comment by `orion-sapienform` on the PR, and its URL in
   `orion_session_turn.pr_comment_url`.
6. The sending session gets Orion's questions as context on its next prompt, once.
7. One round per thread: after deploy, `python3 scripts/smoke_session_threads.py --rule-since <deploy
   time>` passes (no Orion turn since then asked on a thread that already had its round). Before the
   deploy it fails on 18 turns across 8 threads (2026-11-17 22:00 UTC, read-only).

## Recommended next patch

Read two weeks of rounds, then decide: does Orion's turn need its tools at all (a no-tools turn is
cheaper), and should a session's reply reach Orion sooner than the next hour. Wire Cursor hooks if
Cursor sessions open spec PRs.
