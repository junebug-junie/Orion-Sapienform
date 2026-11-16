# Session threads: coding sessions write to Orion, Orion asks questions hourly

Date: 2026-11-15. Status: built in the same PR (Juniper approved the build on 2026-11-15).

## Arsonist summary

Coding sessions write design specs about Orion all day. Orion never sees them until they are merged,
and Juniper cannot see what Orion thinks of them. This adds one path: a session sends its spec to Hub,
Hub gives a receipt at once, and once an hour Orion reads every held message in ONE turn and comes back
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
- **What does a second round see?** The thread's earlier turns (clipped to 3,000 characters each), the
  new message in full, and the newest spec text again. Each round is a fresh Orion session.
- **Cursor sessions?** Not wired (`.cursor/hooks.json` is unchanged). The CLI works from any session.

## Proposed schema / API changes

- **Tables** (`manual_migration_orion_session_thread_v1.sql`, additive, required by orion-hub):
  `orion_session_thread` (one per PR or note; `source_ref` unique; status
  `held | reading | asked | no_questions | failed`) and `orion_session_turn` (speaker
  `session | hub | orion`; `roundup_id`; `delivered_at`; `pr_comment_url`; `meta`).
- **Durable workflow** `session_roundup.orion_turn` (`orion/schemas/session_roundup_run.py`,
  `SessionRoundupBriefV1`, registered). Run id `session-roundup-<utc hour>`. Act family
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

- A message that arrives in hour H is read in the round-up that starts at H+1. The clock is taken after
  the read of held threads.
- One round per hour. The threads are marked `reading` before the run is submitted. A round id already
  on a thread is never submitted again.
- Session text is quoted data inside a labelled fence with a per-round nonce. It cannot close the fence.
- Code cuts Orion's reply at `### thread:<id>` lines. No usable section for a thread: the whole reply
  is stored on it, marked `unparsed`. A repeated or out-of-order heading makes the whole round
  `unparsed`. Empty text is a failed round, never a stored turn.
- A failed round returns its threads to `held`. Three failures in a row leave a thread `failed`.
- A run that has not ended 30 minutes after its deadline fails the round, so one stuck run cannot block
  later rounds.
- Code posts the PR comment. Only on the configured repo, and only on an open pull request whose head
  branch is the thread's branch. `@name` and `#123` are broken. `NO QUESTIONS` and unparsed replies are
  never posted.
- Intake has no login (Hub is host-local). It is bounded: 50 waiting threads, 20 unread messages a
  thread, the same message stored once.

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

## Recommended next patch

Read two weeks of rounds, then decide: does Orion's turn need its tools at all (a no-tools turn is
cheaper), and should a session's reply reach Orion sooner than the next hour. Wire Cursor hooks if
Cursor sessions open spec PRs.
