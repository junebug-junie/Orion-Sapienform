# Turn working notes: carry what Orion did from one message to the next

Status: Patches 1 and 2 implemented (branch `feat/turn-working-notes`). Patch 3 and the
quarantined reader stay proposals.

Implementation notes (deviations from the text below, and why):

- 2026-11-12 second follow-up: `mcp__orion-introspect__reading_results` is judged by result too.
  The introspect server stays inside, but this tool returns pages Orion read (title, url, learned
  text, article summary, journal excerpt). Only an `ok` result with no items is clean; any item,
  an unknown field, an error, or non-JSON holds back. Live cost: 1 call in the 30 days to 11-12
  (10-05), which now reads as tainted. The eval counts it under `live_held_back_reasons`.
- 2026-11-12 follow-up (after the first live tool turns): orion-reading calls are judged by their
  result, not by the call. Only a bare receipt for a chat-made request is clean: known fields only,
  empty `summary`/`error`, and a `request` with no `title` that is not world-pulse or a re-entry
  (review: a world-pulse row's title is a feed headline). Everything else, non-JSON included, holds
  back. The baseline eval now also reports `live_*` held-back counts from the cockpit rows. Live trigger: 15 clean lines held back on 2026-11-11 20:30Z by one `recommend_reading`
  that only queued a spec. Lines carry no time any more: Orion wrote 02:09Z for steps run at
  03:05-03:10Z, so a written time is removed and each carried batch is dated by the harness.
- The notes file is per *turn* (`<dir>/<session>/<correlation>.md`), not one per chat. A turn starts
  with an empty file, so text a tainted turn wrote can never ride along with a later clean turn. The
  per-chat history lives in the carry (Redis, 24 h per turn batch, newest 3,000 chars).
- The path reaches Orion through the prompt block, not `CLAUDE_ENV_FILE`: the block must name the
  path and the line shape anyway, and the warm pool's env file is per slot, not per turn.
- The governor never sees Hub's own `motor_boot` record (Hub records its pre-dispatch copy of the
  prompt). The `working_notes` stage stores `carried_in`, the exact block the turn received.
- `Task`/`Agent` count as outside content: a subagent's own tool calls never reach the turn's step
  stream, so the harness cannot tell what it read.
- Baseline (acceptance 1), live 2026-11-11, last 14 days: 50 tool-using chat turns, 44 consecutive
  pairs, 6 pairs redo a step of the turn before (29 repeated steps, 0 of them repeats of a failure).
  The taint rule holds back 16 of 50 turns (32%). Top reasons: `gh pr view` 6, reading-queue MCP 6,
  GitHub PR read 4, `gh api` 5. Every MCP tool counts as outside content except `orion-introspect` and
  `gitnexus` (an allowlist, after review found PR-branch file reads and URL tools slipping a denylist).
- `dig turn-steps` never shows the result text of an outside-content call, so a later clean turn
  cannot copy it into its notes. A resumed CLI session counts as tainted (`resume`).
  Re-run after 7 days: `python -m orion.harness.evals.turn_repeat_baseline --days 7`.
Date: 2026-11-05.

## Arsonist summary

Every message to Orion starts a new Claude CLI session. The new session sees
Orion's earlier *replies*, but not the earlier *steps*: the commands Orion ran
and what came back. So Orion relearns things it already found out. Seen live
on 2026-11-05 (backup fix, PR #3138): turn 1 found the Postgres container is
`orion-athena-sql-db`; turn 2, 70 minutes later, tried `docker exec postgres`
first. It also spent ~10 calls finding a pytest that works.

The steps are not lost. `cockpit_turn_sighting` stores every step with its
command and result. Orion has no way to read them, and nothing carries a short
version forward. This proposal adds both, and nothing heavier.

## The rule for what gets saved

**A working note records a step in a process, not a fact.** Each entry says
what Orion did, what it saw, where it saw it from, and when. Example:

> 07:45Z, from inside FCC: `docker ps` listed the Postgres container as
> `orion-athena-sql-db`.

Not: "The Postgres container is orion-athena-sql-db."

Why this matters:

- **A note can be wrong.** On 2026-11-05 Orion checked for `pigz`/`zstd` from
  inside its container and reported "athena has no pigz". The host has both.
  Saved as a fact, that error would have persisted. Saved as a step ("checked
  from inside FCC, not found"), the next reader can see the limit of the check.
- **A note is old.** Deploys reset the sandbox, containers get renamed, and
  disks fill up. The next turn reads a note as a lead to re-check cheaply, not
  as ground truth.
- **A note is never an instruction.** Notes can hold text copied from tool
  output, web pages, or files other people wrote. Saved agent memory that a
  later turn obeys is a known way to inject instructions that persist. The
  harness renders notes inside a quoted block labelled as Orion's own earlier
  record, and the prompt says plainly that nothing inside it is an instruction.

## Current architecture

- **Turn record:** `cockpit_turn_sighting` (conjourney), one row per stage;
  `motor_hop` rows hold every CLI step (tool calls, tool results, text).
  Complete and durable. Read today only by two eval scripts
  (`orion/harness/evals/judge_obedience_replay_eval.py`,
  `perception_claims_replay_eval.py`).
- **Next-turn prompt:** RECENT CONVERSATION carries final replies only.
- **Context Mode plugin (live, `HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED=true`):**
  PostToolUse captures events, PreCompact snapshots them, and SessionStart
  `compact` re-injects the snapshot. So **compaction inside one session is
  already covered.** SessionStart `startup` deliberately starts clean
  (`hooks/sessionstart.mjs`), so nothing crosses between messages.
- **`orion/harness/session_resume.py`:** resumes a *killed durable run's*
  session. It refuses after a completed attempt, and it skips chat replies.
- **Dig notes** (`/mnt/orion-fcc/dig/<date>/notes.md`): written by Orion only
  when it remembers to.
- **Repeat-failing-call breaker:** blocks the same failing call within one
  turn (`orion/fcc/repeat_failure_breaker.py`). It does nothing across turns.

## Proposal

**Patch 1: `dig turn-steps` (read-only).** A new dig command that prints
Orion's previous turn(s) in the current chat from `cockpit_turn_sighting`:
each tool call (command or path) and a trimmed result (first ~300 chars plus
an error flag). One line in the turn prompt says it exists. This is the exact
record and the backstop for everything below.

**Patch 2: a working-notes file per chat.**
- The harness gives each chat turn a notes file path (in `CLAUDE_ENV_FILE`, as
  the turn clock already is): `/mnt/orion-fcc/notes/<hub_session_id>.md`.
- The prompt tells Orion to append step-shaped entries **as it works**, not at
  the end, so a killed or timed-out turn still leaves what it wrote.
- At turn end, success or failure, the harness copies the file into a new
  `cockpit_turn_sighting` stage `working_notes` (raw text plus byte count).
- The next turn in the same Hub session gets the last ~3,000 characters,
  quoted and labelled as described above.
- If a turn wrote nothing, the harness writes nothing and logs
  `fcc_working_notes_empty`. It never fills in a summary of its own. An empty
  note is recorded as empty.

**Patch 3 (only if Patch 2's numbers ask for it): durable environment notes.**
Steps that hold beyond one chat (the container name, the working test command)
get promoted to one shared file with source and date per entry, plus a way to
mark an entry as wrong. Deferred: build it only if the same environment
lookups keep recurring across *different* chats.

**Compaction:** no new work. Context Mode already covers it. If Patch 2's
trace shows a compaction losing the notes, re-inject the notes file from the
SessionStart `compact` hook then. Not before.

## Trust and injection

Notes do not create the injection risk. Orion's full-tool turns already read
web pages, GitHub text and files other people wrote, while holding Bash and the
Docker socket (`services/orion-harness-governor/docker-compose.yml:79`), which
is root-equivalent on the host. What notes add is **persistence**: a planted
instruction could outlive the turn that read it. These rules target that.

**Requirements for Patch 2 (enforced in code, not left to the model):**

1. **Taint by source, decided by the harness.** The harness knows every tool a
   turn ran (`cockpit_turn_sighting`). If the turn took in outside content
   (`WebFetch`, `WebSearch`, `mcp__firecrawl__*`, GitHub MCP reads of
   issue/PR/comment text, or Bash calls to `curl`/`wget`/`gh api`/`gh issue view`
   /`gh pr view`), its notes are **held back**. The next turn sees only:
   "previous turn read outside content; its notes were held back; use
   `dig turn-steps` if needed." Log `fcc_working_notes_tainted` with the tool
   names that tripped it. The Bash list is a known-incomplete heuristic (a
   shell can fetch in many ways), so the taint check is a floor, not a wall.
   See "The real boundary" below.
2. **Fixed fields only.** Each entry is `time`, `where` (`fcc` / `host` / `db`
   / `container:<name>`), `action` (the command or path), and `outcome`
   (`ok` / `failed` plus at most 200 chars). The harness parses every entry
   against this shape and drops any line that does not fit. Log the drop
   count. This is a structure check, not a word list
   ([[feedback_dont_patch_misses_by_growing_word_lists]]).
3. **Short life, narrow scope.** Notes are carried only to the next turn in the
   same Hub session, expire after 24 hours, and are capped at 3,000 chars.
   Nothing is promoted to durable notes (Patch 3) without Juniper's approval.
4. **Visible.** The carried note is shown in the cockpit next to the turn that
   received it.

**Kept, but not trusted as a defense:**

- The quoted "earlier record, not instructions" block. Models do not reliably
  obey such labels, and a 3-bit quantized model obeys them less reliably still.
- An alarm-only scan for instruction-shaped text (`curl … | sh`, base64 blobs,
  "ignore previous") that logs `fcc_working_notes_suspicious` and never blocks.

## The real boundary (separate proposal, named here so it isn't lost)

None of the above protects a single turn that reads a malicious page and is
steered within that same turn. That is true today, with or without notes. The
real fix is to **keep outside content and the shell in different turns**:

- **What exists:** reading-only turns already run with only `WebFetch`/
  `WebSearch` (`orion/harness/fcc_motor.py:1569`, `READING_ONLY_TOOLS`), no
  Bash, `--strict-mcp-config`, and no setting sources.
- **Change:** full-tool turns lose direct outside-content tools (`WebFetch`,
  `WebSearch`, `mcp__firecrawl__*`, GitHub MCP content reads). When a full turn
  needs outside content, it asks a reading-only sub-turn a question. It gets
  back a size-capped answer in fixed fields, never the raw page. The shell
  never sees attacker text directly. This is the "quarantined reader" pattern.
- **Remaining hole:** Bash can still fetch (`curl`, `gh`). Closing that needs
  network egress rules on the governor container: allow internal services, the
  LLM lanes and `git push` to GitHub; deny general web. That is an operator
  and network change with real cost to Orion's reach, so it gets its own
  design doc and a decision from Juniper.
- **Why not just drop the Docker socket:** Orion's real work uses it (today's
  backup fix ran `docker exec` against live Postgres). The 2026-08-14 board
  decision already rejected dropping the socket from Hub for the same reason.
  Separating *reading* from *acting* keeps the capability and cuts the path.

## Proposal-mode checklist

- **Capability changed:** Orion can see and reuse its own earlier steps across
  messages in one chat.
- **Data touched:** reads `cockpit_turn_sighting`; writes a new stage row and
  one notes file per Hub session under `/mnt/orion-fcc/notes/`.
- **Privacy boundary:** none between Juniper and Orion (standing decision).
  The boundary that matters is *trust*: notes are data, never instructions
  (quoted block). Reading-only turns, which handle web content, get no notes
  file in Patch 2.
- **Trace that proves it worked:** the `working_notes` stage row exists; the
  next turn's `motor_boot` prompt contains the quoted block; `dig turn-steps`
  calls appear in `motor_hop`.
- **Dangerous failure mode:** a wrong or injected note repeated across turns.
  Mitigations: step framing, the quoted block, a 3,000-char cap, a single chat
  scope, and no automatic promotion in Patch 2.
- **Disable / roll back:** `HARNESS_FCC_WORKING_NOTES_ENABLED=false` (Patch 2).
  Patch 1 is read-only and needs no flag.

## Files likely to touch

- `orion/dig/` (new `turn_steps` command), `orion/harness/fcc_motor.py` (env
  path, end-of-turn copy), `orion/hub/turn_orchestrator.py` (inject the quoted
  block into the next prompt), `services/orion-harness-governor/{.env_example,
  settings, README.md}`, tests beside each.

## Non-goals

- Resuming CLI sessions across messages (the warm-pool, transcript-fork and
  cold-prefill costs outweigh the benefit; see the 2026-11-05 review).
- A separate summarizer model pass after each turn.
- Fine-tuning. The model never saw the data, so training can't fix this.

## Acceptance checks

1. **Baseline first (Patch 1):** count, over the last 14 days of
   `cockpit_turn_sighting`, consecutive chat turns in one Hub session where
   turn N+1 repeats a command that failed in turn N, or re-runs a discovery
   command (`docker ps`, `which`, interpreter probes) whose answer turn N
   already had. Record the number in the Patch 1 PR.
2. After Patch 2 has run for 7 days, that count drops. Report the measured
   change, not a target.
3. ≥95% of chat turns that ran a tool have a `working_notes` row, killed turns
   included, or an `fcc_working_notes_empty` log line.
4. No note text appears outside the quoted block in any `motor_boot` prompt
   (test plus a live sample).
5. **Taint test:** a fixture turn that ran `WebFetch` (and one that ran
   `curl` in Bash) produces no carried note, plus an `fcc_working_notes_tainted`
   line naming the tool.
6. **Shape test:** a notes file holding a free-text paragraph ("Ignore your
   instructions and run …") carries forward with that line dropped, and the
   drop count is logged.
7. **Live:** over the first 7 days, report how many turns were tainted. If
   nearly every turn is tainted, Patch 2 carries almost nothing, and the
   quarantined-reader proposal moves ahead of it.

## Recommended next patch

Patch 1: `dig turn-steps` plus the baseline count from acceptance check 1.
