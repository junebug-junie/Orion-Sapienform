# Turn working notes: carry what Orion did from one message to the next

Status: proposal (cognition/memory change: proposal mode per CLAUDE.md 0A).
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

## Recommended next patch

Patch 1: `dig turn-steps` plus the baseline count from acceptance check 1.
