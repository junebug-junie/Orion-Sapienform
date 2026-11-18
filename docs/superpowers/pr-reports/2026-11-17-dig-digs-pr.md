# feat(dig): `dig digs <date>`, the day's executed dig commands (day-dream Patch 1)

Design: `docs/superpowers/specs/2026-11-17-nightly-day-dream-design.md` (PR #3605), section "Patch 1". Juniper
approved the design and its defaults. This is the nightly day-dream's input: what Orion dug that day.

## Summary

- **New read-only command `python -P -m orion.dig digs <date>`.** For one Denver day it prints one line per dig
  command Orion actually ran: the time (Denver), where it ran (a chat session or a background workflow), the
  command, the first line of its result, and a ref. No model call.
- **Only commands that ran count.** A tool call counts when its shell line runs `python [opts] -m orion.dig`.
  The dig help inside every motor prompt is not a call (the naive text match finds ~340 motor_hop rows a
  day). A note that only quotes a dig command is not a call either: on 2026-11-17 Orion wrote 48 notes with
  `printf '...python -P -m orion.dig help...'` or `cat >> notes <<'EOF'`, and a plain regex counted all of them.
- **Background runs are read from the governor's `claude` transcripts.** The spec named
  `harness_turn_trace.run_artifact` as the source. It holds no tool steps (checked live: no key of it holds a
  tool_use, the same finding as `dig take-ons`, #3639), and a durable run writes no `motor_hop` rows. The
  transcripts are the only record, as in `dig take-ons`.
- **Each background call is tied to its run.** First by the `Strain refs: hub:turn:<id>` line in the run's own
  prompt (then `harness_turn_trace.workflow`), else through Hub's resume registry
  (`/root/.claude/orion-resume/*.json`, every `<run_id>:<source_tag>[:...]` key and its last generation), which
  names a durable step's transcript exactly. A warm-pool chat session, which has no turn id in its prompts,
  reads `chat (no turn id)`. 100% of the calls on 2026-11-15, 11-16 and 11-17 say where they ran.
- **The line rule is code.** `digs.check_lines` refuses any line that is not time + dig command + ref, and
  `render` runs it on its own output. A source that cannot be read is an `UNREAD` line in the tail, never 0.

## Outcome moved

Before: nothing could list what Orion dug in a day. `dig turn-steps` sees one chat at a time and no background
run; a text search of `orion.dig` is ~340 false hits a day.
After: `dig digs 2026-11-16` lists all 117 executed dig commands of that day (3 in chat, 114 in background runs:
pr_field_note 62, curiosity_self_inquiry 23, curiosity_self_sense_eval 13, reminiscence 15, ...) in 2.4 s, as
`orion_readonly`, and every count matches an independent hand count. The printed command is the dig
command that ran, cut at the next shell operator.

## Current architecture

- Dig kit: `orion/dig/__main__.py` (CLI, 40-line / 4,000-char cap, result file under `$ORION_DIG_HOME`).
- Chat turn tool blocks: `cockpit_turn_sighting` `stage='motor_hop'` (read by `dig turn-steps`).
- Background runs (durable steps, curiosity, field notes, reminiscence): no tool steps in Postgres; the
  `claude` CLI transcript in the governor's `/root/.claude/projects/*/<session>.jsonl` is the only record.
- `dig take-ons` already reads those transcripts by resume-key session name.

## Architecture touched

- `orion/dig` only (new module + CLI wiring). No service, schema, bus or env change.
- Reads: `cockpit_turn_sighting`, `chat_history_log`, `harness_turn_trace` (two SELECTs, READ ONLY,
  `orion_readonly`), and the governor's transcript and resume-registry files (plain reads).
- **Deviation from the spec (Patch 1):** the spec names `harness_turn_trace.run_artifact` as the background
  source and asks for a grants file. `run_artifact` holds no tool steps, so the transcripts are read instead,
  and no grant is needed. For Patch 2: `dream.day` must run `dig digs` where the transcripts are readable
  (inside the governor); anywhere else the background half reads `UNREAD`.

## Files changed

- `orion/dig/digs.py`: new. SQL builders, the shell-aware `dig_invocation`/`is_dig_command`, the streamed
  transcript scan, the merge, the registry linking, the line rule, the renderer.
- `orion/dig/__main__.py`: `cmd_digs`, dispatch, one help line.
- `orion/dig/tests/test_digs.py`: new unit tests (see below).
- `orion/dig/tests/test_dig.py`: help cap 4250 -> 4500 for the new line (main was at 4193).
- `orion/dig/evals/run_digs_live_eval.py`: new live read-only eval.
- `orion/dig/README.md`: the `digs <date>` row.
- `services/orion-hub/static/catalog/orion-abilities.json`: the `digs` entry in Hub's abilities catalog (CI's `test_every_dig_command_has_an_entry` caught it missing on the first push).
- `.github/workflows/dig-kit-tests.yml`: triggers on `orion/harness/session_resume.py`; the dig step names
  `test_digs.py` (it runs the whole `orion/dig/tests` folder).

## Schema / bus / API changes

- Added: none (a CLI command).
- Removed: none.
- Renamed: none.
- Behavior changed: none for existing commands.
- Compatibility notes: refs printed are `[chat:<id>]`, `[harness_turn_trace:<corr>]`, `[durable_run:<run_id>]`,
  `[transcript:<session>]`, `[cockpit_turn_sighting:<corr>]`. They are not `orion.refs` kinds (not openable by
  `dig arc|act`); they name the row or file the line came from.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change).
- skipped keys requiring operator action: none.

**Grants:** none needed. `orion_readonly` already reads all four tables (checked live 2026-11-17:
`has_table_privilege` true for `cockpit_turn_sighting`, `chat_history_log`, `harness_turn_trace`), so no grants
file was added.

## Tests run

```text
PYTHONPATH=. python -m pytest -q -p no:cacheprovider orion/dig/tests \
  orion/harness/tests/test_stance_scope_hierarchy.py orion/harness/tests/test_harness_prefix.py
353 passed (orion/dig/tests; 52 in test_digs.py), plus the two harness prefix files: all green
```

`test_digs.py` covers: a non-array content shape (string, dict, null, non-dict message, junk line, mixed
array); a dig command whose result is missing (`-> (no result recorded)`); help text inside a prompt and in
assistant prose not counted; a note quoting a dig command (printf, heredoc, echo, an unquoted echo, a shell
comment) not counted; `bash -lc`, wrappers (`FOO=1 timeout -s KILL 120 python ...`) and `if ...; then` counted;
two known misses (`-Pm`, a quoted module) pinned; the line rule
refusing 8 kinds of bad line (summary, prose, no ref, no time, not a dig command, wrong ref kind, trailing
text); window edges and failed results; sidechain calls; resumed history counted once; cockpit-over-transcript
and duplicated-cockpit-hop dedupe; registry linking incl. a 3-part `intention_carry` key and a later
generation; the scan stopping past the day; the printed command being the one that ran, not a quoted note;
a warm-pool chat session not inheriting an earlier turn id;
the CLI end to end (read-only SQL only, a transcript begun after the day never opened); UNREAD never 0.

## Evals run

Live, read-only, `orion_readonly`, in a throwaway container from the governor image with the transcript
volume mounted read-only (`--rm`, `:ro`):

```text
python -P orion/dig/evals/run_digs_live_eval.py --day 2026-11-16
117 dig command(s) ran on 2026-11-16 (times Denver): 3 in chat, 114 in background runs; read in 1.6 s
PASS E1 under 3 s: 1.61 s
PASS E2 line rule: 117 lines, each time + dig command + ref
PASS E3 nothing UNREAD: {}
PASS E4 chat = hand SQL: dig 3 vs hand 3: 0 refused by the shell check
PASS E5 total = independent scan: dig 117 vs hand 118: 1 refused by the shell check (1 explained)
PASS E6 refs are real rows: {'harness_turn_trace': 37, 'durable_run': 10, 'chat': 1}; missing []
PASS E7 where is known: 117/117 = 100%
PASS E8 not empty: 117 calls, 117 with a result line
info: naive text match (motor_hop rows whose raw mentions orion.dig) = 339, not counted
```

The one refused call on 2026-11-16 is a `cat > ... <<'EOF'` Cypher write whose heredoc text quotes a dig
command. Also run for 2026-11-15 (137 calls; all PASS, 2.3 s, 100% linked; E4 hand SQL 5 vs dig 2: the 3
extra are `cat >> notes <<'EOF'` / `printf` notes, each explained) and 2026-11-17 (241 calls so far; all PASS,
1.8 s; 50 refused notes, every one explained by the eval's separate quote/heredoc test).

Hand SQL used for E4 (2026-11-16 = 3, dig's chat count = 3):

```sql
SELECT DISTINCT x->>'id' FROM cockpit_turn_sighting s,
  jsonb_array_elements(CASE jsonb_typeof(s.raw->'step'->'raw'->'message'->'content')
    WHEN 'array' THEN s.raw->'step'->'raw'->'message'->'content' ELSE '[]'::jsonb END) x
 WHERE s.stage = 'motor_hop' AND s.ts >= '2026-11-16 07:00' AND s.ts < '2026-11-17 07:00'
   AND x->>'type' = 'tool_use'
   AND x->'input'->>'command' ~ '(^|[^[:alnum:]_.])-m[[:space:]]+orion[.]dig($|[^[:alnum:]_.])';
```

## Docker/build/smoke checks

```text
docker run --rm -v <worktree>:/w:ro -v orion-harness-governor_harness-claude-config:/root/.claude:ro ... \
  orion-harness-governor-harness-governor python -P -m orion.dig digs 2026-11-16
117 dig command(s) ran on 2026-11-16 (times Denver): 3 in chat, 114 in background runs; read in 1.6 s
```

No image rebuild: the dig kit runs from the FCC sandbox, which the post-pull rebuild resets to `origin/main`.

## Review findings fixed

Review by a separate review subagent on the first commit; no blocker. Fixed:

- Finding: the printed command anchored on the first dig words, so a quoted note before the real call was shown.
  - Fix: `dig_invocation` returns the command that ran (re-quoted, up to the next operator); `_command` shows it.
  - Evidence: `test_printed_command_is_the_one_that_ran_not_a_quoted_note`.
- Finding: durable steps with 3-part resume keys (`intention_carry:<concern>`, self-fit) never linked; the
  `(run x tag x generation)` search was brute force with a generation cap.
  - Fix: read Hub's resume registry (`orion-resume/*.json`) and match keys exactly; the `durable_run_workflow`
    query is gone.
  - Evidence: registry test with a 3-part key; live 2026-11-15 linked share 99% -> 100%.
- Finding: `echo python -P -m orion.dig ...` and a `# ...` comment counted as executed.
  - Fix: `python` only in command position (after an operator, keyword, assignment or wrapper); `#` comments cut.
  - Evidence: 9 new `test_is_dig_command` cases.
- Finding: whole transcript files read into memory; no early stop.
  - Fix: streamed line by line; stops at the first event past the day plus the 2 h result grace; calls before
    the day skipped before JSON parsing.
  - Evidence: `test_scan_stops_past_the_day_and_skips_calls_before_it`; live E1 2.2 s -> 1.6 s on 2026-11-16.
- Finding: a duplicated cockpit hop would print twice. Fix: `merge` dedupes cockpit rows too. Evidence: test.
- Finding: `bash -lc '...'` missed. Fix: any short flag set containing `c`. Evidence: test case.
- Finding: `1 << shift` in Python code read as a heredoc. Fix: a heredoc needs its terminator line. Evidence: test.
- Finding: a line-rule failure would crash with a traceback. Fix: `cmd_digs` raises `DigError` (still refuses).
- Finding: the tail counted warm-pool and cockpit chat calls as background. Fix: counted as chat.
- Finding (CI, Hub full tests): `digs` had no entry in Hub's abilities catalog.
  - Fix: added the entry under "Past & history". Evidence: `test_orion_abilities_catalog.py` 10 passed.
- Finding: spec vs code on sources and grants. Fix: stated under "Architecture touched", with the Patch 2 note.
- Not changed: `python -Pm orion.dig` and `python -m 'orion.dig'` are known misses, pinned in tests (none seen
  live). A call within seconds of midnight could in theory land on both days (cockpit `ts` vs transcript
  timestamp); dedupe is per day.

## Restart required

```text
No restart required. After merge, the post-pull run resets Orion's sandbox (/mnt/orion-fcc/repo) to main,
which makes `dig digs` live for Orion's turns. Check: git -C /mnt/orion-fcc/repo log -1 --oneline
```

## Risks / concerns

- Severity: low. Concern: run-from-host reads only chat turns. Mitigation: the transcripts live in the
  governor (`/root/.claude`); from a host shell the tail says `UNREAD background runs (claude transcripts)`,
  never 0. Orion runs it in a turn's Bash, where they are readable.
- Severity: low. Concern: transcripts are the only record of background tool calls. If the governor's claude
  config volume is pruned, those days read as an empty background list. Mitigation: none here; same exposure
  as `dig take-ons` (#3639).
- Severity: low. Concern: `chat` as a workflow. On 2026-11-17, 107 turns carry `harness_turn_trace.workflow =
  'chat'` but only 7 have a `chat_history_log` row or `motor_hop` rows (the rest look like eval traffic writing
  `memory-scoreboard-eval` notes). `dig digs` shows them as `chat` with a `[harness_turn_trace:]` ref, which is
  what the record says. Not filed as a bug.
- Issues opened: none. Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3675
