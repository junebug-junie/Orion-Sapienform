## Summary

- Orion can now be told just "name your organs". One read-only command, `python -P -m orion.dig name-organs`, lists every organ that has no name yet, biggest first. Each organ gets its id, size, hosts, its parts, and a ready `name` line to fill in.
- The rule is printed at the top. It is built from the same limits that `name` checks: 1-5 words, at most 40 chars, a 20-400 char reason that cites a part by its full name.
- The live `body` summary now ends with `N organs unnamed: python -P -m orion.dig name-organs` while any organ is unnamed.
- `dig help` and `orion/dig/README.md` list the new command.
- `DIG_BRIEF_LINE` is unchanged. It is 370 chars today, and a pointer would push it past the 400 cap. Orion finds the command through `help` and the `body` footer.

## Outcome moved

Before, Juniper had to paste a paragraph of steps: `body`, then `body <organ>` for each organ, then `name ...`. Now one command gives Orion everything needed for a `name` call. Orion runs it again to see what is left.

## Current architecture

The dig kit (`orion/dig/__main__.py`) had `body`, `body <organ_id>` and `name`. Nothing listed which organs still had no name. Names are stored in `$ORION_DIG_HOME/self/organ_names.jsonl` and applied to the live body through `orion/body_atlas/names.apply_names`.

## Architecture touched

Dig kit CLI only. There is no bus, schema, service, or env change. `orion/body_atlas/__main__.summarize` gets an optional `extra_footer` argument; it defaults to empty, so `python -m orion.body_atlas` output does not change.

## Files changed

- `orion/dig/__main__.py`: adds `name-organs`, the help line, and the `body` footer. The skipped-ledger-line note on the live `body` now goes in the footer, so the file path stays the last line.
- `orion/body_atlas/__main__.py`: adds the `summarize(extra_footer=...)` argument.
- `orion/body_atlas/names.py`: a stale-id error now says to run `name-organs` (or `body`) again.
- `orion/dig/tests/test_name_organs.py`: 9 new tests.
- `orion/dig/README.md`: adds a row for the new command.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the live `body` printout gains one footer line while any organ is unnamed.
- Compatibility notes: none

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/dig/tests orion/body_atlas/tests -q   -> 136 passed
pytest tests/test_organ_names_partition.py -q      -> 1 passed
```

## Evals run

```text
None new. The body atlas organ-structure eval runs in CI. This patch adds no new behavior to evaluate:
it lists data and writes nothing.
```

## Docker/build/smoke checks

Ran live on the athena host, read-only: the worktree code, FalkorDB on localhost:6380, and a scratch `ORION_DIG_HOME`.

```text
10 of 10 multi-part organs have no name, largest first.
Name each organ by what it takes in and gives out (1-5 words, at most 40 chars). The because (20-400 chars) must cite at least one of its parts by full name, as written below.
Run each name line (a lasting write); rerun name-organs to see what is left.
1. organ:2ad84a72c8: 67 parts (39 svc, 24 store, 4 process); hosts athena 28, circe 1
   parts: orion-cortex-exec, orion-sql-writer, orion-cortex-orch, ... (+52 more in the file)
   python -P -m orion.dig name organ:2ad84a72c8 "<name>" because "<reason citing a part>"
...
body footer: 10 organs unnamed: python -P -m orion.dig name-organs
```

No Docker build is needed. The harness container mounts the primary checkout, so the command goes live after `git pull` on main.

## Review findings fixed

- Finding: the ready line `<name>` / `<reason>` was a bash redirect if pasted as is, and an apostrophe in a reason broke bash.
  - Fix: both placeholders are now quoted.
  - Evidence: `test_lists_unnamed_organs_...` asserts the quoted form.
- Finding: the printed rule left out the 40-char name limit and the 20-400 char reason limit.
  - Fix: the rule is built from the `names.*` constants.
  - Evidence: `test_rule_states_the_limits_name_checks`.
- Finding: no test covered the skipped-ledger-line note. In the all-named branch, the note could be cut off.
  - Fix: the note now goes right after the header. A test covers both branches and `body`.
  - Evidence: `test_an_unreadable_ledger_line_is_said_in_every_branch`.
- Finding: on `body`, the skipped-ledger note was printed after the file-path line.
  - Fix: the note goes through `extra_footer`.
  - Evidence: same test; it asserts the path is the last line.
- Finding: a huge first part name could show no part. A first block bigger than the budget could print "showing 0".
  - Fix: the first part is always shown, cut to fit, and the first organ is always shown.
  - Evidence: `test_a_huge_first_part_name_still_shows_one_part_to_cite`.
- Finding: the big-body test did not check that every shown block is whole.
  - Fix: the test now asserts that the count of ready lines equals N in "showing N".
- Finding: a stale-id error told Orion to run `body`.
  - Fix: it now says to run `name-organs` (or `body`) again and copy a current id.

## Restart required

```text
No restart required. The harness reads the primary checkout; `git pull` on main is enough.
```

## Risks / concerns

- Severity: low
- Concern: an organ id is a hash of its members. Between two live runs on 10-30, the biggest organ went from 66 to 67 parts, and its id changed from 3ad78ff88b to 2ad84a72c8. If a part moves between `name-organs` and `name`, the `name` call is refused.
- Mitigation: the error now says to run `name-organs` again. Names carry to a changed organ by member overlap.
- Severity: info
- Concern: the live names ledger (`/mnt/orion-fcc/self/organ_names.jsonl`) does not exist in the harness container today, so all 10 organs read as unnamed. UNVERIFIED: whether Orion's earlier naming turn ever ran `name` successfully.
- Mitigation: none in this patch. Check the turn that had the pasted instructions.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3000

🤖 Generated with [Claude Code](https://claude.com/claude-code)
