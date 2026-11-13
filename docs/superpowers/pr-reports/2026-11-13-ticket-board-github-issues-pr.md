## Summary

- Open follow-ups now live in GitHub Issues, not in one agent's memory file.
- New labels: kind (`follow-up`, `verify-live`, `design`), `area:*`, and `needs-juniper`.
- The old memory "OPEN QUEUE" moved into issues #3255-#3266 (item 12 was already fixed on main, so it was dropped).
- New SessionStart hook prints the open-ticket count and every `needs-juniper` ticket at the start of each session.
- CLAUDE.md "Ticket board" subsection: file on discovery, search first, close with evidence. Section 11 now points eval-gap follow-ups at real issues.

## Outcome moved

Divergent paths no longer depend on an agent remembering them. Juniper can see them on GitHub, and every session is shown what waits on her.

## Current architecture

Deferred work was in a private agent memory file, in PR reports, and on the host-local agent board. None of these could be filtered or seen as one list.

## Architecture touched

Claude Code SessionStart hooks only. No service, bus, schema, or env changes.

## Files changed

- `scripts/hooks/session_start_open_tickets.py`: new hook. It runs `gh issue list -R <repo>` with an 8 s timeout and fails silently.
- `tests/test_session_start_open_tickets.py`: tests the summary format, the cap on listed tickets, the explicit repo argument, and silence when `gh` fails.
- `.claude/settings.json`: registers the hook.
- `CLAUDE.md`: adds the "Ticket board" subsection and updates the section 11 wording.

## Schema / bus / API changes

- None.

## Env/config changes

- None.

## Tests run

```text
pytest tests/test_session_start_open_tickets.py -q  -> 5 passed
hook run live from repo and from /tmp -> lists 5 needs-juniper tickets
```

## Evals run

```text
Not applicable (tooling hook, no cognition path).
```

## Docker/build/smoke checks

```text
Not applicable.
```

## Review findings fixed

- Finding: `gh` picked the repo from its cwd, so the hook printed nothing when it started outside the repo.
  - Fix: pass `-R junebug-junie/Orion-Sapienform` explicitly.
  - Evidence: run from /tmp now prints the summary; test `test_main_targets_repo_explicitly`.
- Finding: the CLAUDE.md board line described a Project board that can't exist yet, because the token lacks the `project` scope.
  - Fix: the line now says the board is planned and points at #3268. Labels are the board until then.
  - Evidence: `gh project list` refuses with a missing-scope error.
- Finding: section 11 still said to put follow-up issues in the PR description.
  - Fix: changed it to "file a `follow-up` issue and list it in the PR report".
  - Evidence: CLAUDE.md diff.

## Restart required

```text
No restart required. New Claude Code sessions pick up the hook after merge.
```

## Risks / concerns

- Severity: low
- Concern: 16 older issues (#642-#2286) have no labels yet.
- Mitigation: Juniper sorts them once the board exists (#3268).
- Issues opened: #3255-#3266, #3268. Issues closed: none.
