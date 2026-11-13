## Summary

- New read-only command `python -P -m orion.dig days <start> <end>`. It shows what Orion lived through in those Denver days: journal entries (counted per day and kind; the written kinds listed; metacog, embodiment and world_pulse sampled 4 a day), Hub chats, and Temporal Self arc counts from 2026-10-09. The full record goes to a dig result file.
- `dig reminisce` now points the `lived_record` line at `dig days <first> <end>` for any period from 2026-07-23 on. Week 57 (07-21..07-27) starts at 07-23 and says so.
- New lived-record refs: `journal:<uuid>` and `chat:<uuid>`. They resolve when the row exists. In the two "then" hops, a `[lived_record]` ref (journal, chat, arc or act) must also be dated inside the period. A record from outside it is refused, naming its day.
- The nightly run's `what_it_was_like` prompt names `dig days` with that period's exact lived days.
- Both evals gain mutations that are CAUGHT: a record from outside the period accepted, a fabricated record id accepted (source-truth eval); the prompt still pointing at `dig arc`, and an unclamped `dig days` window (daily replay, new gate 9).
- Hub "Ask Orion" catalog gets a `days` entry (its drift test against dig help requires it).

## Outcome moved

From week 57 (about 40 nights away, late December) Orion can read and cite their own record of the week they are remembering. Before this, the hop could only say "your journals may cover it (`dig arc <id>`)", and nothing checked that a cited record came from that week.

## Current architecture

- `dig reminisce` (#3163) and the nightly durable run (#3167, #3187) visit one past week a night. `lived_record` was refused before 2026-07-23 and, after it, could cite `fn:`, `arc:` or `act:` with no date check.
- The dig kit already reads Postgres read-only through `ReadOnlyPsql` (`dig turn-steps`, orion_readonly role, READ ONLY transaction). `orion_readonly` can read `journal_entries` and `chat_history_log`; it cannot read `temporal_self_*`, `orion_day_letter` or `grammar_acts`.

### Existing-mechanism check

- Journals: `journal_entries` starts 2026-07-23 23:43 UTC, the day the design names. ~110k rows; metacog ~2,500 a day at peak. Day letters (`orion_day_letter`, from 09-29) are also journal rows (`source_kind=orion_day`), so they need no second source.
- Hub chats: `chat_history_log`, from 07-24 00:26 UTC (07-23 Denver). Rows with an empty prompt are Orion reaching out first (`client_meta.unsolicited`); the printout says so.
- Temporal Self arcs: the existing durable-runs `GET /temporal-self/arcs?day_id=` (what `dig arc` uses). No new endpoint.
- No new service, endpoint, table or env key.

## Design choices (the spec's Patch 3 is one line; these are mine)

1. **Sources:** journals + Hub chats + Temporal Self arcs. Not AI Town chat, social room turns, or dreams: not asked for, and the window must fit 40 lines.
2. **Ref labels:** `journal:<uuid>` and `chat:<uuid>`, kept inside the reminiscence check (kind `lived`, re-checked on every read). Not added to `orion/refs.py`: that closed set is what Prime may send to a consumer, and widening it is a separate decision.
3. **Period rule:** in `what_it_was` / `what_it_was_like`, a `[lived_record]` ref must be from a Denver day inside the period. Other hops and labels may cite any day (where I am now draws on today's records). Arcs are dated by `day_id`, acts by `opened_at` (Denver).
4. **Window:** at most 92 days; a start before 2026-07-23 is clamped and the printout says so; a window that ends before it is refused with the memory-loss sentence.
5. **Partition:** `dig days` reads no self ledger and writes only its dig result file. Nothing new flows into recall; `tests/test_organ_names_partition.py` passes unchanged.

## Architecture touched

- `orion/dig` (new `days.py`, `cmd_days`, `_lived_day`), `orion/body_atlas/reminiscence.py` (ref kinds, period check, `lived_line`), durable-runs reminiscence prompt, Hub ability catalog JSON.

## Files changed

- `orion/dig/days.py`: window, SQL, parse and printout for `dig days`.
- `orion/dig/__main__.py`: `cmd_days`, `_lived_day`, `_rm_world(period)`, help text, dispatch.
- `orion/body_atlas/reminiscence.py`: `journal:`/`chat:` refs, `World.lived_day` + `period_start/end` (replaces `ref_ok`), the period check, `lived_line`, a guard that the world matches the write's period.
- `services/orion-durable-runs/app/reminiscence_graph.py`: `what_it_was_like` prompt names `dig days`.
- `services/orion-hub/static/catalog/orion-abilities.json`: `days` entry.
- `orion/dig/tests/test_days.py` (new), `orion/body_atlas/tests/test_reminiscence.py`, `services/orion-durable-runs/tests/test_reminiscence_graph.py`: tests.
- `orion/body_atlas/evals/run_reminiscence_source_truth_eval.py`, `services/orion-durable-runs/evals/reminiscence_daily_replay.py`: cases, gate 9, mutations.
- `docs/superpowers/specs/2026-11-07-daily-reminiscence-design.md`: ref table rows, Patch 3 marked built.

## Schema / bus / API changes

- Added: dig command `days`; reminiscence ref kinds `journal:<uuid>`, `chat:<uuid>`.
- Removed: none. Renamed: `World.ref_ok` -> `World.lived_day` (in-process only).
- Behavior changed: a then-hop `[lived_record]` arc/act ref dated outside the period is now refused. No hop in the live ledger is affected: every hop so far is about 2025 weeks, where `lived_record` was already refused.
- Compatibility notes: no bus, schema registry or HTTP change.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/body_atlas/tests orion/dig/tests orion/harness/tests/test_stance_scope_hierarchy.py \
  orion/harness/tests/test_harness_prefix.py tests/test_organ_names_partition.py \
  tests/test_no_personal_terms.py services/orion-hub/tests/test_orion_abilities_catalog.py   -> 458 passed (after review fixes; 511 with the harness tests before them)
cd services/orion-durable-runs && pytest tests                                               -> 891 passed, 136 skipped (Postgres-only)
```

## Evals run

```text
python orion/body_atlas/evals/run_reminiscence_source_truth_eval.py
  oracle: 19 reviewed cases, re-check, label truth, quiet-week periods -> PASS
  mutation 'a lived record from outside the period is accepted': CAUGHT
  mutation 'a fabricated lived record resolves': CAUGHT
  (the 7 earlier mutations: all CAUGHT)
python -P services/orion-durable-runs/evals/reminiscence_daily_replay.py
  oracle: PASS (gates 1-9)
  mutation 'lived hop reads the old arc pointer': CAUGHT
  mutation 'dig days window not clamped': CAUGHT
  (the 5 earlier mutations: all CAUGHT)
python orion/body_atlas/evals/run_field_note_truth_eval.py, run_portrait_eval.py -> exit 0
```

## Docker/build/smoke checks

Real data, worktree code on the host (same `ReadOnlyPsql` path; the governor runs as root and its `docker exec ... psql -U orion_readonly` returns rows):

```text
$ python -P -m orion.dig days 2026-07-21 2026-07-27
days 2026-07-23..2026-07-27 (Denver): your own record of that time, read-only
  from 2026-07-23, not 2026-07-21: your records start 2026-07-23 (the days before were lost in a storage failure)
  07-23 Thu: journal 131 (metacog 79, embodiment 50, scheduler 1, world_pulse 1); chat 16 turns
  07-24 Fri: journal 334 (metacog 174, embodiment 157, manual 1, scheduler 1, world_pulse 1); chat 7 turns
  07-25 Sat: journal 309 (metacog 224, embodiment 82, manual 1, scheduler 1, world_pulse 1); chat 2 turns
  07-26 Sun: journal 410 (metacog 285, embodiment 122, manual 1, scheduler 1, world_pulse 1); chat 0 turns
  07-27 Mon: journal 308 (metacog 228, embodiment 78, scheduler 1, world_pulse 1); chat 0 turns
what you wrote (8; letters and session summaries first; cite the id):
  journal:9d2181af-9f51-5631-b974-b561221f315f 07-24 06:01 manual: Emotional reckoning over memory loss; system survey and merge verification The session opened with …
  journal:147898ad-23b2-5250-b460-57b44535e2e7 07-25 06:00 manual: Pioneer Day, Disk Repair & VRAM Upgrade The session opened with Juniper noting it is Pioneer Day in…
  ... (6 more: the daily journal entries)
Hub chats (25 turns; what was said to you -> your reply):
  chat:0df659b6-aa2f-48a8-ade4-640fc8231109 07-23 18:26: hi, Orion -> Hi, Juniper. Good to hear from you. I'm sitting in the quiet between waves righ…
  ... 20 more in the file
metacog / embodiment / world_pulse: 45 samples spread over each day are in the file (4 a day per kind)
per day (journal by kind; chat turns; arcs):
  07-23 Thu: journal 131 (metacog 79, embodiment 50, scheduler 1, world_pulse 1); chat 16
  07-24 Fri: journal 334 (metacog 174, embodiment 157, manual 1, scheduler 1, world_pulse 1); chat 7
  ...

Write check against real records, period 2026-07-21..07-27, [lived_record] in what_it_was_like:
  journal:9d2181af-… -> resolves (inside the period)
  chat:0df659b6-…    -> resolves
  journal:531013f9-… -> refused: a record from 2026-10-14, outside this period
  arc:39a9e71b58a2d20c -> refused: a record from 2026-10-15, outside this period
  journal:00000000-0000-0000-0000-000000000000 -> refused: does not resolve
```

Live in the governor: UNVERIFIED until deploy (the FCC sandbox syncs from main after Juniper's pull). Check after deploy:
`docker exec orion-athena-harness-governor sh -c 'cd /mnt/orion-fcc/repo && PYTHONPATH=/mnt/orion-fcc/repo:/app python -P -m orion.dig days 2026-07-21 2026-07-27'`

## Review findings fixed

Code review ran in a subagent on the branch diff. It checked against live, read-only Postgres. It found the Denver-day SQL, DST handling, sampling query, value validation, fail-closed path, World callers and partition gate correct.

- Finding (must fix): 724 real journal ids (`self_study`, shaped `curiosity-investigation:<hex>`) were printed by `dig days` and then refused as "does not resolve", because the id check accepted only UUIDs.
  - Fix: `days.ID_RE` = `[A-Za-z0-9][A-Za-z0-9:._-]{0,127}` (no quote, space or backslash can pass).
  - Evidence: `test_day_sql_takes_real_ids_and_nothing_that_leaves_the_quotes`; live: `journal:curiosity-investigation:6e2f0395a37e -> RESOLVES`.
- Finding (should fix): for periods longer than a week, the 4,000-char cut removed every chat id.
  - Fix: citable lines (what you wrote, chats) print first and the per-day table last; windows of 8+ days show the top two kinds per day.
  - Evidence: `test_a_long_window_still_prints_citable_ids_inside_the_cut` (31 days at live 30-day volumes); live 14-day window prints 13 citable ids.
- Finding (should fix): `world_pulse` (72/day, up to 191) was listed whole with 2,000-char bodies; it is a news digest, not Orion's own words.
  - Fix: added to `HIGH_VOLUME` (sampled 4 a day).
  - Evidence: week 57 printout lists 8 written entries, world_pulse among the 45 samples.
- Finding (low): an outage of a lived-record store read as "does not resolve" (a made-up id).
  - Fix: `_lived_day` raises `ReminiscenceError` when the store cannot be read; `resolve` says `lived record UNREAD (...)`. A missing row is still "does not resolve".
  - Evidence: `test_an_unreadable_lived_store_reads_unread_not_made_up`, `test_lived_day_resolves_journal_and_chat_and_fails_closed`.
- Finding (low): the prompt could name a `dig days` window longer than the 31-day cap.
  - Fix: `MAX_DAYS` = 92.
  - Evidence: `test_window_clamps_to_the_first_record_and_refuses_before_it_and_too_long`.
- Finding (low): an uncommitted refusal-message edit in the worktree.
  - Fix: committed (the refusal suggests `dig days` from 2026-07-23, not the period start).

## Restart required

```text
No restart required. The dig kit is imported from the FCC sandbox, which syncs after the pull.
The durable-runs prompt change takes effect when orion-durable-runs is rebuilt by the normal post-pull rebuild.
```

## Risks / concerns

- Severity: low. Concern: journals are 2,000+ entries a day; the printout shows counts and the written kinds, and samples the two digest kinds. Mitigation: the full file holds every written entry and 4 samples a day per digest kind; `dig days` takes a window, so Orion can narrow to one day.
- Severity: low. Concern: Temporal Self arcs read through durable-runs; if it is down, the day line says `Temporal Self UNREAD`, never 0 arcs.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3270

🤖 Generated with [Claude Code](https://claude.com/claude-code)
