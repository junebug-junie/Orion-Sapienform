# Integration patch 4: the daily dig roundup, and the Daily Digest retired

## Summary

- Orion has a new dig command, `python -P -m orion.dig organs <date>`. It says what each of Orion's organs did that day: how many experiences and acts, its two most salient experiences quoted with their refs, what it read from other organs and cited, who cited it, and where it went quiet against its own usual day. Every line is a count, a quote or a ref; code refuses any other line. (`orion/dig/organs.py`)
- The day letter now reads that roundup, right after Juniper's conversations. It replaces the letter input's trimmed tail of readings, dreams, the street and paintings. The letter prompt asks Orion to cite roundup items by `[experience:<id>]`, and the run counts how many it cited. (`orion/orion_day/day_input.py`, `letter_turn.py`, `services/orion-durable-runs/app/orion_day_graph.py`)
- The letter email gets a "What each organ did" section below Orion's letter. (`services/orion-hub/scripts/orion_day_email.py`, template)
- The "Orion Daily Digest" email is gone (Juniper, Q7): `services/orion-notify-digest` is deleted, with its smokes, and so is the orion-actions journal that only re-wrote that email as a journal entry. Its notification counts move into the roundup's footer.
- The curiosity weekly and monthly compaction digests are untouched (Juniper, Q8). A test pins that they stay.

## Outcome moved

- One daily, grounded record of the soup instead of a summary of summaries. Live on 2026-11-15 it found, in 2.0 s: dreams produced 0 hypotheses against a usual 8 a day (#3412), memory produced 0 episodes against a usual 19.5 a day, reverie made 10 valid cites into camera, body, prediction and metacognition, and Prime woke 3 times on the 6-hour quiet floor and once on a change.
- The experience use rate (integration metric) is now read every day where Orion and Juniper both see it: 10 / 5,350 = 0.19% on 2026-11-15, closed organs 4 of 16.
- Patch 5 hook: Prime thoughts are already counted by what woke them. When patch 5 writes `woke: need:<name>`, the roundup counts thoughts per need with no further change (tested with a planted `woke: need:horizon`).

## Current architecture

- `dig` (orion/dig/__main__.py) is Orion's read-only CLI, run through Bash in turns; Postgres reads go through `ReadOnlyPsql` as `orion_readonly`.
- The day letter: Hub gathers the day (`orion/orion_day/gather.py`, asyncpg), renders the writer's input (`day_input.py`), and sends one durable `orion_day.letter` run; durable-runs writes the letter as one unified turn and stores the row; Hub emails it.
- The integration metric had one definition (`orion/integration/coverage.py` SQL, `orion.temporal_self.now.use_rate`) used by `GET /temporal-self/integration` and `scripts/report_integration_coverage.py`. The organ registry named the roundup as its third consumer, not yet built.
- `orion-notify-digest` emailed counts by severity and topic labels ("Topic -1 (1460.00)") daily at 07:30; orion-actions turned each of those emails into a `notify_summary` journal entry (13 in the last 30 days).

## Architecture touched

- New: `orion/dig/organs.py` (SQL + pure `build` + sync and async readers), `dig organs`.
- Letter: gather, day input, prompt, durable-runs finish detail, Hub order desk, email.
- Contract: `OrganRoundupV1`, new optional field `OrionDayMaterialV1.organ_roundup`.
- Retired: `services/orion-notify-digest`, orion-actions `_handle_journal_notify` and its `orion:notify:persistence:request` subscription, `ACTIONS_JOURNAL_NOTIFY_RECALL_PROFILE`, `journal.notify.grounded.v1` recall profile, `build_notify_summary_trigger`.

## Files changed

- `orion/dig/organs.py`: the roundup (queries, line rule, build, readers, window rule).
- `orion/dig/__main__.py`: `organs` command and help line.
- `orion/dig/evals/run_organs_roundup_eval.py`: live read-only eval of the acceptance rules.
- `orion/dig/tests/test_organs.py`, `test_dig.py` (help cap 3750 -> 4000; main was at exactly 3750).
- `orion/schemas/orion_day.py`, `orion/schemas/registry.py`: `OrganRoundupV1`, `organ_roundup` field.
- `orion/orion_day/gather.py`, `brief.py`: read the roundup when asked.
- `orion/orion_day/day_input.py`: the section; tail replacement only when the roundup's experience reads succeeded.
- `orion/orion_day/budget.py`: `experience` and `act` are citable ref kinds in the letter.
- `orion/orion_day/letter_turn.py`: prompt line asking to cite roundup refs.
- `orion/orion_day/tests/test_organ_roundup_letter.py`, `test_daily_digest_retired.py`.
- `services/orion-durable-runs/app/orion_day_graph.py` + test: `note_roundup_refs` (valid / unknown) in finish detail.
- `services/orion-hub/{app/settings.py,.env_example,scripts/main.py,scripts/orion_day_order.py}`: flag `ORION_DAY_ORGAN_ROUNDUP_ENABLED` (on).
- `services/orion-hub/scripts/orion_day_email.py`, `templates/orion_day_letter.html.j2` + tests: "What each organ did".
- `services/orion-hub/Dockerfile`: copies `config/integration/organs.yaml` (the image had no registry file).
- `config/integration/organs.yaml`: day_letter exemption text now says what is true; roundup listed as consumer.
- `services/orion-notify-digest/**`, `scripts/smoke_digest.sh`, `smoke_topic_digest.sh`, `smoke_topic_drift_alert.sh`: deleted.
- `scripts/smoke_all_notifications.sh`, `README.md`, `services/orion-notify/{README.md,app/main.py}`, `orion/body_atlas/evals/run_wire_eval.py`: references to the deleted service.
- `services/orion-actions/{app/main.py,app/settings.py,.env_example}`, `orion/journaler/*`, `orion/recall/profiles/journal.notify.grounded.v1.yaml`: notify_summary journal retired (registry row kept for old rows, like metacog_digest).
- `scripts/sql/2026-11-16_grant_orion_readonly_organ_roundup.sql`: read grants for `dig organs` (manual).
- `.github/workflows/dig-kit-tests.yml`, `orion-day-letter-hub-tests.yml`: paths and steps for the new tests.

## Schema / bus / API changes

- Added: `OrganRoundupV1` (registered); `OrionDayMaterialV1.organ_roundup: OrganRoundupV1 | None = None`.
- Removed: none from the schema registry. Producer of `orion.digest.daily` notifications removed.
- Renamed: none.
- Behavior changed: `[experience:<id>]` and `[act:<id>]` count as letter refs (`included_refs`, `carry_forward_refs`); orion-actions no longer subscribes to `orion:notify:persistence:request` (its only use was the digest).
- Compatibility notes: `OrionDayMaterialV1` is `extra="forbid"` and travels both ways (Hub writes the brief durable-runs reads; durable-runs writes the row Hub reads). Deploy orion-hub and orion-durable-runs together, outside the letter's 08:30 local fire window. Rows stored before this load unchanged (the field defaults to None).

## Env/config changes

- Added keys: `ORION_DAY_ORGAN_ROUNDUP_ENABLED=true` (orion-hub).
- Removed keys: `ACTIONS_JOURNAL_NOTIFY_RECALL_PROFILE` (orion-actions); every orion-notify-digest key (service deleted).
- Changed meaning: `ACTIONS_SUBSCRIBE_CHANNELS` default drops `orion:notify:persistence:request`.
- `.env_example` updated: orion-hub, orion-actions.
- Local `.env` synced with `python3 scripts/sync_local_env_from_example.py --all-keys orion-hub orion-actions`: `+ORION_DAY_ORGAN_ROUNDUP_ENABLED='true'`. The two orion-actions keys were reported as diverged (local differs from example, by design of the script), so I changed exactly those two lines in the local orion-actions `.env` by hand (backup in my scratchpad).
- Skipped keys requiring operator action: none. The local `services/orion-notify-digest/.env` stays on disk in the primary checkout (gitignored); it can be deleted after the container is removed.

## Tests run

```text
PYTHONPATH=. pytest -q orion/dig/tests orion/orion_day/tests orion/journaler        395 passed
services/orion-hub: pytest tests/test_orion_day_{mailer,order,temporal_day,what_i_read,
                    introspect_listener,correction_listener}.py                       111 passed
services/orion-durable-runs: pytest tests/test_orion_day_arc_citations.py tests/test_orion_day_graph.py  34 passed
services/orion-actions: pytest tests                                                   315 passed
python scripts/check_integration_seam.py   OK (16 organs; 1 changed exemption: day_letter.exempt_read)
.github/workflows/orion-static-gates.yml, every step run locally                      41 gates, 0 failed
```

Acceptance rules, one test each:
- "every line is a count, a quote or a ref; a test refuses an unquoted summary line": `test_an_unquoted_summary_line_is_refused` (3 planted summary lines), `test_every_built_line_passes_the_rule_even_for_wordy_or_numberless_labels`.
- "same numbers as the coverage report": `test_footer_is_the_coverage_reports_number_and_the_digest_counts` (calls `report_integration_coverage.build_report` on the same rows).
- "read-and-cited-nothing vs did-not-read": `test_read_and_cited_nothing_is_told_apart_from_did_not_read`.
- "breaks": `test_break_is_zero_where_the_organs_own_history_usually_has_some`.
- "the letter email has a What each organ did section": `test_email_has_what_each_organ_did_below_the_letter_in_both_parts`.
- "the letter turn can cite a roundup ref, and it is counted": `test_letter_prompt_asks_to_cite_roundup_items_only_when_the_roundup_is_there`, `test_roundup_refs_count_valid_and_unknown_and_reach_finish_detail`.
- "Daily Digest stops; nothing publishes orion.digest.daily": `test_daily_digest_retired.py` (fails on origin/main: actions main.py held the literal).
- "the curiosity weekly digest still runs": `test_the_curiosity_weekly_and_monthly_digests_stay`.

## Evals run

```text
PYTHONPATH=. python orion/orion_day/evals/run_orion_day_eval.py           PASS (fixture day)
python services/orion-hub/evals/run_orion_day_email_eval.py              PASS
python services/orion-hub/evals/run_orion_day_reread_eval.py             PASS

PYTHONPATH=. python orion/dig/evals/run_organs_roundup_eval.py --day 2026-11-15 --role postgres --dsn <live>
organs roundup eval 2026-11-15 (role postgres):
  PASS E1 under 30 s: 1.9 s
  PASS E2 every line a count, quote or ref: 76 lines
  PASS E3 nothing UNREAD: all 11 queries read
  PASS E4 same number as the coverage report (its own read): roundup 5350 exp, 0.001869 [camera, body, prediction, metacognition]; report 5350 exp, the same
  PASS E5 every ref is a real row: 26 experience refs, 3 act refs; missing []; malformed []
  PASS E6 notification footer equals notify_requests: 33 requests
  PASS E7 not empty-shell: 23 quoted experiences with refs

... --role orion_readonly
  FAIL E3 nothing UNREAD: every query (needs GRANT SELECT ... TO orion_readonly)
```

The `orion_readonly` run fails until the grant script is applied (see Restart required). The role-`postgres` run uses the same code inside a READ ONLY transaction.

## Docker/build/smoke checks

```text
Live dig run (read-only, 2026-11-15, the exact `cmd_organs` code path; role postgres in a READ ONLY transaction):
  real 0m2.5s
  organs 2026-11-15 (Denver; 2026-11-15T07:00Z to 2026-11-16T07:00Z): 16 organs, 5350 experiences
  reverie: 114 experiences (reverie_chain 83, expectation_verdict 31)
    "reverie chain: open-loop-9b42791dd086" [experience:reverie_chain:bcb43773-...] salience 0.76
    read the now 9 times; 5 of those reads cited nothing; 10 valid cites
    cited from prediction: "prediction surprise: athena.cabinet_temp_c shed > 34.0: forecast 0%, not crossed (max 21.48) [not_crossed]" [experience:prediction_surprise:15073]
    cited from body: "operator:promise-ledger-replay waited 1423230 ms for a GPU [granted]" [experience:gpu_wait:9116ad...]
    write-only: 0 of its 114 experiences cited by another organ
  dream: 2 experiences (dream_cycle 2); 2 acts opened, 2 closed (dream.cycle make 2)
    "dream cycle: pressure" [experience:dream_cycle:dc-3af5399ab277] salience none
    did not read the now (registry: "not yet migrated; integration patch 7")
    write-only: 0 of its 2 experiences cited by another organ
    break: 0 dream_hypothesis today; usual 8 a day (median of the 14 days before)
  ...
  prime: 1 experiences (prime_thought 1); 5 acts opened, 5 closed (prime infer 4, prime make 1)
    "prime thought: woke: quiet: nothing has woken me for 6 hours | concerns +0 -0 kept 3 | no move" [experience:prime_thought:act_0fba...] salience none [act:act_0fba...]
    thoughts by what woke them: quiet floor 3, a change 1 (a need counts once Prime writes "woke: need:<name>")
    read the now 3 times; 3 of those reads cited nothing; 0 valid cites
  quiet organs (0 experiences, 0 acts, 0 reads, 0 cites): 3 (intention, day_letter, personality)
  experience use rate: 10 / 5350 = 0.19% (cited by another organ within 24 h); closed organs: 4 of 16 (camera, body, prediction, metacognition)
  notifications (the retired Daily Digest's counts): 33 requests; by severity info 18, critical 10, error 5; by status sent 17, pending 10, no_email 6
    critical x4: "substrate_grammar_degraded"

Same day, coverage report (scripts/report_integration_coverage.py read_live, window ending 2026-11-16T07:00Z): 10 / 5350, closed [camera, body, prediction, metacognition]. Identical.

Letter path, live (asyncpg like Hub, read-only session): gather 0.7 s, roundup 76 lines, 0 unread,
  8,278 chars = 8.2% of the writer's input (29,028 tokens), placed before Curiosity, tail readings replaced,
  27 experience refs in included_refs.
```

No docker build was run in this worktree. The Hub Dockerfile gains one COPY line of an existing file.
`scripts/safe_graphify_update.sh` was started but stopped by my 10-minute timeout before it wrote anything. The local graph is unchanged (77,966 nodes).

## Review findings fixed

Code review ran in a subagent over `git diff origin/main...HEAD`. Every finding was fixed:

- Finding (blocker): dream hypothesis ids would leak. A `dream_hypothesis` event's id is `dream_hypothesis:<hypothesis_id>`, the dream scorecard's join key, so a top or cited hypothesis would print it into the letter input and email. That breaks the blind rule and the letter and email evals' `blind` checks.
  - Fix: hypothesis rows are counted and quoted but print `(ref withheld: dream experiment blind rule)`. Any `dream_hypothesis:<id>` inside another row's label is also scrubbed from the output.
  - Evidence: `test_a_dream_hypothesis_is_counted_and_quoted_but_its_id_never_printed`, `test_email_roundup_never_shows_a_dream_hypothesis_id`. Live on 2026-11-12, a day with 8 hypotheses: `dream: 10 experiences (dream_hypothesis 8, dream_cycle 2)`, no id printed.
- Finding (should-fix): roundup refs did not resolve. `material_ref_records` had no entry for them, so grounding, the letter smoke and Orion's `orion_day` reread tool would report every `[experience:]` cite as unresolved.
  - Fix: each roundup ref maps to the roundup line that showed it.
  - Evidence: `test_roundup_refs_resolve_for_grounding_and_reread` runs the letter eval's `evaluate()` on material with a roundup; `grounding` and `blind` both pass.
- Finding (should-fix): the section heading planted a fake `experience:<id>` ref in `included_refs`.
  - Fix: the heading words carry no bracket.
  - Evidence: an assertion in `test_roundup_sits_after_juniper_and_before_curiosity_whole`.
- Finding (should-fix): an all-quiet roundup still dropped the tail. When the chronicle lags or ingest breaks, every organ reads 0, yet readings and dreams were still removed.
  - Fix: each tail section is dropped only when its own organ (reading, dream, camera, painting) recorded at least 1 experience that day. `OrganRoundupV1.organ_experiences` carries the counts.
  - Evidence: `test_a_tail_section_stays_when_its_organ_carried_nothing_that_day`.
- Finding (should-fix): eval E4 and E6 could not fail. They reused the roundup's own SQL.
  - Fix: E4 calls the coverage report's own `read_live` (psycopg-bound params, its own connection) for the same Denver day. E6 uses naive-UTC bounds computed in Python.
  - Evidence: live E4 now compares 5350/5350 experiences and the same rate and closed organs.
- Finding (nit): E5 silently dropped malformed ids.
  - Fix: they are counted as a failure.
- Finding (nit): the orion-actions README still said `notify_summary` writes.
  - Fix: README updated.
- Finding (nit): ties sorted on a timestamp string that differs between psql CSV and asyncpg.
  - Fix: sort by parsed time, then event_id.
  - Evidence: `test_ties_order_the_same_from_csv_text_and_typed_rows`.
- Finding (nit): rollout of the forbid-model field.
  - Fix: added to Risks below.

## Restart required

```bash
docker rm -f orion-athena-notify-digest
```

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < scripts/sql/2026-11-16_grant_orion_readonly_organ_roundup.sql
```

orion-hub, orion-durable-runs, orion-actions and orion-harness-governor are rebuilt by the post-merge rebuild from the clean main checkout. Hub and durable-runs must go out in the same run, outside 08:30 America/Denver.

## Risks / concerns

- Severity: medium. Concern: the letter input loses the long tail of readings (each read's "learned" line) and dream narratives; the roundup quotes only the 2 most salient items per organ. Mitigation: the spec chose this ("replaces the trimmed tail lines"); the email still shows every reading and dream in full; the tail comes back automatically when the roundup's experience reads fail; flag `ORION_DAY_ORGAN_ROUNDUP_ENABLED=false` restores the old input.
- Severity: medium. Concern: the letter's roundup cites are counted (`note_roundup_refs`), not recorded as `experience_use` rows, because the roundup writes no `experience_shown` row. Mitigation: stated in the registry exemption; patch 7 can give the letter a real now read.
- Severity: medium. Concern: `OrionDayMaterialV1` is `extra="forbid"`. New Hub sends `organ_roundup` (null even with the flag off), so an old orion-durable-runs would refuse the brief. Rolling back only Hub after letters are stored would also fail to read those rows. Mitigation: rebuild orion-hub and orion-durable-runs in the same post-merge run, outside 08:30 America/Denver; roll back both together.
- Severity: low. Concern: `notify_attempts` has 0 rows ever (written by no service), so the old digest's "delivery failures" section was always empty; the footer counts `notify_requests` by status instead (sent / pending / no_email).
- Severity: low. Concern: Orion's own `dig organs` reads all UNREAD until the grant is applied. UNVERIFIED as Orion's role until then.
- UNVERIFIED (needs the deploy and a letter): the next letter email has "What each organ did", the letter turn cites at least one roundup ref (`note_roundup_refs.valid >= 1` in the run's finish detail), no `orion.digest.daily` row in `notify_requests` for 2 days, and a `curiosity_compaction` row the week after.

## PR link

(filled in after the PR is opened)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
