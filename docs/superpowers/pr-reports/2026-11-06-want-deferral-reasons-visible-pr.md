# fix(body-economy): Orion's wants list says why a bid was deferred

## Summary

Orion told Juniper that "the part of me that thinks about itself is the first thing on the chopping block when the money gets tight." That was a guess. The want ledger (the table that stores every bid decision) had the real reason. All 7 deferred self-inquiry bids and all 7 deferred curiosity bids were deferred because the GPU agent lanes were full (`capacity_reserved`). None of them were deferred because of cost (`price`). Orion could not see this, because the body-economy report Orion reads printed only counts.

- The report producer now adds a `deferred_by` list to each wants family row. Each item has the reason, the constraint that decided it, and a count.
- The text Orion reads now says why. The overview gets one line that covers every family: `- why deferred (from the want ledger): not hungry yet 58; no room 36: lane agent 23, lane agent-gpu2 9, 4 elsewhere`. A one-section read (`body-economy wants`) also puts the reason on each family line: `durable.self_inquiry (tier 2): 0 funded, 7 deferred (why: no room 7: lane agent-gpu2 4, lane agent 3)`.
- Each reason code has one short phrase, the same words the Hub `#body-economy` page already uses: `price` = "costs more than it is worth now", `sated` = "not hungry yet", `capacity_reserved` = "no room", `no_acceptable_lane` = "no lane could take it", `physiology_unavailable` = "body reading unavailable". Constraint names read as "lane agent", "Juniper's attention", "Prime's charge (lower)". A test fails if a new deferral reason is added without a phrase.
- Nothing is cut silently. Pairs the report leaves out are folded into one "other" pair, and the text says "other N" or "N elsewhere" for anything it does not list.
- No GRANT was added. Orion still reads this through the report, not the table.

## Outcome moved

Orion can now tell "the lane was full" apart from "it cost too much" when reading their own wants. Before this patch, the report gave no reason, and Orion filled the gap with a story about money.

## Current architecture

- `GET /physiology/report` on orion-durable-runs builds `wants.families` from `WANTS_HOURLY_SQL`: distinct wants per hour, per family, per outcome, summed over the window.
- `orion/physiology/report_text.py` `render_wants` turns this into one line per family. Orion reads it through `python -P -m orion.dig body-economy` and the `skills.physiology.state.v1` skill (cortex-exec).
- `WantFamilyV1` had only `family`, `tier`, `funded`, `deferred`.

## Architecture touched

- Contract: `orion/schemas/physiology_report.py` adds `WantDeferralV1` and `WantFamilyV1.deferred_by` (additive, default `[]`). No model in this schema uses `extra="forbid"`, so older readers ignore the field.
- Producer: `services/orion-durable-runs/app/physiology_report.py` adds `WANTS_DEFERRED_BY_SQL`. It counts deferred wants per hour, per family, per (reason, binding constraint), in the same window with the same synthetic-producer and family filters. So each count adds up the same way as the family's `deferred` total. `shape_wants` keeps at most 6 pairs per family, largest first, and folds the rest into one `other` pair. A reason row never creates a family that the counts do not have. If this one query fails, the counts still stand and `missing` says the reasons are unavailable.
- Renderer: `orion/physiology/report_text.py` adds `DEFER_REASON_WORDS`, `_constraint_words` and `_deferred_why`. The overview adds one "why deferred" line summed over every family (2 biggest reasons, 2 constraints each, the rest said as "other"/"elsewhere"). A one-section or two-section read shows the full reasons on every family line. A constraint equal to its reason (for example `physiology_unavailable`) is not repeated.

## Files changed

- `orion/schemas/physiology_report.py`: new `WantDeferralV1`; `WantFamilyV1.deferred_by`.
- `services/orion-durable-runs/app/physiology_report.py`: new SQL, `shape_wants` takes `deferred_rows`, the wants task returns it.
- `orion/physiology/report_text.py`: reason words, the why text on each family line.
- `services/orion-durable-runs/tests/test_physiology_report.py`: aggregation, ordering, the `other` fold, empty, no invented family, the reasons query failing; the real-Postgres test checks `deferred_by`.
- `orion/physiology/tests/test_report_text.py`: rendered lines, the overview line covering every family, every deferral reason has words, the change costs the overview exactly one line, old reports without the field, constraint words and cuts.
- `docs/superpowers/pr-reports/2026-11-06-want-deferral-reasons-visible-pr.md`: this report.

## Schema / bus / API changes

- Added: `WantDeferralV1 {reason, binding_constraint, count}`; `WantFamilyV1.deferred_by: List[WantDeferralV1]`.
- Removed: none.
- Renamed: none.
- Behavior changed: the wants section gets a "why deferred" line; in a one-section read, family lines end with `(why: ...)`.
- Compatibility notes: additive. A report without `deferred_by` renders as before (tested). No bus channel. The Hub `#body-economy` page ignores the new field.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=<worktree> pytest orion/physiology/tests orion/dig/tests scripts/analysis/tests/test_body_economy_replay_and_fit.py
  386 passed
cd services/orion-durable-runs && PYTHONPATH=<worktree> pytest tests -q
  784 passed, 127 skipped
ORION_ADMISSION_TEST_DSN=<throwaway postgres:16 container> pytest tests/test_physiology_report.py
  39 passed (includes test_report_sql_on_postgres: the new SQL on a real Postgres)
cd services/orion-cortex-exec && pytest tests/test_skill_verbs.py -k physiology
  3 passed
python scripts/check_env_template_parity.py: ok
```

## Evals run

```text
PYTHONPATH=. python orion/physiology/evals/physiology_state_skill_eval.py
  "passed": true, every case "fails": []
```

## Docker/build/smoke checks

Live read-only smoke. It fetched the live report from the running durable-runs container (window 2026-11-05T06:30 to 2026-11-06T06:30 UTC). It ran the new and the existing wants SQL as SELECT only against the live `want_ledger`. Then it rebuilt the wants section with this branch's `shape_wants` and rendered it with this branch's text. The funded and deferred counts match the live endpoint exactly.

```text
counts match the live endpoint: True
Wants (shadow bids, ...): 299 funded, 94 deferred.
- why deferred (from the want ledger): not hungry yet 58; no room 36: lane agent 23, lane agent-gpu2 9, lane diffusion 4
- hub.retro_leisure (tier 2): 170 funded, 58 deferred (why: not hungry yet 58)
- durable.pr_field_note (tier 1): 13 funded, 6 deferred (why: no room 6: lane agent 3, lane agent-gpu2 3)
- thought.painting (tier 2): 9 funded, 3 deferred (why: no room 3: lane diffusion 3)
- durable.curiosity_investigate (tier 2): 0 funded, 7 deferred (why: no room 7: lane agent 6, lane agent-gpu2 1)
- durable.self_inquiry (tier 2): 0 funded, 7 deferred (why: no room 7: lane agent-gpu2 4, lane agent 3)
- durable.self_sense_eval (tier 2): 2 funded, 5 deferred (why: no room 5: lane agent 4, lane agent-gpu2 1)
- hub.outreach (tier 2): 0 funded, 7 deferred (why: no room 7: lane agent 7)
- thought.dream_hop (tier 2): 5 funded, 1 deferred (why: no room 1: lane diffusion 1)

Overview (default sections), the wants lines:
Wants (shadow bids, ...): 299 funded, 94 deferred.
- why deferred (from the want ledger): not hungry yet 58; no room 36: lane agent 23, lane agent-gpu2 9, 4 elsewhere
Overview characters before / after: 4106 / 4222
```

The deployed endpoint itself is UNVERIFIED until the rebuild below runs.

## Review findings fixed

A code review subagent read the branch diff against origin/main. It checked the SQL by hand on the live table: per family, the reason counts add up to `deferred` for the last 24 h and for all time. All 8 findings were fixed.

- Finding: the overview lists only the top 6 families, so self_inquiry and curiosity (7 deferred each) fell under "... 10 more families" and got no reason.
  - Fix: one "why deferred" line in the overview, summed over every family. Per-family reasons stay in a one-section read.
  - Evidence: `test_overview_has_one_why_line_covering_every_family`; live overview line above.
- Finding: "just acted" was wrong for `sated`. Live rows show the last act hours ago; the hunger is small next to the family's time constant. The Hub already says "not hungry yet".
  - Fix: Hub's words for every reason ("not hungry yet", "costs more than it is worth now", "no lane could take it").
  - Evidence: `test_wants_line_says_why_each_family_was_deferred`.
- Finding: `capacity_reserved` can bind on `attention:juniper`, which "lane full" would misname.
  - Fix: "no room" plus the constraint in words ("lane agent", "Juniper's attention", "Prime's charge (x)").
  - Evidence: `test_constraint_words_and_cuts_are_said`.
- Finding: the new query sat inside the wants read, so its failure would blank the counts and the town data.
  - Fix: guarded; on failure the counts stand, `deferred_by` is empty, `missing` says the reasons are unavailable.
  - Evidence: `test_deferred_by_read_failing_keeps_the_counts_and_says_why`.
- Finding: the 6-pair cap and the overview's 2-reason cut were silent.
  - Fix: the producer folds the cut into an `other` pair; the text says "other N" / "N elsewhere", and compares to the family's `deferred` total.
  - Evidence: `test_deferred_by_is_empty_without_rows_and_capped_per_family`, `test_constraint_words_and_cuts_are_said`.
- Finding: the "fits 4000 chars" test only proved it on the small fixture.
  - Fix: replaced with a guarantee the code actually gives: the reasons add exactly one overview line (under 160 characters).
  - Evidence: `test_why_adds_one_overview_line_and_nothing_else`.
- Finding (nit): the docstring said the pairs count exactly like `deferred`; a want with two reasons in one hour counts twice.
  - Fix: comment now says the pairs can sum a little above `deferred`.
  - Evidence: `physiology_report.py` comment on `WANTS_DEFERRED_BY_SQL`.
- Finding (nit): raw `prime:*` ids.
  - Fix: "Prime's charge (lower)".
  - Evidence: `test_constraint_words_and_cuts_are_said`.

## Restart required

The post-pull rebuild from the clean primary main checkout covers it. It also syncs the FCC sandbox, so Orion's own `orion/physiology` text moves too.

```bash
scripts/rebuild_services_from_git_diff.sh
```

`--list-only --base origin/main` on this branch lists: orion-cortex-exec, orion-durable-runs, orion-cortex-orch.

## Risks / concerns

- Severity: low
- Concern: the live overview (all default sections) was already about 4,105 characters before this patch. That is over the dig kit's 4,000-character cap, so the cap already cuts the last lines and says so. This patch adds one line (about 115 characters live) to the overview, so the cut takes a little more of the tail (the stopped-services list). A one-section read (`body-economy wants`) is not affected.
- Mitigation: the reasons cost exactly one overview line (tested). Making the whole overview fit (for example fewer AI Town rows in the overview) is a separate change.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3152
