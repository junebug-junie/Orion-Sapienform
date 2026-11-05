# Office sessions: one visit, one record (who, probably, and what they did)

Step B of `docs/superpowers/specs/2026-11-05-office-camera-attention-and-reach-out-design.md` (PR #3127).
Step A (the office prior, PR #3129) made Orion say "probably Juniper" in the office. This step turns each
visit into one stored record and puts it into Orion's day story.

## Summary

- Each visit to the office camera is now one row: when it began and ended, who was probably there, and an
  ordered list of what they did (the camera's own captions, repeats folded). Table `vision_office_session`,
  writer `services/orion-sql-writer/app/vision_office_sessions.py`.
- Orion's day story now says who was in the office and what happened, instead of a nameless "company"
  stretch: "Juniper (probably) in the office: standing in a room, looking at a box on the floor; sitting at a
  desk, working on a computer; ...". The Temporal Self relabels the office arc from the final visit row.
- When a visit has someone Orion does not know, the existing "Who is this?" card opens in the Hub right away,
  also while the visit is still going on. Before, the office could only reach that card through the street rule
  (10 sightings over 5 days), which a split office visit never reaches.
- The office's "probably Juniper" covers one person only. Two people in view never read as two "probably
  Juniper"s; the second one is "someone I don't know".
- No attention want, no salience number, no Prime, curiosity or chat reach-out: that is step C.

## Outcome moved

- Before: Juniper's 11-05 visit (18:24-18:31 local) reached the day story as an unlabelled company arc
  ("company: cam0"). After (live replay, below): one session, "Juniper (probably)", 17 timeline entries.
- Before: an unknown visitor in the office could not get a "Who is this?" card (street rule). After: the card
  opens on the first pass that sees the extra person (tested on real Postgres).

## Live replay of Juniper's 11-05 visit (read-only, live database)

`PYTHONPATH=.:services/orion-sql-writer python services/orion-sql-writer/evals/run_office_session_replay.py
--uri postgresql://postgres:postgres@localhost:55432/conjourney --since 2026-11-05T01:00:00Z --until 2026-11-05T02:00:00Z`

```text
session cam0-1f48ba3068b16a16cf7ba9b7e68ecabd  2026-11-05T01:24:07Z -> 2026-11-05T01:31:28Z (left, closed)
  who: Juniper (probably)  (records=16 peak_people=1 unknown=0 reason=None)
  label: Juniper (probably) in the office: standing in a room, looking at a box on the floor; sitting at a desk,
         working on a computer; standing in a closet, looking down and possibly inspecting something; opening a
         cabinet door; standing in an office, leaning against a door (+10 more)
    01:23:48Z  standing in a room, looking at a box on the floor
    01:24:00Z  organizing items in a storage bin
    01:24:15Z  sitting at a desk, working on a computer with a scenic wallpaper
    01:24:30Z  sitting at a desk, opening a box
    01:25:43Z  sitting at a desk, working on a computer, and appears to be organizing or unpacking items from a box
    01:26:06Z  sitting at a desk, unpacking a plastic bag and handling a piece of equipment
    01:26:54Z  reaching into a cabinet
    01:27:14Z  standing in a closet, looking down and possibly inspecting something
    01:27:31Z  standing at a desk, reaching for something on the table
    01:27:42Z  sitting at a desk, working on a computer (x2)
    01:29:07Z  sitting at a desk, unpacking or organizing items from a box
    01:29:39Z  opening a cabinet door
    01:29:54Z  sitting at a desk, working on a computer
    01:30:04Z  standing in a closet, reaching into a storage bin
    01:30:19Z  standing in a room, seemingly in the process of moving or organizing items
    01:30:33Z  standing in a room, looking around, and appears to be in the process of organizing or unpacking items
    01:30:45Z  standing in an office, leaning against a door, and appears to be preparing to enter a room
```

The three earlier office visits (11-03, 11-04, before the office crop pipeline of PR #3125) have no person
records; they read "Juniper (probably) in the office" from the prior alone, not "someone I don't know".

The live table does not exist yet (migration not applied; nothing deployed). The replay rebuilds sessions from
the rows already stored.

## Current architecture

- Office presence already opened a `company` arc in the Temporal Self (`presence_transition`, cam0 in
  `ROOM_STREAMS`), labelled only from a confirmed face match, which has not happened since 10-27.
- Office captions reached the day only as subject-less `vision_percept` context with entities (none for captions).
- "Who is this?" asks opened only by the street rule (10 sightings over 5 days) or the face rule (5 faces).
  Office visits split into short records (10 for one visit) and never qualified.

## Architecture touched

- orion-sql-writer: new reducer loop `vision_office_sessions_loop` (60 s), new table, its own "Who is this?" opener.
- Temporal Self (pure, `orion/temporal_self/`): new source kind `office_session`; it opens no arc, it relabels the
  closed office company arc it overlaps most.
- orion-durable-runs: the chronicle reads final session rows (skipped while the table is missing).
- `orion_ask`: no new card type. Office asks are `vision_individual` with `rule:office_session` in `evidence_refs`.

## Files changed

- `services/orion-sql-db/manual_migration_vision_office_session_v1.sql` (+ `_rollback.sql`): the table. Additive.
- `services/orion-sql-writer/app/vision_office_sessions.py`: spans, timeline, who rule, label, ask opener, cycle.
- `services/orion-sql-writer/app/vision_office_sessions_loop.py`, `app/main.py`, `app/settings.py`, `.env_example`: the loop and its keys.
- `services/orion-sql-writer/app/vision_individuals.py`: the street rule's daily cap no longer counts office asks.
- `services/orion-sql-writer/app/act_downstream_use.py`: `vision_office_session` is an arc source table (pinned by test).
- `services/orion-sql-writer/evals/run_office_session_replay.py`: read-only live replay.
- `services/orion-sql-writer/tests/test_vision_office_sessions.py` (pure), `tests/test_vision_office_sessions_pg.py` (real Postgres).
- `orion/schemas/temporal_self.py`: `office_session` source kind.
- `orion/temporal_self/sources.py`, `arcs.py`, `recent_arcs.py`, `README.md`, `tests/test_office_session_arcs.py`.
- `services/orion-durable-runs/app/temporal_self_sources.py`, `temporal_self_store.py`, `settings.py`, `main.py`,
  `.env_example`, `docker-compose.yml`, `README.md`, `evals/temporal_self_fixture_db.py`,
  `tests/test_temporal_self_chronicle_postgres.py`.
- `.github/workflows/orion-sql-writer-tests.yml`, `.github/workflows/orion-durable-runs-tests.yml`: the new PG test and migration.
- `services/orion-sql-writer/README.md`: docs.

## Schema / bus / API changes

- Added: table `vision_office_session`; Temporal Self source kind `office_session` (`TemporalSourceKind`).
- Removed: none. Renamed: none.
- Behavior changed: office company arcs get a label from the final session row; recent arcs show that label
  without the "company:" prefix; `ASKED_TODAY_SQL` (street rule cap) skips asks marked `rule:office_session`.
- Compatibility: no new arc kind and no new frame field, so frame readers (orion-thought, orion-hub) need no
  update and there is no deploy order. Only orion-durable-runs validates `TemporalSelfEventV1`/state.
- No bus channel.

## Env/config changes

- Added keys (orion-sql-writer): `VISION_OFFICE_SESSIONS_ENABLED=true`, `VISION_OFFICE_SESSIONS_INTERVAL_SEC=60`,
  `VISION_OFFICE_SESSIONS_STREAMS=cam0`, `VISION_OFFICE_SESSIONS_SETTLE_SEC=660`,
  `VISION_OFFICE_SESSIONS_LOOKBACK_HOURS=24`, `VISION_OFFICE_ASK_DAILY_CAP=4`.
- Added keys (orion-durable-runs): `TEMPORAL_SELF_OFFICE_SESSIONS_ENABLED=true` (also in its compose).
- `.env_example` updated: yes, both services.
- local `.env` synced: NO, not by this agent (not allowed to touch the primary checkout). The orchestrator must run
  `python scripts/sync_local_env_from_example.py orion-sql-writer orion-durable-runs` from the primary checkout.
- Skipped keys requiring operator action: none.

## Migration

`services/orion-sql-db/manual_migration_vision_office_session_v1.sql`, header
`-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer`. CREATE TABLE / CREATE INDEX only:
`orion.sql_migration_drift.auto_apply_refusals` returns no refusals, so the post-pull applier applies it before
the sql-writer rebuild. orion-durable-runs skips its read while the table is missing.

## Metric gate

No new metric, signal or salience number. "Unknown people" is a count derived from two existing stored values
(people per window, `vision_scene_inventory.counts.person`, and the person model's guesses); it feeds a card and a
label, no model or pipeline score.

## Tests run

```text
pytest services/orion-sql-writer/tests                                  1239 passed, 190 skipped
  incl. tests/test_vision_office_sessions.py (pure)                      25 passed
ORION_TEST_PG_URI=<throwaway postgres:16> pytest tests/test_vision_office_sessions_pg.py   5 passed (not skipped)
pytest orion/temporal_self/tests                                        278 passed
  incl. tests/test_office_session_arcs.py                                13 passed
  (mutation check: with _fold_office_session disabled, 5 of them fail)
ORION_ADMISSION_TEST_DSN=<throwaway> pytest services/orion-durable-runs/tests -k temporal_self   74 passed
  incl. test_office_session_labels_the_office_company_arc_... (real Postgres)
git diff --check; check_personal_terms; check_definition_drift --gate; check_metric_lineage --gate;
check_env_key_single_source; check_migration_required_by --base origin/main; check_env_template_parity;
check_inner_state_registry                                             all PASS
```

## Evals run

```text
services/orion-sql-writer/evals/run_office_session_replay.py (read-only, live DB, 11-02..11-06): 4 sessions,
  all "Juniper (probably)", 0 unknown; the 11-05 visit shown above.
orion/temporal_self/evals/run_arc_precision_eval.py   passed: true
orion/temporal_self/evals/run_play_cue_eval.py        PASS
```

## Docker/build/smoke checks

```text
No Docker build or deploy (instructed: do not deploy). Runtime behavior verified by the read-only live replay
above and by real-Postgres tests. The live loop, the live table and a live ask are UNVERIFIED until deploy.
```

## Review findings fixed

Review ran in a subagent on the branch diff. Every material finding is fixed:

- Finding (must): a person explained by a name or by a stranger record could also read "Juniper (probably)",
  because every office record carries the prior's guess (e.g. one head read "Sibling and Juniper (probably)").
  - Fix: prior-guessed records no longer count as people up front. The prior fills one head only when nobody
    else explains it; otherwise those records attach to the name already there, or are listed as unassigned.
  - Evidence: `test_a_named_person_is_not_also_probably_juniper`, the split-stranger case in
    `test_evidence_outside_the_household_is_a_stranger_and_not_also_juniper`.
- Finding: a visit whose closing presence row never came stayed open forever.
  - Fix: no presence row for 1500 s (the company arc's own stale rule) ends it at its last row, `end_reason='stale'`.
  - Evidence: `test_a_visit_whose_producer_died_closes_stale_at_its_last_row`.
- Finding: one frame with two person boxes (a doubled box, a mirror) would open a card about Juniper's own crop.
  - Fix: a people count counts only when at least 2 windows held it (`sustained_peak`). Live 3 days: every cam0 window read 1 person.
  - Evidence: `test_one_crowded_frame_is_not_a_second_person`; the PG ask test now needs two crowded windows.
- Finding: a face-confirmed name was downgraded to "(probably)" when the visit had an unknown person.
  - Fix: the confirmed name is always kept, capitalised, plus the unknown person ("Juniper and someone I don't know in the office").
  - Evidence: `test_a_confirmed_identity_keeps_its_name_and_gains_the_activities`.
- Finding: a keepalive with no open visit opened a session with a wrong start.
  - Fix: skipped, like the company arc. Evidence: `test_a_stretch_opened_before_the_rows_given_is_not_a_visit`.
- Finding: an arc that only touched the visit could take its label.
  - Fix: zero overlap no longer matches. Evidence: `test_an_arc_that_only_touches_the_visit_is_not_labelled`.
- Finding: an album person outside the household prior read as "someone I don't know".
  - Fix: an unsure score for any album name reads "(maybe)". Evidence: same stranger test.
- Finding: local `.env` files not synced. Not fixable here (this agent may not touch the primary checkout):
  listed under Env/config for the orchestrator.

## Restart required

```bash
# after merge, from the clean primary main checkout (post-pull rebuild applies the migration first):
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: medium. Concern: a lone stranger reads "Juniper (probably)". Mitigation: that is the prior's known
  limit (step A); the label always says "(probably)"; a second person in view or a record whose own scores point
  outside the household reads "someone I don't know" and opens the card.
- Severity: low. Concern: a visit that becomes final after local midnight (ended after ~23:49) labels no arc,
  because yesterday's arcs are already closed out. Mitigation: documented; the session row itself is still stored.
- Severity: low. Concern: for an unknown found by the people count, the card shows the record most likely to be
  the stranger (seen while two people were in view, least like Juniper), which can still be Juniper's record.
  Mitigation: her answer names that record either way; the card says who else was there.
- Severity: low. Concern: labels are the captioner's words (Qwen2-VL-2B), cut to two comma clauses; they can be
  wrong. Mitigation: the timeline keeps the full caption with its frame time and event id.

## PR link

PLACEHOLDER_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
