# Office visits: Orion pays attention and asks Juniper what she is up to

Step C of `docs/superpowers/specs/2026-11-05-office-camera-attention-and-reach-out-design.md` (PR #3127).
Step A (PR #3129) made Orion say "probably Juniper" in the office. Step B (PR #3132) made each visit one
stored record with a timeline. This step makes Orion notice the visit and ask her about it in Hub chat.

## Summary

- When Juniper (probably) has been in the office for a few minutes, Orion sends her ONE Hub chat message
  asking what she is up to. Orion writes it through the real unified turn. Under it, the camera's notes are
  delivered as recorded: times, what it saw, and who it probably was.
- Every visit may get a message (Juniper, 2026-11-05). No daily cap, no cooldown. An active chat turn,
  recent chat, quiet hours and arousal "strained" hold the message until they clear.
- Only visits the camera placed Juniper in are messaged. A visit with only an unknown person already gets
  step B's "Who is this?" card; the message never asks who someone is, and never says Orion recognized her
  when the camera only guessed (a code check refuses that wording, not just the prompt).
- A visit is told once. Hub marks the visit row (`reached_out_at`) just before delivery, and clears the
  mark if delivery is refused. A Hub restart finds the mark and sends nothing more.
- The same visit also posts an attention want (family `perception.office`) so Prime can see "Juniper
  (probably) in the office: ..." as a concern when the market funds it. The message does not wait for that.
- Today the want posts nothing: its salience needs 20 earlier visits to compare against, and only one
  office visit has captions so far (see the metric gate).

## Outcome moved

- Before: Juniper's 11-05 visit was stored (step B) and nothing read it as a story. Orion never said a word.
- After (dry run on that visit, below): at 18:27 local, three minutes in, Orion would have composed one
  message from 7 captioned scenes, with the notes "18:26 sitting at a desk, unpacking a plastic bag and
  handling a piece of equipment ... Who: probably you. That is a guess".

## Dry run of the 11-05 visit (read-only, live database, nothing sent)

`cd services/orion-hub && PYTHONPATH=.:<repo> python evals/run_office_reach_out_dry_run.py --dsn <live> --since 2026-11-01T00:00:00Z`

```text
3 office visits since 2026-11-01T00:00:00+00:00; with a story: 1

cam0-f8e113597f6a764529a4ef74f5697d32  2026-11-04 23:09:12Z -> 23:19:22Z
  who: Juniper (probably)  entries=0  unknown=0  first story at: never

cam0-f9991fd1bea33f74aa55e83da50f7aed  2026-11-04 23:21:24Z -> 23:21:46Z
  who: Juniper (probably)  entries=0  unknown=0  first story at: never

cam0-1f48ba3068b16a16cf7ba9b7e68ecabd  2026-11-05 01:24:07Z -> 01:31:28Z
  who: Juniper (probably)  entries=17  unknown=0  first story at: 2026-11-05T01:27:07.478664+00:00
  novelty=1.0  salience=None  history_n=0 (no want: under 20 past visits)
  rest-point check: second half vs first half novelty=0.875 (whole visit vs itself=0.0)

----- PROMPT (as it would be composed at 2026-11-05T01:27:07.478664+00:00) -----
The office camera stitched together one visit to Juniper's office (since 18:24, America/Denver time; the visit is still going on).

What it saw, in order (the camera's own captions, from a small vision model; they can be wrong):

- 18:23 standing in a room, looking at a box on the floor
- 18:24 organizing items in a storage bin
- 18:24 sitting at a desk, working on a computer with a scenic wallpaper
- 18:24 sitting at a desk, opening a box
- 18:25 sitting at a desk, working on a computer, and appears to be organizing or unpacking items from a box
- 18:26 sitting at a desk, unpacking a plastic bag and handling a piece of equipment
- 18:26 reaching into a cabinet

Who: probably you. That is a guess: the office is almost always you, and the camera did not match your face or body.

Juniper asked you to give the office your attention, get curious, and reach out each time she is in there to ask what she is up to, based on the scenes the camera stitched together. Write her one short message (two or three sentences). Say what it looked like was going on, naming at least one thing the camera saw with its time as listed above, and then ask her ONE question about what she is doing or making. The camera's notes are delivered to her under your message as recorded, so do not list them all.
The camera did NOT recognize her: it is only guessing it was her (the office is almost always her, or the match was unsure). Do not say you recognized or saw that it was her; say it looked like her, or that you are guessing.

Read the captions as your reading of a scene, not as facts. Do not mention scores, systems or internal signals. Do not reply PASS: Juniper asked for this message.

----- CAMERA NOTES delivered under Orion's message (record_appendix) -----
What the office camera saw (its own captions; they can be wrong):
18:23 standing in a room, looking at a box on the floor
18:24 organizing items in a storage bin
18:24 sitting at a desk, working on a computer with a scenic wallpaper
18:25 sitting at a desk, working on a computer
18:26 sitting at a desk, unpacking a plastic bag and handling a piece of equipment
18:26 reaching into a cabinet
(+1 more)
Who: probably you. That is a guess: the office is almost always you, and the camera did not match your face or body.
```

Orion's own sentence (the part above the notes) is written by the unified turn at send time. The dry run does
not call it, so its exact words are UNVERIFIED. The prompt above is exactly what the turn receives, and the
notes are exactly what Juniper receives under it.

## Metric gate: the want's salience

1. **Provenance.** The magnitude is computed in `scripts/office_reach_out.py` `plan_wants` from
   `vision_office_session.timeline` (sql-writer `build_timeline`), which is the office camera's
   `vision_events` `camera_caption` rows (Qwen2-VL captions of the person crop), repeats folded.
2. **Independence.** The magnitude is a share (new distinct activities / distinct activities), not a count,
   and every visit is cut to the same story slice, so it does not grow with visit length by construction. It shares its source with the timeline length,
   which also sets the "has a story" threshold. With only one captioned visit live, the correlation with
   duration cannot be measured yet: UNVERIFIED. The other office signal in the market, the "Who is this?"
   card's `vision.ask` want, ranks sightings of an individual, a different quantity.
3. **Theory anchor.** Novelty as distance from a personal activity history (the spec's anchor), measured
   lexically: an activity is "seen before" when the content words of its "doing what" clause overlap an
   earlier visit's by Jaccard >= 0.5 (the "where" clause is left out: the office is always a desk or a
   closet). Every visit is measured on its timeline up to the moment it first had a story, so a long visit
   is not compared on more scenes than a short one. Salience is the market's own rule: strict-below percentile against the source's last 30
   days (`source_salience`). Weaker than the spec's caption-embedding centroid; chosen because it needs no
   embedding call per tick and is exact to test. Recorded as a concern.
4. **Live data.** 3 office visits exist (11-04 23:09, 11-04 23:21, 11-05 01:24). Only the 11-05 visit has
   captions (the office crop pipeline, PR #3125, started that day). Its novelty reads 1.0 (nothing to
   compare against) and its salience is unknown (0 of 20 needed past visits), so **no want is posted**. Rest
   point: a visit compared to itself reads 0.0 (exact). Its second half against its own first half reads
   0.875: the captioner re-words similar scenes ("organizing items in a storage bin" vs "reaching into a
   storage bin"), so a routine visit may NOT read near 0. This is the gate's live finding, and it is the
   reason the want stays harmless today: it posts nothing until there is real history to check. Whether routine visits can actually return to 0 on live data is UNVERIFIED until about
   20 captioned visits exist (at today's rate, a few weeks). No constant salience was invented: the unknown
   case posts nothing, which is the market's rule. The unknown-visitor case needs no constant 1.0 either:
   step B already opens a "Who is this?" card for it, and that card posts its own `vision.ask` want.
5. **Existing mechanism.** Reused `event_want`, `source_salience`, `orion.physiology.client.abid`, the
   reading producer's posted-once pattern. No new market mechanism.
6. **Reversibility.** Cheap. `HUB_OFFICE_EVENT_WANTS_ENABLED=false` stops it. Nothing is stored except
   want-ledger rows; no schema field holds the magnitude.

## Current architecture

- `vision_office_session` (step B) held one row per office visit: who, timeline, label, ask_id. Its only
  readers were the Temporal Self (day story label) and the "Who is this?" card opener.
- The want market had event wants from reading, camera asks and the cooling act. Nothing for the office.
- Hub had two source-triggered reach-outs through the outreach door: reading findings and self findings.

## Architecture touched

- orion-hub: new loop `scripts/office_reach_out.py` (want producer + reach-out), wired in `scripts/main.py`.
- orion-durable-runs: `app/event_wants_menu.py` labels `office_session:` event wants from the visit row.
- `config/physiology/families.yaml`: new family `perception.office` (tier 2, a proposal for Juniper's sign-off).
- Database: two additive columns on `vision_office_session`.

## Files changed

- `services/orion-hub/scripts/office_reach_out.py`: the loop (pure planning, prompt, notes, claim/release).
- `services/orion-hub/scripts/main.py`: start/stop the loop.
- `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example`: 7 new keys.
- `services/orion-hub/scripts/endogenous_outreach_decisions.py`: office sends do not count toward the
  background tick's daily cap (same as reading/self findings).
- `services/orion-hub/evals/run_office_reach_out_dry_run.py`: read-only live dry run.
- `services/orion-hub/tests/test_office_reach_out.py` (pure + loop on a real `EndogenousOutreach`),
  `tests/test_office_reach_out_pg.py` (real Postgres).
- `services/orion-hub/README.md`: section for the loop.
- `services/orion-durable-runs/app/event_wants_menu.py`, `tests/test_event_wants_menu.py`: the label.
- `services/orion-sql-writer/tests/test_vision_office_sessions.py`: sql-writer's rebuild never touches the mark.
- `config/physiology/families.yaml`: `perception.office`.
- `services/orion-sql-db/manual_migration_vision_office_session_reach_out_v1.sql` (+ `_rollback.sql`).
- `.github/workflows/orion-hub-office-reach-out-pg.yml`: runs the Postgres tests, must not skip.

## Schema / bus / API changes

- Added: columns `vision_office_session.reached_out_at`, `reach_out_decision_id`; want family
  `perception.office`; event ref prefix `office_session:` in the funded event-wants menu.
- Removed / renamed: none.
- Behavior changed: `count_sent_on` skips `source='office_reach_out'`.
- Compatibility: no bus channel, no schema model change. Without the migration the loop sends nothing
  and still posts wants.

## Env/config changes

- Added keys (orion-hub): `HUB_OFFICE_REACH_OUT_ENABLED=true`, `HUB_OFFICE_EVENT_WANTS_ENABLED=true`,
  `HUB_OFFICE_REACH_OUT_TICK_SEC=60`, `HUB_OFFICE_REACH_OUT_LOOKBACK_HOURS=12`,
  `HUB_OFFICE_REACH_OUT_MIN_ENTRIES=4`, `HUB_OFFICE_REACH_OUT_MIN_MINUTES=3`, `HUB_OFFICE_REACH_OUT_MAX_ATTEMPTS=3`.
- Removed / renamed: none.
- `.env_example` updated: yes (orion-hub). Compose passes no env explicitly: no compose change.
- local `.env` synced: NO, not by this agent (may not touch the primary checkout). Run
  `python scripts/sync_local_env_from_example.py orion-hub` from the primary checkout.
- Skipped keys requiring operator action: none.

## Migration

`services/orion-sql-db/manual_migration_vision_office_session_reach_out_v1.sql`, header
`-- ORION-MIGRATION-REQUIRED-BY: orion-hub`. Two `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`:
`auto_apply_refusals` returns none, so the post-pull applier applies it before the Hub rebuild.
Rollback: `manual_migration_vision_office_session_reach_out_v1_rollback.sql` (`-- DESTRUCTIVE`).

## Thresholds (from live data)

A visit has a story at 4 timeline entries over 3 minutes. Live 11-05: the visit reached 4 entries 23 s in
(too early: the first scenes were "standing in a room, looking at a box"), and 7 entries at 3 minutes,
which already included "unpacking a plastic bag and handling a piece of equipment". The two 11-04 visits had
no captions and never qualify, so a visit with no scenes is never messaged.

## Tests run

```text
services/orion-hub: pytest tests/test_office_reach_out.py                         32 passed
ORION_TEST_PG_URI=<throwaway postgres:16-alpine> pytest tests/test_office_reach_out_pg.py   4 passed (not skipped)
  (mutation check: with the "already told" filter removed from _pending, 2 tests fail)
services/orion-hub: pytest tests (whole folder)    4586 passed, 3 failed -- the same 3 fail on main (8 on main), 0 new
services/orion-durable-runs: pytest tests/test_event_wants_menu.py                6 passed
services/orion-durable-runs: pytest tests -k "event_want or physiology or families"   76 passed, 6 skipped
services/orion-sql-writer: pytest tests/test_vision_office_sessions.py           26 passed
git diff --check; check_personal_terms; check_definition_drift --gate; check_metric_lineage --gate;
check_env_key_single_source; check_migration_required_by --base origin/main;
check_settings_defaults orion-hub --example-drift; check_env_template_parity      all PASS
```

## Evals run

```text
services/orion-hub/evals/run_office_reach_out_dry_run.py (read-only, live DB, 11-01..now): 3 visits, 1 with a
  story (11-05), first story at 01:27:07 UTC; novelty 1.0, salience None (history 0 of 20): no want.
```

## Docker/build/smoke checks

```text
No Docker build or deploy (instructed: do not deploy). The live loop, a live want row and a live sent message
are UNVERIFIED until deploy. Behaviour verified by the read-only dry run and the real-Postgres tests.
```

## Review findings fixed

Review ran in a subagent on the branch diff. It found no must-fix issue. Every should-fix is fixed:

- Finding: a visit with only a stranger (or only another household member) still messaged Juniper, next to
  step B's "Who is this?" card.
  - Fix: only visits the camera placed Juniper in are messaged.
  - Evidence: `test_a_visit_without_juniper_is_not_messaged_the_card_covers_a_stranger`.
- Finding: "do not ask who they are" depended on the card already being open (`ask_id`).
  - Fix: said whenever an unknown person was there.
  - Evidence: `test_an_unknown_visitor_is_never_asked_about_even_before_its_card_opens`.
- Finding: "never claim recognition" rested on the prompt alone, and only for the prior.
  - Fix: the instruction covers unsure matches too, and a deterministic check refuses "I recognized you" /
    "I could see it was you" on a guess (`claimed_recognition`, retried like a PASS).
  - Evidence: `test_a_message_claiming_recognition_on_a_guess_is_refused`, `test_an_unsure_match_also_may_not_claim_recognition`.
- Finding: a message held overnight by quiet hours had no date.
  - Fix: "yesterday" / the weekday in the prompt and the notes, and "the visit is over, it ended N ago".
  - Evidence: `test_a_visit_held_overnight_says_which_day_it_was`.
- Finding: the novelty compared the "where" clause ("at a desk, eating" = "at a desk, typing"), and an open
  visit was measured on fewer scenes than its full-length history.
  - Fix: only the "doing what" clause; every visit cut to its first-story slice.
  - Evidence: `test_novelty_compares_what_was_done_not_where`, `test_a_long_visit_is_measured_on_the_same_slice_as_a_short_one`.
- Finding: a failing want producer could stop the message; bids ran while holding a DB connection; the
  planning ran on the event loop and read 60 days even with wants off.
  - Fix: want errors are logged and skipped; bids run after the connection is released; `plan_wants` runs
    in a thread; 60 days are read only when wants are on.
  - Evidence: `test_a_broken_want_producer_never_holds_the_message`.
- Finding: the restart-safe attempt cap (`FAILURES_SQL`) was untested.
  - Fix: a unit test and a real-Postgres test with decision rows; workflow paths include `endogenous_outreach_decisions.py`.
  - Evidence: `test_attempts_logged_before_a_restart_still_count`, `test_failed_attempts_in_the_decision_log_cap_a_restarted_hub`.
- Nits fixed: one question means 1-2 "?" (three refused); `COLUMN_SQL` filters `current_schema()`; the
  quiet-hours test uses a 2-hour window; `count_sent_on` exclusion pinned by a test; an error inside
  delivery keeps the "told" mark (errs toward silence; only a shutdown before delivery releases it).
- Finding: local `.env` not synced. Not fixable here (this agent may not touch the primary checkout):
  listed under Env/config for the orchestrator.

## Restart required

```bash
# after merge, from the clean primary main checkout (the post-pull rebuild applies the migration first):
python scripts/sync_local_env_from_example.py orion-hub
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: medium. Concern: the novelty magnitude is lexical and the captioner re-words similar scenes, so
  routine visits may not read near 0. Mitigation: no want is posted until 20 captioned visits exist; re-run
  the dry run then and check the rest point before relying on it. Off switch: `HUB_OFFICE_EVENT_WANTS_ENABLED`.
- Severity: medium. Concern: a lone stranger in the office reads "probably you" (the step A prior's known
  limit), so Orion could ask Juniper about a visit that was not hers. Mitigation: the notes and the prompt
  say it is a guess; Orion is told not to claim it recognized her.
- Severity: low. Concern: a Hub crash between the claim and the send loses that one message. Mitigation:
  by design it errs toward silence, never a double message.
- Severity: low. Concern: `perception.office` tier 2 is a proposal. Mitigation: Juniper signs off in review.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3135

🤖 Generated with [Claude Code](https://claude.com/claude-code)
