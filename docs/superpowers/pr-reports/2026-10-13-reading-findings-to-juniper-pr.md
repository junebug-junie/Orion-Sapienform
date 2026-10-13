## Summary

- Orion's Day letter gets a **What I read** section: for every read of the day, what the piece says, what Orion took from it, and what the read changed (a link made, a belief moved), or one line saying nothing changed.
- When a read changes Orion, Orion now **writes Juniper a message** in their own words, through the same outreach path curiosity findings use. One message per read; several changes are bundled.
- New typed seam `ReadingFindingV1` (`kind`: `link_made` | `belief_moved`), built only from stored records. The resonance design (PR #2642) adds a third kind as one enum value plus a producer.
- Stage 2 now writes `article_summary` (3-5 sentences about the source itself, kept apart from Orion's reaction) so Juniper can learn from the reads too. Shown in the letter and delivered under each outreach message as recorded.
- No daily cap or cooldown for this kind (Juniper 2026-10-13). Quiet hours, turn-in-flight, recent-chat and arousal `strained` still apply; a blocked message waits, it is not dropped.

## Outcome moved

Before: a read could make a link or feed a belief change and Juniper heard about it only by reading the full "Readings" dump of the letter (Orion's summary, nothing about what changed). After: the letter says per read what changed, and Orion reaches out when something did.

Live 7-day replay (read-only, production code, real Postgres + Orion's graph): 19 verified reads, **1 message** would have gone out (the 2026-10-11 NASA flyover read, its accepted link to the F-5 jets claim). 0 `belief_moved`: 39 prior moves in that week, 26 with text the moving run wrote itself, **none cites any read**. The belief trigger is wired and tested but has no live behavior to fire on yet (see Risks). 0 reads have `article_summary` yet (field is new).

## Current architecture

- Letter: `orion/orion_day/gather.py` reads verified reads (`reading_items_between`) into `ReadingItemV1` (title, url, learned); the email renders them in full under "Readings". Nothing about links or beliefs.
- Links: Stage 2 (`services/orion-hub/scripts/world_pulse_read_stage2.py`) journals claims; accepted ones get a decision + materialization in `substrate_graph_journal`, receipts in `stage2_result_json.relationship_claims`.
- Beliefs: readings never write priors (#2199). Curiosity runs do; `curiosity_run_outcomes.per_prior` records before/after; `Prior.why` / `formed_from` hold the run's own words.
- Outreach: `EndogenousOutreach.offer_message` (curiosity Door-A) with `skip_schedule_gates` (skips quiet hours too).

## Architecture touched

- Contract: new `orion/schemas/reading_finding.py`; additive fields on `ReadingItemV1` (`request_id`, `article_summary`, `findings`) and `WorldPulseReadStage2ResultV1.article_summary`. No bus channel, no migration.
- Producer: `orion/world_pulse_read/findings.py` (SQL + one parameterised `GRAPH.RO_QUERY`).
- Consumers: letter gather/budget/email; `services/orion-hub/scripts/reading_finding_outreach.py`.
- Outreach seam: `outreach_block_reason(skip_rate_gates=)`, `offer_message(skip_rate_gates=, provenance=, record_appendix=)`, `EndogenousOutreach.compose`, `refresh_human_gates`.

## Files changed

- `orion/schemas/reading_finding.py`: `ReadingFindingV1`, `ReadingFindingBundleV1` (with `article_summary`).
- `orion/world_pulse_read/findings.py`: link/belief finders, SQL, prior lookup, `first_sentences`.
- `orion/world_pulse_read/finding_prompt.py`: composition prompt (evidence only, PASS allowed) and the recorded appendix.
- `orion/world_pulse_read/introspect.py`: `article_summary` in item extra (same verified-read gate as `learned`).
- `orion/schemas/world_pulse_read.py`, `services/orion-hub/scripts/world_pulse_read_stage2.py`: `article_summary` field + prompt.
- `orion/schemas/orion_day.py`, `orion/orion_day/{gather,budget,brief}.py`: findings attached to each reading; LLM view line; named gap `sources.reading_beliefs`.
- `services/orion-hub/scripts/orion_day_email.py`, `templates/orion_day_letter.html.j2`: "What I read" section (HTML + text).
- `services/orion-hub/scripts/orion_day_letter.py`, `scripts/main.py`: graph reader wiring; new loop start/stop.
- `services/orion-hub/scripts/reading_finding_outreach.py`: the loop.
- `services/orion-hub/scripts/endogenous_outreach.py`, `endogenous_outreach_decisions.py`: rate-gate skip, provenance/appendix, cap count excludes the new source.
- `services/orion-hub/app/settings.py`, `.env_example`, `README.md`: four `HUB_READING_FINDING_OUTREACH_*` keys, docs.
- Tests: `orion/world_pulse_read/tests/test_findings.py`, `orion/orion_day/tests/test_reading_findings_letter.py` (+ fixtures), `services/orion-hub/tests/test_reading_finding_outreach.py`, `services/orion-hub/tests/test_orion_day_what_i_read.py`.
- Evals: `services/orion-hub/evals/run_reading_finding_replay.py` (live, read-only), `services/orion-hub/evals/test_reading_finding_eval.py` (offline).
- CI: reading, day-letter and durable-runs workflows run the new tests/eval and trigger on the new files.

## Schema / bus / API changes

- Added: `ReadingFindingV1`, `ReadingFindingBundleV1` (not bus payloads; not registered: producer and consumers are all in Hub). `ReadingItemV1.request_id/article_summary/findings`, `WorldPulseReadStage2ResultV1.article_summary` (all default-empty).
- Removed / renamed: none.
- Behavior changed: `count_sent_on` also excludes `source='reading_finding_outreach'`.
- Compatibility: `OrionDayMaterialV1` is `extra="forbid"` and read by orion-durable-runs: **rebuild orion-durable-runs before or with orion-hub** (same rule as `corrections`). Old letter rows validate (test).

## Env/config changes

- Added keys (orion-hub): `HUB_READING_FINDING_OUTREACH_ENABLED=true`, `HUB_READING_FINDING_OUTREACH_TICK_SEC=300`, `HUB_READING_FINDING_OUTREACH_LOOKBACK_HOURS=48`, `HUB_READING_FINDING_OUTREACH_MAX_ATTEMPTS=3`.
- `.env_example` updated: yes. Compose: keys arrive via `env_file` (no compose change needed).
- local `.env` synced with `python scripts/sync_local_env_from_example.py orion-hub --all-keys`: yes, 4 keys added to the primary checkout's `services/orion-hub/.env`.
- skipped keys requiring operator action: none.

## Tests run

```text
python -m pytest -q orion/orion_day/tests orion/world_pulse_read/tests        -> 238+ passed (incl. 21 new)
python -m pytest -q services/orion-hub/tests/test_reading_finding_outreach.py \
  services/orion-hub/tests/test_orion_day_what_i_read.py services/orion-hub/tests/test_endogenous_outreach.py \
  services/orion-hub/tests/test_orion_day_letter.py services/orion-hub/tests/test_world_pulse_read_stage2.py \
  services/orion-hub/tests/test_curiosity_investigation.py                    -> all passed
python -m pytest -q services/orion-hub/tests (full)                          -> 33 failures, identical set on origin/main (pre-existing, none new)
python -m pytest -q tests/scripts/test_schema_skew_discovery.py              -> 18 passed
scripts/check_env_template_parity.py PASS; check_settings_defaults.py orion-hub --example-drift OK;
check_metric_lineage.py --gate PASS; check_definition_drift.py --gate PASS; check_env_key_single_source OK
```

## Evals run

```text
python -m pytest -q services/orion-hub/evals/test_reading_finding_eval.py   -> passed (planted week: 3 messages, bundling, held unprojected link, honest nothing-line)
PYTHONPATH=. .venv/bin/python services/orion-hub/evals/run_reading_finding_replay.py --days 7   (LIVE, read-only)
  reads=19 (with article summary: 0) findings={'link_made': 1, 'belief_moved': 0} held_unprojected=0 beliefs_unchecked=None
  messages that would have gone out: 1
  - 2026-10-11T14:13Z  NASA to Conduct Flyover, Engage NFL Fans at MetLife Stadium
      Link made: Per advisory M26-080, NASA Administrator Jared Isaacman pilots one of the three F-5 fighter jets ...
  other 18 reads: "no link accepted; no curiosity run cited this read"
  belief check: 39 prior moves in the window, 26 with run-written text, 0 cite any read
services/orion-hub/evals/run_orion_day_email_eval.py PASS; orion/orion_day/evals/run_orion_day_eval.py PASS;
run_orion_day_reread_eval.py PASS; run_orion_day_correction_eval.py PASS
```

## Docker/build/smoke checks

```text
Not run (no deploy, per task). UNVERIFIED live: the loop and letter section have not run in a container.
```

## Review findings fixed

Code review ran in a subagent against `git diff origin/main...HEAD`.

- Finding: one fixed correlation id per bundle was reused on every compose retry; an abandoned agent-lane attempt is cancelled by correlation id and the fcc motor would kill the retry on spawn.
  - Fix: fresh `uuid4` per attempt; the stable id is `meta.bundle_key`.
  - Evidence: `test_each_attempt_gets_its_own_correlation_id`.
- Finding: a belief moved more than 48h after its read was never told (reads were only watched for 48h).
  - Fix: reads up to 7 days old are watched; the 48h lookback now applies to when the change was recorded.
  - Evidence: `test_a_belief_moved_days_after_an_old_read_is_still_told`.
- Finding: the letter said "nothing came of this read" / "no belief moved" when the check failed or could not see; a link plus an unreadable graph dropped the gap.
  - Fix: "nothing came of it" only when everything was checked; gaps render as "Not checked: ..."; the nothing-line now says what was checked ("no curiosity run cited this read").
  - Evidence: `test_an_unchecked_read_never_says_nothing_came_of_it`, `test_a_link_plus_an_unreadable_graph_keeps_the_gap`.
- Finding: belief_moved is effectively dead on live data (0 of 180 priors cite a read).
  - Fix: recorded as a live-data finding here and in the README; wording above. Making attribution structural (curiosity citing `reading:<seed_id>`) is a follow-up, not in this patch.
  - Evidence: replay output above.
- Finding: OrionDayMaterialV1 forbid rollout.
  - Fix: restart order below (durable-runs first); brief round-trip test.
  - Evidence: `test_findings_survive_the_durable_brief_round_trip`.
- Finding (nits): URL prefix matching credited longer paths; same assertion from two reads showed on one; link projected after midnight fell out of its day; DECISIONS_SQL scanned all rows; unbounded log-guard set; silent prior-id truncation; missing gate unit test and restart test.
  - Fix: whole-token URL match; key `(seed_id, assertion_id)`; `observed_at` = accept time; `reason = ANY(...)` filter (uses the reason index); guard set cleared past 1000; truncation logged; tests added.
  - Evidence: `test_a_url_names_only_its_own_page_not_a_longer_one`, `test_two_reads_making_the_same_link_each_show_it`, `test_a_link_belongs_to_the_moment_it_was_accepted`, `test_rate_gate_skip_keeps_every_gate_that_protects_juniper`, `test_a_restart_does_not_resend_a_told_read`.
- Not changed: two "seconds since Juniper spoke" reads per send (pre-check before compose, re-check after); kept deliberately, the compose turn can take minutes.

## Restart required

orion-durable-runs first (it validates the letter material with `extra="forbid"`), then orion-hub. No migration.

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-durable-runs up -d --build
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium. Concern: `belief_moved` has never matched live (0 of 26 run-written prior texts in 7 days cite a read). Readings do not reach curiosity runs as citable material, so this kind will stay silent until they do. Mitigation: wired, tested, and the letter says "no belief moved" only when the graph was actually read.
- Severity: low. Concern: `article_summary` is model-written (Stage 2) and not checked against the fetched text. Mitigation: labelled as the read's summary of the source, kept apart from Orion's reaction; the prompt forbids anything not in fetched text and asks for empty on 404/paywall.
- Severity: low. Concern: a change recorded more than 48h ago, or on a read older than 7 days, is never messaged. Mitigation: env knob for the 48h; the letter still shows it for the read's own day.
- Severity: low. Concern: after quiet hours (or right after deploy) every pending bundle is sent back to back; nothing paces them. Mitigation: consistent with "as much as they want"; today's rate is about one message a week.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
