# Orion's Day, Hub half: scheduling, the HTML email, carry-forward into curiosity

Stacked on #2435 (`feat/orion-day-letter-core`). Base of this PR: `feat/orion-day-letter-core`.

## Summary

- Each morning after 08:30 Denver time, Hub asks orion-durable-runs to write yesterday's letter. It does this as a queued job that has to wait its turn for a GPU. If the job fails, Hub tries again as attempt n+1, up to 6 times, then posts one in-app notice. Hub keeps no memory of its own for this: it reads the attempt number back from durable-runs, and it knows the day is done because `emailed_at` is set on the letter's database row.
- Once the letter is written, Hub emails it once. The email re-renders the stored row, so no model call is repeated. The row is marked sent only when orion-notify answers `email_status == "sent"`. The email is HTML with inline styles only, plus a full plain-text copy. Nothing is shortened. Up to 6 reverie images are embedded in the message itself.
- Orion's note and the "Carrying forward into curiosity" text are two separate, visually distinct sections. Only the carry-forward text ever reaches curiosity.
- Curiosity's regular investigate line claims the freshest carry-forward once and shows it under its own heading in the kickoff prompt. If the run is cancelled, or a durable run ends before its turn starts, the claim is released.
- Changes to where things live: the scheduler loop is `scripts/orion_day_letter.py`, started from `scripts/main.py`. The renderer is `scripts/orion_day_email.py` plus the template `templates/orion_day_letter.html.j2`. The carry-forward claim/release logic is `orion/orion_day/carry_forward.py`. There are 14 new env keys, 2 new dependencies (`markdown-it-py`, `Pillow`), tests, an eval, and a CI workflow.

## Outcome moved

Before this PR, #2435 could write a letter, but nothing scheduled it, emailed it, or fed it back to Orion. With this PR the loop closes: gather, then an admitted run, then the stored row, then one email, then one curiosity offer. Every step can be seen in Postgres (`orion_day_letter.emailed_at`, `carry_forward_offered_run_id`) or in the durable registry (`orion-day-<date>-<n>`).

## Current architecture

- Hub already runs the curiosity and reading loops as in-process tasks. Reading submits durable runs over HTTP (`POST /runs`, `HUB_READING_DURABLE_URL`). Curiosity listens on `orion:durable:run:state`.
- The dream-hypothesis offer (`orion/dream/hypotheses.py`) is the template followed here for an offer-once TAKE/RELEASE in SQL.
- #2435 supplies `build_orion_day_brief`, `build_orion_day_request`, `fetch_letter`, the `orion_day_letter` table, and the `orion_day.letter` workflow.

## Architecture touched

- orion-hub: a new loop, the renderer and template, `main.py` wiring, and settings. `curiosity_investigation.py` gains the carry-forward claim and release, plus `run_state_hooks` so the letter reuses the existing durable-state listener instead of opening a second subscription.
- Shared: `orion/orion_day/carry_forward.py` (new), and a `carry_forward=` section in `orion/curiosity/kickoff_prompt.py`.
- No changes to schemas, bus channels, or the registry.

## Files changed

- `services/orion-hub/scripts/orion_day_letter.py`: the scheduler. The active-letter window, attempt number derived from `GET /runs`, submit, send, the email lock, and the notice.
- `services/orion-hub/scripts/orion_day_email.py`: markdown to email-safe HTML, image selection and JPEG transcoding, the plain-text part, and the `NotificationRequest`.
- `services/orion-hub/templates/orion_day_letter.html.j2`: the email layout.
- `orion/orion_day/carry_forward.py`: the TAKE/RELEASE SQL and the prompt section.
- `orion/curiosity/kickoff_prompt.py`: the `carry_forward=` parameter and its own section.
- `services/orion-hub/scripts/curiosity_investigation.py`: the claim (regular line only), the releases, and `run_state_hooks`.
- `services/orion-hub/scripts/main.py`: builds and starts the loop after the memory pool exists, and stops it on shutdown.
- `services/orion-hub/app/settings.py`, `.env_example`, `README.md`: 14 keys and documentation.
- `services/orion-hub/requirements.txt`: `markdown-it-py`, `Pillow`.
- `services/orion-hub/tests/test_orion_day_letter.py`: 50 tests.
- `services/orion-hub/evals/run_orion_day_email_eval.py`, `evals/test_orion_day_email_eval.py`: the full-text eval.
- `.github/workflows/orion-day-letter-hub-tests.yml`: CI.
- `orion/schema_skew_discovery.py`: declares orion-durable-runs as the writer of `OrionDayLetterV1` (the schema-skew gate).

## Schema / bus / API changes

- Added: none. Hub reads and writes the `orion_day_letter` columns that #2435 created for exactly this purpose: `emailed_at`, `email_notification_id`, `carry_forward_offered_at`, `carry_forward_offered_run_id`.
- Behaviour changed: `build_kickoff_prompt(carry_forward=None)` is a new optional parameter, so existing callers are unaffected.
- Compatibility: the notify event kinds are `orion_day.letter` (email) and `orion_day.letter.exhausted` (in-app).

## Env/config changes

- Added keys (orion-hub): `HUB_ORION_DAY_ENABLED=true`, `HUB_ORION_DAY_EMAIL_ENABLED=true`, `HUB_ORION_DAY_HOUR_LOCAL=8`, `HUB_ORION_DAY_MINUTE_LOCAL=30`, `HUB_ORION_DAY_TICK_SEC=300`, `HUB_ORION_DAY_DURABLE_URL=http://127.0.0.1:8124`, `HUB_ORION_DAY_MAX_ATTEMPTS=6`, `HUB_ORION_DAY_TIMEOUT_SEC=1800`, `HUB_ORION_DAY_CARRY_FORWARD_TTL_HOURS=36`, `HUB_ORION_DAY_EMAIL_RETRY_SEC=1800`, `HUB_ORION_DAY_NOTIFY_TIMEOUT_SEC=60`, `HUB_ORION_DAY_MAX_IMAGES=6`, `HUB_ORION_DAY_IMAGE_MAX_BYTES=450000`, `HUB_CURIOSITY_CARRY_FORWARD_ENABLED=true`.
- `.env_example` updated: yes. No compose `environment:` entries were added, because hub's `env_file: .env` already carries every key (`check_service_env_compose_parity.py orion-hub`: N/A, all 452 keys reach the container).
- Local `.env` synced with `python3 scripts/sync_local_env_from_example.py --all-keys orion-hub`: yes. All 14 keys are in `/mnt/scripts/Orion-Sapienform/services/orion-hub/.env`.
- Skipped keys requiring operator action: none. One pre-existing, unrelated divergence was reported: `HUB_ROOM_CLAUDE_ENABLED`.
- Carry-forward freshness: #2435's brief defaults to 48 h. Hub sends the approved 36 h on every brief.

## Tests run

```text
pytest services/orion-hub/tests/test_orion_day_letter.py services/orion-hub/evals/test_orion_day_email_eval.py \
  services/orion-hub/tests/test_curiosity_investigation.py services/orion-hub/tests/test_curiosity_dream_hypotheses.py \
  services/orion-hub/tests/test_curiosity_self_inquiry.py                                   243 passed
pytest tests/test_dream_hypotheses.py orion/orion_day/tests tests/test_curiosity_peer_kickoff.py   71 passed
full hub suite (services/orion-hub/tests, final merged head)  3201 passed, 38 failed, 72 skipped
  -> the same 38 node ids run on origin/feat/orion-day-letter-core (no hub diff): 38 failed. Pre-existing, not this PR
     (route/UI tests: memory_consolidation_draft_routes, llm_route_selector, substrate_effect_*, ...).
scripts/check_env_template_parity.py      PASS (94 services)
scripts/check_chat_route_poachers.py      PASS
git diff --check                          clean
```

What the tests cover:
- An emailed row causes nothing to happen (quota met).
- A failed or abandoned attempt is retried as n+1, with the number derived from the durable registry by a fresh loop instance each time, which proves restart safety.
- The attempt cap raises exactly one notice. An operator cancel is not retried.
- The durable deadline equals Hub's hard stop. The letter is abandoned at the next slot.
- An email retry re-renders from the row and never submits a run. The row is stamped only on `sent`. `skipped` waits until the hard stop.
- A timed-out reply or a failed stamp is never resent. The advisory lock blocks a second Hub.
- An empty day is skipped. A missing table reads as unavailable. Refused submits back off and then notify.
- Rendering: the note and carry-forward sections are distinct; all material sections are present at full text; there are no truncation markers and no `<details>`, `<style>`, or `<script>`; cid references match attachments exactly; the most salient images are chosen and transcoded to JPEG at most 1024 px and under the byte cap; a missing file is skipped; image paths cannot leave the storage dir; raw HTML is escaped; `javascript:` URLs are not linked; markdown images are not fetched; no dream hypothesis id appears.
- Carry-forward: claimed once, newest first; released by run; never offered once expired; the take SQL never selects `note_md`; the prompt section is its own and comes after the dream section; the note never appears in the prompt; the kill switch works; a cancelled turn releases; an empty generation keeps the claim; a durable run that ended before `run.started` releases, while a started or unknown run keeps the claim; only `_investigate` claims (AST check); retries resend the frozen prompt (`_prompt_for_attempt`).

## Evals run

```text
python services/orion-hub/evals/run_orion_day_email_eval.py                        PASS (fixture day)
python services/orion-hub/evals/run_orion_day_email_eval.py \
  --material <live 2026-09-29, gathered read-only> --images-dir /mnt/storage-lukewarm/orion/reverie-visual
  letter 2026-09-29: 132 full-text items, 6 images, html ~265 KB, text ~208 KB   PASS
mutation: capping every rendered body at 300 chars -> eval reports 3 html failures (it catches truncation)
```

The eval pulls every full-text item straight from the material model: curiosity write-ups, self-definitions and lived answers, failures, self-sense questions and answers, reading titles, what was learned and why-now, reading journals, dreams, hypothesis claims and whys, image captions, the GitHub and chat digests, and world-news items. It checks that each one appears in full in both the HTML and the plain-text part, ignoring whitespace and markup.

## Docker/build/smoke checks

```text
Not deployed (per instructions). No docker build run: runtime depends on #2435's migration + durable-runs/cortex
images first. The hub container currently has markdown-it-py 4.2.0 (transitive) but NOT Pillow -- the rebuild
installs it from requirements.txt.
Live read-only: gathered 2026-09-29 from Postgres with default_transaction_read_only=on:
  8 curiosity runs, 9 failed, 16 self-sense, 9 readings, 8 reading journals, 6 dream hypotheses,
  958 reverie thoughts / 290 chains, 15 visual reveries (all 15 files present), world digest;
  chat + github compactor EMPTY for that day. orion_day_letter table: not present yet (migration not applied).
Preview (not emailed): rendered from that live material with note/carry-forward text marked PLACEHOLDER,
kept in the session scratchpad (not committed -- it holds Orion's self-sense answers and journals).
Headless-Chromium screenshot checked: header band, note, green-bordered carry-forward box, curiosity write-ups
with markdown lists/code/bold rendered, images inline.
```

## Review findings fixed

The code-review subagent reviewed `origin/feat/orion-day-letter-core..784c56bdf`. It found no must-fix items. The should-fix items and nits were all addressed in f62bcaf11:

- Finding: the durable deadline (local midnight on L+2) came before Hub's hard stop (08:30 on L+2), so a run abandoned at midnight would be resubmitted with a deadline already in the past and would burn through every attempt.
  - Fix: the request's `deadline_at` is now Hub's hard stop.
  - Evidence: `test_the_durable_deadline_is_the_hub_hard_stop_not_midnight`.
- Finding: an email could go out twice. notify sends SMTP synchronously and does not dedupe, so a timed-out reply after a real send, or a failed stamp after `sent`, would lead to a resend.
  - Fix: the send runs under a per-day Postgres advisory lock, and the row is re-read under that lock. A timed-out reply or a stamp that still fails after 3 tries puts the day in an "outcome unknown" state, which is never resent automatically.
  - Evidence: `test_a_timed_out_reply_is_never_resent_automatically`, `test_a_failed_stamp_after_sent_is_retried_then_never_resent`, `test_another_process_holding_the_email_lock_blocks_the_send`.
- Finding: `skipped` (SMTP not configured, or policy declined) was retried every 30 minutes, each time also posting an in-app copy.
  - Fix: `skipped` now waits until the hard stop.
  - Evidence: `test_skipped_email_waits_until_the_hard_stop`.
- Finding: a submit that was refused every time stayed silent all day.
  - Fix: after 3 refusals, Hub sends the same one-time notice.
  - Evidence: `test_repeated_submit_refusals_raise_one_notice`.
- Finding: carry-forward claimed by a durable curiosity run that never reached its turn was never given back.
  - Fix: on a `failed`, `abandoned` or `cancelled` curiosity terminal, Hub checks `GET /runs/{id}` history for `run.started` and releases the claim only if the run never started. Any doubt keeps the claim.
  - Evidence: `test_an_unstarted_durable_curiosity_run_gives_its_carry_forward_back`, `test_a_started_or_unknown_run_keeps_the_claim`.
- Finding: path traversal was possible through the sha256 fallback filename.
  - Fix: basename plus a check that the resolved path stays inside the storage dir.
  - Evidence: `test_sha_fallback_path_cannot_escape_the_image_dir`.
- Finding: Gmail clipping. Handled as far as code can: a warning is logged over 102 KB and inline styles were trimmed. It is still a decision for Juniper; see Risks.
- Nits fixed:
  - Exceptions from hook-started ticks are now logged.
  - Rendering moved off the event loop.
  - The loop now starts after the memory pool is created and outside the bus block.
  - Non-http URLs are not linked, and markdown remote images are turned off.
  - Naive datetimes are read as UTC.
  - `color-scheme` is set to `light`.
  - CI paths now include `main.py`, `settings.py` and `.env_example`, and the push trigger is path-filtered.

CI fix: `tests/scripts/test_schema_skew_discovery.py` flagged `OrionDayLetterV1` as read by orion-hub with no declared writer. It is now declared in `orion/schema_skew_discovery.py::DECLARED_WRITERS`, with orion-durable-runs as the writer (the persist node inserts the row).

Base update: merged `origin/feat/orion-day-letter-core` at e708151ec (now includes main's #2419/#2429/#2431). The only conflict was hub `.env_example`: two independent appends, both kept. The metric lock is unchanged (`check_definition_drift.py`: no definition changes). Post-merge focused run: 327 passed.

## Restart required

Deploy after all of #2435's order (migration, sql-writer and actions, cortex-orch and cortex-exec, durable-runs). Hub goes last:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Live checks after the Hub deploy:

```bash
docker logs orion-athena-hub 2>&1 | grep -E "orion_day_letter started|orion_day_(submitted|emailed|retry|empty|store_unavailable)"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \
  "SELECT letter_date, run_id, emailed_at, email_notification_id, carry_forward_expires_at, carry_forward_offered_run_id FROM orion_day_letter ORDER BY letter_date DESC LIMIT 3"
curl -s http://127.0.0.1:8124/runs/orion-day-$(date -d yesterday +%F)-1 | jq '{status, orion_day}'
docker logs orion-athena-hub 2>&1 | grep curiosity_carry_forward_offered
```

## Risks / concerns

- Severity: medium. Concern: **Gmail clips heavy days.** The live 2026-09-29 letter is about 265 KB of HTML (about 227 KB of it is content). Gmail hides everything past about 102 KB behind "View entire message". Nothing is lost, but it is one extra click. Mitigation today: a warning is logged. The alternative, keeping note, carry-forward and summaries in the body and attaching the full material, changes the "everything in the email body" design, so it is Juniper's call.
- Severity: low. Concern: the 08:30 slot is shared with orion-actions' daily journal (`ACTIONS_DAILY_PULSE_*`). They do not collide in practice. The pulse goes through cortex-exec on the `metacog` route, while the letter is an admitted `agent`-lane durable run at `background` priority that queues behind whatever the GPU pool is serving. Email goes through notify synchronously per request with no shared slot. Since #2429 (merged into the base), Daily Pulse generation is paused by default (`ACTIONS_DAILY_PULSE_ENABLED=false`), but the daily journal still fires at 08:30. If both email, Juniper gets two emails around 08:30. Mitigation: move `HUB_ORION_DAY_MINUTE_LOCAL` if she prefers them staggered.
- Severity: low. Concern: the in-memory guards (the empty-day cache, the exhaustion notice, "outcome unknown") reset on a Hub restart. The worst case is one repeated notice, or one resend of a letter whose earlier send outcome was unknown. Every decision about what to submit and whether to email comes from Postgres and durable state.
- Severity: low. Concern: the dream-hypothesis claim has the same "durable run never started" gap that carry-forward now closes. It was left unchanged on purpose, because changing it would move the blind scorecard's denominator.
- Severity: low. Concern: notify also publishes an in-app copy of every email request. That is the existing notify behaviour, and it is one copy per successful send.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2437

🤖 Generated with [Claude Code](https://claude.com/claude-code)
