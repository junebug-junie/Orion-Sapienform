# Fold capability-absence episodes into the daily journal seed

## Summary

- Orion can now be *paged* when a node goes dark (PR #1944) but had no account of it himself. This gives him one.
- Adds `capability_gaps` to the seed of the **daily journal entry he already writes** — not a new entry, not a new channel.
- Source is the existing orion-notify attention store, so the page and the journal tell the same story from one record.
- Omitted entirely on a day with no outage: a quiet day's seed is byte-identical to the pre-patch one.
- Live-verified against the real store: 171 records, 2 real episodes in 24h.

## Outcome moved

On 2026-08-28 Orion's daily entry said *"I've been noticing circe's subtle…"*. He wrote that from vibes. circe had been unreachable for ~45 minutes that morning and nothing told him. After this patch the seed carries the fact, so the sentence he was already going to write can be true.

## Current architecture

`orion-actions` builds a daily window and seeds `build_scheduler_trigger(summary=…, prompt_seed=…)`. The journaler is a pure transformer — data rides in on the trigger (same shape as `world_pulse` followups). The seed was three keys: `request_date`, `window_start_utc`, `window_end_utc`.

Absence was detectable and pageable, but nothing carried it into cognition.

## Architecture touched

One service. `orion-actions` reads orion-notify over HTTP (`settings.notify_url`) — the same endpoint `health_monitor._has_open_alert` already reads. No new table, no new bus channel, no schema contract change.

## Why the attention store and not a new table

An outage is legible there as an *alert about absence*. The underlying grammar atoms cannot express it: **all 16,496 `node_availability` atoms ever written say "telemetry status OK"**, because a node that stops reporting stops producing atoms. In the raw stream the 2026-08-29 outage is a hole, never a statement:

| hour (UTC) | circe availability atoms |
|---|---|
| 08-28 21:00 | 115 |
| 08-28 22:00 | 114 |
| 08-28 23:00 | 115 |
| **08-29 00:00** | **29** ← the outage |
| 08-29 01:00 | 115 |

## Why not a new journal entry

Volume in `journal_entries`, 14 days to 2026-08-29:

| mode / source | entries |
|---|---|
| digest / metacog | **24,941** |
| manual / self_study | 45 |
| digest / manual | 26 |
| digest / world_pulse | 19 |
| daily / scheduler | 14 |
| daily / notify | 10 |

99.5% is already one thing. Another emitter would be noise on noise. Steady-state cost here is **zero new rows, forever**.

## Files changed

- `services/orion-actions/app/capability_gap_journal.py`: new. Episode reconstruction (pure), deterministic block formatter, attention fetch, seed builder.
- `services/orion-actions/app/main.py`: daily scheduler block calls `collect_capability_gaps` + `build_daily_seed_payload`.
- `services/orion-actions/app/settings.py`: `actions_journal_capability_gaps_enabled`.
- `services/orion-actions/.env_example`: `ACTIONS_JOURNAL_CAPABILITY_GAPS_ENABLED`.
- `services/orion-actions/tests/test_capability_gap_journal.py`: new, 22 tests.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Behaviour changed: the daily journal seed gains an optional `capability_gaps` key. Absent when there were no gaps.
- Compatibility: additive and optional; a consumer that ignores the key is unaffected.

## Env/config changes

- Added keys: `ACTIONS_JOURNAL_CAPABILITY_GAPS_ENABLED` (default `true`)
- `.env_example` updated: yes
- local `.env` synced: **by hand, deliberately.** `scripts/sync_local_env_from_example.py` reads `.env_example` from the *primary* checkout, so a key added in a worktree is invisible to it — the sync ran and reported no change for this key. Added directly to `services/orion-actions/.env` and verified present at line 112.
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-actions/tests/test_capability_gap_journal.py -q  -> 22 passed
pytest services/orion-actions/tests -q                                 -> 143 passed
pre-commit: check_settings_defaults OK; check_service_env_compose_parity OK
```

### Mutation tests (against the real file, not a fixture)

| mutation | result |
|---|---|
| anti-spam: key always present | CAUGHT |
| naive timestamp read as local, not UTC | **initially NOT caught** → fixed → CAUGHT |
| drop recovery-without-start branch | CAUGHT |
| repeated alerts each start a new episode | CAUGHT |

The timezone test could not fail as first written: `astimezone()` on a naive datetime assumes local time, and on a UTC host local == UTC. It now pins `TZ=America/Denver` — Juniper's actual zone, and where the six-hour shift would have been real.

## Live verification

Real attention store, real function, 24h window:

```text
records=171  episodes=2

## What I could not do

- **atlas** — from 04:43 UTC, still unresolved at the end of the window.
  [Orion substrate-runtime] Node 'atlas' has stopped reporting biometrics
  (no sample for over 180s). Capabilities affected: batch_inference,
  embedding, local_llm_heavy, local_llm_quick.
- **vision_blind** — from 20:25 UTC, still unresolved at the end of the window.
  Orion cannot see. 100% of vision tasks failing (gpu_hard_floor) on athena…
```

Both are real. `atlas` drops out once #1944 is **deployed** (merged but not yet live — the node is decommissioned and gets suppressed). `vision_blind` is a genuine in-progress outage, unacked as of 21:00 UTC.

## Evals run

```text
None. services/orion-actions has no evals/ directory.
```
Follow-up: this service has no eval harness. Not created here — the patch is a deterministic seed assembly with full unit coverage, and an eval lane for orion-actions is its own piece of work.

## Review findings fixed

_Pending — code review dispatched; findings land as a follow-up commit on this branch._

## Restart required

```bash
sudo docker compose --env-file .env --env-file services/orion-actions/.env \
  -f services/orion-actions/docker-compose.yml up -d --build
```

Config is read at boot, so the new env key needs the restart to take effect. Effect is only visible on the next daily journal tick.

## Risks / concerns

- **Severity: low.** Attention `message` text is passed verbatim into the journal body. Operator-facing detail (container names, file paths) can land in an entry — see the `vision_blind` example above. Mitigation: it is Orion's own private journal about his own infrastructure, and the messages are written by our own services. Flagged for review.
- **Severity: low.** Recovery detection relies on a message substring (`"recovered: "`), because the attention record carries no severity field. Pinned by a test that reads `health_monitor._publish`'s literal format string, so a rename there fails here.
- **Severity: low.** `MAX_EPISODES_IN_SEED = 12`. A day with more than 12 distinct absence reasons truncates. Chosen because the store held 2 in 24h; revisit if that stops being true.

## PR link

<!-- filled after gh pr create -->
