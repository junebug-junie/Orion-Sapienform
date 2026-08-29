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
- `services/orion-actions/tests/test_capability_gap_journal.py`: new, 30 tests.

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
pytest services/orion-actions/tests -q                                 -> 151 passed
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

Adversarial review returned **BLOCKED** with six verified defects. All fixed; each fix mutation-tested against the real file.

- **BLOCKER — `httpx` is not installed in the orion-actions image, and the import sat outside the `try`.**
  - Verified live: `docker exec orion-athena-actions python -c "import httpx"` → `ModuleNotFoundError`. `requirements.txt` has `requests`, not `httpx`.
  - Blast radius was not just this feature: the exception escaped into the scheduler's shared handler at `main.py:2238`, so the first daily tick after deploy would have killed the journal dispatch *and* the workflow-schedule claim, then re-raised every 45s all day. Tests passed only because the repo venv happens to have httpx.
  - Fix: `requests` on a worker thread (already a dependency; mirrors `orion-vision-host/app/liveness.py` against this same endpoint), import moved inside the `try`, plus a dedicated `try/except` at the `main.py` call site so nothing here can ever reach the shared handler.
  - Evidence: module imports and runs to completion **inside the live container**; with notify unreachable it returns `[]` and logs, no exception.

- **CRITICAL — `vision_blind` episodes could never close; three real outages collapsed into one false span.**
  - `orion-vision-host/app/liveness.py:342` announces recovery under a *different* reason (`vision_recovered`, severity `info`) with no `"recovered: "` substring, so the liveness filter discarded it before it could close anything. Separately, folding repeat alerts was wrong for a producer that re-arms.
  - Fix: `RECOVERY_REASON_BY_ALERT` maps recovery reasons onto their alert; folding removed entirely (it existed to absorb an edge-triggered restart re-fire, but `_has_open_alert` already prevents that — the absence sweep fired 142 times on 08-29 and produced exactly **one** record).
  - Follow-on found by re-running live: `vision_recovered` has **never been emitted — 0 rows, ever**, so episodes still never closed and a 24h window inherited all nine since 08-21. Added an upper-bound close: vision-host must clear `_alerting` before re-arming, so a later alert proves the earlier gap ended — reported as `duration_upper_bound_minutes` with `resolved: false`, never as a hard duration.
  - Evidence: live in-container run went **9 permanently-open episodes → 5 correctly-bounded ones**.

- **HIGH — an outage in progress for the whole window produced nothing.** Records were filtered by the window, so day two of a multi-day outage had neither an in-window alert nor an in-window recovery and vanished. Fix: build episodes from full history, filter by *overlap*.

- **MEDIUM — `MAX_EPISODES_IN_SEED` truncation dropped the newest episode**, and no test covered the cap. Fix: select unresolved-first then newest, restore chronological order for display; test pins which episode survives.

- **MEDIUM — the docstring claim "the attention record carries no severity field" was false.** `severity` is declared on `ChatAttentionState` and present in the live payload; the claim came from my own truncated key listing (`sorted(keys)[:14]` cut it off). Severity is now the primary recovery signal, with the marker as a secondary — which also closes the hole where a critical alert whose interpolated capability list contained `"recovered: "` was journaled as a gap that *ended*.

- **MEDIUM — `format_capability_gap_block` was dead code** whose docstring claimed a deterministic-fallback guarantee it did not provide (nothing called it; three tests exercised unreachable code). Deleted, with the tests. **Follow-up:** there is now no deterministic fallback if the composer ignores the seed key — wiring one needs an injection point in `orion/journaler`, which is a different service boundary.

- **LOW — producer message went into the prompt verbatim and unbounded.** Now truncated to `MAX_DETAIL_CHARS = 320`.

- **LOW — the 200-record fetch is orion-notify's hard ceiling** (`Query(le=200)`). Now logs a warning when the slice comes back full, so a burst pushing the window out of range is visible rather than silently reporting no gaps.

### Claims the review confirmed held up

Timezone handling and its non-vacuous test; the byte-identical-seed property on the pure-function path; env parity including the kill switch actually reaching the container; no bus/schema contract change; the service boundary; and `reason` being matchable as written.

### Mutation results after the fixes

| mutation | result |
|---|---|
| drop `vision_recovered` pairing | CAUGHT |
| drop the later-alert upper bound | CAUGHT |
| ignore severity, marker only | CAUGHT |
| match marker regardless of severity | CAUGHT |
| slice chronologically (drops newest) | CAUGHT |
| filter records by window, not overlap | CAUGHT |
| no detail truncation | CAUGHT |

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
