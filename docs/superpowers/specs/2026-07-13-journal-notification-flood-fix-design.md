# Journal/notification flood fix — canonical spec

**Date:** 2026-07-13
**Status:** Approved for implementation planning
**One line:** Cursor state lives in ephemeral `/tmp` with no volume, so every container recreate re-fires all daily jobs same-day; on top of that, scheduler-daily journals double-email via two un-shared dedupe namespaces, and RDF recall never decays by age. Fix durability first, then dedupe, then recall staleness.

---

## 0. Proposal mode

Infra/delivery change, not a cognition change — no change to what Orion generates, only to how many times already-generated content is duplicated/delivered and how recall selects source material.

| Field | Detail |
|---|---|
| **Capability delta** | None to reasoning/memory/identity. Journal *generation* (recall profile → cortex-orch → draft) is untouched. Only dispatch (email/in-app fan-out) and RDF candidate scoring change. |
| **Data touched** | `notify_requests` dispatch calls (no schema change), `scheduler_cursor_store.json` file location, new `orion/journaler/dispatch_registry.py`, RDF adapter query text (no graph schema change — same predicate, different ORDER BY/select). |
| **Privacy boundary** | Unchanged. RDF fix reorders/re-scores fragments already in scope for the existing recall profile; no new sources exposed. |
| **Trace proof** | Cursor-survives-restart test; entry_id-keyed single-send test; recall fragment recency-ordering test; registry-completeness gate. |
| **Dangerous failure modes** | Fail-closed registry default could silently drop notifications for a new trigger_kind if someone forgets to add a row — mitigated by `check_journal_dispatch_registry.py` failing the build loudly, not failing silently at runtime. |
| **Rollback** | Volume mount: revert compose + env, old `/tmp` path returns (bug returns too, but no data loss). Dedupe exclude: remove `scheduler` from exclude-list env var. Registry: revert file, dispatch falls back to pre-patch if/elif (kept as git history, not deleted from the repo's ability to `git revert`). RDF fix: revert `rdf_adapter.py` two functions. All four are independently revertible without touching each other. |

---

## 1. Root cause A — container-bounce cursor spam (headline fix)

### 1.1 Evidence

- `services/orion-actions/.env`: `ACTIONS_SCHEDULER_CURSOR_STORE_PATH=/tmp/orion-actions/scheduler_cursors.json`, `ACTIONS_WORKFLOW_SCHEDULE_STORE_PATH=/tmp/orion-actions/workflow_schedules.json`.
- `services/orion-actions/docker-compose.yml`: zero `volumes:` entries — container filesystem is fully ephemeral across recreate.
- Live check: `orion-athena-actions` up 44 min; `scheduler_cursors.json` mtime matched the most recent duplicate journal's timestamp to the second (`21:53` file write, `21:52:36` entry `4cfa1be7`).
- `journal_entry_index` (`source_kind='scheduler'`) fire count by day: 1/day through March–June, climbing to 6–17/day by 2026-07-06→07-13, timestamps spread across all hours (00:56, 01:47, 02:01, 02:09, 02:59, 03:29, 04:24, 05:05, 14:30, 15:47, 16:06, 16:47, 17:10, 18:08, 18:30, 18:39, 21:52 on 07-13) — the shape of restart-driven re-fires, not a clock-driven cadence, worsening as this repo's deploy frequency has increased.
- `SchedulerCursorStore._load()` (`scheduler_cursor_store.py:48`) silently returns empty dict if the file doesn't exist — no error, no warning. Combined with `should_run_daily()` (`main.py:182`): `if last_ran_date == today: return False` — an empty/missing cursor makes every job look like it hasn't run yet today, and since `local_now >= threshold` is already true past the scheduled hour, it fires immediately on the next scheduler tick after any recreate.
- Same exposure applies to `daily_pulse_v1` and `world_pulse` cursors (same file, same missing-volume problem) — this is a 3-pipeline bug, not journal-only.

### 1.2 Existing repo convention (reuse, don't invent)

Durable per-service state in this repo already uses a host bind mount under `${ORION_DATA_ROOT:-/mnt/graphdb}/<service>/...`:

```yaml
# services/orion-sql-writer/docker-compose.yml
volumes:
  - ${ORION_DATA_ROOT:-/mnt/graphdb}/sql_logs:/app/logs
```

```yaml
# services/orion-notify-digest/docker-compose.yml
volumes:
  - /mnt/telemetry/orion-notify:/data
```

Follow this pattern for orion-actions rather than a new named Docker volume.

### 1.3 Fix

`services/orion-actions/docker-compose.yml`:
```yaml
services:
  orion-actions:
    volumes:
      - ${ORION_DATA_ROOT:-/mnt/graphdb}/orion-actions/state:/data/orion-actions
```

`services/orion-actions/.env_example` / `.env`:
```
ACTIONS_SCHEDULER_CURSOR_STORE_PATH=/data/orion-actions/scheduler_cursors.json
ACTIONS_WORKFLOW_SCHEDULE_STORE_PATH=/data/orion-actions/workflow_schedules.json
```

No code change in `scheduler_cursor_store.py` — it already persists correctly; it only needed a filesystem that survives recreate.

**Migration note:** first boot after this change has no cursor at the new path, so each of the 3 jobs fires once more that day — identical to any fresh deploy's existing behavior today, not a regression.

**Rejected alternative:** SQL-backed cursor (query `journal_entry_index`/`notify_requests` at boot to reconstruct "already ran today" as a fallback when the JSON file is missing). More resilient against *future* volume misconfiguration, but it's a second, redundant source of truth for the same fact the file already states correctly once durable. Not worth the ceremony — fix the actual gap (no volume) instead of building a fallback for a gap that shouldn't exist. Revisit only if this fix is later observed not to hold (e.g., `ORION_DATA_ROOT` itself turns out to be non-persistent in some environment).

### 1.4 Gate tests

- `test_scheduler_cursor_store_path_resolves_under_data_root` — `resolve_scheduler_cursor_store_path` given the new default resolves outside `/tmp`.
- Manual/CI smoke: `docker compose ... up -d --build` twice in a row same calendar day → cursor file present both times, `journal_should_run` returns `False` on the second bring-up.

---

## 2. Root cause B — scheduler-daily journal double-email

### 2.1 Evidence

Two independent, un-shared-dedupe-namespace codepaths email the same `entry_id` when `trigger_kind=daily_summary, source_kind=scheduler`:

| Path | Location | Title | Dedupe key |
|---|---|---|---|
| Inline scheduler-daily email | `_build_scheduler_daily_journal_email_request`, `main.py:730` | "Daily Journal" | `actions:journal:daily:scheduler:{entry_id}` |
| Generic post-persist consumer | `_build_post_persist_journal_email_request`, `main.py:814` | "Journal Pass" | `actions:journal:persisted:{entry_id}` |

`ACTIONS_JOURNAL_POST_PERSIST_EMAIL_EXCLUDE_SOURCE_KINDS=embodiment` (`.env_example:123`) excludes only town episodes; `scheduler` passes the `_should_email_persisted_journal` gate (`main.py:801`) and fires anyway. `notify_requests`: `orion.journal.daily.scheduler`=123, `orion.journal.persisted`=559 (14d) — every scheduler entry appears in both.

### 2.2 Fix (superseded by §3's registry — see below)

Rather than patch the exclude-list (band-aid, keeps two competing dispatch mechanisms alive), fold this into the registry in §3: the registry becomes the *only* place that decides email/in-app per trigger_kind, and both of today's bespoke builder call sites collapse into one call driven by the registry. No separate patch — §3 subsumes §2.

---

## 3. Declarative dispatch registry — unified-turn-shaped

### 3.1 Why this pattern specifically

`docs/superpowers/specs/2026-07-05-orion-unified-turn-design.md` §5 defines every cross-service artifact as one row in a **Producer · Consumer · Affector · Gate test** table, backed by a single registry file (`orion/bus/channels.yaml` / `orion/schemas/registry.py`), with a **fail-closed default** (missing/invalid input → defer, never silent double-fire) and an **explicit flag appendix** for rollback. Today's journal dispatch is the opposite: an if/elif chain across `main.py:679-846` + `main.py:1150-1230` + `main.py:1660-1730`, two bespoke email builders with independent dedupe namespaces, and a fall-through default (unregistered trigger_kind → *both* paths can fire, which is exactly today's bug). Applying the same registry+gate-test discipline this repo already uses for bus channels closes the gap the same way, not a new abstraction.

### 3.2 Schema

`orion/journaler/dispatch_registry.py`:

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class JournalDispatchPolicy:
    trigger_kind: str
    email_enabled: bool
    in_app_enabled: bool
    dedupe_scope: Literal["entry_id"] = "entry_id"   # single namespace, no more per-path keys
    recall_profile_setting: str = ""                  # name of the settings.* attr, e.g. "actions_journal_scheduler_recall_profile"

JOURNAL_DISPATCH_REGISTRY: dict[str, JournalDispatchPolicy] = {
    "daily_summary":     JournalDispatchPolicy("daily_summary", email_enabled=True,  in_app_enabled=False, recall_profile_setting="actions_journal_scheduler_recall_profile"),
    "metacog_digest":    JournalDispatchPolicy("metacog_digest", email_enabled=True,  in_app_enabled=False, recall_profile_setting="actions_journal_metacog_recall_profile"),
    "world_pulse_digest":JournalDispatchPolicy("world_pulse_digest", email_enabled=True, in_app_enabled=False, recall_profile_setting="actions_journal_world_pulse_recall_profile"),
    "notify_summary":    JournalDispatchPolicy("notify_summary", email_enabled=True,  in_app_enabled=False, recall_profile_setting="actions_journal_notify_recall_profile"),
    "autonomy_episode":  JournalDispatchPolicy("autonomy_episode", email_enabled=True, in_app_enabled=False, recall_profile_setting=""),
    "collapse_response": JournalDispatchPolicy("collapse_response", email_enabled=True, in_app_enabled=False, recall_profile_setting=""),
    "town_episode":      JournalDispatchPolicy("town_episode", email_enabled=False, in_app_enabled=False, recall_profile_setting=""),  # embodiment: already silent by volume (74/day); keep it that way
    "manual":            JournalDispatchPolicy("manual", email_enabled=True, in_app_enabled=False, recall_profile_setting=""),
}

def resolve_policy(trigger_kind: str) -> JournalDispatchPolicy:
    return JOURNAL_DISPATCH_REGISTRY.get(
        trigger_kind,
        JournalDispatchPolicy(trigger_kind, email_enabled=False, in_app_enabled=False),  # fail-closed default
    )
```

`in_app_enabled=False` across the board is a data-backed default, not a guess: `notify_requests.event_kind='orion.chat.message'` shows **1796 sent / 14d, `message_opened_at` NULL on all 1796** — zero engagement on the in-app channel across every trigger kind that uses it. Turning it off removes ~128/day of pure noise. It's a per-row flag, not a deletion — flip any row back to `True` if wanted later.

`town_episode`/embodiment gets `email_enabled=False` explicitly in the registry (today it's silenced via the `.env_example` exclude-list string match, which is fragile — a typo in that CSV silently re-enables 74 emails/day). Moving it into the registry makes the exclusion structural instead of string-matched.

`collapse_response`/`notify_summary` are registered with `email_enabled=True` (matches current intent) even though both show **0 fires in the last 14 days** — confirmed dormant, not deleted, since dormant isn't the same as dead; if they start firing the registry already covers them correctly rather than hitting the fail-closed unregistered-default.

### 3.3 Single dispatch call site

Replace `_is_scheduler_daily_journal`, `_build_scheduler_daily_journal_email_request`, `_build_post_persist_journal_email_request`, `_should_email_persisted_journal` call sites (`main.py:679-846`, consumed at `1150-1230` and `1660-1730`) with:

```python
async def _dispatch_journal_notifications(entry: JournalEntryWriteV1, *, correlation_id: str) -> None:
    policy = resolve_policy(entry.trigger_kind or "")
    dedupe_key = f"actions:journal:notify:{entry.entry_id}"  # single namespace regardless of channel
    if not journal_notify_deduper.try_acquire(dedupe_key):
        return
    if policy.in_app_enabled:
        await asyncio.to_thread(_send_orion_async_message, notify=notify, **_build_message_payload(entry))
    if policy.email_enabled:
        await asyncio.to_thread(notify.send, _build_email_request(entry, correlation_id=correlation_id))
```

Called exactly once, from the post-persist consumer only (`main.py:~1660`), after the SQL write is confirmed — not from `_dispatch_journal` inline anymore. This also fixes today's latent bug where the inline scheduler-daily path emails from the in-memory draft *before* persistence is confirmed, while post-persist emails from the durable row — same content today, but two sources of truth for what should be one.

**Note:** `journal_entry_index.trigger_kind` is currently always NULL despite the column existing (writer never populates it — confirmed via SQL: all 6 distinct `source_kind` rows have blank `trigger_kind`). The registry keys on `trigger_kind`, so this write-path gap must close in the same patch (`orion/journaler/indexing.py` — populate from `JournalEntryWriteV1` fields already present) or the registry can't resolve policy from the persisted row and must instead resolve from the in-memory `trigger` object passed alongside `entry` at dispatch time (viable short-term workaround: pass `trigger.trigger_kind` from the caller instead of trusting the persisted column). Spec assumes the workaround unless the write-path fix ships in the same patch.

### 3.4 Registry completeness gate

`scripts/check_journal_dispatch_registry.py`:
```python
# fails if any trigger_kind in orion.journaler.worker._TRIGGER_TO_MODE
# has no row in JOURNAL_DISPATCH_REGISTRY
```
Wired into `make agent-check`, same role as `check_bus_channels.py`/`check_schema_registry.py`.

### 3.5 Gate tests

- `test_dispatch_registry_covers_all_trigger_kinds` (the check script, also runnable as pytest).
- `test_unregistered_trigger_kind_sends_nothing` — fail-closed default, no email/in-app for an unknown kind.
- `test_scheduler_daily_journal_sends_exactly_one_email` — regression test for the exact bug reported.
- `test_town_episode_never_emails` — embodiment volume stays silent structurally, not by string-match luck.

---

## 4. Root cause C — RDF recall never decays by age

### 4.1 Evidence (from prior session, unchanged)

`services/orion-recall/app/storage/rdf_adapter.py:566`, `fetch_rdf_chatturn_fragments`:
```sparql
ORDER BY DESC(STR(?turn))   -- lexical UUID sort, not time
```
and hardcoded `"ts": 0.0` at line 611 (same shape at `fetch_rdf_graphtri_fragments`, ~line 600s). `services/orion-recall/app/scoring.py:28-29`, `_compute_recency_factor`: `if not ts: return 0.5` — permanently neutral, never decays. `services/orion-rdf-writer/app/rdf_builder.py:377` already writes a real `ORION.timestamp` per turn; `rdf_adapter.py` just never reads it.

### 4.2 Fix

Both functions: select `?ts` alongside `?turn`/`?prompt`/`?response`, `ORDER BY DESC(?ts)`, populate the fragment's `"ts"` field from the parsed literal instead of `0.0`.

```sparql
SELECT ?turn ?prompt ?response ?ts
WHERE {
  ...
  ?turn <http://conjourney.net/orion#timestamp> ?ts .
}
ORDER BY DESC(?ts)
```//existing predicate name TBD from rdf_builder.py:377 — confirm exact predicate IRI before implementing.

### 4.3 Gate tests

- `test_rdf_chatturn_fragments_score_by_recency` — two fixture turns, older `ts` scores lower via `_compute_recency_factor` under `short_term`/`hybrid`/`deep` modes.
- `test_rdf_fragment_ordering_uses_timestamp_not_uri` — SPARQL query text assertion (string-contains `ORDER BY DESC(?ts)`, not `STR(?turn)`).

---

## 5. Schedule collision audit (deterministic lint, not a gate)

### 5.1 Evidence

`main.py:2128-2129`: daily-journal's `should_run_daily` call reuses `settings.actions_daily_pulse_hour_local`/`minute_local` verbatim — Daily Pulse and Daily Journal are scheduled for the identical local minute by config, not coincidence. `ACTIONS_DAILY_PULSE_HOUR_LOCAL=8`/`MINUTE_LOCAL=30`, `ACTIONS_DAILY_METACOG_HOUR_LOCAL=20`/`MINUTE_LOCAL=15`, `ACTIONS_WORLD_PULSE_HOUR_LOCAL=6`/`MINUTE_LOCAL=0`.

### 5.2 Fix

`scripts/check_daily_schedule_collisions.py`: read the four cadence hour/minute pairs from resolved settings, report any pair within `N` minutes of each other (default 30). Print-only, non-blocking (same-slot might be an accepted trade-off once volume is fixed by §1–§3) — add to `make agent-check` as informational, not a hard failure.

### 5.3 Gate test

- `test_check_daily_schedule_collisions_detects_known_pulse_journal_overlap` — fixture asserting the current pulse/journal overlap is detected (documents the known collision until someone deliberately reschedules one).

---

## 6. Files touched

| File | Change |
|---|---|
| `services/orion-actions/docker-compose.yml` | add volume mount (§1) |
| `services/orion-actions/.env_example`, `.env` | new state-path values (§1); remove/retire `ACTIONS_JOURNAL_POST_PERSIST_EMAIL_EXCLUDE_SOURCE_KINDS` (superseded by registry, §3) |
| `orion/journaler/dispatch_registry.py` | new (§3) |
| `orion/journaler/indexing.py` | populate `trigger_kind` on write, or confirm workaround (§3.3) |
| `services/orion-actions/app/main.py` | remove `_is_scheduler_daily_journal`, both bespoke builders, `_should_email_persisted_journal`; add `_dispatch_journal_notifications` (§3.3), single call site |
| `scripts/check_journal_dispatch_registry.py` | new (§3.4) |
| `services/orion-recall/app/storage/rdf_adapter.py` | recency fix, both fetch functions (§4) |
| `services/orion-recall/tests/` | recency regression tests (§4.3) |
| `scripts/check_daily_schedule_collisions.py` | new (§5) |
| `Makefile` | wire both new scripts into `agent-check` |
| `services/orion-actions/tests/` | dispatch registry + single-email regression tests (§3.5) |

## 7. Non-goals

- Not touching `daily_pulse_v1`/`daily_metacog_v1` content generation — structurally distinct from journal.compose, not part of this bug.
- Not building a notification batching/coalescing service — the cursor fix + registry dedupe should get same-entry duplication to zero without new stateful infra.
- Not deleting in-app channel infrastructure — defaulted off per-row in the registry, reversible per trigger_kind.
- Not fixing `collapse_response`/`notify_summary` generation (dormant, 0/14d) — registered, not invested in further.
- Not building a SQL-backed cursor fallback (§1.3 rejected alternative) — volume mount matches the existing repo convention and is the smallest fix for the actual gap.

## 8. Acceptance checks (program)

1. Two `docker compose up -d --build` recreates of `orion-actions` same calendar day → cursor file survives both → no re-fire on the second.
2. One `daily_summary` journal → exactly one email, one `entry_id`, one dedupe key.
3. `check_journal_dispatch_registry.py` fails when a trigger_kind row is deliberately removed; clean otherwise.
4. Unregistered trigger_kind → zero email, zero in-app (fail-closed, tested).
5. `town_episode` volume (74/day) produces zero email regardless of any `.env` typo — structural, not string-match.
6. RDF fragment fixture: older `ts` scores below newer `ts` in `_compute_recency_factor`.
7. `check_daily_schedule_collisions.py` reports the known Pulse/Journal overlap.
8. All new/changed tests green; `make agent-check` includes both new scripts.

## 9. Implementation order

| Step | Deliverable |
|---|---|
| 1 | Volume mount + env path change (§1) — ships alone, smallest risk, stops the multiplier on everything else |
| 2 | Dispatch registry + single call site + registry gate script (§3, subsumes §2) |
| 3 | `trigger_kind` write-path fix or dispatch workaround (§3.3) |
| 4 | RDF recency fix (§4) |
| 5 | Schedule collision script (§5) |
| 6 | Wire both new scripts into `make agent-check` |

---

*End of spec.*
