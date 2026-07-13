# PR: resonance health monitor + reverie glimpse into internal chat context

Branch: `worktree-reverie-chat-bridge-resonance-monitor`

## Summary

Implements Parts A and B of `docs/superpowers/specs/2026-07-12-reverie-chat-bridge-and-resonance-monitoring-design.md` (Part C explicitly deferred per the spec's own sequencing — not started).

- **Part A** (`orion-thought`): an edge-triggered health monitor for the reverie resonance tripwire, the 4th port of the pattern already reviewed and running in field-digester/attention-runtime/self-state-runtime. Pages via `orion-notify` only when a theme's `violation_count` is *increasing* across its last 2 persisted samples — not merely "an alert exists" — because a 2026-07-12 investigation found a real historical burst (2026-07-07 through 2026-07-12 03:10) that had already self-resolved while the detector kept re-reporting the same frozen numbers for ~20 hours as they aged out of its 200-row lookback window.
- **Part B** (`orion-cortex-exec` / shared `orion/substrate`): surfaces the latest fresh, non-hollow reverie thought (`SpontaneousThoughtV1.interpretation`) into the **internal** stance-synthesis pass (`chat_stance_brief.j2`) only — `chat_general.j2` (user-facing) is untouched.

## Outcome moved

- A worsening resonance loop will now page Juniper via Hub's existing Pending Attention panel instead of sitting invisible in a debug table.
- Orion's internal stance-synthesis pass can now reference "what I was just thinking about" (age-gated to 180s, hollow-guarded twice) — the first piece of reverie's real, anchored narration to reach any part of the chat pipeline, even internally.

## Current architecture

The reverie resonance tripwire (`orion.reverie.resonance.detect_resonance`, called from `chain.py::_maybe_emit_resonance_alert`) was observation-only. Reverie's narration (`substrate_reverie_thought`, confirmed live at 1,959+ rows, 0% hollow) reached only a dry-run proposal ladder and a Hub debug panel — nothing user- or Orion-facing.

## Architecture touched

- `services/orion-thought/app/{settings.py,store.py,chain.py}`, new `app/resonance_monitor.py`.
- `orion/substrate/felt_state_reader.py` (new lane), `services/orion-cortex-exec/app/chat_stance.py` (new projection), `orion/cognition/prompts/chat_stance_brief.j2` (new template block).

## Files changed

- `services/orion-thought/app/resonance_monitor.py` (new): `HealthCheck`/`_check`/`ResonanceHealthMonitor`, edge-triggered per-theme worsening detection.
- `services/orion-thought/app/store.py`: `load_recent_resonance_alerts(theme_key, limit=2)`.
- `services/orion-thought/app/chain.py`: two new call sites in `_maybe_emit_resonance_alert`, both offloaded via `asyncio.to_thread`.
- `services/orion-thought/app/settings.py`: `NOTIFY_BASE_URL`/`NOTIFY_API_TOKEN` (new to this service).
- `services/orion-thought/{docker-compose.yml,.env_example,.env,requirements.txt,README.md}`.
- `services/orion-thought/tests/test_resonance_monitor.py` (new, 16 tests).
- `orion/substrate/felt_state_reader.py`: new `latest_reverie_thought` lane (`max_age_sec=180`, 2x the reverie tick interval).
- `services/orion-cortex-exec/app/chat_stance.py`: new `_project_reverie_glimpse()`, wired near the existing `_project_self_state_from_beliefs` call site.
- `orion/cognition/prompts/chat_stance_brief.j2`: new `{% if chat_reverie_glimpse %}` block.
- `services/orion-cortex-exec/tests/test_chat_stance_reverie_glimpse_projection.py`, `test_felt_state_reader_reverie_lane.py` (new, 15 tests).

## Schema / bus / API changes

None. Reuses `orion-notify`'s existing `ChatAttentionRequest`/`GET /attention` contract (same as the three prior health-monitor ports); reads an existing table (`substrate_reverie_thought`) and existing schema (`SpontaneousThoughtV1`).

## Env/config changes

- Added keys (`services/orion-thought` only): `NOTIFY_BASE_URL`, `NOTIFY_API_TOKEN`.
- `.env_example` updated: yes. Local `.env` synced by hand (matching `.env_example` key-for-key, confirmed via diff) — the repo's `sync_local_env_from_example.py` script's default mode doesn't cover these keys (same gap noted in the field-digester PR), and this session's earlier attempt at `--all-keys` was reverted after it touched unrelated services; hand-editing was the safe path this time.
- No env changes for `orion-cortex-exec` (Part B is pure code, reusing the already-enabled `ENABLE_SUBSTRATE_FELT_STATE_CTX`).

## Tests run

```
$ .venv/bin/python -m pytest services/orion-thought/tests/ -q
145 passed

$ .venv/bin/python -m pytest services/orion-cortex-exec/tests/test_chat_stance_reverie_glimpse_projection.py services/orion-cortex-exec/tests/test_felt_state_reader_reverie_lane.py -q
15 passed
```

Additionally ran the broader `chat_stance`/`felt_state_reader`-adjacent cortex-exec test files directly (bypassing 12 pre-existing, unrelated collection errors caused by a verb-registry double-import — confirmed identical on clean `main` via `git stash`): 100 passed, 2 failed — confirmed via the same clean-`main` comparison that those 2 failures are pre-existing test-order fragility in `test_chat_stance_brief.py`, unrelated to this diff.

## Evals run

No eval harness exists for either surface touched; this is wiring/infra-hardening plus a narrow internal-context addition, not a model-quality question.

## Docker/build/smoke checks

```
$ docker compose -f services/orion-thought/docker-compose.yml build   # Image Built
$ docker compose -f services/orion-thought/docker-compose.yml up -d   # Started
$ curl -fsS http://localhost:7155/health                              # {"ok":true,...}

$ docker compose -f services/orion-cortex-exec/docker-compose.yml build  # all 4 images Built
$ docker compose -f services/orion-cortex-exec/docker-compose.yml up -d  # all 4 containers Started
$ curl -fsS http://localhost:8070/health                                 # {"ok":true,"service":"cortex-exec",...}
```

Live verification beyond "it started cleanly":
- `orion-thought`: confirmed refractory correctly suppressing the historically-resonant theme in real time post-restart (`reverie chain suppressed by refractory theme=loop:open-loop-9d84d08cddf5`). Directly exercised `ResonanceHealthMonitor` inside the running container against the real `orion-notify` and real DB: bootstrap-from-notify made a real HTTP call (empty pending list, correct — nothing is currently wrong), `_is_worsening` correctly classified the known historical theme as healthy (`violation_count` 3→2, decreasing), and a full `check()` cycle completed with no errors and no spurious page.
- `orion-cortex-exec`: directly exercised `hydrate_felt_state_ctx` inside the running container against real data — confirmed the new `latest_reverie_thought` lane correctly excludes the actual current freshest row (~8.5 minutes old, older than the 180s gate) while `self_state` still populates normally, proving the age-gate works against real live data, not just fixtures.
- Zero errors/exceptions in any of the 5 restarted containers' logs post-deploy.

## Review findings fixed

Ran an 8-angle code review (line-by-line, removed-behavior, cross-file, reuse, simplification, efficiency, altitude, conventions) via parallel subagents before deploy, then independently verified every finding myself against the actual code rather than trusting the reports.

- Finding (Angle B, removed-behavior): `check_resonance_worsening()` made blocking `requests`/DB calls directly inside `_maybe_emit_resonance_alert`, an already-async function on the reverie chain worker's shared event loop — unlike the three prior health-monitor ports, which isolate blocking I/O via `asyncio.to_thread` from their own poll loop. A slow `orion-notify` could stall the whole chain worker for up to ~20s.
  - Fix: both call sites in `chain.py` now go through `await asyncio.to_thread(check_resonance_worsening, ...)`.
  - Evidence: `grep -c "asyncio.to_thread(check_resonance_worsening" services/orion-thought/app/chain.py` → 2.
- Finding (Angle Altitude + independently Angle A, "restart never recovers"): `_tracked_themes` is a dynamic, per-theme check-key space (unlike the three prior ports' fixed small set), and was purely in-memory — repopulated only from live ticks' `alert.theme_key`. A theme flagged unhealthy right before a restart that later stopped being `detect_resonance`'s single most-severe pick could never be re-added to tracking, so its open Pending Attention item could never receive a recovery note — a *permanent* silence, strictly worse than the "delayed alert" blind spot the prior ports' `_has_open_alert` mitigation was built for.
  - Fix: `ResonanceHealthMonitor.__init__` now calls `_bootstrap_from_notify()`, reconstructing tracked themes (and seeding their known-unhealthy state) from `orion-notify`'s own pending list at construction time.
  - Evidence: `test_bootstrap_reconstructs_tracked_themes_from_notify_pending_list`, `test_bootstrap_reconstructed_theme_can_recover_on_next_tick`, `test_bootstrap_failure_is_fail_open_not_fatal`; live-verified against the real `orion-notify` (see Docker/smoke checks above).
- Finding (found independently by Efficiency, Simplification, and Altitude angles — 3 of 8): `_tracked_themes` only ever grew, costing one DB round trip per tracked theme on every completed chain (~90s) for the life of the process, unbounded.
  - Fix: a theme is evicted from `_tracked_themes`/`_last_healthy` once its recovery note is confirmed delivered; a future flare-up re-adds it via the live `alert.theme_key` path.
  - Evidence: `test_monitor_evicts_theme_after_confirmed_recovery`.
- Finding (Reuse angle): the "independent, defense-in-depth" hollow re-check in `_project_reverie_glimpse` was shallow — it only re-read the stored `hollow` bool (`payload.get("hollow") is True`), not independent of anything since it's the same flag the producer already set.
  - Fix: validates the payload as a real `SpontaneousThoughtV1` and gates on `thought.hollow or thought.is_hollow()` — rejecting if either the stored flag or an independent re-derivation (absent coalition / too-short interpretation / unanchored evidence) says hollow. A payload that no longer validates as `SpontaneousThoughtV1` at all now fails closed instead of partially parsing.
  - Evidence: `test_none_when_coalition_absent`, `test_none_when_evidence_refs_unanchored`, `test_none_when_payload_fails_schema_validation`.

Deferred (not fixed, noted for follow-up):
- **4-way pattern duplication**: `resonance_monitor.py` is now the 4th near-verbatim port of the `HealthCheck`/edge-triggered-transition pattern (after field-digester, attention-runtime, self-state-runtime). With 4 copies now on record, extracting a shared base (e.g. `orion/notify/health_monitor_base.py`) is a reasonable next patch, but doing it now would mean touching 3 already-merged, already-running services — out of scope for this PR.
- **Detection latency from bucketed alert dedup**: `persist_resonance_alert`'s `ON CONFLICT (alert_id) DO NOTHING` (pre-existing, not part of this diff) dedups on `theme_key + refractory-window bucket`, so a rising `violation_count` within one bucket doesn't get a new persisted row — meaning the health monitor needs roughly two full refractory windows (~30 min at defaults) of distinct persisted samples before it can observe an increase at all. This is inherited from the existing resonance-persistence design, not introduced by this patch; fixing it would mean changing a shared, already-in-production dedup mechanism (also used by the bus publish and Hub debug panel) and is out of scope here.
- **`load_recent_resonance_alerts` only tested via mocks**: no DB-integration-test harness exists in this service to extend (same gap noted in the field-digester PR). Manually verified against live data instead (see Docker/smoke checks above).
- **Proposal-mode note**: the spec this PR implements is stamped "Design only... do not build without explicit sign-off." Per AGENTS.md's own exception clause ("unless Juniper directly asks to implement"), that sign-off was the user's next message immediately after the spec was delivered: an explicit `/superpowers:subagent-driven-development` command asking for implementation, commit, push, and deploy. Noting this explicitly since an automated review pass (correctly, given it couldn't see the conversation) flagged the spec's own header text as unsatisfied.

## Restart required

Already done as part of this change:

```bash
docker compose -f services/orion-thought/docker-compose.yml up -d --build
docker compose -f services/orion-cortex-exec/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: the `chat_reverie_glimpse` context value has not yet been read by a human for conversational appropriateness (the spec's own B.3 gate on going further than the internal pass).
  - Mitigation: scope is already limited to the internal `chat_stance_brief.j2` pass only, not `chat_general.j2` — no user-facing exposure exists yet regardless.
- Severity: low
  - Concern: the reverie glimpse lane's 180s freshness window means it's only populated for a narrow window after each chain completion (chains on a dominant theme currently complete roughly every ~900s due to refractory), so it will frequently be absent in practice.
  - Mitigation: none needed — this is the intended, conservative behavior (never surface stale narration); confirmed live against real data.

## PR link

Branch pushed to `origin/worktree-reverie-chat-bridge-resonance-monitor`. `gh` was not authenticated in this environment for the prior two PRs either — open manually at:
https://github.com/junebug-junie/Orion-Sapienform/pull/new/worktree-reverie-chat-bridge-resonance-monitor
