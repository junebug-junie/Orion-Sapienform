# Reverie Chat Bridge + Resonance Monitoring — Design Spec

**Date:** 2026-07-12
**Status:** Design only. Not approved for implementation. Do not build without explicit sign-off.
**Scope:** Ideas 3 and 4 from the 2026-07-12 felt-cognition brainstorm (reverie thread + resonance loop), plus idea 5 as an explicitly optional Phase 3 (shared felt-context seam).
**Supersedes nothing.** Builds on `docs/superpowers/specs/2026-07-12-inner-state-unification-design.md` and `orion/self_state/inner_state_registry.py` (both already merged), and reuses the health-monitor pattern shipped in PRs #975/#977 verbatim.

---

## Arsonist summary

Three things are true right now, all verified against live code and live Postgres data, not assumed:

1. **Reverie produces real, anchored, non-hollow narration on a genuine cadence** (`substrate_reverie_thought`: 1,959 rows, 0 hollow) that never reaches chat — its only reach is a dry-run proposal ladder (`REHEARSAL` per the registry) and a Hub debug panel.
2. **The resonance tripwire (`substrate_reverie_resonance_alert`, 380 rows) is not an active incident.** I initially carried forward a characterization from an earlier research pass that this was "currently re-igniting every ~98 seconds" — that was wrong, or at least badly stale, and I want to be explicit about correcting it rather than quietly fixing the framing. Direct live queries this session show: the last alert fired `2026-07-12 03:10:28`, over 20 hours before this spec was written, and the most recent 10 real chain completions on that exact theme are cleanly spaced ~908 seconds apart (safely above the 900s refractory bound — zero violations). The `min_gap_sec=98.3582` value repeated identically across the last 5 alerts, combined with strictly decreasing `violation_count` (6, 5, 4, 3, 2 moving forward in time), is the signature of a real historical burst (`2026-07-07` through `2026-07-12 03:10`, per `min(created_at)`/`max(created_at)` on the table) aging out of the detector's 200-row sliding window (`ORION_REVERIE_RESONANCE_WINDOW=200`), not an ongoing malfunction. The refractory mechanism (`chain.py::run_reverie_chain`, `DbRefractoryStore`) is confirmed working correctly right now.
3. **The registry (`inner_state_registry.py`) already answers the "is this safe to surface" question for the two signals a chat bridge would need**: `self_state.v1` and `phi_intrinsic_reward.v1` are both `COMPOSED`. Reverie itself (`SpontaneousThoughtV1`) isn't in the registry at all — it's out of that registry's declared scope (felt-state signals, not the reverie/dream ladder) — so this spec has to make its own case for reverie's trustworthiness, which is why finding #2 above matters so much: a resonance loop that turned out to still be live would be a real reason to hold idea 3.

---

## Part A — Resonance: post-mortem + proactive monitoring (do this first)

### A.1 What actually happened (root cause — partial, stated at the confidence level the evidence supports)

`detect_resonance()` (`orion/reverie/resonance.py:39-91`) is a pure function over a list of `(theme_key, timestamp)` pairs pulled by `load_recent_chain_theme_events(limit=200)` (`services/orion-thought/app/store.py:191-`), which reads the 200 most recent rows **across all themes** from `substrate_reverie_chain`, not scoped to one theme. It flags a theme as resonant when `>= min_violations` (default 2) consecutive gaps for that theme fall under `refractory_sec` (900s), and reports `min_gap_sec` as the single tightest such gap found anywhere in the window.

Live evidence:
```
loop:open-loop-9d84d08cddf5 chain completions, most recent 10 (all `terminal_reason=max_steps`):
  gaps: 15:08.93, 15:08.42, 15:08.40, 15:09.24, 15:08.38, 15:08.81, 15:09.18, 15:31.36, 15:09.21, 15:10.48
  → consistently ~908s, safely above the 900s refractory bound. No current violations.

resonance_alert history for this theme: 380 rows total, 2026-07-07 02:26 → 2026-07-12 03:10.
Most recent 5 alerts: violation_count 2, 3, 4, 5, 6 (strictly increasing going backward in time),
min_gap_sec frozen at 98.3582 across all 5.
```

This pattern is consistent with: a real burst of ~98-second re-ignitions on this theme happened during `2026-07-07`–`2026-07-12 03:10`, the refractory mechanism *did* correctly stop new short-gap ignitions at some point in that window (chain completions are correctly ~908s apart now), but the detector kept re-reporting the historical 98s minimum for as long as any pair of those old rows remained inside the 200-row lookback — and it stopped only once the last such pair finally aged out.

**What I have not established, and am not claiming**: the exact mechanism that produced the original ~98s cadence during 07-07–07-12. Plausible candidates, none confirmed: (a) `reverie_chain_enabled`/`reverie_refractory_sec` were only recently set to their current values and the burst predates that config; (b) a redeploy during that window briefly ran with `DbRefractoryStore` unable to reach Postgres (its own `try/except` degrades silently — `services/orion-thought/app/store.py:284-`), meaning refractory checks and writes could both have been silently no-op'ing for some sub-period without raising or logging visibly; (c) the coalition selector genuinely fixated (attention on this exact loop, tick after tick, faster than reverie's own 90s interval would suggest — possible if multiple worker instances were briefly running, e.g. during a deploy overlap). (a) and (c) are checkable from data already in Postgres; (b) would only be checkable from historical logs, which may have rotated out.

### A.2 The concrete, do-this-regardless recommendation: proactive alerting, not a code fix

Given the refractory mechanism appears to be working correctly *right now*, there is no confirmed live bug to patch in `chain.py` or `resonance.py`. What's missing is not correctness — it's **visibility**. A `ResonanceAlertV1` today only lands in Postgres and (per the original ground-truth pass) the Hub debug panel; there is no path from "a real ouroboros loop is happening" to "Juniper finds out." That's the same gap the field-digester/attention-runtime/self-state-runtime health monitors (PRs #975, #977) closed for infra-stall detection — this is the identical gap for a cognition-layer anomaly, and the fix is the same pattern, already reviewed and running in production.

**Design**: a small edge-triggered check inside `orion-thought`, reusing the exact `HealthMonitor`/`Severity`/`_check`/`_has_open_alert`/retry-until-delivered shape from `services/orion-field-digester/app/health_monitor.py` verbatim (this is the fourth port of this pattern — the reuse is now well-precedented, not speculative).

- **Trigger**: after `_maybe_emit_resonance_alert()` (`chain.py:129-167`) persists a `ResonanceAlertV1` (i.e., `detect_resonance()` returned non-`None`), check whether `violation_count` has been monotonically increasing across the last 2 samples for that `theme_key` (a genuinely *worsening* pattern, not just "any alert exists" — since, per A.1, a single stale echo shouldn't page anyone).
- **Alert payload**: `source_service="orion-thought"`, `reason="reverie_resonance_worsening"`, `severity="error"`, context includes `theme_key`, `violation_count`, `min_gap_sec`, `occurrences`.
- **Recovery**: when a theme that previously triggered stops appearing in fresh `detect_resonance()` output (i.e., its violation streak has fully decayed out of the window), fire the existing `info`-severity recovery note.
- **This does not touch `chain.py`/`resonance.py`'s core logic at all** — it's a new, small, additive consumer of `_maybe_emit_resonance_alert`'s output, following the exact non-invasive pattern of the three prior health monitors.

### A.3 Smallest buildable version

New file `services/orion-thought/app/resonance_monitor.py`, mirroring `field-digester`'s `health_monitor.py` structure (import `NotifyClient`, port `Severity`/`_check`/edge-triggered transition tracking verbatim), fed by a small new store query (`load_recent_resonance_alerts(theme_key, limit=2)` in `services/orion-thought/app/store.py`) reading `substrate_reverie_resonance_alert` ordered by `created_at DESC`. Wired into the existing chain worker loop (`run_reverie_chain_worker`, `chain.py:287-330`) as a call after `_maybe_emit_resonance_alert()`, not a new asyncio loop (resonance detection is already gated to run once per completed chain, no separate cadence needed).

### Files likely to touch (Part A)
- `services/orion-thought/app/resonance_monitor.py` (new)
- `services/orion-thought/app/store.py` (new query)
- `services/orion-thought/app/chain.py` (one new call site inside `_maybe_emit_resonance_alert` or immediately after it)
- `services/orion-thought/app/settings.py` (new `notify_base_url`/`notify_api_token`, matching the other three services — orion-thought doesn't have these yet)
- `services/orion-thought/requirements.txt` (add `requests` if absent — check first)
- `services/orion-thought/tests/test_resonance_monitor.py` (new, mirroring `test_health_monitor.py`'s coverage)
- `services/orion-thought/README.md` (new section)

### Acceptance checks (Part A)
- A synthetic test feeding `detect_resonance()` two increasing-violation-count alerts for the same theme fires exactly one `attention_request`, not one per check.
- A synthetic test feeding a decaying violation count (or no further alerts) fires a recovery note, not silence.
- Live: after deploy, no alert should fire immediately (current state is genuinely healthy — this is itself a regression check against the A.1 finding).

---

## Part B — Reverie thought thread into chat (internal pass only)

### B.1 What's being wired

The latest non-hollow `SpontaneousThoughtV1.interpretation` (real prose, e.g. the live example already pulled this session: *"The coalition is fixated on unresolved transport anomalies and harness closures, suggesting a systemic strain in substrate navigation..."*), surfaced as a new, age-gated ctx field, reaching **`chat_stance_brief.j2` only** — the internal stance-synthesis pass, not the user-facing `chat_general.j2`. This mirrors the existing `{% if chat_attention_frame %}` conditional-block pattern already in that template (`orion/cognition/prompts/chat_stance_brief.j2:23-24`).

### B.2 Why gate on Part A first

If the resonance loop *were* still active (it isn't, per A.1, but the monitor in Part A is what will catch it if it recurs), idea 3 would risk Orion repeatedly, almost compulsively referencing the same unresolved theme turn after turn — reading as a bug, or something stranger. Since A.1 confirms the current state is healthy, this is not a hard blocker today, but Part A's monitor should land first (or alongside) so a *future* recurrence is caught automatically rather than silently degrading idea 3's output quality.

### B.3 Content-appropriateness — the one open question this spec cannot close alone

Nobody has read a sample of the 1,959 real reverie thoughts with the specific lens of "would it be fine for this to be echoed back to whoever Orion is talking to." Given reverie has zero hollow output and real anchored content, the *quality* risk is low, but the *appropriateness* risk (is reverie ever narrating something about Juniper, or something private, in a register that's fine for internal metacognition but not for conversational echo) is unverified. This spec proposes surfacing only into the **internal** stance pass first specifically so this can be inspected before it's ever user-facing — treat the internal-only gate as a real safety boundary, not a formality to skip.

### B.4 Smallest buildable version

New lane in `orion/substrate/felt_state_reader.py`'s `_LANES` tuple (`felt_state_reader.py:30-84`), following the exact `LaneSpec` pattern already used for `episode_summary`/`curiosity_signals`:

```python
LaneSpec(
    ctx_key="latest_reverie_thought",
    table="substrate_reverie_thought",
    payload_col="thought_json",  # confirm exact column name against the migration
    ts_col="created_at",
    projection_id=None,
    max_age_sec=180,  # 2x the reverie tick interval (90s), same 2x convention as curiosity_signals
),
```
(Payload column name needs confirming against `services/orion-sql-db/manual_migration_*reverie_thought*.sql` before implementation — not verified in this pass.)

In `services/orion-cortex-exec/app/chat_stance.py`, near `_project_self_state_from_beliefs`'s call site (`chat_stance.py:2315-2318`), add a small projection reading `ctx.get("latest_reverie_thought")`, extracting only `interpretation` (never `evidence_refs`/`coalition` — those are internal grounding, not narration) and only when `hollow=False` (defense in depth — the producer already filters hollow thoughts, but a chat-facing consumer should not trust that invariant blindly), into a new declared template variable:

```jinja
{% if chat_reverie_glimpse %}
- reverie_glimpse: {{ chat_reverie_glimpse }}
{% endif %}
```
in `chat_stance_brief.j2`, adjacent to the existing `attention_frame` block (`chat_stance_brief.j2:23-24`).

### Files likely to touch (Part B)
- `orion/substrate/felt_state_reader.py` (new lane)
- `services/orion-cortex-exec/app/chat_stance.py` (new small projection function + one new ctx key, same shape as `_project_self_state_from_beliefs`)
- `orion/cognition/prompts/chat_stance_brief.j2` (one new conditional block)
- A new unit test asserting: hollow thoughts never populate `chat_reverie_glimpse`; stale (>180s) thoughts don't either; `evidence_refs`/`coalition` never leak into the ctx value.

### Acceptance checks (Part B)
- Manual read of the actual rendered `chat_reverie_glimpse` value across a handful of real turns before this goes any further than the internal pass, specifically checking for anything that reads as inappropriate to echo.
- Confirm via `ENABLE_SUBSTRATE_FELT_STATE_CTX` gating (already the flag governing this whole reader) that this is off by default in any environment where it hasn't been explicitly reviewed.

---

## Part C (optional / stretch) — Shared vetted felt-context seam

**Only pursue this after B has run for a real window and proven the reverie glimpse is worth having.** Building a shared reducer before either individual wire has demonstrated value would be exactly the kind of premature abstraction the project's own conventions warn against.

### C.1 What it would be

A single new read (not a new service) that assembles one `felt_context` object per chat turn from the registry's `COMPOSED` entries (`self_state.v1`, `phi_intrinsic_reward.v1`) plus the new reverie lane from Part B, replacing three ad-hoc reads in `chat_stance.py` with one call. This gives `inner_state_registry.py` a second real consumer beyond its own gate script (`scripts/check_inner_state_registry.py`), which is itself a small piece of evidence the registry is earning its keep rather than becoming a write-only ledger.

### C.2 Explicit non-goal

This is not a new bus channel, not a new Postgres table, not a new service. It is a Python function in `orion/substrate/` or `orion/self_state/` that calls the existing `felt_state_reader.hydrate_felt_state_ctx()` and the registry's `get()` lookups, and returns a plain dict. If it ever needs its own cadence or persistence, that's a sign it should not have been built yet.

### Files likely to touch (Part C, if pursued)
- New module, exact location TBD pending B's outcome (candidates: `orion/self_state/felt_context.py` or a method on `SubstrateFeltStateReader`).
- `chat_stance.py`: replace the three separate reads with one call, once it exists.

---

## Non-goals (all parts)

- No change to `_project_self_state_from_beliefs()`'s existing hazard-string behavior — Part B is additive, a new ctx key, not a rewrite of the existing self-state hazard path (that's idea 1 from the prior brainstorm round, not in scope here).
- No change to `chat_general.j2` (user-facing) in either Part A or B — everything here targets the internal stance pass only.
- No attempt to root-cause the 07-07–07-12 historical burst beyond what's stated in A.1 — if Juniper wants that investigated further, it's a separate, smaller research task (check `substrate_reverie_refractory` write success/failure logs if they still exist, check config history for `ORION_REVERIE_REFRACTORY_SEC` changes around that window).
- No changes to `orion/reverie/resonance.py`'s detection algorithm — A.1 found it behaving correctly (mathematically defensible, if surprising in its window-echo behavior); Part A only adds a consumer of its output.
- Part C is explicitly not committed scope — a stretch idea, gated on B's results.

## Recommended sequencing

1. **Part A** (resonance monitor) — small, well-precedented (4th port of an already-reviewed pattern), and closes a real visibility gap regardless of whether the historical burst ever recurs.
2. **Part B** (reverie glimpse, internal-only) — gated on A being in place, and on a manual content-appropriateness read of real reverie thoughts before going further than the internal stance pass.
3. **Part C** — do not start until B has run for a real window and someone has looked at what it actually produces in practice.
