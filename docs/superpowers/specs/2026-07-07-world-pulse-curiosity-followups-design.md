# World-Pulse Curiosity Followups ("Orion went looking") — Design

- Date: 2026-07-07
- Status: Draft (pending spec review)
- Owner: Juniper / Orion
- Related services: `orion-world-pulse` (producer), `orion-spark-concept-induction` (reuse consumer), shared `orion/schemas/*` + `orion/autonomy/*`

## Arsonist summary

Today the world-pulse digest reports coverage gaps but does nothing about them inside the run: it publishes `DailyWorldPulseV1`, and only *later* — reactively, off the published `world.pulse.run.result.v1` — does concept-induction metabolize the gap, fetch articles via Firecrawl, and write an episode journal. That reactive loop now works end-to-end (verified 2026-07-07 via the durable run-result stream), but its findings never make it back into the digest a human reads. The digest says "hardware_compute_gpu: missing" and stops; the articles Orion actually went and found live only in a journal entry.

This design closes that loop **in the same run**: when the world-pulse run detects under-covered sections, it fetches gap-filling articles inline, attaches them to the digest as a dedicated **"Orion went looking"** block with clear provenance, and carries the same findings forward on the published run result so the reactive episode-journal loop **reuses** them instead of fetching again. One fetch, two consumers: the human-facing digest and Orion's own episode journal.

## Current architecture

### The two loops today

| Loop | Where | Trigger | Output |
|---|---|---|---|
| Digest build | `services/orion-world-pulse/app/services/pipeline.py` `run_world_pulse` (sync) | scheduled/manual run | `DailyWorldPulseV1` + `WorldPulseRunResultV1`, published to SQL/hub and XADD'd to `orion:stream:world_pulse:run:result` |
| Reactive curiosity | `orion/spark/concept_induction/bus_worker.py` `handle_envelope` → `maybe_execute_substrate_act_after_metabolism` (`orion/autonomy/policy_act.py`) | consumes `world.pulse.run.result.v1` from the durable stream | Firecrawl fetch (`execute_readonly_fetch`) → `ActionOutcomeRefV1` → episode journal (`journal.entry.write.v1`) |

### Coverage detection (already deterministic)

`_compute_coverage` (`pipeline.py:82-127`) already returns `missing_required`, `missing_recommended`, and per-section `SectionCoverageV1`. It runs at `pipeline.py:449`, **after** `build_digest` (`pipeline.py:431`) and before `_finalize_digest_aggregates` (`pipeline.py:455`). This is the natural insertion seam: the digest object exists, coverage is known, and finalize still runs.

### Reactive fetch machinery (reused as-is)

- `EpisodeFetchRequest` + `execute_readonly_fetch` (`orion/autonomy/episode_fetch.py`) — pure request → `ActionOutcomeRefV1` with `articles: [FetchedArticleRefV1]`, per-article + aggregate `salience`, `query`.
- `resolve_fetch_backend` (`orion/autonomy/fetch_backend_resolve.py`) — Firecrawl via `FIRECRAWL_API_KEY` / `~/.fcc/.env`, honors `ORION_EPISODE_FETCH_BACKEND`.
- `evaluate_capability("web.fetch.readonly", ctx)` (`orion/autonomy/capability_policy.py`) — the existing gate.
- `gap_terms_from_signals` / `score_article_salience` (`orion/autonomy/salience.py`) — deterministic term-overlap scoring.

### The gap

`build_digest` (`services/orion-world-pulse/app/services/digest.py:16`) has no notion of curiosity findings, `DailyWorldPulseV1` has no field to hold them, and the reactive loop always fetches fresh — even for a section the run could have filled once.

## Design decisions (locked in brainstorm)

1. **Timing:** same-run inline fetch (the world-pulse run fills gaps before publishing).
2. **Fetch coexistence:** single shared fetch — the inline fetch's findings ride the run result so the reactive loop reuses them (no duplicate Firecrawl call for that section).
3. **Digest integration:** a dedicated "Orion went looking" block (`curiosity_followups`), separate from RSS-sourced items, with explicit provenance.
4. **Gap scope:** any of the 9 sections that are under-covered (`status != "covered"`), not just required ones.
5. **Guardrails:** generous / no min-salience — fetch for every under-covered section, keep every returned article; salience is computed for display + reuse, never used as an inclusion filter.
6. **Gating:** reuse the existing capability policy (`web.fetch.readonly`) **plus** one new world-pulse enable flag.
7. **Dry runs:** skip the fetch on dry runs (fetching is a billable external side effect).
8. **Reuse transport:** findings ride on the published `world.pulse.run.result.v1` (via `DailyWorldPulseV1.curiosity_followups`); the concept-induction worker reads them and reuses.

## Goals

1. A world-pulse run that detects under-covered sections fetches gap-filling articles **inline** (non-dry-run only) and attaches them to the digest.
2. The digest exposes a first-class **"Orion went looking"** structure: per under-covered section, the driving gap, the query, and the fetched articles (title, url, description, salience).
3. The reactive episode-journal loop **reuses** the inline findings for the matching gap section instead of issuing a second Firecrawl call; it still fetches live when no prefetched finding exists (backward-compatible fallback).
4. The inline fetch is gated by the existing `web.fetch.readonly` capability policy plus a new `WORLD_PULSE_CURIOSITY_FETCH_ENABLED` flag, and is skipped on dry runs.

## Non-goals

- Changing `_compute_coverage`: curiosity-fetched articles do **not** count toward RSS coverage. A section stays "missing" in `section_coverage`/`section_rollups`; the followup is separate provenance. (This is deliberate — it keeps the gap signal firing so the reactive loop still metabolizes, and it keeps "what our sources covered" honest.)
- Persisting curiosity findings into the SQL article/claim/event projections. They ride in the digest JSON + run result only; a SQL projection is a follow-up.
- Full-text scraping / Firecrawl `scrapeOptions`. Title + description is enough for the block and the seed.
- Model/embedding salience. The existing deterministic term-overlap scorer is reused unchanged.
- Changing capability-policy thresholds or the drive/metabolism math.
- Rendering polish in the email/hub templates beyond wiring the block through (kept minimal; see Seam 4).

## Design

Five thin seams. Seams 1–2 are the producer (world-pulse), Seam 3 is the schema contract, Seam 4 is rendering, Seam 5 is the consumer reuse.

### Seam 1 — Schema: the "Orion went looking" contract

**`orion/schemas/world_pulse.py`** — add two models (mirroring `FetchedArticleRefV1`/`ActionOutcomeRefV1` shape, but kept inside the world_pulse schema package to avoid a `schemas → autonomy` import dependency) and one field on the digest:

```python
class CuriosityFindingV1(_WPBase):
    url: str
    title: str = ""
    description: str = ""
    salience: float = Field(default=0.0, ge=0.0, le=1.0)

class CuriosityFollowupV1(_WPBase):
    section: str                     # one of the 9 DailyWorldPulseSectionsV1 keys
    driving_gap: SectionCoverageState  # "missing" | "source_unavailable" | "no_articles"
    query: str
    articles: list[CuriosityFindingV1] = Field(default_factory=list)
    action_id: str | None = None       # ties to the ActionOutcomeRefV1 for reuse/audit
    correlation_id: str | None = None  # world-pulse run_id

# in DailyWorldPulseV1 (defaulted → backward-compatible under extra="forbid"):
    curiosity_followups: list[CuriosityFollowupV1] = Field(default_factory=list)
```

`WorldPulseRunResultV1` needs no change: it already carries `digest: DailyWorldPulseV1`, so the followups travel with it.

### Seam 2 — Producer: inline gap-fill fetch in the pipeline

**`services/orion-world-pulse/app/services/pipeline.py`** — new helper `run_curiosity_followups(...)`, called in `run_world_pulse` **after** `_compute_coverage` (line 449) and **before** `_finalize_digest_aggregates` (line 455). Gate order (all must pass, else return `[]`):

1. `settings.world_pulse_curiosity_fetch_enabled` is true.
2. `not dry` (the run's effective `dry`, computed at `pipeline.py:140`).
3. There is ≥1 under-covered section (`status != "covered"` across the 9 sections).
4. Capability gate: build a `CapabilityEvaluationContext` from the coverage gaps and call `evaluate_capability("web.fetch.readonly", ctx)`; require `outcome == "allowed" and auto_execute`.

For the capability context, world-pulse has no drive loop, so it synthesizes a deterministic, gap-derived context (this is the "reuse the existing policy" seam — the policy still decides, world-pulse just supplies honest inputs):

- one `FrontierInvocationSignalV1(signal_type="world_coverage_gap", focal_node_refs=["section:<name>"], signal_strength=1.0, ...)` per under-covered section;
- a synthetic `GoalProposalV1` with `drive_origin="predictive"`, `proposal_status="planned"` (readonly is not `external`/`write`, so no promote gate), reusing `goal_proposal_from_episode_intent`-style construction;
- `predictive_pressure=1.0`, `curiosity_strength=1.0` (generous — decision #5).

Effective on/off therefore = `WORLD_PULSE_CURIOSITY_FETCH_ENABLED` **and** `ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED` (the master readonly switch the policy already checks).

Then, per under-covered section, build and run one fetch reusing the existing machinery:

```python
req = EpisodeFetchRequest(
    subject="orion",
    goal_artifact_id=f"world-pulse-gap-{run_id}-{section}",
    spawned_correlation_id=run_id,
    query=build_readonly_fetch_query(section_signals),      # "<label> recent news coverage"
    max_articles=settings.world_pulse_curiosity_max_articles_per_section,  # generous default
    gap_terms=tuple(sorted(gap_terms_from_signals(section_signals, fallback_query=query))),
)
outcome = asyncio.run(execute_readonly_fetch(req, fetch_backend=resolve_fetch_backend()))
```

`asyncio.run` inside the sync `run_world_pulse` is safe and consistent with the existing pattern (`emit_runtime.py:108`, `publish_hub.py:57`), because the `/api/world-pulse/run` route calls `run_world_pulse` synchronously (`app/routers/runs.py:52`) → FastAPI runs it in a worker thread with no running loop. **Verify during implementation** that the route stays a sync `def` (if it becomes `async`, replace with a thread-offloaded bridge).

Map each `ActionOutcomeRefV1` → `CuriosityFollowupV1` (articles → `CuriosityFindingV1`, `action_id`, `correlation_id=run_id`, `driving_gap=section_coverage[section].status`). No salience floor — keep every article (decision #5). Set `digest.curiosity_followups` before finalize.

Guardrails / cost bounds: a per-section article cap (`world_pulse_curiosity_max_articles_per_section`) and an overall section cap are the only bounds; each fetch is wrapped so a Firecrawl error for one section degrades to an empty followup (or is skipped) and **never fails the run** (mirror the `emit_runtime`/`publish_hub` try/except pattern). Emit a `world_pulse_curiosity_followups` metric (sections fetched, articles found).

### Seam 3 — `build_digest` wiring

**`services/orion-world-pulse/app/services/digest.py`** — `build_digest` gains an optional `curiosity_followups: list[CuriosityFollowupV1] | None = None` param, defaulting to `None` → `[]`, set on the returned `DailyWorldPulseV1`. Because coverage/fetch happen *after* `build_digest` in the current pipeline order, the pipeline sets `digest.curiosity_followups` post-build (either by passing into a second construction or by direct assignment before finalize). Keep it a plain attribute set — no restructuring of the build order.

### Seam 4 — Rendering ("Orion went looking" block)

**Hub / email renderers** (`services/orion-world-pulse/app/services/renderers.py` + hub message assembly). Add a compact block after the sections when `curiosity_followups` is non-empty:

```
Orion went looking (gaps our sources missed)
  hardware compute gpu — "hardware compute gpu recent news coverage"
    • [0.67] NVIDIA … — https://…
    • [0.00] RTX 50 … — https://…
```

- `HubWorldPulseMessageV1.structured_payload` carries the raw `curiosity_followups` for programmatic consumers; `rendered_markdown` gets the human block.
- Provenance is explicit: the block is labeled as Orion's own gap-driven fetch, distinct from tracked RSS items. Salience shown per article; a `0.00` reads as low-overlap, not "unscored" (mirrors the episode-seed convention).
- Frontend/template checklist (per repo rule): rendered template updated, block only renders when non-empty, and a renderer test covers the populated + empty cases.

### Seam 5 — Consumer reuse (single shared fetch)

**`orion/autonomy/policy_act.py`** — `maybe_execute_substrate_act_after_metabolism` gains `prefetched_outcome: ActionOutcomeRefV1 | None = None`. When provided, it **skips** `maybe_execute_readonly_fetch_after_goal` (no live Firecrawl call), records the prefetched outcome into `SubstrateActResultV1` (`fetch_attempted=True`, `fetch_outcome=prefetched_outcome`), and proceeds to the existing journal compose. When `None`, behavior is byte-identical to today (live fetch fallback).

**`orion/spark/concept_induction/bus_worker.py`** — in the `world.pulse.run.result.v1` branch (the stream `_handle_wp_stream_envelope` → `handle_envelope` path), after metabolism decides to act on a gap section, look up `digest.curiosity_followups` for that section. If found, reconstruct an `ActionOutcomeRefV1` from the `CuriosityFollowupV1` (`CuriosityFindingV1[] → FetchedArticleRefV1[]`, `query`, aggregate `salience = max(...)`, `action_id`) and pass it as `prefetched_outcome`. The narrative seed (`build_episode_narrative_seed`) then grounds on the shared articles; the journal reads identically to a live-fetch episode.

Matching rule: reuse only when the followup's `section` equals the gap section the metabolism selected. No match → live fetch (today's path). This bounds reuse to exactly the shared gap and keeps the loop working when world-pulse fetch is disabled/dry.

## Schema / bus / API changes

- Added models: `CuriosityFindingV1`, `CuriosityFollowupV1` (`orion/schemas/world_pulse.py`).
- Changed model: `DailyWorldPulseV1` gains `curiosity_followups` (defaulted → backward-compatible).
- `WorldPulseRunResultV1`: unchanged (carries the digest).
- Bus channels: none added/removed. `world.pulse.run.result.v1` payload grows a defaulted field — additive, non-breaking. If `DailyWorldPulseV1`/`WorldPulseRunResultV1` are in `orion/schemas/registry.py`, re-validate the registry entry; run `python scripts/check_schema_registry.py` and `python scripts/check_bus_channels.py`.
- API: `/api/world-pulse/run` response (`WorldPulseRunResultV1`) gains the field transparently.

## Config / env changes

New keys on **`services/orion-world-pulse`** (`settings.py` + `.env_example`, then `python scripts/sync_local_env_from_example.py`):

| Key | Default | Meaning |
|---|---|---|
| `WORLD_PULSE_CURIOSITY_FETCH_ENABLED` | `false` | Master enable for the inline gap-fill fetch. |
| `WORLD_PULSE_CURIOSITY_MAX_ARTICLES_PER_SECTION` | e.g. `5` | Per-section article cap (cost bound; generous, no salience floor). |
| `WORLD_PULSE_CURIOSITY_MAX_SECTIONS` | e.g. `9` | Optional cap on sections fetched per run. |

Reused existing keys (already present in the runtime, no change): `ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED`, `FIRECRAWL_API_KEY` / `ORION_FCC_ENV_PATH`, `ORION_EPISODE_FETCH_BACKEND`, `WORLD_PULSE_DRY_RUN`.

concept-induction (`orion-spark-concept-induction`): no new env — reuse is automatic when `curiosity_followups` are present on the run result.

Bus URL stays `redis://100.92.216.81:6379/0` (repo mandate).

## Data flow (after)

```
run_world_pulse (non-dry, flag on)
  build_digest -> _compute_coverage -> under-covered sections
    -> [gate] WORLD_PULSE_CURIOSITY_FETCH_ENABLED + evaluate_capability(web.fetch.readonly)
    -> per section: execute_readonly_fetch (Firecrawl) -> ActionOutcomeRefV1{articles,salience,query}
    -> CuriosityFollowupV1[]  -> digest.curiosity_followups           (Seams 1,2,3)
  _finalize_digest_aggregates -> publish SQL/hub + XADD run:result     (Seam 4 renders block)
        |
        v  world.pulse.run.result.v1 (now carries curiosity_followups)
  concept-induction stream consumer -> metabolize gap (section X)
    -> match digest.curiosity_followups[section==X]?
         yes -> prefetched_outcome (no 2nd Firecrawl call)            (Seam 5)
         no  -> live execute_readonly_fetch (today's fallback)
    -> build_episode_narrative_seed -> journal.entry.write.v1
```

## Error handling / edge cases

- Fetch disabled / dry run / no under-covered sections → `curiosity_followups=[]`; digest and reactive loop behave exactly as today.
- Firecrawl error / timeout for a section → that section's followup is empty or skipped; the run never fails (wrapped like `emit_runtime`/`publish_hub`).
- Capability policy denies (`ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED` off) → no fetch, empty followups; logged with the decision reason.
- Reuse mismatch (no followup for the metabolized section) → live fetch fallback; no crash.
- Backend returns no articles → followup present with `articles=[]`, `driving_gap` still recorded (honest "looked, found nothing").
- Oversized descriptions → truncated in the rendered block (reuse the seed's ~300-char cap convention); raw kept in `structured_payload`.
- Async-in-sync: if the run route becomes `async`, `asyncio.run` breaks — covered by the implementation-time verify + a thread-offload fallback.

## Testing

### Unit (gate lane)
- `world_pulse` schema: `DailyWorldPulseV1` round-trips with/without `curiosity_followups`; `CuriosityFollowupV1` validates section + articles.
- `pipeline` (`run_curiosity_followups`): with a stubbed fetch backend + a registry that leaves a section under-covered → followup produced for that section; **dry-run → no fetch, empty followups**; flag-off → no fetch; capability-denied → no fetch. (Extend `tests/test_world_pulse_pipeline.py`.)
- `digest.build_digest`: passes `curiosity_followups` through unchanged.
- `renderers`: populated block renders section + query + ≥1 article w/ salience; empty followups render nothing.
- `policy_act`: `maybe_execute_substrate_act_after_metabolism` with `prefetched_outcome` set → no fetch-backend call, journal composed from prefetched articles; with `None` → today's live-fetch path (regression).
- `bus_worker`: run result with a matching `curiosity_followups[section]` → reuse (no Firecrawl); no match → live fetch.

### Live smoke
- Non-dry run with `WORLD_PULSE_CURIOSITY_FETCH_ENABLED=true` and a real gap (e.g. `hardware_compute_gpu`): confirm (a) digest JSON has `curiosity_followups` with real articles, (b) the hub/email block renders, (c) the reactive episode journal for that run reuses the same articles — the `action_id` in the journal's fetch outcome matches the followup's `action_id`, and there is **one** Firecrawl call for that section, not two.

## Acceptance checks

- [ ] Non-dry run with a gap produces `digest.curiosity_followups` with ≥1 real article for ≥1 under-covered section.
- [ ] Dry run and flag-off runs produce **no** fetch and empty followups.
- [ ] The "Orion went looking" block renders in hub/email only when followups exist, with per-article salience and explicit provenance.
- [ ] For a shared gap section, the reactive episode journal reuses the inline findings (matching `action_id`), with exactly one Firecrawl call for that section.
- [ ] With followups absent (disabled/dry), the reactive loop still fetches live and journals (no regression).
- [ ] Inline fetch is gated by `WORLD_PULSE_CURIOSITY_FETCH_ENABLED` + `web.fetch.readonly` capability; denial is logged with reason.
- [ ] Schema/registry/channel checks pass; new unit tests + existing world-pulse/policy/bus_worker tests pass.
- [ ] `.env_example` updated and local `.env` synced.

## Rollback / disable

- Off by default: `WORLD_PULSE_CURIOSITY_FETCH_ENABLED=false` (or `ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED=false`) fully disables the inline fetch; the digest reverts to today's shape and the reactive loop keeps working via live fetch.
- Seams 1/3/5 are additive/defaulted; the `curiosity_followups` field is harmless when empty. Reverting Seam 5's `prefetched_outcome` restores pure live-fetch.

## Files likely to touch

- `orion/schemas/world_pulse.py` — `CuriosityFindingV1`, `CuriosityFollowupV1`, `DailyWorldPulseV1.curiosity_followups`.
- `services/orion-world-pulse/app/services/pipeline.py` — `run_curiosity_followups`, wire after `_compute_coverage`.
- `services/orion-world-pulse/app/services/digest.py` — `build_digest` optional param.
- `services/orion-world-pulse/app/services/renderers.py` (+ hub message assembly) — the block.
- `services/orion-world-pulse/app/settings.py` + `.env_example` — new flags; then `scripts/sync_local_env_from_example.py`.
- `orion/autonomy/policy_act.py` — `prefetched_outcome` on `maybe_execute_substrate_act_after_metabolism`.
- `orion/spark/concept_induction/bus_worker.py` — read `curiosity_followups`, reconstruct outcome, pass prefetched.
- `orion/schemas/registry.py` — only if the world-pulse digest/run-result models are registered (verify).
- Tests: `services/orion-world-pulse/tests/` (pipeline, digest, renderers), `orion/autonomy/tests/test_policy_act.py`, concept-induction bus_worker tests.

## Follow-ups (out of scope)

1. Persist curiosity findings into a SQL projection for historical querying.
2. Count curiosity-fetched articles toward a *separate* "Orion-sourced" coverage metric (without polluting RSS coverage).
3. Reuse across *multiple* gap sections in one reactive pass (today's metabolism acts on one goal/section per run).
4. Optional Firecrawl markdown scrape for deeper block/journal grounding.
5. Model/embedding salience upgrade (shared with the episode-journal follow-up list).
