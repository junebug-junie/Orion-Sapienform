# Grounded Autonomy Episode Journal — Design

- Date: 2026-07-07
- Status: Approved (pending spec review)
- Owner: Juniper / Orion
- Related services: `orion-spark-concept-induction` (worker), shared `orion/autonomy/*`

## Arsonist summary

The autonomy episode journal reads like it "forgot" why it fetched articles and never mentions what it read. It did not forget — it was never given the episode. The readonly-fetch pipeline is a lossy funnel: it strips article content at the search backend, collapses the fetch to a count string, discards the originating curiosity signal, and hands the compose LLM a one-line seed (`"fetch outcome: fetched 2 article(s)"`). Everything the journal would need to name the "why" and the "what" is thrown away before compose runs. Fix the funnel and thread real evidence — plus a deterministic salience score — into the seed.

## Current architecture (the lossy funnel)

| Hop | File / symbol | What is dropped |
|---|---|---|
| Search backend | `orion/autonomy/fetch_backends.py:38-47` `firecrawl_search_backend` | Firecrawl `/v1/search` returns `title` + `description` per result; only `url` is kept |
| Fetch execution | `orion/autonomy/episode_fetch.py:53` `execute_readonly_fetch` | Result reduced to `_build_summary()` = `"fetched N article(s)"`; even URLs do not survive into the outcome |
| Outcome model | `orion/autonomy/models.py:153` `ActionOutcomeRefV1` | No article fields, no query, no relevance signal |
| Compose gate | `orion/autonomy/policy_act.py:115` `maybe_compose_autonomy_episode_after_fetch` | `del curiosity_signals` — the "why" is explicitly discarded |
| Synthetic goal | `orion/autonomy/policy_act.py:203` `goal_proposal_from_episode_intent` | `goal_statement = "Substrate episode intent (synthetic goal for policy)."` — placeholder |
| Narrative seed | `orion/autonomy/policy_act.py:156` | `narrative_seed = f"fetch outcome: {fetch_outcome.summary}"` — the entire grounding the LLM receives |

Result: the compose LLM grounds only on the count string plus whatever incidental recall it pulls (e.g. real PR-title SQL logs), so it cannot summarize the articles or connect them to the curiosity that drove the fetch.

### Relevant existing contracts

- `FrontierInvocationSignalV1` (`orion/core/schemas/frontier_curiosity.py:28`): `signal_type`, `focal_node_refs` (e.g. `"section:hardware_compute_gpu"`), `signal_strength` (0-1), `evidence_summary`, `confidence`.
- `EpisodeFetchRequest` (`orion/autonomy/episode_fetch.py:14`): `subject`, `goal_artifact_id`, `spawned_correlation_id`, `query`, `max_articles=2`.
- `dispatch_autonomy_episode_journal` (`orion/autonomy/episode_journal.py:24`): consumes `narrative_seed`, composes via cortex RPC, publishes `journal.entry.write.v1`.
- `append_action_outcome` persists `ActionOutcomeRefV1` to the local outcome store; the worker separately emits the flat `ActionOutcomeEmitV1` to SQL.

## Goals

1. The episode journal names the **why**: the curiosity gap (type, focal section, strength) and the query it drove.
2. The episode journal summarizes the **what**: the actual fetched articles (title, url, short description).
3. The episode journal assesses **satiation** in-prose: did these articles close the gap, and what is still missing.
4. A **deterministic salience score** (0-1) per article, computed from gap-term overlap, is threaded into the seed and persisted with the outcome so the journal (and future gating/ranking) can reason over relevance rather than guess.

## Non-goals

- Model-based / embedding salience scoring. The deterministic term-overlap score is the first rung; an LLM/embedding scorer is a later seam. (No-regex-swamp note: the scorer is a narrow deterministic sensor with a single scoring function, not a cognition architecture.)
- Full-text article scraping (Firecrawl `scrapeOptions` / markdown). `title` + `description` is enough for grounding; markdown is a cost/latency follow-up.
- Persisting per-article detail into the **SQL** `action_outcomes` projection. The local outcome store (JSON) carries the new fields automatically; a JSONB SQL column is a follow-up.
- Enriching the synthetic `goal_statement`. The seed is the compose lever; goal enrichment is out of scope.
- Changing the capability policy / gating thresholds.

## Design

Three thin seams, evidence-first, each independently testable.

### Seam 1 — Stop dropping evidence

**`orion/autonomy/fetch_backends.py`** — in `firecrawl_search_backend`, capture per-result `title`, `description`, and `url`. Return both the existing `urls: [str]` (back-compat) and a new `articles: [{"url","title","description"}]`. Missing fields default to empty strings; `success` semantics unchanged (`success and bool(urls)`).

**`orion/autonomy/models.py`** — add a small nested model and extend the outcome (defaults keep it backward-compatible under `extra="forbid"`):

```python
class FetchedArticleRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    url: str
    title: str = ""
    description: str = ""
    salience: float = Field(default=0.0, ge=0.0, le=1.0)

class ActionOutcomeRefV1(BaseModel):
    # existing fields unchanged ...
    query: str | None = None
    articles: list[FetchedArticleRefV1] = Field(default_factory=list)
    salience: float = Field(default=0.0, ge=0.0, le=1.0)  # aggregate = max(article saliences)
```

**`orion/autonomy/episode_fetch.py`** — `EpisodeFetchRequest` gains `gap_terms: tuple[str, ...] = ()` so the fetch path can score without reaching back for signals. `execute_readonly_fetch` parses `result["articles"]` into `FetchedArticleRefV1[]`, records `req.query`, computes per-article salience (Seam 2) using `req.gap_terms`, and sets the outcome-level `salience` aggregate (max over articles) — all **before** `append_action_outcome`, so the persisted outcome carries salience. `_build_summary` stays as the human count string.

The caller `maybe_execute_readonly_fetch_after_goal` (`orion/autonomy/policy_act.py`) already holds `curiosity_signals`; it derives `gap_terms` via `gap_terms_from_signals(...)` and sets them on the `EpisodeFetchRequest`. This keeps `execute_readonly_fetch` a pure function of its request.

### Seam 2 — Deterministic salience scorer

New pure function, e.g. `orion/autonomy/salience.py`:

```python
def gap_terms_from_signals(signals) -> set[str]: ...
def score_article_salience(article_text: str, gap_terms: set[str]) -> float: ...
```

- **Gap terms:** for each `world_coverage_gap` signal, take `focal_node_refs` starting with `"section:"`, strip the prefix, replace `_`→space, tokenize (lowercase, alphanumeric). Union across signals. Fall back to query tokens if no section refs.
- **Score:** `article_text = f"{title} {description}".lower()`; `score = |gap_terms ∩ article_tokens| / |gap_terms|` when `gap_terms` is non-empty, else `0.0`. Clamped to `[0,1]`. Deterministic and order-independent.
- **Edge cases:** empty gap terms → all saliences `0.0`, seed marks them "unscored"; empty article text → `0.0`.

`gap_terms_from_signals` runs in `maybe_execute_readonly_fetch_after_goal` (where the signals live) and the terms are passed into `EpisodeFetchRequest.gap_terms`; `score_article_salience` runs inside `execute_readonly_fetch` before the outcome is built and persisted, so salience is stored per article + as the outcome aggregate.

### Seam 3 — Grounded narrative seed

**`orion/autonomy/policy_act.py`** — remove `del curiosity_signals` in `maybe_compose_autonomy_episode_after_fetch`; build a structured, multi-line seed via a dedicated helper `build_episode_narrative_seed(goal, curiosity_signals, fetch_outcome)`:

```
Why: predictive coverage gap in "hardware compute gpu" (strength 0.71).
Query: "hardware compute gpu recent news coverage"
Fetched 2 article(s):
  1. [salience 0.67] <title> — <url>
     <description>
  2. [salience 0.33] <title> — <url>
     <description>
Reflect: summarize each article and assess whether it closes the gap that
drove this fetch. Name what is still missing. Do not invent sources.
```

The failure branch keeps the existing `"fetch failed: ..."` form. The seed is the only compose input we control, so it carries why + what + salience + the explicit satiation-assessment ask. The "Do not invent sources" line reduces confabulation.

## Schema / bus / API changes

- Added model: `FetchedArticleRefV1`.
- Changed model: `ActionOutcomeRefV1` gains `query`, `articles`, `salience` (all defaulted → backward-compatible).
- No bus channel changes. No new events. `ActionOutcomeEmitV1` (SQL flat projection) is unchanged; new fields ride only in the local outcome store and the in-process `SubstrateActResultV1.fetch_outcome`.
- If `ActionOutcomeRefV1` is in `orion/schemas/registry.py`, update the registry entry; otherwise no registry change (verify during implementation).

## Data flow (after)

```
world.pulse.run.result.v1
  -> metabolism (curiosity signals: world_coverage_gap @ section:hardware_compute_gpu)
  -> capability gate (web.fetch.readonly allowed)
  -> firecrawl_search_backend -> {urls, articles:[{url,title,description}]}   (Seam 1)
  -> score_article_salience per article using gap terms                        (Seam 2)
  -> ActionOutcomeRefV1{query, articles[+salience], salience}                  (Seam 1)
  -> build_episode_narrative_seed(why + articles + salience + ask)             (Seam 3)
  -> dispatch_autonomy_episode_journal(narrative_seed) -> cortex compose
  -> journal.entry.write.v1  (now names articles + curiosity + satiation)
```

## Error handling / edge cases

- Backend returns no `articles` (older stub / error): `articles=[]`, seed falls back to the count/URL form; journal still composes.
- No gap signals with section refs: salience `0.0`, seed marks articles "unscored"; no crash.
- Article missing title/description: empty strings, salience `0.0`.
- Journal compose still isolated by the existing try/except (`policy_act.py:250`) — a compose failure never discards the (now richer) fetch outcome.
- Oversized descriptions: truncate each to a sane cap (e.g. 300 chars) in the seed to protect the compose context budget.

## Testing

### Unit (gate lane)
- `fetch_backends`: parses `title`/`description`/`url`; preserves `urls`; `success` semantics unchanged.
- `salience`: `gap_terms_from_signals` extracts section terms; `score_article_salience` returns expected fractions; empty-gap and empty-text → `0.0`; clamped.
- `episode_fetch`: outcome carries `query`, `articles` (with salience), and aggregate `salience` = max.
- `policy_act`: `build_episode_narrative_seed` contains the gap section **and** at least one real article title **and** a salience marker (regression: not just a count). Failure branch unchanged.

### Live smoke
- Trigger a world-pulse run; read the composed journal; confirm it names the actual article(s), ties them to the curiosity gap, and gives a satiation assessment. Capture the `substrate_episode_journal_dispatched` line + the journal body.

## Acceptance checks

- [ ] Journal body names ≥1 fetched article title/url from the run.
- [ ] Journal body states the curiosity/gap that drove the fetch.
- [ ] Journal body assesses satiation (did it close the gap / what's missing).
- [ ] `ActionOutcomeRefV1` in the local outcome store carries `query`, `articles`, per-article `salience`, aggregate `salience`.
- [ ] All new unit tests pass; existing episode-journal/policy tests still pass.
- [ ] No new confabulated sources in a live sample (the seed's real articles are the only ones named).

## Rollback / disable

- The whole path is already gated by `ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED`. Disabling it reverts to no episode journal.
- Seams 1-2 are additive/defaulted; reverting Seam 3 (seed builder) restores the old count-only seed without touching the schema.

## Files likely to touch

- `orion/autonomy/fetch_backends.py` — capture title/description.
- `orion/autonomy/models.py` — `FetchedArticleRefV1`, extend `ActionOutcomeRefV1`.
- `orion/autonomy/salience.py` (new) — deterministic scorer.
- `orion/autonomy/episode_fetch.py` — carry articles/query/salience into outcome.
- `orion/autonomy/policy_act.py` — grounded seed builder; stop discarding curiosity signals.
- `orion/autonomy/tests/` — new/updated tests (`test_fetch_backends`, `test_salience`, `test_episode_fetch`, `test_policy_act`).
- `orion/schemas/registry.py` — only if `ActionOutcomeRefV1` is registered (verify).

## Follow-ups (out of scope, filed as next patches)

1. Model/embedding salience scorer (upgrade from term overlap), usable for fetch ranking and surprise.
2. Persist per-article detail + salience into the SQL `action_outcomes` projection (JSONB column).
3. Optional Firecrawl `scrapeOptions` markdown for deeper article grounding.
4. Enrich synthetic `goal_statement` with the gap so non-journal consumers see the "why".
