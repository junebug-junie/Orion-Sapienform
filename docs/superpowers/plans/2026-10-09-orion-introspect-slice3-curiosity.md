# orion-introspect slice 3: `curiosity` tool — plan

Spec: `docs/superpowers/specs/2026-09-28-orion-introspect-mcp-design.md`
(section "Curiosity decisions (chat, 2026-10-09)"). Mirror slice 2 (dreams,
PR #2537; commits `2fa028d12`, `8cc643ad5`, `5ea8e86f5`, `236412258`,
`e02d88f75`, `3390c2953`, `d6e03c6ec`) unless this plan says otherwise.

## Facts this plan rests on (checked live 2026-10-09)

- A run's write-up is `journal_entries.source_ref = 'curiosity:<run_id>'`
  (titles `Curiosity` 225, `Self-inquiry` 144; body avg 6.2k chars, max 38k;
  most open with a preamble then `## Answer`).
- `GET /curiosity/api/run/<id>` (the `read_run_payload` path) answers in
  ~0.09 s; `read_runs_payload(days=90)` in ~2 s.
- `curiosity_run_outcomes.run_id` uses the same ids (91 rows since 09-26).
- `curiosity_self_questions`: 20 rows, `status='open'`.

## Step 0 — fix the Curiosity tab leak (own commit, regression test first)

`read_runs_payload` reads every `durable_admission_runs` row in the window;
`run_story._group` only drops `self_study.reflect`. Live 90-day view: 915
"runs", ~350 of them `reading.turn`, `reverie.visual`, `memory.episode_distill`,
`compactor.digest`, `orion_day.letter`, `journal.compose`, all shown as
"World question". Fix in `_group`: drop any admission row whose
`request.workflow` is present and not in `CURIOSITY_WORKFLOWS` (and its
resource events), generalizing the reflect exclusion. Test with fixture rows
for a `reading.turn` and a `reverie.visual` run.
→ verify: live `/curiosity/api/runs?days=90` totals after deploy contain no
non-curiosity run ids (`UNVERIFIED` until then).

## Step 1 — contract

- `orion/schemas/introspect.py`: `IntrospectBusOperation` / `IntrospectOperation`
  gain `"curiosity"`. `CuriosityArguments` (`extra="forbid"`):
  `query` (normalize_query), `run_id` (pattern = `orion.curiosity.atlas`
  run-id regex; import or duplicate with a test pinning equality),
  `kind: Literal["run","self_question"] = "run"`,
  `line: Literal["investigate","self_inquiry","self_sense_eval"] | None`,
  `limit`, `since` (tz-aware). `run_id` excludes query/since/line/kind;
  `kind="self_question"` excludes query/run_id/line.
  `clip_json_text(text, budget)` helper: longest prefix whose
  `json.dumps(..., ensure_ascii=False)` length is within `budget`;
  `CURIOSITY_FULL_JSON_BUDGET = 9000`.
- `orion/introspect/transport.py`: `CURIOSITY_REQUEST_CHANNEL`.
- `orion/bus/channels.yaml`: `orion:introspect:curiosity:request`
  (producer governor, consumer orion-hub); add orion-hub to producers of
  `orion:introspect:result:*` and `orion:vector:semantic:upsert`.
- `orion/schemas/registry.py`; skew declaration for `CuriosityArguments`
  (see `d6e03c6ec`); metric lock re-lock (new channels = real definition change).
- Tests: arg bounds/exclusions, worst-case full-run item (quotes + newlines +
  accents) serialized under `MCP_TOOL_RESULT_MAX_CHARS`.

## Step 2 — Hub responder

`services/orion-hub/scripts/curiosity_introspect.py` (pure: story payload →
`IntrospectItemV1`, self-question rows → items, write-up extraction) and
`services/orion-hub/scripts/curiosity_introspect_listener.py` (bus, search,
index loop), started in `main.py` next to `ReadingListener`, same
`memory_pg_pool` provider and the same `WorldviewReader` the curiosity routes use.

Modes:
- `run_id` → `read_run_payload`; not found → `ok, items=[], total=0`;
  `available=False` → unknown. Text = full `journal_body` via
  `clip_json_text(..., CURIOSITY_FULL_JSON_BUDGET)`.
- recent → `read_runs_payload(days = 90, or from since, clamped)` filtered by
  `line`; drop `running`; newest `limit`; `total_available` = runs in window.
- `query` → rank journals in `orion_curiosity` → re-read each hit through
  `read_run_payload` (≤ limit hits, ranked order, `similarity` in extra),
  `line`/`since` applied after re-read. Empty + index behind → unknown
  (dreams `index_complete_as_of` rule).
- `kind=self_question` → open `curiosity_self_questions`, newest first,
  `record`, extra `{family, ask_count, last_asked_at, pinned}`.

Run item: `id=run_id`, `kind="curiosity_run"`, `occurred_at` = started or
finished (skip runs with neither, still counted). With a write-up:
`unsettled`, text = `## Answer` section else opening, 900 chars. Without:
`record`, text = one plain sentence (line label, status, error/outcome_kind).
`extra`: `line`, `status`, `error`, `hops`, `findings`, `revisions`,
`prior_touched` (claim clipped to SHORT_FIELD_CAP, from/to), `outcome_kind`,
`reach_out` decision, and from `curiosity_run_outcomes` when a row exists
`{turn_ok, n_tested, n_moved, n_formed, unknown_reason}`.

All Postgres reads read-only; failures → `ok=False` with a fixed error string
(no SQL/DSN); log `introspect op=curiosity corr= mode= items= total=`.

Search index: docs = curiosity journal entries, doc id = run_id, text =
`## Answer` section (or opening) clipped 1800, meta `occurred_ts`.
Settings `HUB_CURIOSITY_SEARCH_{CHROMA_URL,EMBED_URL,COLLECTION,MIN_SIMILARITY,
INDEX_INTERVAL_SEC,INDEX_BATCH}` in settings.py + `.env_example` + README,
then `python scripts/sync_local_env_from_example.py` (writes the primary
checkout's `.env`; confirm the keys landed there). Similarity floor from a
calibration eval like `services/orion-dream/evals/run_dream_search_calibration.py`
run on live journals; record the number in the PR.

## Step 3 — tool, brief, smoke, docs

- `orion/introspect/tools.py`: `curiosity` ToolSpec + invoke path.
- `orion/introspect/brief.py`: one line (call before describing past runs;
  write-ups are what you concluded then; failed runs are listed; unknown ≠ none).
- `scripts/smoke_introspect.py --tool curiosity`.
- Governor + Hub READMEs (introspect overview coverage test), CI workflow
  includes the new Hub tests.

## Acceptance

- Gate tests for steps 0–3 pass; `check_schema_registry`, `check_bus_channels`,
  metric lock, env parity pass.
- Eval: calibration run on live journals (precision at the chosen floor).
- Live: `scripts/smoke_introspect.py --tool curiosity` after Hub deploy from
  main. `UNVERIFIED` until run.
