# Live body map + self-improvement reading (design, proposal mode)

Date: 2026-10-14. Status: design only, nothing implemented. Proposal mode applies (CLAUDE.md 0A):
Part 1 changes Orion's self-model, Part 2 adds a curiosity line.

Juniper's asks, in her words:

- Part 1: "Orion needs to be able to understand their physical topology dynamically ... I don't want
  things just written in a file as this topology changes."
- Part 2: follow threads about Orion's own hardware/software, reread as comments arrive (max 5
  rereads, 3 days apart, new comments only), and say "worth trying" or "nothing worth trying".
  "I don't care that someone set up 2x V100s on a Chinese board -- I care about software setups
  and optimizations."

Evidence below was gathered read-only on 2026-10-14 between 00:39 and 00:45 UTC. Anything not
checked live is marked UNVERIFIED.

---

## Arsonist summary

Most of Part 1 already exists. It is just split across three services that never get joined, and
it is missing the one part Juniper actually named: the launch flags.

- **The GPU pool already finds out which model is actually running.** `orion/gpu_pool/discovery.py`
  joins three things: the worker's announcement, the model file the profile names, and what
  llama.cpp's own `GET /props` says is loaded. A role counts as `confirmed` only when all three
  agree. Live, it found chat, agent, metacog and fast `confirmed`, `agent-gpu2` `unloaded`, and
  `agent-deep` (hecate) `down`.
- **Biometrics already reports per-node GPU temperature, use, memory and power**, and it is fresh
  (`/snapshot` `freshness_s` 0.17).
- **Nothing reports how each model server was started.** That means flash-attn, `--n-gpu-layers`,
  tensor split, batch size, threads and reasoning flags. llama.cpp's HTTP API does not expose
  them: `/props` gives `model_path`, `total_slots`, `n_ctx`, `build_info`, `modalities` and
  `model_ftype`, and `/metrics` returns 501 unless the server is started with `--metrics`. The
  real command line is visible only inside the container, and it was read live over SSH.
- **Four static files describe the body, and all four are wrong somewhere today:**
  1. `config/biometrics/node_catalog.yaml` gives circe `role: burst_gpu` (it is the main brain)
     and gives hecate all capabilities `false` (it serves `agent-deep`).
  2. `config/field/orion_field_topology.v1.yaml`, which `self_study.py` reads as Orion's
     "physical_topology" facts, has no hecate at all.
  3. The pool YAML does not cover athena's two GPUs (Tesla P4 + T10), which run all of vision.
  4. The memory note `reference_circe_gpu_inventory_and_lane_map` says circe has seven GPUs.
     Live `nvidia-smi` shows four.

  That is exactly the failure Juniper described.
- **Part 2's hard part is getting into Reddit, not the reading.** Reddit closed self-service Data
  API access in November 2025. New apps need an approval request, and Reddit points
  non-commercial builders at Devvit, which is why Juniper landed on Devvit templates. The policy
  details are UNVERIFIED (secondary sources only). Everything after the fetch already exists:
  - the document snapshot path (`orion/world_pulse_read/documents.py`), which turns fetched
    text into a content-addressed read;
  - the Stage 1/2 durable reading turns;
  - the `ReadingFindingV1` seam;
  - `reading_finding_outreach.py`, which already skips the daily cap and still respects quiet
    hours;
  - durable-runs timed interrupts (`orion_day_graph.py` `retry_wait`).

The plan:

1. **One reducer**, `orion-body-map`, joins pool state, biometrics and a new "how I was launched"
   report into one projection. Every field records where it came from and how old it is.
2. **Drift is a first-class output.** It compares what the config files say with what is
   actually running.
3. **Self-improvement reading** is a curiosity line, not a new loop. It feeds the reading
   pipeline and adds a single judging step, run against the body map.

## Current architecture

### Part 1: who knows what about the body today

| Source | What it knows | Live? | Gap |
| --- | --- | --- | --- |
| `orion-gpu-pool` (`GET :8127/v1/pool`, bus `orion:gpu_pool:state`, persisted by sql-writer `GpuPoolStateSQL`, 30-day history) | cards (vram, host, nvidia index, lent), roles -> `DiscoveredRoleV1` (status, profile, model_file, model_path, slots, ctx/slot, vision, checked_at), leases (31 live), queue/backlog | yes, every few seconds | Only knows circe + hecate cards in `config/gpu_pool.yaml`; athena's P4/T10 invisible. No launch flags. No temp/util. |
| `orion/gpu_pool/discovery.py` | announce + profile file + `/props` agreement; vLLM via `/orion/server-info` middleware (`services/orion-vllm-host/app/discovery.py`: model, max_model_len, max_num_seqs, TP/PP/DP size) | yes | llama.cpp side reads only model file, slots, ctx, vision. |
| `LlmWorkerAnnounceV1` (`orion:llm:worker:announce`, every 30 s from `services/orion-llamacpp-host/app/main.py:_announce_worker`) | host, role, profile_name, port, cuda_visible_devices, service_name, started_at | yes | Does not carry the argv the launcher built, though the launcher builds it (`main.py` ~L385-L573). |
| llama.cpp HTTP | `/props`: model_path, total_slots, n_ctx, build_info (`b10398-8e7f22b67` on chat), modalities, model_ftype (`Q5_K - Medium`); `/v1/models` meta: n_params, size, n_ctx_train; `/slots`: per-slot n_ctx, is_processing, sampling params | yes | No flash-attn, ngl, tensor split, batch, threads. `/metrics` disabled (501). |
| Container argv (circe, `ps` inside container) | the full command line, e.g. chat: `--ctx-size 65536 --n-gpu-layers 99 --threads 18 --parallel 1 --batch-size 1024 --flash-attn on --ubatch-size 1024 --mmproj ...` | yes, verified via SSH | Nothing in Orion reads it. |
| `orion-biometrics` (`GET :8100/snapshot`, `orion:biometrics:*`) | per-node chassis watts, per-GPU temp/util/mem, freshness; `nodes` has athena + circe | yes | hecate appears only as PDU watts (96 W) in `measurements_by_node`; no node summary row (biometrics there not reporting at check time, UNVERIFIED why). |
| `config/biometrics/node_catalog.yaml` | expected_online, role, capabilities | static | Stale roles/capabilities (above). `expected_online` is a real operator *intent* and must stay an input (hecate on-demand). |
| `config/field/orion_field_topology.v1.yaml` -> `services/orion-cortex-exec/app/self_study.py` `physical_topology` facts | nodes athena/circe/prometheus, edges | static, last edited 2026-10-07 | No hecate. This is what Orion currently learns as "my body". |
| LLM gateway `GET :8210/routes` | route -> served_by, backend, status, model | yes, *generated from pool state* (`orion.gpu_pool.route_view`), retiring (counted by `routes_compat_reads`) | Do not build on it; read pool state directly. |
| `orion-durable-runs` `/health` | active_runs, admitted runs, regulation/temporal-self | yes | Not joined to which card/role a run holds (that is in pool leases). |
| orion-introspect (`orion/introspect/tools.py`) | reading_results, dreams, curiosity, orion_day, record_correction, node_history | yes | No body tool. |
| Anatomy inspection (PR #2151) | organs/channels/verbs from `orion_bus_synapse`; proposed `build_orion_anatomy_snapshot()` | docs only | Software anatomy, not hardware. Complementary; not built. |

**Live snapshot, 2026-10-14 ~00:40 UTC (the body map's first test fixture):**

- **circe** has four GPUs (`nvidia-smi`):
  - 0: V100-PCIE-32GB, 27.9 GB used, 48 C
  - 1: V100-PCIE-32GB, 24.9 GB, 47 C
  - 2: PG500-216, 1.2 GB, 52 C
  - 3: V100-SXM2-32GB, 15.4 GB, 78 C, 91% util

  It runs these containers: `atlas-llamacpp-{chat,agent,metacog,fast}`, `world-model`,
  `diffusion-host` and `gpu-lane-controller`.
- **What runs on circe:**
  - chat: Qwen3.6-35B-A3B Q5_K_M on gpu0 (`CUDA_VISIBLE_DEVICES_OVERRIDE=0`), 1 slot x 65536.
  - agent: Qwen3.8-27B Q4_K_XL on gpu1, 1 x 131072, `--reasoning on`.
  - metacog/fast: Qwen3-8B Q5/Q4 on gpu3, 4 x 4096 each.
- **Drift found by hand:** the chat profile is named `...-2xv100-32gb-...`, but it runs on one
  card with no `--tensor-split`.
- **athena** has a Tesla P4 (63 C, 97% util) and a T10, which run the `vision-*` containers and
  `whisper-tts`. None of this is in the pool.
- **hecate** does not answer ping. PDU power reads 96 W, so the box is powered but `agent-deep` is
  `down`. This matches "on-demand" (memory `project_hecate_on_demand_not_24_7`). The body map
  must show it as `expected_offline`, not as an outage.

### Part 2: the reading pipeline today

- Queue/seeds: `world_pulse_read_seed` (`orion/world_pulse_read/queue.py`, `seeds.py`). The same
  URL is deduped as `already_read`.
- Documents: Hub stores the exact text content-addressed in `reading_document_snapshot` and
  rewrites the source to `file:///abs/path?sha256=<hex>`. The same bytes dedupe; changed bytes
  make a new read. **This is the seam for rereads**: each pass's "new comments only" text is new
  bytes, so it becomes a new read, with nothing to bypass.
- Turns: Stage 1/2 durable turns (`orion/world_pulse_read/durable.py`, `reading_durable_turn`),
  with bounded retry in `retry.py`.
- Findings: `ReadingFindingV1` / `ReadingFindingBundleV1` (`orion/schemas/reading_finding.py`,
  PR #2645). There are two kinds, `belief_moved` and `link_made`, and `resonance` is expected.
  Bundles carry `article_summary` for Juniper.
- Outreach: `services/orion-hub/scripts/reading_finding_outreach.py` goes through the real
  `EndogenousOutreach.compose` + `offer_message` with `skip_rate_gates=True`. Quiet hours, arousal
  `strained`, turn-in-flight and recent-chat gates still apply, and a blocked finding is retried
  later rather than lost.
- Curiosity lines: `orion/curiosity/self_inquiry.py` is the precedent, "a second LINE inside the
  existing curiosity loop, not a second loop", with its own budget.
- Timed waits on durable runs: `orion_day_graph.py` `retry_wait` interrupts until `retry_at`. A
  3-day wait has never been exercised (UNVERIFIED that `admission_runtime` resumes timer
  interrupts days later without a pool event; Phase 4 acceptance check).
- Reddit: unauthenticated fetch is blocked ("Your request has been blocked due to a network
  policy", 2026-10-14).

## Missing questions

Juniper answered these on 2026-10-14 (relayed):

- Follow the **whole V100 subreddit**, because "it isn't a big community or high velocity". The
  link was truncated, so the name is recorded as `r/V100`, UNVERIFIED until she confirms it.
- She is registering a Reddit app.
- There are two kinds of source: self-improvement and interests (Part 2 config below).

Still open:

1. **Exact subreddit name.** Is it `r/V100`, `r/Volta`, or another name?
2. **Reddit approval.** Since November 2025, a script app at reddit.com/prefs/apps may not be
   enough on its own. Reddit asks non-Devvit personal tools to file an access request. Will she
   file it, and does she want the request text drafted? (Policy details UNVERIFIED.)
3. **Proposed sources to trim:**
   - self-improvement: r/LocalLLaMA, r/LocalLLM, r/homelab;
   - interests: r/ArtificialSentience, r/consciousness, r/cogsci, r/compmathneuro,
     r/neuroscience, r/MachineLearning, r/philosophyofmind;
   - non-Reddit: GitHub issues/discussions for `ggml-org/llama.cpp` and `vllm-project/vllm`,
     and the Level1Techs / ServeTheHome forums.
4. **Experiment proposals.** Should "worth trying" proposals go only to outreach, or also to a Hub
   panel she can mark tried / not tried? The mark is what lets Orion learn whether its judgement
   was right.
5. **Athena's GPUs in the pool.** They are not pool cards today. The body map shows them from
   biometrics plus probes either way. Making them pool-managed is a separate decision; this
   design does not do it.

## Proposed schema / API changes

### Part 1: body map

**A. Launch report (producer-side, on the launcher).** Add one optional field to
`LlmWorkerAnnounceV1`:

```python
launch: LlmLaunchFactsV1 | None = None   # what this launcher actually exec'd

class LlmLaunchFactsV1(BaseModel):          # extra="forbid"
    argv: list[str]                         # exact cmd list passed to subprocess, secrets redacted
    binary: str                             # e.g. /app/llama-server
    build_info: str | None                  # from `llama-server --version` the launcher already runs
    skipped_flags: list[str] = []           # profile asked, launcher dropped (e.g. the --flash-attn
                                            # value skipped for b5332-compatible builds, main.py ~L451)
    argv_sha256: str
```

The launcher is the only process that knows what it actually executed. A live server cannot be
asked, and reading the profile would mean trusting config again. `skipped_flags` makes the
launcher's own silent adjustments visible.

The schema is `extra="forbid"`, so readers must be updated before writers. That means the pool
and the body map must accept the field before any launcher sends it (memory
`feedback_additive_schema_fields_are_a_consumer_first_migration_on_forbid_models`).

The argv is parsed into typed fields by a pure function, `parse_llamacpp_argv()`:

- `ctx`, `n_gpu_layers`, `flash_attn`, `tensor_split`
- `batch` / `ubatch`, `threads`, `parallel`
- `cache_type_k` / `cache_type_v`, `mmproj`
- `reasoning`, `spec_draft`

Unknown flags are kept verbatim under `other`. They are never guessed.

For vLLM, extend the existing `/orion/server-info` middleware with the remaining engine fields:
`dtype`, `quantization`, `gpu_memory_utilization`, `enforce_eager`, `kv_cache_dtype`. These are
read from the initialized `vllm_config`, not from env.

**B. Body map projection** (`orion/schemas/body_map.py`, registered):

```python
class ObservedV1(BaseModel, Generic[T]):    # every leaf
    value: T | None                          # None = unknown, never 0
    source: str                              # "pool.discovery" | "props" | "announce.launch" |
                                             # "biometrics.snapshot" | "vllm.server_info" | "config:<path>"
    observed_at: datetime | None
    stale: bool                              # observed_at older than the source's own ttl
    state: Literal["observed", "unknown", "stale", "expected_offline"]

class BodyNodeV1:     node, expected_online (config input), reachable: ObservedV1[bool],
                      watts: ObservedV1[float], gpus: list[BodyGpuV1], servers: list[BodyServerV1]
class BodyGpuV1:      host, index, name, vram_total_gb, vram_used_gb, temp_c, util_pct,
                      pool_card: str | None, lent: ObservedV1[bool], holds: list[lease summary]
class BodyServerV1:   host, port, engine ("llamacpp"|"vllm"|"service"), role, pool_status,
                      model_path, quant (model_ftype), build_info, slots, ctx_per_slot,
                      launch: parsed flags (each ObservedV1, source "announce.launch"),
                      busy_slots (from /slots is_processing), cards
class BodyRouteV1:    route, class, priority, landing_role (pool route_view), served_by_server
class BodyDriftV1:    drift_id (deterministic: "<kind>:<subject>"), kind, subject,
                      config_says, live_says, config_source, live_source, first_seen, last_seen
class BodyMapV1:      schema_version "body_map.v1", generated_at, nodes, routes,
                      durable_load (active/admitted runs from durable-runs /health), drift, sources
                      (per-source last_ok_at / last_error)
```

Drift kinds, all deterministic comparisons. Each one ships only if it fires on today's live data
or a recorded fixture (no keyword cathedral):

- `profile_name_vs_launch`: the profile name says `2xv100` but there is no tensor split. Live
  today, chat.
- `catalog_capability_vs_live`: the catalog says `local_llm_heavy: false` but the node serves an
  LLM. Live today, hecate (when on).
- `topology_file_missing_node`: a node is live in biometrics or the pool but absent from
  `orion_field_topology.v1.yaml`. Live today, hecate.
- `pool_card_vs_nvidia`: a pool card's index or VRAM does not match `nvidia-smi`. Not firing
  today.
- `unpooled_gpu`: a GPU is live in biometrics with no pool card. Live today, athena P4/T10. This
  is informational, not an error.
- `launch_flag_skipped`: `skipped_flags` is non-empty.
- `model_file_mismatch`: already computed by discovery as `mismatch`. Reused, not recomputed.

**C. Bus/channel.**

- `orion:body_map:state`: a `BodyMapV1` event, published on change and at least every 60 s.
- `orion:body_map:drift`: a `BodyDriftV1` event when a drift opens or closes.

Both go in `orion/bus/channels.yaml` and the registry. sql-writer persists the state history the
same way as `GpuPoolStateSQL`, with retention.

**D. Read surfaces.**

- introspect tool `body_map(node?: str, include_drift: bool)`, added to `orion/introspect/tools.py`.
- A compact form, `body_map_brief_lines()`, of at most 12 lines. Example:
  `circe: 4 GPUs, chat Qwen3.6-35B Q5_K_M gpu0 65k flash-attn on; hecate: expected offline`.
  Curiosity and the situation brief read it instead of the static field-topology file.
- Hub: a read-only Body panel.
- `self_study.py` `physical_topology` switches its source from the static YAML to the latest
  `BodyMapV1`. The YAML stays as the input for edges and weights.

### Part 2: self-improvement reading

**Sources config (input, not code)** in `config/reading_sources.yaml`:

```yaml
reddit:
  self_improvement:          # judged against the body map; experiment step on
    - {sub: V100, mode: whole_sub, max_new_threads_per_day: 10}         # name UNVERIFIED
    - {sub: LocalLLaMA, mode: body_map_filter, max_new_threads_per_day: 5}   # proposed
    - {sub: LocalLLM, mode: body_map_filter, max_new_threads_per_day: 3}     # proposed
    - {sub: homelab, mode: body_map_filter, max_new_threads_per_day: 3}      # proposed (cooling/noise)
  interests:                 # normal reading + resonance, no experiment step
    - {sub: singularity, max_new_threads_per_day: 3}
    - {sub: transhumanism, max_new_threads_per_day: 3}
    # proposed, Juniper to trim: ArtificialSentience, consciousness, cogsci, compmathneuro,
    # neuroscience, MachineLearning, philosophyofmind
reread: {max_passes: 5, interval_days: 3, new_comments_only: true}
```

**`body_map_filter` takes its terms from the body map, not from a word list.** At filter time the
terms are read from the live `BodyMapV1`:

- GPU names (`V100`, `PG500-216`, `Tesla P4`, `T10`) and their arch (`Volta`, `sm_70`, which
  comes from the GPU name via one small table, the only hand-kept part);
- engine names (`llama.cpp`, `vllm`) and build tags;
- model families/quants actually loaded (`Qwen3.6-35B-A3B`, `Q5_K_M`);
- launch flags actually in use (`flash-attn`, `tensor-split`, `ubatch`).

When the body changes, the filter changes with it. The filter is a cheap pre-selection; the
judge carries the real decision. A miss is fixed by fixing the body map, never by adding words
(memory `feedback_dont_patch_misses_by_growing_word_lists`).

**Connector, deterministic and outside the LLM.** `orion/reading_sources/reddit.py` uses the
Reddit Data API over OAuth:

- **Credentials:** a personal script app. `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`,
  `REDDIT_USERNAME` and `REDDIT_PASSWORD` (or `REDDIT_REFRESH_TOKEN`) live in the Hub `.env`.
  `.env_example` gets empty placeholders only.
- **User agent:** a descriptive `REDDIT_USER_AGENT`, for example
  `orion-reader/0.1 (personal, non-commercial; by u/<juniper>)`.
- **Rate limiting:** honor the `X-Ratelimit-Remaining` / `X-Ratelimit-Reset` headers. Published
  guidance is about 100 queries/min per client (UNVERIFIED); we target far below that, at most
  about 30 requests per sweep.
- **Kill switch:** `READING_REDDIT_ENABLED`.
- **No evasion:** no scraping, no old.reddit or `.json` endpoints without auth, no third-party
  mirrors or proxies, and no rotating user agents. If OAuth fails, the connector stops and
  records `reddit_auth_failed`; it does not fall back to anything else.
- **Output:** for each thread pass it writes one document snapshot. The text is the thread title
  and body on pass 0, then only the comments with `created_utc > cursor` on passes 1-5, each with
  its author-score and permalink. It then submits a read through the existing
  `operator.submit_read` path with a `file://...?sha256=` ref and `why_now` =
  `self_improvement:<sub>:<thread_id>:pass<N>`.
- **Operator README:** say plainly that Devvit templates are not what we need. This is a local
  script app, plus an access request if Reddit requires one.

**Fallback if API access is denied or not yet approved:**

- Juniper pastes thread URLs into the Hub Reading tab. For Reddit URLs, the connector refuses
  without credentials and Hub shows "needs Reddit API approval". Nothing is fetched by other
  means.
- Non-Reddit sources (GitHub issues via the GitHub API, forums with ordinary robots-respecting
  fetch) work regardless. A GitHub issue's comments get the same reread treatment.

**Reread schedule = one durable run per followed thread.** The workflow is `reading.follow.v1`
in `services/orion-durable-runs/app/`, with run_id `follow:<source>:<thread_id>`:

```text
fetch_pass(N) -> snapshot + submit_read (or "no new comments" -> skip) -> wait_until(now+3d)
  -> ... until N == 5 -> finish
```

- The run holds **no GPU lease**: fetching is HTTP. The reads it submits go through the existing
  admitted reading turns.
- The wait reuses the `retry_wait` interrupt pattern. There is no new queue table; the cursor
  (last comment `created_utc`, pass N) lives in run state.

**Judge step (self_improvement sources only).** After Stage 2 of a self-improvement read, one more
durable turn runs, `reading.self_improvement_judge.v1`. It takes three inputs:

- the read's evidence text;
- `body_map_brief_lines()` plus the relevant `BodyServerV1` launch facts;
- the prior verdicts for this thread.

It outputs `SelfImprovementVerdictV1`:

```python
verdict: Literal["worth_trying", "nothing_worth_trying", "already_doing", "not_applicable"]
claim_quotes: list[str]          # verbatim spans from the thread; checked by reading_quote_rule-style
                                 # substring verification, a verdict with an unverifiable quote is dropped
target: str | None               # body-map subject, e.g. "circe:8011 chat"
change: str | None               # exact change, e.g. "--cache-type-k q8_0 --cache-type-v q8_0"
current_value: str | None        # from the body map (with source), so "already_doing" is checkable
expected_gain: str | None        # what the thread claims, attributed, not Orion's invention
measure: str | None              # how to check: an existing gate (llama-bench, lane occupancy recorder,
                                 # scripts/analysis/record_lane_occupancy.py, turn latency metric)
rollback: str | None
hardware_only: bool              # "2x V100 on a Chinese board" -> true -> no outreach
```

**Deterministic post-checks** run after the judge:

- If `current_value` already equals `change` according to the body map, the verdict is forced to
  `already_doing`.
- A `worth_trying` verdict with no `target` that exists in the body map is downgraded to
  `not_applicable`.
- `hardware_only` drops the item.

**New finding kind.** `ReadingFindingKind` gains `experiment_proposal` and `nothing_worth_trying`.
That is the extension the schema docstring already anticipates: consumers group by `seed_id` and
render `headline` / `evidence_text`.

**Outreach**, through the existing `reading_finding_outreach.py`:

- It uses `skip_rate_gates=True`, so there is no daily cap; quiet hours still apply.
- The message carries: thread link, a 3-5 sentence `article_summary` of what the thread says, then
  the verdict.
- A **"worth trying"** message also gives the change, the expected gain as the thread claims it,
  how to measure it, and how to roll it back.
- A **"nothing worth trying"** message is a single heads-up line after the summary.
- `not_applicable` and `hardware_only` verdicts send nothing. They are recorded and visible in the
  Reading tab.

**Never self-applies.** The judge has no tool that writes config, profiles, compose files or the
pool. A proposal is text plus a record. Do not route proposals into the self-experiment machinery
(memory `project_self_experiments_never_dispatched_theater_2026-10-01`).

**Learning loop (only if Juniper answers yes to open question 4).**

- A Hub control lets her mark a proposal `tried_helped`, `tried_no_change`, `tried_worse` or
  `skipped`, and records which body-map field moved.
- The next body-map diff confirms the change actually landed: the launch argv changed. That is
  the evidence; her click alone is not.
- Outcomes feed back into the judge as prior verdicts.

## Files likely to touch

Part 1:

- `orion/schemas/gpu_pool.py`: `LlmLaunchFactsV1`, the optional `launch` on `LlmWorkerAnnounceV1`
- `services/orion-llamacpp-host/app/main.py`: put `cmd` and `skipped_flags` in the announce;
  redact secrets (`--api-key`)
- `services/orion-vllm-host/app/discovery.py`: more `server-info` fields
- `orion/body_map/{probes.py,reducer.py,drift.py,argv.py,brief.py}`: pure reducer + parsers
- `services/orion-body-map/`: a new thin service, or a worker inside `orion-gpu-pool`
  (decided in Phase 1)
- `orion/schemas/body_map.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`
- `services/orion-sql-writer/app/settings.py` + model: persist `body_map.state.v1`
- `orion/introspect/tools.py`: the `body_map` tool
- `services/orion-cortex-exec/app/self_study.py`: switch the source
- `services/orion-hub/`: Body panel (template + static JS + UI test)

Part 2:

- `config/reading_sources.yaml`
- `orion/reading_sources/{reddit.py,github.py,filter.py}`
- `services/orion-durable-runs/app/reading_follow_graph.py`
- `services/orion-durable-runs/app/self_improvement_judge_graph.py` (or a stage-3 node in
  `reading_graph.py`)
- `orion/schemas/reading_finding.py`: the new kinds; `orion/world_pulse_read/findings.py`: producer
- `services/orion-hub/scripts/reading_finding_outreach.py`: render the new kinds
- `services/orion-hub/.env_example` + local `.env` sync (`REDDIT_*`, `READING_REDDIT_ENABLED`,
  `READING_FOLLOW_ENABLED`)
- `services/orion-hub/README.md`: the operator note on Reddit app vs Devvit

## Non-goals

- The body map does not control anything. The pool stays the only thing that grants, loads or
  evicts.
- It does not replace `config/llm_profiles.yaml` or `gpu_pool.yaml`. Those stay the universe of
  the possible, and the body map shows what is.
- No SSH from the reducer to read argv. The launcher reports what it ran; SSH was used only for
  this design's verification.
- No making athena's GPUs pool-managed.
- No flipping hecate's `expected_online`. It stays an operator intent.
- No automated application of any reading-derived change, ever.
- No Reddit access without approved OAuth, no scraping, no mirrors.
- No word-list topic filters.
- No caps on learning outreach.

## Acceptance checks

**Phase 1 (launch facts):**
- Gate: `parse_llamacpp_argv` unit tests use today's real chat and agent argv as fixtures (the
  ones captured above).
- Gate: `extra="forbid"` round-trip of an announce with `launch`.
- Gate: a test that an old pool rejects nothing (consumer-first order).
- Live: after the redeploy, the pool's or body map's announce cache shows `flash_attn=on`,
  `n_gpu_layers=99`, `ctx=65536` for chat, with `source=announce.launch`.

**Phase 2 (reducer + projection):**
- Gate: a reducer fixture built from today's snapshot, where hecate is down and expected offline.
  It must produce `state="expected_offline"`, never `0 GPUs` or `0 W used`.
- Gate: a missing biometrics GPU temp gives `value=None, state="unknown"`.
- Gate: a stale probe older than its ttl gives `stale=true`.
- Live: `orion:body_map:state` is consumed by sql-writer and rows land. The introspect
  `body_map` tool returns circe's four GPUs with the right models.

**Phase 2 (drift):**
- Live: drift fires on today's three real cases: `profile_name_vs_launch` for chat,
  `topology_file_missing_node` for hecate, and `unpooled_gpu` for athena.
- Live: each closes when fixed.
- Gate: no drift fires on a fixture where config and live agree.

**Phase 3 (consumers):**
- Live: `self_study` `physical_topology` includes hecate.
- Live: the curiosity/situation brief shows the compact lines.
- UI: the Hub Body panel renders against a real `BodyMapV1`, with a UI test on the stale/unknown
  rendering.

**Phase 4 (Reddit + follow):**
- Gate: the connector refuses without credentials (`reddit_auth_failed`, no fallback fetch).
- Gate: rate-limit headers are honored (mocked).
- Gate: the pass-N snapshot contains only comments newer than the cursor.
- Gate: a pass with no new comments submits no read.
- Live: `reading.follow.v1` resumes after a timer interrupt. First check this with a 10-minute
  interval in a test run, because the multi-day resume is UNVERIFIED today.
- Live: a real r/V100 thread produces pass 0 and pass 1 reads with different sha256 values.

**Phase 5 (judge + outreach):**
- Eval: Juniper hand-labels a replay set of at least 20 real threads or comments, mixing
  software tips, already-doing tips, hardware show-and-tell and noise. Report the judge's
  agreement with her labels. Rule of thumb: `hardware_only` and `already_doing` should be nearly
  perfect because they are post-checked, and `worth_trying` precision is reported, not assumed.
- Gate: a verdict whose quote is not a substring of the thread is dropped.
- Live: one real outreach message has the link, the summary, and either a verdict with change,
  measure and rollback, or a one-line heads-up. Its decision row carries `seed_id` and
  `finding_ids`.

## Proposal-mode items

- **Capability change.**
  - Orion can see their actual running body: nodes, GPUs, model servers, launch flags, routes,
    holds, and where config and reality disagree.
  - Orion can follow software threads about their own stack and tell Juniper what is worth
    trying.
- **Data touched.**
  - Part 1 reads pool state, biometrics, announces, llama.cpp/vLLM HTTP and durable-runs health,
    and writes one new projection plus its history.
  - Part 2 reads public Reddit/GitHub threads through approved APIs and writes reading snapshots,
    reads, findings and outreach rows.
- **Privacy boundary.**
  - Reddit content is public, but attribution matters: store usernames only as part of permalinks,
    never build profiles of users, and do not post or vote. The OAuth scope is `read` only.
  - Credentials live only in the local `.env`.
  - Launch argv is redacted for `--api-key` and any `*_TOKEN`/`*_KEY` values before it leaves the
    launcher.
  - There is no Juniper-to-Orion privacy boundary (memory
    `feedback_no_privacy_boundary_between_juniper_and_orion`).
- **Proof trace.**
  - `orion:body_map:state` rows in Postgres, and introspect `body_map` output naming
    `source` + `observed_at` per field.
  - Drift events that open and close on the real cases.
  - For reading: the follow run's checkpoints (pass N, cursor), snapshot sha256 values, and
    outreach decision rows with `finding_ids`.
- **Dangerous failure modes.**
  - Absence reads as calm, for example "hecate 0 W used, all fine". This is prevented by the
    `ObservedV1` unknown/stale/expected_offline states and tested in Phase 2.
  - The body map becomes a second scheduler. This is prevented by being read-only, with no
    control surface.
  - The judge invents a gain or a flag. This is prevented by verbatim-quote verification and the
    body-map target/current-value post-checks.
  - Unapproved Reddit access gets the account banned. This is prevented by OAuth-only access and
    a hard stop on auth failure.
  - A proposal is applied unattended. This is prevented by having no write tools in the judge and
    an explicit non-goal.
- **Disable / rollback.**
  - `BODY_MAP_ENABLED=false` stops the reducer, and consumers fall back to showing "body map
    unavailable" (not the static YAML; kill means kill).
  - `READING_REDDIT_ENABLED=false` / `READING_FOLLOW_ENABLED=false`. Follow runs cancel through
    the durable-runs operator.
  - The new finding kinds are additive, and old consumers ignore them by grouping by `seed_id`.

## Recommended next patch

Phases, each one PR:

1. **Launch facts.** Consumers first: the schema field plus pool/body readers that accept it. Then
   the launcher that sends `launch` with argv/skipped_flags, plus the vLLM server-info fields.
   This is the smallest patch that answers "is flash-attn actually on?" from the running process.
2. **Body map reducer + projection + drift.** A pure reducer over pool state, biometrics, announce
   and durable health; the bus channel, registry and sql-writer persistence; and the
   introspect `body_map` tool.
3. **Consumers.** `self_study` reads the body map; the compact brief for curiosity/situation; the
   Hub Body panel.
4. **Sources + Reddit connector + follow runs.** Can start in parallel with 2 once Juniper has API
   approval. The GitHub connector can ship first because it needs no approval.
5. **Judge + finding kinds + outreach rendering**, plus the hand-labelled replay eval.

Start with Phase 1: it is two files, it is reversible, and it closes the one gap Juniper named
that nothing in Orion can see today.

## Sources

- Reddit API policy (secondary, UNVERIFIED against Reddit's own text):
  [Responsible Builder Policy thread, r/redditdev mirror, Nov 2025](https://r.datuan.dev/r/redditdev/comments/1oug31u/introducing_the_responsible_builder_policy_new/nrxva1s/?context=3),
  [data365: Reddit OAuth API documentation](https://data365.co/blog/reddit-oauth-api-documentation),
  [keyapi: Reddit Data API overhaul](https://www.keyapi.ai/blog/reddit-data-api-overhaul-major-changes-and-migration-guide/),
  [Medium: Reddit's API in 2026](https://medium.com/@alex_79882/reddits-api-is-officially-dead-in-2026-here-s-what-i-use-instead-f88ee5b809c8).
  The first link is a third-party mirror. It is cited only as a pointer to the announcement and is
  not a fetch path.
