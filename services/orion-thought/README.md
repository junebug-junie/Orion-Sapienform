# orion-thought

Bus worker service for unified Hub turns. Listens on `orion:thought:request`, runs cortex `stance_react`, applies stance quality and disposition policy, replies with `ThoughtEventV1`, and publishes audit artifacts.

Also publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s), independent of the bus worker/reverie/reasoning tasks above.

## Channels

| Env key | Default | Role |
|---------|---------|------|
| `CHANNEL_THOUGHT_REQUEST` | `orion:thought:request` | RPC intake from Hub |
| `CHANNEL_THOUGHT_RESULT_PREFIX` | `orion:thought:result:` | Reply channel prefix (Hub sets `reply_to`) |
| `CHANNEL_THOUGHT_ARTIFACT` | `orion:thought:artifact` | Audit publish after each thought |
| `CHANNEL_CORTEX_EXEC_REQUEST` | `orion:cortex:exec:request` | Cortex exec plan RPC |
| `CHANNEL_CORTEX_EXEC_RESULT_PREFIX` | `orion:exec:result` | Cortex exec reply prefix |
| (contract constant) | `orion:reverie:visual:step:request` | Durable `reverie.visual` stage intake from orion-durable-runs |

## Flow

```text
LISTEN orion:thought:request
  → validate StanceReactRequestV1
  → run_stance_react() — cortex verb stance_react (brain tier)
  → parse_stance_react_payload + apply_stance_react_pipeline
  → REPLY ThoughtEventV1 on reply_to
  → PUBLISH orion:thought:artifact
```

An admitted turn supplies optional `StanceReactRequestV1.gpu_lease` (the durable run's GPU pool
hold ref, `GpuLeaseRefV1`). Hub sends the same hold the turn's motor runs under; Thought forwards it
in the Cortex Exec context and sets both `llm_route` and `llm_lane` to `agent` (the hold's work
class, never its role), so the gateway attaches stance to the existing hold instead of queueing it
behind the run. The hold takes precedence over the caller's route preference. Requests without a
hold keep their prior route behavior. The older durable token (`resource_lease` /
`ResourceLeaseV1`) was deleted in GPU pool stage 4.6.

## Local checks

The `stance_react` prompt treats downstream tasks and their requested output
schemas as input data, not as instructions for this assessment. It renders an
identical `stance_inputs.user_message` only once while retaining distinct context.
The imperative still directs the harness to perform the original task.

Opt-in model eval (normal gateway admission, no reading/job/journal writes):

```bash
python -m orion.thought.evals.stance_task_boundary \
  --task-file /path/to/downstream-task.txt --gateway-url http://localhost:8210
```

This checks raw stance fields, evidence grounding, output shape and completion
inside 235 seconds. Inspect the reported imperative for task preservation;
schema validity alone is not semantic correctness. It reconstructs a minimal
stance context, not an exact replay of historical transient context. Rebuild
`orion-cortex-exec` to deploy prompt changes; the renderer lives there.
The optional `--reasoning-effort` flag is a diagnostic-only per-request override
for workers that support it; omitting it preserves the worker default.

```bash
PYTHONPATH=services/orion-thought:. ./orion_dev/bin/python -m pytest services/orion-thought/tests/ -v

docker compose \
  --env-file .env \
  --env-file services/orion-thought/.env \
  -f services/orion-thought/docker-compose.yml config
```

## Health

`GET http://localhost:7155/health`

## RPC-health publish (mesh transport coverage)

Every `RPC_HEALTH_PUBLISH_INTERVAL_SEC` (default 30) orion-thought publishes one
`RpcHealthSnapshotV1` (`instance="main"`) on `orion:rpc_health:snapshot`, with the per-hop
`channel_latency` breakdown when `RPC_HEALTH_CHANNEL_LATENCY_ENABLED=true` (A0 of
`docs/superpowers/specs/2026-09-24-metacog-capture-and-transport-ewma-baseline-design.md`).
Both default `true`; `RPC_HEALTH_PUBLISH_ENABLED=false` turns it off.

This service has no single RPC bus -- the per-request handler and `/visual-chain/run-once`
open a bus per call, and the reverie / reverie-chain / visual-chain workers each hold their
own. Each folds its window into a process-wide sink (`app/rpc_health.py`) -- per tick for
the workers, in `finally` for per-call buses -- and one dedicated publisher bus drains it.

Hop keys:

| Hop key | Call |
|---|---|
| `CHANNEL_CORTEX_EXEC_REQUEST` (and lane channels) | `CortexExecClient.execute_plan` (reverie, stance, visual legacy exec) |
| `CHANNEL_VISION_HOST_REQUEST` | `visual_chain.request_caption` (vision-host `caption_frame`) |
| `http:<mind host[:port]>/v1/mind/run` | `mind_enrichment.run_mind_for_thought` (504 / httpx timeout = timeout) |

Consumer-first rollout: `RpcHealthSnapshotV1` is `extra="forbid"`, so rebuild
`orion-signal-gateway` and `orion-equilibrium-service` on PR #2312's build before
deploying this with channel latency on.

## Reveries that ask (2026-11-24, Patch 1)

Spec: `docs/superpowers/specs/2026-11-18-reverie-questions-design.md`. Code: `app/reverie.py` (the poll),
`app/reverie_seeds.py` (pure gate, seeds, picker), `app/store.py` (`reverie_question` reads and writes),
verb `reverie_question` (`orion/cognition/verbs/reverie_question.yaml`, prompt `reverie_question.j2`).

A reverie runs only when Orion is **quiet**, starts from **one piece of real material**, and leaves **one
question** for curiosity to dig, or says why there is none.

```text
every ORION_REVERIE_INTERVAL_SEC (60 s):
  quiet gate   arousal == idle (Redis orion:regulation:latest)
               AND no attention winner for ORION_REVERIE_QUIET_MIN_CALM_SEC (600 s)
               AND no reverie_question row since the stretch began (one per quiet stretch)
               AND fewer than ORION_REVERIE_MAX_PER_DAY rows in 24 h (8)
  seed         least recently used kind with material:
                 absence     a named person's walkway routine scored `missed`; clean (no person seen) vs
                             unrecognised (a person seen, not matched) -- never "did not come"
                 lost_bet    a self-bet calibration gap: faculty x p-bin, n >= 20, |gap| >= 0.15, > 2 SE
                 far_memory  newest lived memory + one from centered-cosine ranks ORION_REVERIE_FAR_MEMORY_BAND
                             (5:25) of orion_lived_memory, live records only (recall's serving gate)
               a ref used in the last ORION_REVERIE_THREAD_LAPSE_DAYS (14) is spent
  question     one reverie_question call on the agent lane; sees the seed and Orion's now only
  check        refs must be the seed's; no near-duplicate (word-set Jaccard >= 0.6) of the pending focus,
               an open self-question or a reverie question of the lapse window; no machinery words
  persist      one reverie_question row (question, none_reason, or a refusal with its reason)
  route        accepted -> curiosity focus slot via Hub's queue_curiosity_focus (invocation_context
               reverie: origin self, never replaces a pending focus, ORION_REVERIE_FOCUS_MAX_PER_DAY = 3)
  read back    the question's cites of the now -> POST /temporal-self/now/use (#3499)
```

A refused agent lane (shed, deadline, outage, timeout) writes no row and does not spend the stretch;
after 3 refusals in one stretch it is given up. Every poll that writes nothing logs
`reverie_skipped reason=<why> count=<n>` (reasons: `arousal_not_idle`, `calm_too_short`, `stretch_spent`,
`daily_cap`, `lane_refused_stretch`, `llm_unavailable`, `no_material`, ...). A written reverie logs
`reverie_question q=... kind=... accepted=... refused=... routed=...`.

The chronicle reads `reverie_question` as the `reverie_question` experience (orion-durable-runs
`app/temporal_self_sources.py`); nothing else is published on the bus.

Off switches: `ORION_REVERIE_MAX_PER_DAY=0` (writes nothing), `ORION_REVERIE_FOCUS_MAX_PER_DAY=0` (records
questions, never sends them), `ORION_REVERIE_ENABLED=false` (no worker).

Needs `services/orion-sql-db/manual_migration_reverie_question.sql`.

**Retired 2026-11-24 (kill means kill).** The coalition-narration reverie (`reverie_narrate`, the 4-step
chain, theme refractory, EMA salience, no-coalition streak), the resonance monitor and its page, the
room-camera expectations and their judge (`reverie_expectation_judge`), the semantic-lift concern cards,
the reverie-scope `AttentionSalienceTraceV1` emit, and the `reverie.chain.v1` / `reverie.thought.v1` /
compaction-request / resonance-alert publishes. The old `substrate_reverie_*` tables stay for history;
nothing writes them.

## Attention salience (GWT-coalition Borda rank-aggregation)

`orion/substrate/attention/salience.py` computes chat-level/open-loop salience by
Borda rank-aggregating two real, evidence-derived signals (`evidence_strength`,
`evidence_breadth`) across the loops competing in one tick. orion-thought no longer emits salience
traces: the reverie-scope `AttentionSalienceTraceV1` emit went with the narration reverie (2026-11-24);
chat-scope traces are cortex-exec's.

## Mind stance enrichment (unified turn)

Set `ORION_THOUGHT_MIND_ENRICHMENT_ENABLED=true` (default `false`) to run
`orion-mind` before `stance_react` and inject an advisory self/attention
`mind_coloring` block into the verb context. `stance_react` stays the sole
author of `ThoughtEventV1` and reconciles the coloring (existing inputs win —
it never forces chat framing on technical/agent turns).

Module: `app/mind_enrichment.py` (snapshot builder, fail-open HTTP client,
allow-list coloring selector, artifact publisher).

Flags:

| Env key | Default | Role |
|---------|---------|------|
| `ORION_THOUGHT_MIND_ENRICHMENT_ENABLED` | `false` | Master switch |
| `ORION_MIND_BASE_URL` | `http://orion-mind:6611` | Mind endpoint |
| `ORION_THOUGHT_MIND_TIMEOUT_SEC` | `210` | HTTP read timeout (must exceed `WALL_MS/1000`) |
| `ORION_THOUGHT_MIND_WALL_MS` | `180000` | Mind policy wall time (must be ≥ `3 × MIND_LLM_TIMEOUT_SEC × 1000`) |
| `ORION_THOUGHT_MIND_ROUTER_PROFILE` | `default` | Mind router profile |
| `ORION_THOUGHT_MIND_MAX_RESPONSE_BYTES` | `2000000` | Response body cap |
| `ORION_THOUGHT_MIND_ARTIFACT_PUBLISH_ENABLED` | `false` | Publish `mind_runs` artifact (`mode=orion`) |
| `ORION_THOUGHT_MIND_COLORING_MAX_ITEMS` | `3` | Coloring list cap |

**`drive_state_compact` facet: removed 2026-07-30.** This used to fetch the
latest `drive_audits` row (`subject = 'orion'`) via
`fetch_drive_state_facet_for_thought` (mirroring `orion-cortex-orch`'s own
`drive_state_compact` facet in `services/orion-cortex-orch/app/mind_runtime.py`)
before building the Mind request. DriveEngine (`drive_audits`' only producer)
was deleted in the same sprint, making the table write-never — the facet
could only ever hand Mind a frozen historical row dressed up as live
self-state, so it was removed outright along with
`ORION_THOUGHT_MIND_DRIVE_STATE_FETCH_TIMEOUT_SEC`.

**Preconditions (silent no-op if unmet):**
1. `orion-mind` must have `MIND_LLM_SYNTHESIS_ENABLED=true` — `meaningful_synthesis`
   (the only quality that fires coloring) is produced only by Mind's LLM path.
2. `orion-thought` must be able to reach `ORION_MIND_BASE_URL`.

**Budget invariant:** `orion-mind` runs 3 sequential LLM phases, each capped by
`MIND_LLM_TIMEOUT_SEC` (default 60s on the `orion-mind` service). If
`ORION_THOUGHT_MIND_WALL_MS` is below ~`3 × MIND_LLM_TIMEOUT_SEC × 1000` (180000)
synthesis is cut off mid-pipeline and the Mind always degrades to `contract_only`
(the coloring never fires) — an empty-shell no-op. The default `180000` allows
all three phases; `ORION_THOUGHT_MIND_TIMEOUT_SEC` (HTTP read) must
exceed `WALL_MS/1000` so Mind's own fail-open result is returned instead of the
client aborting. `orion-thought` logs a `mind_enrichment_config` warning at boot
if either invariant is violated while enrichment is enabled.

The min-viable wall is derived from `MIND_LLM_TIMEOUT_SEC_ASSUMED` (60s) in
`app/settings.py`. That mirrors `orion-mind`'s `MIND_LLM_TIMEOUT_SEC` (a separate
service, not readable from here) — if you change the per-phase timeout on
`orion-mind`, re-derive `MIND_ENRICHMENT_MIN_VIABLE_WALL_MS` and the wall default
here, or the boot warning will under-fire.

**Latency caveat:** enrichment runs synchronously before `stance_react`. A stuck
Mind LLM now fails open after ~3×60s of phase clamping (bounded by the 210s HTTP
read) rather than the old ~12s, i.e. up to ~180–210s added to a live user turn in
the worst case. Pair any rollout (`ORION_THOUGHT_MIND_ENRICHMENT_ENABLED=true`)
with a turn-level budget/circuit-breaker on the caller.

Everything fails open: Mind unconfigured / unreachable / slow / low-quality →
byte-identical to today's stance behavior.

## "My day" cue for stance_react (Temporal Self patch 4)

Each unified turn, `app/temporal_self_cue.py` fetches orion-durable-runs `GET /temporal-self/frame`
(`TEMPORAL_SELF_FRAME_URL`, `TEMPORAL_SELF_CUE_FETCH_TIMEOUT_SEC=0.5`) concurrently with Mind
enrichment and renders it with `orion.temporal_self.stance_cue.project_stance_cue` (a fixed template,
<= 400 chars, every clause citing `[arc:<id>]`). The text goes into the verb context as
`temporal_self_cue` and renders as `MY DAY SO FAR` in `stance_react.j2`. A frame that is missing,
for another local day, or older than `TEMPORAL_SELF_CUE_MAX_AGE_SEC=900` renders nothing.

The stance's arc citations are gated (`orion.thought.stance_react.gate_day_citations`): shown arc
ids may appear in `evidence_refs`; any other is removed and counted; a sentence naming the shown day
without citing an arc is labelled `UNVERIFIED`. The result is `ThoughtEventV1.day_citations`,
persisted by orion-sql-writer to `thought_decision.day_citations` and shown in the Hub turn trace.

Logs: `temporal_self_cue corr=... chars=... arcs=... fetch_ms=...`,
`temporal_self_cue_skipped corr=... reason=stale|wrong_day|fetch_failed|invalid_frame|disabled`,
`temporal_self_day_citations corr=... label=...`. Kill switch: `TEMPORAL_SELF_STANCE_CUE_ENABLED=false`.

## Personality read-back for stance_react (personality graph, spec rev 2)

What Orion keeps coming back to, in Orion's own words, read back as a record, not an instruction.
`app/personality_cue.py` fetches orion-durable-runs `GET /personality/frame` (`PERSONALITY_FRAME_URL`,
same base as `TEMPORAL_SELF_FRAME_URL`) and renders it with `orion.personality.cue.project_personality_cue`
(a fixed template, no LLM, <= 600 chars, strongest nodes first). No tone or style instruction is added
anywhere: the cue only names what recurred, where, and how often.

- Chat: built concurrently with Mind enrichment and the day cue (`PERSONALITY_CUE_TIMEOUT_SEC=0.5`),
  passed as `personality_cue` and rendered as `WHAT I KEEP COMING BACK TO` in `stance_react.j2`.
- Reverie no longer reads it (2026-11-24: reveries that ask see their seed and the now only).

A frame that is missing, invalid, older than `PERSONALITY_CUE_MAX_AGE_SEC=3600`, or empty renders nothing.
Log (one per build): `personality_cue corr=... rendered=true|false ids=... skip=disabled|fetch_failed|invalid_frame|stale|empty|render_failed`.
Kill switch: `PERSONALITY_CUE_ENABLED=false`.

## Self-monitor track record for stance_react (self-monitor spec step 3)

What Orion is failing at or misjudging right now, from their own settled bets, in plain first-person
sentences. `app/self_track_record_cue.py` reads Redis `orion:self_monitor:latest` on `ORION_BUS_URL`
(the self-monitor state orion-durable-runs publishes hourly) every 5 min in a background thread and
renders it with `orion.self_monitor.cue.project_track_record_cue` (fixed templates, no LLM).

- **What renders:** only faculties whose held status is `broken`, `unreliable` or `miscalibrated`
  (never `reliable`, never `unknown`), worst status first, then most evidence; at most 300 chars.
  When room is left, one clause counts faculties that cannot be graded at all yet. Example:
  "My walkway-grade journal runs failed to start 22 of 28 times this week. My guesses about how many
  steps my turns take hold 6% of the time (I say 80%)."
- **Where:** stance_react (`MY TRACK RECORD`). (Reverie's coalition branch read it until 2026-11-24.)
- **Fail-open:** the turn never waits on Redis; an absent, unparseable, stale (> 3 h) or
  future-stamped state renders nothing (reader contract, `orion/self_monitor/view.py`). Shows nothing
  until orion-durable-runs publishes the key (self-monitor patch 2).
- Logs: `self_track_record corr=... purpose=stance|reverie chars=... shown=...` or
  `self_track_record_skipped reason=disabled|not_loaded|fetch_failed|absent|unparseable|stale|future_stamped|nothing_notable|render_failed`.
- Kill switch: `SELF_TRACK_RECORD_CUE_ENABLED=false` (both prompts exactly as before).
- Eval: `python evals/run_self_track_record_live_eval.py` renders the cue from the live state.

## Self-readings for stance_react (self-calibration spec section C)

Orion's body signals, each scored 0-10 against its own last 7 days, shown to the unified turn's
stance step (`MY BODY RIGHT NOW`). (Reverie's coalition branch read it until 2026-11-24.)
`app/self_readings_cue.py` reads `substrate_node_prediction_error_history` (read-only) into one
process-wide `NodePeHistoryCache` (7-day seed once, then incremental); `orion/attention/self_readings.py`
calibrates with the attention seam's own rule (`world_first.judge_candidate`) and renders a fixed
template, no LLM.

- **Which signals:** glossary prediction_error entries whose `prompt_sites` name
  `orion.attention.self_readings:render_self_readings` (biometrics, bus_synaptic, execution, chat,
  route, codebase, cabinet). The world (camera perception) and placeholders (harness_closure) never.
- **What renders:** only signals high or unusual for themselves, fresh, and (event-written ones)
  within the 300 s orienting window; max 3 lines / 600 chars, each citing
  `[pe:<node>@<observed_at>]` (`[alarm:<node>@...]` for an alarm: cabinet, strain). A calm body renders nothing. When a block shows, unreadable signals
  are named ("Can't read right now: ...") and event signals with no recent event say so.
- **Triggers** (cabinet warming) read as times it went off (off->on transitions, not 30 s checks) in
  the last hour against their own week; the ref names the newest ON row.
- Reverie's concern-card (semantic lift) branch never gets it: that branch discards interpretations
  with mechanism vocabulary.
- With stance on and Mind coloring present, the prompt labels Mind's frontier scores as Orion's own
  estimates, not measurements.
- Fail-open: `SELF_READINGS_FETCH_TIMEOUT_SEC` (0.5 s) bounds read+render; a miss renders nothing.
  Startup warms the cache in a thread. Logs: `self_readings corr=... purpose=stance|reverie chars=...
  shown=...` or `self_readings_skipped reason=...`.
- Kill switch: `SELF_READINGS_STANCE_ENABLED=false` (the prompt byte-identical to before).
- Eval: `evals/test_self_readings_truth_eval.py` (every number shown resolves to its row);
  live replay `python evals/run_self_readings_live_eval.py`.
- What it tends to lead to (spec section E): an unusual reading's line gets "; when this is high, my
  next hour had <a failed turn | a reply I had to repair | an RPC timeout> Nx as often at equal activity (...)
  [assoc:<id>]" from the newest `self_signal_outcome_association` row per (signal, outcome) written by
  orion-equilibrium-service within the last 3 h (read at most every 5 min, same lock/timeout). Only
  rows whose 95% interval excludes 1 with enough hours carry a clause, and only once that verdict held
  on each of the last `SELF_SIGNAL_OUTCOME_CONSECUTIVE_HOLDS` (6) consecutive hourly computations (a
  gap or a different verdict resets; shared `held_latest_by_pair`); a clause is dropped before a
  reading would be. Kill switch: `SELF_SIGNAL_OUTCOME_ASSOCIATION_ENABLED=false` (block byte-identical).

## Reverie VISUAL chain (context rotation + validated episode-memory seed)

`app/visual_chain.py`, alongside `chain.py`. Patch 2 of
`docs/superpowers/specs/2026-08-20-reverie-visual-chain-design.md` — the
orchestration Patch 1.5 (`services/orion-diffusion-host`'s real model wiring)
was a prerequisite for. Enabled 2026-08-25 per Juniper after a live smoke
verified the full loop: real file on disk, real
`reverie_visual_chain`/`reverie_visual_artifact` rows, correct FK, honest
`description=null` rather than a fabricated caption.

**Who starts a waking painting (2026-11-05, ladder retirement PR 2): this service, and only this
service.** `app/painting_self_start.py` ticks every `ORION_PAINTING_SELF_START_TICK_SEC` (60 s):

1. It runs the visual-baseline schedule (`orion/reverie/baseline.py`, policy
   `config/reverie/visual_baseline.v1.yaml`) on the `visual_baseline_checkpoint` row under
   `FOR UPDATE`. A painting is due 5,400 s after the last acknowledged painting. A failed attempt
   is asked again after 600 s. An open attempt (a dream hop too) waits.
2. Arousal `strained` at the moment of issue defers it by 600 s
   (`ORION_VISUAL_BASELINE_AROUSAL_ENABLED`, R3 reader 4).
3. The self-monitor act gate (`SELF_MONITOR_ACT_GATE_ENABLED`) can refuse a painting faculty held
   broken or unreliable, except one re-test probe per 6 h.
4. It places a SHADOW hunger bid on `thought.painting` (a clock want, logged
   `physiology_shadow_bid ... real=submitted`). Never obeyed: painting binds in the market later
   (attention patch 3).
5. It submits one `reverie.visual` durable run to orion-durable-runs (`POST /runs`). The run waits
   for its GPU-pool `diffusion` hold, the generate step checks the thermal gate, and the steps below
   open and close the `thought.painting` `make` act. `dispatch_id` is
   `painting_self_start:<need_id>:<issued_at>`.

`/health` -> `painting_self_start` shows the last tick (`submitted`, `visual_baseline_not_due`,
`arousal_strained`, ...). Replay: `python -P services/orion-thought/evals/painting_self_start_replay.py`
(fixture) or `--live --days 14` (about 10 a day against the real 9.2, in the PR report).

Before this, the proposal arena's `render_scene` template and orion-proposal-runtime's baseline
read were the triggers, through policy and execution dispatch to cortex-exec's
`skills.imagination.render_scene.v1`. Both are gone. The verb stays for a manual run (no
`dispatch_id`: the direct `POST /visual-chain/run-once`).

**The fixed-cadence legacy worker is OFF** (`ORION_VISUAL_CHAIN_ENABLED=false`, retired
2026-08-30). Setting it true makes the self-start loop issue nothing
(`legacy_visual_worker_enabled`), so the two can never both paint.

On whichever cadence actually triggers a run (timer, if re-enabled, or the
on-demand dispatch above): generate an image from `orion-diffusion-host`
(circe, `POST /generate`), store it content-addressed
(`orion.reverie.visual_storage`), hand it to `orion-vision-host`'s existing
`caption_frame` task (via `orion-percept-store` — no vision-host code
change needed, no new task_type; `caption_frame` + `percept_sha256` already
captions any image), and persist both a `reverie_visual_chain` row and a
`reverie_visual_artifact` row. The caption becomes `prior_description`, which
the *next* run's prompt is built from — the enforced continuity column
design doc §2 exists specifically to avoid repeating the text chain's dead
`next_focus`/`drift` fields (nothing ever read them). One run = one step
(`step_index=0` always); the "chain" here is the sequence of runs over time,
not a multi-step ladder like the text chain's — there is no coalition/pressure
signal to climb.

**Patch 2 scope (shipped 2026-08-25):** the mechanical loop and the
`prior_description` wiring only — `build_visual_prompt` was deliberately the
thinnest honest placeholder (`prior_description`, or a fixed seed prompt for
the very first run), not a fabricated stand-in for the context-seeding that
patch did not own.

**Patch 3 context-seeding (design doc §14):** `build_visual_prompt` now also
takes `context_text` — the text reverie chain's own most recent, real
(non-hollow) `substrate_reverie_thought.interpretation`, ONLY once it is
already linked into a *settled* `substrate_reverie_chain` row
(`store.load_latest_reverie_interpretation`, capped at
`ORION_REVERIE_CONTEXT_CHAR_LIMIT`=240 chars by default, word-boundary
truncation via `orion.cognition.compactor.truncate`, and only considered
fresh within `ORION_REVERIE_CONTEXT_MAX_AGE_SEC`=900s — without a staleness
bound, a stalled/disabled text-reverie worker would leave the same old
thought answering forever, presented as "Orion is currently thinking"
long after it stopped being current; see `settings.py`'s field comment for
why 900s specifically, not the felt_state_reader.py/proposal-runtime
thresholds already elsewhere in this repo). A deliberately narrow first slice of
the design doc §1's full "recent activity / chat / dream" list: already-
summarized content that already reaches the Hub Reverie tab's Text sub-view
— but only true once chain-linked (review finding: a thought row is written
immediately on generation, before its enclosing chain settles; querying it
directly would source content `text_recent` might never surface). Hollow
status is re-validated in Python via `SpontaneousThoughtV1.is_hollow()`
rather than trusted from a stored flag, matching `chat_stance.py`'s existing
`_project_reverie_glimpse` discipline for this same table. Continuity and the
context-seed are blended when both are present; the fixed seed string is now
a true last resort, hit only when neither exists yet (a fresh install).
`context_text` is recorded as its own `chain_json` key (not just baked into
`prompt` prose) on both the success and `generation_failed` paths, and
surfaced as its own field in the Hub Reverie tab. Live-verified against the
real `conjourney` database 2026-08-26 (design doc §14). Raw chat/dream
sourcing remains a separate, later change.

**Patch 4 continuity reset (design doc §15):** live-caught 2026-08-27, hours
after Patch 3 shipped -- `prior_description` continuity had locked onto one
visual attractor ("ancient Roman aqueduct") across 10+ runs / 100+ minutes,
unmoved by context-seeding: a short abstract clause has nowhere near the
prompt weight of a long, concrete continuity description, and abstract
cognitive-state narration isn't strongly visualizable content regardless of
prompt order. Fix is a deterministic reset, not a reweighting guess:
`resolve_visual_chain_continuity` tracks how many CONSECUTIVE runs used real
continuity (`chain_json.continuity_streak`, read alongside
`prior_description` in one round trip by `store.
load_latest_visual_chain_continuity_state` -- review finding: two separate
reads of the same latest row wasted a query and left a theoretical race);
once that streak reaches `ORION_VISUAL_CHAIN_CONTINUITY_MAX_RUNS` (default
3), the next run forces continuity to drop from its own prompt -- re-seeding
from `context_text`, or the fixed seed if neither exists -- then continuity
resumes normally. On a reset run, a failed generation/caption never
resurrects the stale pre-reset `prior_description` (a second review
finding, fixed the same way: `continuity_fallback` is `None` on a reset
run, the old value only on a normal one). No off switch by design (0 means
reset every run). `continuity_streak`/`continuity_reset` recorded in
`chain_json` on both the success and `generation_failed` paths, surfaced in
the Hub Reverie tab alongside `context_text`.

**Patch 5 self-study context-seed (design doc §16):** same session as Patch
4 -- Juniper directly asked for the visual chain to draw on "actual memory
or a recent chat or something from Orion's self study analysis." All three
candidates were live-checked before any code was written: `memory_
crystallizations` ("actual memory") was DECLINED -- its `summary`/`subject`
columns hold verbatim personal chat content with no safe filter, including
a real sample naming a family member's medical history. The
`chat_history_compactor` digest ("recent chat") was DECLINED on cadence
grounds -- it's a daily-schedule producer with no evidence it has ever
fired in production, not a fit for a ~600s-cadence consumer. Self-study
analysis was BUILT: `build_visual_prompt` gains a third optional input,
`self_study_text` (`store.load_latest_self_study_reflection`), real
quantified self-observation from `self_study_analysis.py`'s four
deterministic window-contrast analyses (concept induction, vision events,
affective state, co-creation signals) -- confirmed safe by reading real
bodies before writing any code. The actual privacy boundary is
`store._SAFE_SELF_STUDY_SOURCE_PREFIXES`, an ALLOWLIST of only those four
producers' `source_ref` prefixes -- `source_kind='self_study'` also covers
a sibling free-form "Curiosity" reflection confirmed live to quote
sensitive personal content, which the allowlist deliberately excludes.
**Repointed 2026-11-07 (ladder retirement PR 6):** those four analyses had one
trigger, the retired dispatch rung, and stopped on 11-05. The seed now reads
curiosity's self-inquiry answers (`self_concept_history`,
`produced_by='curiosity_self_inquiry'`), and the prefix allowlist went with the
old source (Juniper, 2026-10-07: no privacy boundary between her and Orion).
`self_study_text` recorded in `chain_json` on both paths, surfaced in the
Hub tab alongside `context_text`. `build_visual_prompt`'s old if/elif
branches were refactored into a list-join composition (a third optional
input would have meant 8 branches) -- verified byte-identical output for
every pre-Patch-5 combination via exact-string test assertions.

**Patch 6 memory-crystallization context-seed (design doc §17):** same day,
Juniper corrected Patch 5's `memory_crystallizations` call -- the "no new
privacy surface" concern assumed a second audience for that content that
does not exist: this route has no external port mapping and no auth, so
Juniper is the only possible viewer, and also the original source of
everything the table holds. `build_visual_prompt` gains a fourth optional
input, `memory_text` (`store.load_latest_memory_crystallization`),
verbatim `summary` text from the most recent `status='active'` row --
deliberately NOT content-filtered, unlike `self_study_text`. `memory_text`
recorded in `chain_json` on both paths, surfaced in the Hub tab.

**Stage 2 PR G episode-memory cutover (design doc
`2026-10-06-memory-stage2-referent-graph-design.md` §5):** the memory slot
now reads `episode_memory` by default through `load_episode_memory_seed`.
Only active, unexpired `happened`/`about_juniper` memories created in the
last seven days are eligible; every evidence quote must be verified, no
`downgraded_voice`/`rejected_invalid` event may exist, and the memory must
be either low-stakes automatic or explicitly confirmed. Selection rotates
by least recent visual use, with never-used memories first. The exact
`memory_id` is recorded as a `VisualSourceV1` under
`context_selection.source_kind='episode_memory'`, including generation
failures; `chain_json.memory_source_kind` also identifies the loaded memory
slot on runs where a different slot won the round-robin.
`REVERIE_VISUAL_MEMORY_SEED_SOURCE=crystallization` is the bounded
rollback until the legacy reader is removed at cutover.

**Patch 7 context-slot rotation (design doc §18) -- the real root cause:**
same day, live report: "the memory got washed out and Orion just continued
generating stars." Root cause verified with the REAL tokenizer: SDXL-turbo's
CLIP text encoder truncates at 77 tokens, silently -- the actual reported
prompt tokenized to 191 tokens; the model never saw `memory_text` or even
the style suffix. Every context-seed added after Patch 3 had, in practice,
almost never reached the model regardless of how correctly it was computed.
Fix: `build_visual_prompt` now takes a SELECTED slot
(`context_slot_name`/`context_slot_text`, breaking change from the old
4-input signature) instead of concatenating all three -- `select_context_
slot` round-robins among whichever of {context, self_study, memory}
currently have content, persisting the rotation counter in `chain_json.
context_slot_rotation` (read in the same combined round trip as
`prior_description`/`continuity_streak`). Each slot's char cap was
independently re-derived against the real tokenizer:
`MAX_SELF_STUDY_CONTEXT_CHARS` 400→150, `MAX_MEMORY_CRYSTALLIZATION_
CONTEXT_CHARS` 400→180 (`MAX_REVERIE_CONTEXT_CHARS` was already fine
unchanged). `chain_json.context_slot_used` names which one actually
entered the prompt; the other two are still recorded, just visually marked
"not used this run" in the Hub tab. `orion-diffusion-host`'s
`_run_generation` also gained `_log_prompt_token_budget`, logging a
WARNING with real token counts whenever a prompt exceeds either of SDXL's
two text-encoder budgets -- visibility only, using the pipeline's own
already-loaded tokenizer(s), zero new dependency.

**Single-flight, no backlog** (design doc §4 acceptance check): the worker
loop's own sequential shape (run, then sleep, then run again — same as
`chain.py`/`reverie.py`) makes overlap structurally impossible; there is no
separate scheduler event outside the loop. `run_visual_chain_once` also holds
a process-local `asyncio.Lock` and no-ops if already held, as defense-in-depth
independent of caller discipline (mirrors `orion-diffusion-host`'s own
`_generation_lock`).

**Honest degradation, never fabrication:**
- diffusion-host call fails → `terminal_reason="generation_failed"`, no
  artifact row (nothing was generated).
- re-observation (percept upload or vision-host RPC) fails → the image is
  still real and gets stored with `description=None` (never a fabricated
  caption), and the *previous* run's `prior_description` carries forward
  unchanged rather than propagating a null description and losing continuity
  for one bad step.

Requires `manual_migration_reverie_visual_chain.sql` applied (Patch 1),
`orion-diffusion-host` live on circe
(`services/orion-diffusion-host/README.md`), and (fixed after Patch 2's
initial merge — the compose file shipped with no `volumes:` block at all)
`docker-compose.yml`'s bind mount of
`/mnt/storage-lukewarm/orion/reverie-visual` — without it,
`store_visual_artifact` writes into the container's own ephemeral
filesystem at that path, not the real host disk, and every generated image
silently vanishes on the next container restart/recreate.

**Re-observation target moved 2026-08-26.** Originally a new producer on
the existing shared `orion:exec:request:VisionHostService` channel
(`single_consumer: true` means one *consumer*, not one producer, so this
was a safe multi-producer addition at the time). Moved to circe's
dedicated `orion:exec:request:VisionHostService:circe-vl` lane
(`services/orion-vision-host/docker-compose.circe-qwen.yml`) after live
confirmation that athena's shared instance (BLIP-base) cannot produce a
caption real enough to clear `sanitize_caption`'s quality bar for a
*generated* image — 3/3 real ticks on the shared channel came back
uncaptioned; the very first tick against circe's Qwen2-VL-2B-Instruct
produced a real, detailed caption and `prior_description` advanced for the
first time in this feature's life. Replies still land on the shared
`orion:vision:reply:*` wildcard pattern either way.

Flags:

| Env key | Default | Role |
|---------|---------|------|
| `ORION_VISUAL_CHAIN_ENABLED` | `true` | Master switch |
| `ORION_VISUAL_CHAIN_INTERVAL_SEC` | `600` | Trigger cadence (real cadence is `max(this, run duration)`) |
| `ORION_VISUAL_CHAIN_STALENESS_THRESHOLD_MIN` | `45` | Watchdog threshold (2026-09-04): if the newest `reverie_visual_chain` row is older than this, fire an orion-notify `critical` attention request (email + Hub attention item) -- catches the worker wedging before ever reaching the run-deadline/single-flight lock below, which a live 24h+ silent wedge with zero errors proved possible |
| `ORION_VISUAL_PAINTING_GAP_CHECK_ENABLED` | `true` | Painting-gap check (2026-10-10), own loop, not gated on `ORION_VISUAL_CHAIN_ENABLED` (off in production, which is why the staleness watchdog never ran during the 27.6 h 2026-10-09/10 outage) |
| `ORION_VISUAL_PAINTING_GAP_THRESHOLD_HOURS` | `12` | If no real painting (a `reverie_visual_chain` row with a `production_receipt`, timed by its `produced_at`) has landed in this many hours, fire an orion-notify `error` attention request (emails if left unacked). Calibrated in-sample on 2026-09-14..10-10: no gap fell between 7.6 h and 14.7 h |
| `ORION_VISUAL_PAINTING_GAP_CHECK_INTERVAL_SEC` | `600` | How often the painting-gap loop re-reads the last painting's age |
| `ORION_VISUAL_ATTEMPT_STREAK_CHECK_ENABLED` | `true` | Failed-attempt streak check (critical card) in the same loop |
| `ORION_VISUAL_ATTEMPT_FAILURE_STREAK_THRESHOLD` | `6` | Painting attempts in a row with no image before the card |
| `ORION_VISUAL_CHAIN_WATCHDOG_CHECK_INTERVAL_SEC` | `600` | How often the watchdog re-checks that staleness, independently of the worker's own loop |
| `ORION_DIFFUSION_HOST_BASE_URL` | `http://100.112.254.99:8014` | circe's diffusion host |
| `ORION_VISUAL_CHAIN_DIFFUSION_TIMEOUT_SEC` | `120` | `/generate` HTTP timeout (raised from `30` 2026-08-28 -- tuned for sdxl-turbo's near-instant single step; FLUX.1-schnell's real generation measured 49-56s live, timing out every tick until fixed) |
| `ORION_VISUAL_CHAIN_STORAGE_DIR` | `/mnt/storage-lukewarm/orion/reverie-visual` | Content-addressed image store |
| `ORION_VISUAL_CHAIN_PERCEPT_STORE_URL` | `http://orion-athena-percept-store:8000/percepts` | Cross-host hop to vision-host |
| `ORION_VISUAL_CHAIN_PERCEPT_STORE_TOKEN` | *(empty)* | `X-Orion-Percept-Token`, if the store requires one |
| `ORION_VISUAL_CHAIN_PERCEPT_UPLOAD_TIMEOUT_SEC` | `10` | Percept upload timeout |
| `CHANNEL_VISION_HOST_REQUEST` | `orion:exec:request:VisionHostService:circe-vl` | circe's dedicated Qwen2-VL vision-host lane |
| `CHANNEL_VISION_REPLY_PREFIX` | `orion:vision:reply` | Per-call reply channel prefix |
| `ORION_VISUAL_CHAIN_CAPTION_TIMEOUT_SEC` | `60` | Vision-host RPC timeout |
| `ORION_REVERIE_CONTEXT_CHAR_LIMIT` | `240` | Max chars of the text-reverie chain's context-seed woven into the prompt |
| `ORION_REVERIE_CONTEXT_MAX_AGE_SEC` | `900` | How stale that context-seed thought can be before it's treated as absent rather than "current" |
| `ORION_VISUAL_CHAIN_CONTINUITY_MAX_RUNS` | `3` | Consecutive continuity-carrying runs before a forced reset (Patch 4, design doc §15); no off switch, 0 resets every run |
| `ORION_SELF_STUDY_CONTEXT_CHAR_LIMIT` | `150` | Max chars of the self-study analysis context-seed woven into the prompt (Patch 5, design doc §16; cut from 400 by Patch 7, design doc §18 -- 400 chars of self-study's dense jargon alone tokenizes to 104 tokens with framing, already over SDXL's 77-token budget by itself) |
| `ORION_SELF_STUDY_CONTEXT_MAX_AGE_SEC` | `21600` | How stale that self-study analysis can be before it's treated as absent (6h -- these analyses fire on their own 6-72h cadence) |
| `REVERIE_VISUAL_MEMORY_SEED_SOURCE` | `episode_memory` | Memory-slot reader: governed `episode_memory` by default; `crystallization` is a bounded rollback |
| `ORION_MEMORY_CRYSTALLIZATION_CONTEXT_CHAR_LIMIT` | `180` | Max chars of the selected memory source woven into the prompt; historical key retained during cutover |
| `ORION_MEMORY_CRYSTALLIZATION_CONTEXT_MAX_AGE_SEC` | `604800` | Maximum creation age for the selected memory source. Episode memory clamps this to a hard 7-day ceiling; the historical key remains configurable for rollback |

Tests: `tests/test_visual_chain.py` — every hop faked (diffusion HTTP call,
percept upload, vision-host RPC, reverie context-seed, persistence); one test
runs two sequential calls and asserts the second run's diffusion prompt
demonstrably contains the first run's persisted description (design doc §9's
"same-run evidence, not schema presence" acceptance check), and another
asserts `context_text` reaches both the prompt and `chain_json`. Direct
coverage for `load_latest_reverie_interpretation` itself (grounded-candidate
selection, stale-hollow-flag rejection, fall-through past a hollow/
unparsable row to the next real one, word-boundary truncation, empty-row,
never-raises) lives in `tests/test_store.py`, using real `SpontaneousThoughtV1`
fixtures rather than hand-rolled dicts. The chain-linkage `EXISTS` clause
itself needs a real Postgres to exercise (same limitation as this file's
other read-filter functions, e.g. `load_recent_chain_theme_events`'s
`theme_key` filter) -- live-verified instead against the real `conjourney`
database (design doc §14).

Patch 4's `resolve_visual_chain_continuity` is a pure function with direct
unit coverage (no-prior/under-cap/at-cap/max-runs=0), plus an end-to-end
orchestration test driving `run_visual_chain_once` through a full cap+1
cycle and asserting the run AT the cap's own generated prompt excludes the
prior continuity text -- not just that the resolver says it would in
isolation (design doc §15).

Patch 5's `load_latest_self_study_reflection` has direct coverage in
`tests/test_store.py`: real-body return, char-limit override, max-age-sec
clause presence/absence, empty-row/empty-body, never-raises -- and, the
actual privacy boundary, a dedicated test asserting the SQL only ever binds
the four known-safe `source_ref` prefixes by name (not "some WHERE clause
exists"), with an explicit assertion that `curiosity:` is never among them.
`build_visual_prompt`'s list-join refactor has an exact-string regression
test proving every pre-Patch-5 prompt combination is byte-identical to the
old branches (design doc §16).

Patch 6's `load_latest_memory_crystallization` has direct coverage in
`tests/test_store.py`: real-summary return, char-limit override,
max-age-sec clause presence/absence, `status='active'` filter (a
`rejected`/`proposed` row is never returned), empty-row/empty-summary,
never-raises. `build_visual_prompt`'s fourth clause and `run_visual_chain_
once`'s wiring have the same exact-string/orchestration coverage pattern
as Patch 5's `self_study_text` (design doc §17).

Stage 2 PR G's `load_episode_memory_seed` has direct SQL-contract coverage
for purpose/status/expiry, stakes confirmation, all-verified evidence,
disqualifying events, least-recently-seeded rotation, typed identity,
word-boundary truncation, empty results, and database failure. Orchestration
tests prove the episode `memory_id` survives into both successful and
generation-failed chain receipts; the honesty eval covers the selected
source contract end to end.

Patch 7's `select_context_slot` has direct unit coverage in
`tests/test_visual_chain.py`: nothing available, rotates through all
three, skips unavailable slots, wraps a large rotation index, index
unchanged when nothing available. `build_visual_prompt`'s new two-input
contract (breaking change from four) has exact-string wording coverage
per slot label. Two end-to-end orchestration tests are the actual
regression guard: one proves that with all three context-seeds present
simultaneously, only the rotation-selected one appears in the real
generated prompt string; another drives four successive runs through the
same fake-DB round-trip harness the continuity tests use and asserts the
selected slot visits context → self_study → memory → context in order
(design doc §18). `orion-diffusion-host`'s `tests/test_generate.py` has
direct coverage of `_log_prompt_token_budget` with a fake tokenizer: warns
when over budget, silent when within it, checks both SDXL encoders, never
raises when the pipe has no tokenizer at all.

### Visual chain wedge fix + staleness watchdog (2026-09-04)

Live-confirmed: the single-flight lock in `run_visual_chain_once` wedged for
24+ hours with zero errors logged anywhere — a stuck Postgres call run via
`asyncio.to_thread` can never be cancelled, only abandoned, so it permanently
occupied one worker in the process's shared thread pool. `store.py`'s shared
engine now sets `statement_timeout`/`connect_timeout` (matching the pattern
`_get_expectation_read_engine` already used for the former), closing that off.

Same edge-triggered `orion-notify` pattern as the resonance monitor above,
in a new `app/visual_chain_health_monitor.py`, but run from an *independent*
asyncio task (`run_visual_chain_watchdog`, mirroring
`orion-field-digester/app/worker.py`'s `_health_loop` shape) rather than
being called from inside a completed tick — the whole point being that a
wedged worker can never report on its own staleness. Fires `severity="critical"`
(immediate email, per `orion-notify`'s own convention for non-self-healing
failures) if `reverie_visual_chain`'s newest row is older than
`ORION_VISUAL_CHAIN_STALENESS_THRESHOLD_MIN` (default 45min). Guarded by
both `visual_chain_enabled` and `orion_bus_enabled`, matching the worker's
own guard, so a deployment that never runs this pipeline at all doesn't get
a permanent false alarm.

**This watchdog assumes *something* triggers a run at least every ~45
minutes.** That held under the old fixed-cadence cron. It does not hold under the self-start loop
above, which paints about every 90 minutes; the watchdog is only armed with the legacy worker.

**Painting-gap check (2026-10-10).** A second, independent check,
`visual_painting_gap`: hours since the last painting Orion actually produced
(`store.visual_last_painting_age_hours()`, production receipts only). It runs
in its own loop (`run_visual_painting_gap_watchdog`, started from `main.py`'s
lifespan), gated only on `ORION_VISUAL_PAINTING_GAP_CHECK_ENABLED` (default
on) -- deliberately NOT on `ORION_VISUAL_CHAIN_ENABLED`. That legacy flag is
off in production (paintings come from the durable visual-baseline path), so
the staleness watchdog above never starts there; that, not fresh rows, is why
nothing alerted during the 27.6 h no-painting outage of 2026-10-09/10 (GPU
lane controller refusing every swap). Over
`ORION_VISUAL_PAINTING_GAP_THRESHOLD_HOURS` (default 12) it raises an
`error`-severity attention item whose message lists the likely causes in
order: GPU lane controller refusing swaps (check the mesh-guardian GPU cards
/ `scripts/gpu_pool_actuator_probe.py`), long thermal refusals, worker
wedged. A recovery note follows once a painting lands; a failed DB read is
treated as no observation, not as recovery. Dream carry pictures
(`brief.dream_hop`) write no chain row and never count. The threshold is
calibrated in-sample on 2026-09-14..10-10 (no gap fell between 7.6 h and
14.7 h); replay: `evals/test_painting_gap_replay.py`.

`visual_attempt_failure_streak` (same loop, 2026-10-20): how many painting
attempts in a row (`reverie_visual_attempt`) produced nothing. At
`ORION_VISUAL_ATTEMPT_FAILURE_STREAK_THRESHOLD` (6) it raises a **critical**
card (emailed at once), and a recovery note once an attempt produces an image.
2026-10-16 -> 10-20 had 35 in a row while the only alert was one "error" gap
card. 6 is one more than the longest ordinary streak on 2026-09-14..10-19;
every streak of 6+ was an outage. Replay: `evals/test_attempt_streak_replay.py`.
`ORION_VISUAL_ATTEMPT_STREAK_CHECK_ENABLED=false` turns it off (so does
`ORION_VISUAL_PAINTING_GAP_CHECK_ENABLED=false`: it runs in that loop). Attempts
still `active`/`unknown` (in progress) never count as failures.

## Acknowledged visual activity and baseline execution

`GET /visual-chain/activity` returns `VisualActivityV1`. `history_status=ok` with
no success means confirmed empty history; `unavailable` blocks baseline admission.
Activity counts acknowledged images, including caption failures and repeated image
bytes. Failed/refused/deadline-only rows do not advance it. New chains have an
explicit `production_receipt` key, initially null; a transaction verifies the stored
object and content-addressed artifact, then acknowledges that particular chain.
Historical rows use the stricter artifact/chain/hash/positive-bytes/path join.

The selected context retains its real settled text chain/thought/coalition, journal
entry, crystallization, or prior visual chain identity in `chain_json.context_selection`.
Only the selected reverie is credited with attention grounding. Rotation is not a
voluntary attention override. Excerpts remain within the existing visual artifact
boundary; dispatch receipts carry IDs, never source excerpts.

`POST /visual-chain/run-once` accepts optional `VisualRunRequestV1` (empty body is
compatible). It returns explicit `outcome`, `artifact_persisted`, and an
`execution_receipt`. `ran=True` alone is not production. Thermal policy is unchanged:
hot refuses, missing/stale readings allow degraded operation, and disabled is recorded.

Baseline policy lives in `config/reverie/visual_baseline.v1.yaml`, **disabled by
default**. With it enabled, all endpoint attempts use the durable
`reverie_visual_attempt` claim and a shared retry gap. Replays retain their result.
An unresolved active/unknown attempt has no automatic expiry: cancellation cannot
prove a blocking GPU thread stopped. A committed production receipt permits positive
reconciliation; otherwise operator investigation is required. Never clear a hold
merely because a deadline elapsed.

Before enabling: apply `manual_migration_reverie_visual_attempt.sql`, deploy compatible
policy/dispatch/feedback/SQL/exec readers, keep `ORION_VISUAL_CHAIN_ENABLED=false`, and
verify the complete diffusion/receipt/activity rail. The initial interval is 90 minutes,
retry gap 10 minutes; both remain ordinary scheduling parameters. No env template
changes are needed. See the implementation PR report for evidence and rollout order.

Database integration tests use a disposable PostgreSQL URL:
`ORION_VISUAL_TEST_DATABASE_URL=... python -m pytest services/orion-thought/tests/test_visual_activity.py -q`.
They create isolated schemas and never run by default against the operator database.

Continuity (2026-09-28 fix): the next run's `prior_description`, `continuity_streak`
and `context_slot_rotation` come from the latest chain row that recorded continuity
state. Thermal refusals, resource deferrals and deadline rows carry none and are
skipped; before this, one deferral reset all three to a cold start.

## Durable `reverie.visual` steps (orion-durable-runs)

orion-durable-runs can drive a visual reverie as a graph of stages instead of one
`run-once` call: `prepare -> generate -> caption`, plus `abandon` when the run gives
up. It sends `ReverieVisualStepRequestV1` on `orion:reverie:visual:step:request`
(`REVERIE_VISUAL_STEP_CHANNEL`); this worker replies on the request's `reply_to` with
kind `reverie.visual.step.result.v1` under the request envelope's correlation id.
Design: `docs/superpowers/specs/2026-09-28-visual-reverie-durable-graph-design.md`.
Code: `app/visual_steps.py`. `run-once` is unchanged and remains the rollback path;
both paths write identical chain rows through the same `visual_chain.py` pieces.

- **prepare** claims the attempt for the dispatch and freezes the prompt plan into
  `reverie_visual_attempt.stage_json`. Before a new claim only, a baseline request is
  checked structurally (validated at its own `observed_at`, so a run that waited out
  retries is never ended as stale); the claim's `already_satisfied` check is the real
  double-image guard. The legacy worker being on is a retry
  (`legacy_visual_worker_enabled`), not an ending. A replay returns the frozen plan:
  rotation does not advance and the context is not re-interpreted. No GPU.
- **generate** is the only GPU stage. It validates the run's diffusion hold
  (`holder == durable-runs:<run_id>`), then thermal gate, single-flight lock and
  diffusion under its own deadline (`ORION_VISUAL_CHAIN_STEP_GENERATE_DEADLINE_SEC`,
  default 330, never below `ORION_VISUAL_CHAIN_GPU_LEASE_DEADLINE_SEC` + diffusion timeout + 10).
  GPU pool stage 5.4: the diffusion call attaches a child lease under the validated hold (runs
  in the hold's slot, no second wait; no durable-runs `/capacity` permit). The child lives as
  long as the diffusion thread -- even past a cancelled step, for up to one more diffusion
  timeout -- so world-model cannot take gpu2 mid-render even if the run gives its hold back
  (`config/gpu_pool.yaml` `world.serialize_with: [diffusion]`). A generate outside a durable
  run (`/visual-chain/run-once`, the legacy worker) has no hold, so it takes a pool `diffusion`
  lease for the call; refused, late or pool unreachable is `resource_deferred:gpu_pool...`, never
  an ungated diffusion call. An unreachable diffusion-host is `resource_deferred:diffusion_unreachable`. The image is
  stored on disk and recorded in `stage_json`; a replay with a recorded image makes no
  diffusion call. A generate that started less than 2x the deadline ago blocks another
  (`generate_in_flight`): an abandoned diffusion thread may still be on the card.
  Refusals and failures are retries recorded in `stage_json.deferrals`, never chain
  rows (a row keyed by the attempt would make the production row a no-op). Recording
  a finished render is retried; if the stage store still refuses, the step is a
  `stage_store_unavailable` retry carrying the image's `artifact_sha256`, and the next
  generate in the same process adopts the verified file instead of re-rendering.
- **caption** reloads the image by sha (missing/corrupt -> `needs_generate`),
  re-observes it (cached, so a retry never recaptions), writes the production chain
  row with `chain_id == attempt_id`, acknowledges it, persists the execution receipt,
  and finishes the attempt `produced`.
- **abandon** closes the attempt. It may omit `attempt_id` (prepare's reply was
  lost): the attempt is resolved by `dispatch_id`, and no row means nothing to close.
  Result outcome is `unknown`; the row becomes `abandoned`, which does not block later
  claims. If a generate started inside the in-flight window the row stays `unknown`
  until that generate records its exit. A generate past its step deadline is not
  cancelled: it keeps the lock (inside the run's hold) until diffusion returns (hard ceiling:
  the in-flight window, then `generate_wedged`) and records its own exit. If the
  process died instead, the next claim (durable prepare or legacy run-once) releases
  the row once the window passed.

Backstop for an attempt nothing ever closed (lost abandon, crash mid-run): every
claim, durable or legacy, releases an `active`/`unknown` attempt claimed longer ago
than `ORION_VISUAL_CHAIN_ATTEMPT_MAX_AGE_SEC` (default 7200, above the durable 5400s
retry window) as `abandoned` with result reason `attempt_expired` -- unless a
production receipt reconciles it to `produced`, or a generate is recorded inside the
in-flight window.

Every result carries `elapsed_sec` (generate: diffusion + disk write).
Unexpected errors are retries (`step_exception:<Type>`); a missing stage table or
column is `stage_store_unavailable`, never a recompute. A step request that fails
schema validation is a retry (`invalid_step_request`), so schema skew during a
rolling deploy never ends an in-flight run.

Before routing runs here: apply `manual_migration_reverie_visual_attempt_stage.sql`
(after `manual_migration_reverie_visual_attempt.sql`). DB tests:
`ORION_VISUAL_TEST_DATABASE_URL=... python -m pytest services/orion-thought/tests/test_visual_steps_db.py -q`.


## Optional GPU2 elastic admission

GPU2 diffusion/agent-burst borrowing is additive and defaults off. See the
[ownership ADR](../../docs/architecture/gpu2-elastic-admission.md),
[pre-edit repository/live evidence](../../docs/architecture/gpu2-elastic-evidence.md),
and [consumer-first rollout and rollback](../../docs/runbooks/gpu2-elastic-admission.md)
for this service's exact flags, HTTP contracts and operator commands.
No production env sync, migration, GPU transition or deployment was performed.

## Body economy shadow bid (Patch C3b, 2026-10-27)

The `generate` step of a `reverie.visual` run (`app/visual_steps.py`) asks the body economy
whether it would fund the GPU work, then ignores the answer. The answer is only logged, next to
what the step really did. Nothing changes what the step does.

- Where: `generate_step`, after the replay, adoption and in-flight checks (those are not a new
  want) and before the thermal gate. `prepare`, `caption` and `abandon` never bid.
- Family: `thought.painting`, or `thought.dream_hop` for a dream-carry image hop. Verb `make`, tier 2.
- `want_id`: the generate step's correlation id (one per RPC). When this step opens the run's
  `make` act (its first real generation), that is the act's `grammar_acts.correlation_id`. A later
  generate of the same attempt (after a failed render) runs under the act an earlier step opened:
  its line carries `act_correlation_id=<that step's id>`, read from `stage_json.act_correlation_id`,
  which the opening write now stores beside `act_opened_at`.
- Real outcome: `ran` once the real gates (thermal, single-flight, stage) passed and the GPU call
  started, whatever the render's fate; else `blocked:<the step's reason>` (`thermal_refused`,
  `deferred_busy`, `stage_changed`, ...). A step that returns before its work decides (deadline,
  cancel) logs when the work ends, so a late start is not logged blocked.
- Volume: every generate retry is a new RPC and a new bid. A run held by the thermal gate bids once
  per retry (`_THERMAL_RETRY_AFTER_SEC`, 300 s). Group by `object_ref=durable_run:<run_id>` to count
  wants per painting, not per retry.
- A lost `act_opened` publish (not retried) leaves the act row built from its close, on a different
  correlation id, so that `ran` line joins nothing.
- Log line: `physiology_shadow_bid family=... want_id=... shadow=funded|deferred|none reason=...
  lane=... real=...`. durable-runs writes the decision into `want_ledger`. A funded decision whose
  step was blocked releases its reservation.
- Switch: `THOUGHT_PHYSIOLOGY_SHADOW_BID_ENABLED` (default true). The bid runs as a background task
  and never delays the step.
