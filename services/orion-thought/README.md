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

## Flow

```text
LISTEN orion:thought:request
  → validate StanceReactRequestV1
  → run_stance_react() — cortex verb stance_react (brain tier)
  → parse_stance_react_payload + apply_stance_react_pipeline
  → REPLY ThoughtEventV1 on reply_to
  → PUBLISH orion:thought:artifact
```

## Local checks

```bash
PYTHONPATH=services/orion-thought:. ./orion_dev/bin/python -m pytest services/orion-thought/tests/ -v

docker compose \
  --env-file .env \
  --env-file services/orion-thought/.env \
  -f services/orion-thought/docker-compose.yml config
```

## Health

`GET http://localhost:7155/health`

## Reverie semantic lift

Set `ORION_REVERIE_SEMANTIC_LIFT_ENABLED=true` (default `false`) to lift coalition
`harness_closure:{corr}` pointers into human `ConcernCardV1` text from
`substrate_turn_referent` before `reverie_narrate`. Requires referent rows from
unresolved post-turn closures and routes narration through the background/metacog lane.

## Reverie perception context

Set `ORION_REVERIE_PERCEPTION_ENABLED=true` (default `false`) to feed the most
recent `orion-vision-council` narrative(s) from the `vision_events` table into
the reverie prompt as ungrounded sensory context — reverie is otherwise 100%
blind to the camera. `ORION_REVERIE_PERCEPTION_MAX_AGE_SEC` (default `180`)
bounds staleness; `ORION_REVERIE_PERCEPTION_MAX_EVENTS` (default `3`) caps how
many recent narratives are included. Read-only (`app/vision_reader.py` never
writes `vision_events`), fail-open (any DB error degrades to no percepts, never
raises), and narrative-only — same privacy contract as the situation brief's
`PerceptionContextV1` (`orion/schemas/situation.py`): no entities, tags,
embeddings, or anything identity-bearing crosses into the prompt. Does **not**
widen `evidence_refs` grounding to vision ids; `SpontaneousThoughtV1`'s
anti-hollow guard stays coalition-only.

First, unverified cut of Movement III in
`docs/superpowers/specs/2026-08-12-perception-frontier-design.md` — the
prediction/outcome scorer that design doc also proposes is deliberately not
built yet ("event substrate first": get a real percept into the trace before
building a reducer that scores it). Leave this flag off until a live tick has
been eyeballed for whether the BLIP-base narratives actually add anything
worth narrating.

## Reverie expectation scoring (Movement III)

The second half of Movement III: score the falsifiable predictions perception
context makes possible, instead of only generating them. Requires
`services/orion-sql-db/manual_migration_substrate_reverie_thought_expectation.sql`
applied (`expectation`/`expectation_checkable_by`/`expectation_verdict`/
`expectation_scored_at` columns on `substrate_reverie_thought`).

When `ORION_REVERIE_PERCEPTION_ENABLED=true`, `reverie_narrate.j2` may
optionally have the LLM state one concrete, falsifiable `expectation` about
the room in `SpontaneousThoughtV1.expectation` — genuinely optional, never
forced every tick, capped at 200 chars. That alone is inert: nothing checks it
unless `ORION_REVERIE_EXPECTATION_SCORING_ENABLED=true` (default `false`).

With the flag on, two things happen:

- A tick that narrates a new expectation stamps `expectation_checkable_by =
  now() + ORION_REVERIE_EXPECTATION_CHECK_WINDOW_SEC` (default `1800`
  seconds — loosely paced off the perception-context staleness gate's `900`
  without being the same number, long enough for a plausible next percept).
- Every tick also spends **at most one** bounded judge-LLM call
  (`app/store.py::load_pending_expectations(limit=1)`) resolving the single
  most-overdue expectation whose window has already closed and that has no
  verdict yet:
  - No fresh-enough percept available (same `ORION_REVERIE_PERCEPTION_MAX_AGE_SEC`
    staleness gate as perception context, not a second constant) → writes
    `expectation_verdict="unscored"` directly, no LLM call.
  - A fresh percept is available → one call to the
    `reverie_expectation_judge` verb (`orion/cognition/prompts/
    reverie_expectation_judge.j2`), strict `{"verdict": "confirmed" |
    "disconfirmed" | "unscored"}` JSON, fail-closed to `unscored` on any
    ambiguity, LLM error, or parse failure. A pending expectation is never
    left permanently unresolved by a transient failure — it is stamped
    `unscored` instead of silently retried forever.

Scoring is fully independent of the current tick's own narration: it can
never block, delay, or fail narration, and narration succeeding or failing
has no bearing on whether scoring runs.

Query imagination accuracy directly (no dashboard yet — see the design doc's
explicit non-goal on this):

```sql
select expectation_verdict, count(*)
from substrate_reverie_thought
where expectation_verdict is not null
group by expectation_verdict;
```

## Resonance health monitor (Phase H+)

`ORION_REVERIE_RESONANCE_ALERT_ENABLED` (default `true`) already runs an observation-only
tripwire (`orion.reverie.resonance.detect_resonance`) after every completed reverie chain,
persisting a `ResonanceAlertV1` to `substrate_reverie_resonance_alert` whenever a theme
re-ignites faster than its own refractory bound allows. Until now that alert reached
Postgres and a Hub debug panel only — nobody got paged.

`app/resonance_monitor.py` closes that gap the same way the field-digester /
attention-runtime / self-state-runtime health monitors already do (all three merged,
running in production): an edge-triggered check that pages via `orion-notify`'s
`POST /attention/request` (surfacing in Hub's existing Pending Attention panel) only on
a healthy→unhealthy transition, retries a failed delivery until it actually succeeds, and
checks `orion-notify`'s own pending list before suppressing a first-observation alert (so
a process restart mid-incident can't go permanently silent).

**"Unhealthy" here is not "an alert exists."** A 2026-07-12 investigation confirmed a real
historical resonance burst had already self-resolved by the time it was investigated, but
the detector kept re-reporting the same old `violation_count`/`min_gap_sec` for ~20 hours
afterward, because those old rows hadn't yet aged out of `detect_resonance`'s 200-row
lookback window (`ORION_REVERIE_RESONANCE_WINDOW`). Paging on "an alert exists" would have
paged for ~20 hours about an already-resolved problem. Instead, a theme is only considered
unhealthy when its `violation_count` strictly **increases** across its last 2 persisted
samples — i.e. the loop is actually getting worse right now, not just echoing history.

Reuses `NOTIFY_BASE_URL`/`NOTIFY_API_TOKEN` (same values as the other three services'
health monitors) — no new settings beyond those two.

## Attention salience (GWT-coalition Borda rank-aggregation)

`orion/substrate/attention/salience.py` computes chat-level/open-loop salience by
Borda rank-aggregating two real, evidence-derived signals (`evidence_strength`,
`evidence_breadth`) across the loops competing in one tick -- Global Workspace
Theory / Society-of-Mind coalition formation (Baars 1988, Dehaene 2014), the
same theory anchor already live for Layer 5's Candidate B. `score_loop()`/
`derive_salience()` always read this precomputed `loop.salience` -- there is
exactly one formula, no flag selects between formulas anymore.

2026-07-31 (kill means kill, see `orion/sentience_striving_program/README.md`'s
2026-07-31 entry): the prior hand-picked `SEED_WEIGHTS` linear blend
(`recency`/`recurrence`/`dwell`/`novelty_vs_known`/`habituation`, none with a
real theory anchor) was killed with nothing put back. `ORION_ATTENTION_
HABITUATION_ENABLED` and `ORION_ATTENTION_SALIENCE_WEIGHTS` (JSON weight
override) were removed entirely -- no habituation term or combiner weights
exist to gate/override anymore. **Named, disclosed gap**: habituation was the
only automatic repeat-suppression mechanism in this scoring path; a loop with
strong evidence that nobody has explicitly resolved/dismissed can now re-win
coalition attention indefinitely.

`ORION_ATTENTION_SALIENCE_V2_ENABLED` (default-off) is the one remaining flag
here, narrowed to a single purpose: whether the reverie tick emits
`AttentionSalienceTraceV1` on `orion:attention:salience:trace` (persisted to
`attention_salience_trace`) at all. It no longer selects a salience formula.

Migrations (apply before enabling):
`psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_attention_salience_trace.sql`

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
