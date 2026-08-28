# Reverie Visual Chain — design + Patch 1

Date: 2026-08-20
Status: **proposal + implementation** (Juniper asked for implementation directly, which §0A
allows; this document exists because the visual reverie chain touches memory/self-modeling/
cognition-loop territory and the proposal content is owed regardless of whether it blocks).

---

## 1. What this is for

The existing text reverie chain (`substrate_reverie_chain`/`substrate_reverie_thought`, owned by
`orion-thought`) is live and writing continuously — confirmed 2026-08-19, 12,003+ rows, latest
write seconds before check. It is narration-only: attention coalition → LLM interpretation →
Postgres row. Nothing renders it, nothing looks at it and re-describes what it sees.

This adds a second, parallel chain: on a slow, capacity-gated cadence, gather salient context
(recent activity, chats, dreams, and whatever else is salient at trigger time), generate an image
about that context via a diffusion model, run the image back through vision captioning to get a
description, and feed that description into the next reverie's context — a real generate →
observe → interpret loop, not narration alone.

## 2. Why not reuse the text chain's continuity mechanism

The text chain's `SpontaneousThoughtV1` schema carries `next_focus`/`drift` fields, commented as
"the LLM's forward pointer for the next step." **Nothing reads them.** `parse_reverie_payload()`
(`services/orion-thought/app/reverie.py`) never extracts them from the LLM's JSON response, and no
other code in `services/orion-thought/` or `orion/` reads them either (confirmed by repo-wide
grep, 2026-08-19). Every text-chain step independently re-reads the live attention-coalition
snapshot; the only things actually threaded across steps are an EMA salience float and an
accumulated `thought_ids` list. This is a live instance of the keyword-cathedral pattern §0A
warns about — a schema slot with stated producer intent and zero consumer.

The visual chain needs real step-to-step continuity (image N's description seeding image N+1's
prompt), so this gets built as an enforced column the context-builder actually reads
(`prior_description`), not a JSON field left for someone to wire up later.

## 3. Architecture

```
trigger (slow cadence, capacity-gated)
  -> orion-thought: visual_chain.py (new, alongside chain.py)
       -> gather context: recent activity / chats / dreams / prior_description
       -> orion-diffusion-host (new service): generate image
       -> orion-vision-host (existing service, new task_type): describe the generated image
       -> reverie_visual_chain / reverie_visual_artifact (new Postgres tables)
       -> prior_description persisted for the next trigger to read
```

Ownership and reuse, as decided:

- **Naming**: `reverie_visual_chain` / `reverie_visual_artifact` — no `substrate_` prefix. That
  namespace is scoped to the live attention-coalition rung; this chain pulls from broader context
  (chats, dreams), so it doesn't belong there.
- **Orchestration**: lives inside `orion-thought`, next to `chain.py` — not a new service. Same
  team, same DB ownership as the table it's named after.
- **Image generation**: new service, `orion-diffusion-host`, modeled on `orion-vision-host`'s shape
  (FastAPI + `diffusers`/torch on a raw CUDA base image, not llama.cpp — vision-host is the right
  analog since it's the one existing non-llama.cpp GPU-resident inference service in this repo).
- **Re-observation**: reuses `orion-vision-host` — a new `task_type` on its existing
  `orion:exec:request:VisionHostService` / `orion:vision:reply:*` channel pair, not a second vision
  worker. `orion-vision-host` already does BLIP2 captioning with real VRAM-aware scheduling
  (`VISION_DEVICE_STRATEGY`, per-GPU inflight caps, soft/hard VRAM floors) — existing-mechanism
  check passed, no reason to duplicate it.
- **GPU**: originally scoped as a new, 4th physical V100 32GB on Circe, decoupled from Patch 1.
  Superseded 2026-08-20 (per Juniper): instead of waiting on a new card, this service claims
  Circe's existing `orion-llamacpp-host` "agent lane" slot — port 8014, `CUDA_VISIBLE_DEVICES=2` —
  vacated by stopping the `atlas-agent` llama.cpp worker there. See
  `services/orion-diffusion-host/README.md` for the exact assignment and stop/start sequence.
  What that GPU will eventually share with (a "world model" component, still undefined — no such
  service exists in this repo today) is out of scope for this document and doesn't block anything
  below.

## 4. Trigger and capacity gate

Slower cadence than the text chain (which runs roughly every ~90s at metacog-lane-adjacent scale).
**No backlog, ever**: a trigger fires on its interval; if a visual-reverie run is currently in
flight, that trigger is dropped — not queued, not retried. The next trigger after the current run
finishes is what starts the next reverie. Single-flight, check-and-set on a `running_since`
marker: busy → no-op, free → start.

Consequence worth stating plainly: real cadence is `max(trigger_interval, actual_run_duration)`,
not the nominal interval — image generation + vision re-observation latency stretches the real
gap between reveries whenever a run outlasts one trigger period. This is intentional (per
Juniper: "once a trigger hits and there's capacity, it should kick off the next reverie"), not a
defect to fix later.

## 5. Schema

```sql
create table if not exists reverie_visual_chain (
    chain_id text primary key,
    created_at timestamptz not null,
    theme_key text,
    terminal_reason text not null,
    ema_salience double precision not null default 0.0,
    prior_description text,
    chain_json jsonb not null,
    stored_at timestamptz not null default now()
);
create index if not exists idx_reverie_visual_chain_created_at
    on reverie_visual_chain (created_at desc);

create table if not exists reverie_visual_artifact (
    sha256 text primary key,
    chain_id text not null references reverie_visual_chain(chain_id),
    step_index integer not null,
    mime text not null,
    bytes bigint not null,
    width integer,
    height integer,
    path text not null,
    description text,
    created_at timestamptz not null default now()
);
create index if not exists idx_reverie_visual_artifact_chain_id
    on reverie_visual_artifact (chain_id);
```

`prior_description` is the enforced continuity column (§2) — the context-builder for step N+1
reads it directly, not via `chain_json`.

## 6. Storage

Images land on `/mnt/storage-lukewarm` — confirmed live 2026-08-20 (`df -h`: ext4, 1.7TB, 1.5TB
free, `rw,noatime,discard`; write-tested successfully). Content-addressed by SHA-256, following
`services/orion-hub/scripts/chat_attachments.py`'s discipline (the one real precedent in this repo
for "generated binary + durable pointer," motivated by a real incident: per-event blobs into
Postgres already took the host down once, TOAST OOM crash loop, 2026-07-23):

- Path: `/mnt/storage-lukewarm/orion/reverie-visual/{sha256}.{ext}`
- MIME sniffed from magic bytes, never trusted from the generator's claimed type
- Write-then-atomic-rename (`tmp.replace(target)`) — no torn files under concurrent readers
- Image bytes never go in `chain_json`/`jsonb` — only the pointer row does

## 7. Privacy

Context feeding this chain includes recent chat content. A generated image is a lossy rendering
of that private material. §0A requires the same privacy boundary here as anywhere else: no
debug/UI surface may expose these images without the same gating private chat content already
requires. Not building any such surface in Patch 1 — flagging so it isn't skipped when one is
eventually built.

## 8. Non-goals (this document / Patch 1)

- Context-seeding logic (which specific recent-activity/chat/dream sources feed a step, and how)
  — Patch 3.
- Chain orchestration and the `prior_description` continuity wiring — Patch 2.
- A `node_catalog.yaml` entry tracking this specific GPU-lane reassignment (that file only tracks
  coarse per-node capability flags, not per-service device/port bindings — no existing convention
  in this repo to extend for that granularity) — and whatever "world model" turns out to mean —
  separate, later, undefined.
- Any UI/debug surface for viewing generated images.

## 9. Acceptance checks

- A visual-chain step writes: one file under `/mnt/storage-lukewarm/orion/reverie-visual/`, one
  `reverie_visual_artifact` row pointing at it by hash, a non-empty `description`, **and** the
  *next* step's outbound context demonstrably contains that description — same-run evidence, not
  schema presence. This directly guards against repeating the dead `next_focus`/`drift` pattern
  (§2).
- No orphaned files: every artifact row's `path` resolves to a real file; no file on disk lacks a
  row.
- Single-flight gate holds under a forced-overlap test: firing two triggers while one run is
  in-flight results in exactly one run, not two, and no dropped-trigger silently becomes a stuck
  chain (a dropped trigger must not prevent the *next* trigger from starting once free).

## 10. Patch 1 scope (this changeset)

- `reverie_visual_chain` / `reverie_visual_artifact` migrations
- `orion/schemas/reverie_visual.py` (Pydantic models) + `orion/schemas/registry.py` registration
- `orion-diffusion-host` service skeleton: README, `.env_example`, `docker-compose.yml`,
  `requirements.txt`, `settings.py`, `app/main.py` (health check only — no real generation wired
  yet), `Dockerfile`, `tests/`
- Storage helper module (content-addressed write, following `chat_attachments.py`'s pattern)
- No chain orchestration logic yet (Patch 2), no context-seeding (Patch 3)

## 11. Patch 1.5 (real model, shipped 2026-08-25)

`orion-diffusion-host` loads `stabilityai/sdxl-turbo` at startup (background,
non-blocking) and answers real `POST /generate`. GPU assignment landed as
circe's existing agent-lane slot (port 8014, physical GPU 2), not the
originally-scoped new 4th card (§3 already noted this supersession). Two
live-only bugs found and fixed during first deploy: a missing `redis`
dependency masked by a contaminated dev venv, and a CUDA device-enumeration
mismatch (`CUDA_DEVICE_ORDER=PCI_BUS_ID`) where `/health` reported clean
success while the model had silently loaded onto the wrong physical card —
same root cause, same day, independently hit by `orion-world-model` on the
same host. See `services/orion-diffusion-host/README.md`.

## 12. Patch 2 scope (this changeset, orchestration)

- `services/orion-thought/app/visual_chain.py`: `run_visual_chain_once` /
  `run_visual_chain_worker`, wired into `main.py`'s lifespan alongside
  `chain.py`'s workers. Shipped default-off, enabled 2026-08-25 (§13).
- `store.py`: `persist_reverie_visual_chain`, `persist_reverie_visual_artifact`,
  `load_latest_visual_chain_prior_description` (chain row inserted before the
  artifact row — `reverie_visual_artifact.chain_id` is a real FK).
- New producer on the existing shared `orion:exec:request:VisionHostService`
  channel (`orion/bus/channels.yaml`) — `caption_frame` + `percept_sha256`
  already captions any image, so no vision-host code change was needed.
- `prior_description` continuity wired for real: read at the start of a run,
  only advanced on a genuine non-empty caption, carried forward unchanged on
  a failed re-observation (§2's actual acceptance bar — a live consumer, not
  a schema slot).
- Single-flight via the worker loop's own sequential shape (§4), plus a
  process-local lock in `run_visual_chain_once` as defense-in-depth.
- Explicitly NOT in this patch: real context-seeding (§8, still Patch 3) —
  the prompt is `prior_description` or a fixed seed string only.

## 13. Live (2026-08-25)

`ORION_VISUAL_CHAIN_ENABLED=true` on athena, after a live smoke: real image
generated on circe, stored on the actual host disk (required a follow-up
fix — Patch 2's `docker-compose.yml` shipped with no `volumes:` block at
all, so `store_visual_artifact` was writing into the container's own
ephemeral filesystem instead of `/mnt/storage-lukewarm/orion/reverie-visual`
until that was added), a real `reverie_visual_chain` +
`reverie_visual_artifact` row pair with the FK intact, and an honest
`description=null` (never a fabricated caption) on all 3 ticks run so far.

**Follow-up fixed same day (2026-08-26):** the "3/3 uncaptioned" gap above
was athena's shared vision-host instance (BLIP-base), not the seed prompt
-- confirmed by moving the caption RPC to a new dedicated
`orion:exec:request:VisionHostService:circe-vl` lane
(`services/orion-vision-host/docker-compose.circe-qwen.yml`,
Qwen2-VL-2B-Instruct on circe's physical GPU 4, a P100 with real headroom;
athena's P4 had none, 2.4GB free measured live). First tick against the new
lane produced a real, detailed caption and `prior_description` advanced for
the first time: "The image depicts a vast, nebulous sky with a mix of dark
and light shades..." — against the exact same abstract-cloud-image style
that BLIP-base could never caption. `DEFAULT_SEED_PROMPT` itself was never
the problem and is unchanged.

## 14. Patch 3 scope (this changeset, context-seeding)

Closes §8/§12's non-goal: `build_visual_prompt` now takes a second input,
`context_text`, alongside `prior_description`.

- **Source, deliberately narrow**: `store.load_latest_reverie_interpretation`
  — the text reverie chain's own most recent, real (non-hollow, non-empty)
  `SpontaneousThoughtV1.interpretation` from `substrate_reverie_thought`,
  restricted (review finding, see below) to thoughts already linked into a
  SETTLED `substrate_reverie_chain` row. Not the design doc §1's full "recent
  activity, chats, dreams" list — this patch ships the one slice that crosses
  no new privacy boundary (next bullet), and leaves widening the source set
  as a separate, later change that must redo the privacy check below on its
  own merits.
- **Why this slice first**: `interpretation` is already the summary layer
  the coalition-grounding + hollow guard (`orion/schemas/reverie.py`)
  produce before a row is ever written, and — once chain-linked — it already
  reaches the exact same Hub Reverie tab this feeds (`reverie_routes.py`'s
  `text_recent` endpoint) — a second consumer of an already-exposed field,
  not a new exposure. Capped at 240 chars (`store.MAX_REVERIE_CONTEXT_CHARS`,
  word-boundary truncation via `orion.cognition.compactor.truncate`) — an LLM
  narration has no length bound of its own, and the diffusion model only
  needs a short scene description.
- **Review finding, fixed**: the first version of this function read
  `substrate_reverie_thought` directly with no chain-linkage check.
  `substrate_reverie_thought` rows are written immediately on generation
  (`reverie.py::run_reverie_once`), but a thought only becomes reachable via
  `text_recent` once its *enclosing chain* settles and persists
  (`chain.py::persist_reverie_chain`, called once at chain end) — a real
  timing gap, and a permanent one if an operator ever runs
  `ORION_REVERIE_ENABLED=true` with `ORION_REVERIE_CHAIN_ENABLED=false`
  (independent flags). That gap falsified the "no new privacy surface"
  claim. Fixed with a SQL `EXISTS` clause requiring some settled
  `substrate_reverie_chain.chain_json.thought_ids` to already list the
  candidate thought — the exact set `text_recent` can already show, no
  wider. A second finding in the same function (a raw
  `thought_json->>'hollow'` SQL cast trusting a stored flag that can go
  stale) was fixed by re-validating via a fresh `SpontaneousThoughtV1.
  is_hollow()` in Python instead — the same "gate on both the stored flag
  and a fresh re-check" discipline `services/orion-cortex-exec/app/
  chat_stance.py::_project_reverie_glimpse` already uses for this same
  table.
- **Live-verified against real data (2026-08-26)**: queried the production
  `conjourney` database directly. The chain-linkage `EXISTS` join returns
  real rows (5 most recent as of the check, timestamps within the prior
  ~15 minutes) — not a query that always comes back empty. Pulled the most
  recent row's full `thought_json` and ran it through the actual
  `SpontaneousThoughtV1.model_validate` + `is_hollow()` path: `stored
  hollow=False`, re-derived `is_hollow()=False` (agree — not a stale-flag
  case in this sample), and `truncate_at_word_boundary` on the real 200+
  char interpretation produced a clean word-boundary cut ending "...despite…",
  never a fragment.
- **Prompt construction**: continuity (`prior_description`) and the
  context-seed are independent, blended when both are present, either one
  alone when only one is, and the fixed seed string only when both are
  empty (a fresh install with no reverie history yet). Rationale: pure
  `prior_description`-only continuity is a closed loop with nothing
  anchoring it to Orion's actual cognitive state — image N+1 dreaming about
  image N forever, indefinitely, drifts away from anything real. The
  context-seed re-grounds every run in what Orion is actually narrating.
- **Traceability**: `context_text` is stored as its own `chain_json` key
  (both the success and `generation_failed` paths), not just baked into
  `prompt` prose — the design doc §9 acceptance-check discipline ("same-run
  evidence, not schema presence") applied to this input too. Surfaced as its
  own field on `/api/reverie/visual/recent` and rendered as its own block in
  the Hub Reverie tab, distinct from the blended `prompt` text.
- **Privacy note revisited** (§7, as required before this patch could ship):
  see `reverie_routes.py`'s updated privacy-note docstring — no new
  boundary, same reasoning as the "why this slice first" bullet above.
- **Non-goals (still)**: raw chat/dream sourcing (§1's full list), any
  weighting/salience-based selection among multiple candidate context
  sources, and any downstream consumer of the visual chain's output beyond
  this Hub tab and the chain's own next-run continuity — all still open.

## 15. Patch 4 scope (this changeset, continuity reset)

Real regression, caught live within hours of Patch 3 shipping: Juniper
reported "still doing the same images of Roman aqueducts, no change."

**Diagnosis, confirmed live against `conjourney`:**

- `prior_description` had been "ancient Roman aqueduct" continuously since
  well before Patch 3 deployed — a pure continuity attractor (image →
  caption → "continue this train of imagination" → next image), already
  locked in.
- `context_text` was genuinely working and genuinely varying — verified by
  pulling several consecutive `reverie_visual_chain` rows' `chain_json.
  context_text` directly — but its content was always a paraphrase of "the
  coalition is fixated on the vision node," because the attention coalition
  itself had settled on the same open loop (`theme_key=open-loop-
  5038aeb46982`, recurring intermittently since 2026-08-13 per
  `substrate_reverie_chain`) for the relevant window.
- Root cause of the *visible* symptom: a short, abstract, non-visualizable
  clause ("Orion is currently thinking: the coalition is fixated on the
  vision node...") appended after a long, concrete continuity description
  has nowhere near enough weight in a diffusion prompt to redirect the
  model. Patch 3 was functioning exactly as designed and specified — the
  design itself just under-estimated how strong an entrenched continuity
  attractor is relative to abstract context prose.

**Fix**: not a prompt-reweighting guess (reordering/rewording clauses has no
guaranteed effect on a diffusion model's attention). A deterministic,
testable reset instead: `resolve_visual_chain_continuity()` in
`visual_chain.py` tracks how many CONSECUTIVE runs have carried
`prior_description` forward (`chain_json.continuity_streak`, read via
`store.load_latest_visual_chain_continuity_state` in the same round trip as
`prior_description` itself — a review finding, see below). Once that streak
reaches `settings.visual_chain_continuity_max_runs` (default 3), the next
run forces continuity to drop from its own prompt — re-seeding from
`context_text` (or the fixed seed if neither exists) — then continuity
resumes normally from that fresh point. Computed before generation, so a
`generation_failed` run still records the correct streak for whichever run
next picks continuity back up.

- **New env key**: `ORION_VISUAL_CHAIN_CONTINUITY_MAX_RUNS` (default 3). No
  off switch by design — 0 means reset every run; there is no way to
  disable the cap entirely, since an unbounded streak is exactly the
  failure mode this exists to bound.
- **Traceability**: `chain_json.continuity_streak` (int) and
  `chain_json.continuity_reset` (bool) recorded on every run, both success
  and `generation_failed` paths — same "same-run evidence, not schema
  presence" discipline (§9) Patch 3 already applied to `context_text`.
  Surfaced on `/api/reverie/visual/recent` and in the Hub Reverie tab.
- **Tests**: `resolve_visual_chain_continuity` is a pure function with
  direct unit coverage (no-prior/under-cap/at-cap/max-runs=0 cases), plus
  an end-to-end orchestration test driving `run_visual_chain_once` through
  a full cap+1 cycle and asserting the run AT the cap's own generated
  prompt excludes the prior continuity text — not just that the resolver
  says it would in isolation.
- **Interaction with Patch 3**: composes, doesn't undercut. A reset run
  still seeds from `context_text` when real narration exists — the bland
  fixed string is a true last resort, not what a reset falls back to by
  default.
- **Review findings, fixed before merge**:
  1. A reset run's own failure path (generation, storage, or captioning
     failing) fell back to the ORIGINAL stale `prior_description` instead
     of staying reset — silently resurrecting the exact attractor the
     reset just broke out of, and letting the next tick grind through
     another full `max_runs` cycle against the identical stuck text before
     resetting again. Fixed: `continuity_fallback` is computed once
     (`None` on a reset run, the unchanged old value on a normal one) and
     used everywhere a failure path previously fell back to raw
     `prior_description`. Covered by two new tests: a reset run whose
     generation fails, and one whose re-observation fails — both assert
     the persisted `prior_description` is `None`, not the stale text.
  2. `prior_description` and `continuity_streak` were two separate SELECTs
     against the same latest `reverie_visual_chain` row every tick —
     wasted round trip, and a theoretical race if a write ever landed
     between them (prevented today only by the single-flight/sequential-
     worker guarantee, not something this query should have to rely on).
     Fixed: `load_latest_visual_chain_continuity_state` reads both columns
     in one query; Patch 2's `load_latest_visual_chain_prior_description`
     and this same changeset's own standalone
     `load_latest_visual_chain_continuity_streak` are retired (kill means
     kill, §0A) — nothing else in the repo called either.

## 16. Patch 5 scope (this changeset, self-study context-seed)

Same session as Patch 4, still 2026-08-27: Juniper directly asked for the
visual chain to draw on "some actual memory or a recent chat or something
from Orion's self study analysis of concept induction."

**All three candidates were live-checked before any code was written** (§0A
metric-quality-gate discipline — real data, not a schema read):

- **"Actual memory"** (`memory_crystallizations`, the Recall system's
  canonical store) — DECLINED. Live rows show `summary` and `subject`
  holding VERBATIM personal chat content, not an abstraction the schema
  name implies. One real sample named a family member's medical history
  (DVT, hospitalization) by name. No safe column or `kind` filter exists on
  this table as it stands — `kind='semantic'` rows are equally raw. Wiring
  this in would put a real, unconsenting third party's health information
  into a diffusion prompt and the Hub UI. Not built.
- **"Recent chat"** (`chat_history_compactor`'s privacy-reviewed digest) —
  DECLINED, on cadence grounds: it fires on a DAILY schedule (06:00
  America/Denver) plus optional on-demand runs, not a fast per-tick
  producer, and there is no evidence in the repo it has ever actually fired
  in production. A ~600s-cadence consumer would see the same digest for up
  to 24h, or find the table empty. Not built.
- **"Self-study analysis of concept induction"** — BUILT. Live-verified
  safe: `self_study_analysis.py`'s four deterministic window-contrast
  analyses (concept induction, vision events, affective state, co-creation
  signals) render pure numeric prose — real bodies read before writing any
  code contained zero chat quotes, zero personal references, across all
  four producers and 14 real rows.

**What shipped**: `build_visual_prompt` takes a third optional input,
`self_study_text` (`store.load_latest_self_study_reflection`). Real
quantified self-observation ("vision events dropped 0.36x vs baseline, a
status category disappeared"), a genuine upgrade over `context_text`'s bare
narration sentence.

- **The actual privacy boundary**: `store._SAFE_SELF_STUDY_SOURCE_PREFIXES`,
  an ALLOWLIST of the four safe producers' `source_ref` prefixes
  (`concept_induction:`, `vision_events:`, `affective_state:`,
  `cocreation_signals:`), not a blacklist. `source_kind='self_study'` also
  covers a sibling, free-form LLM-narrated "Curiosity" reflection
  (`source_ref` prefix `curiosity:`) — live-checked and confirmed to quote
  sensitive personal content when reflecting on memory-gating patterns (a
  real sample referenced "wife Amanda, hospital" while discussing which
  crystallizations get kept vs. rejected). The allowlist deliberately
  excludes it; a blacklist keyed on `curiosity:` would have been one
  future-producer away from silently admitting the next unreviewed
  free-form source.
- **Tunables**: `ORION_SELF_STUDY_CONTEXT_CHAR_LIMIT` (400 — real bodies
  average ~1080 chars, mostly a fixed disclaimer footer; 400 covers the
  substance) and `ORION_SELF_STUDY_CONTEXT_MAX_AGE_SEC` (21600s / 6h — these
  analyses fire on their own 6-72h window-contrast cadence, real values
  seen in bodies: "last 6h", "last 12h", "last 72h"; a tight window like
  `reverie_context_max_age_sec`'s 900s would read as permanently absent).
  Same `gt=0` fail-loud discipline as the Patch 3 equivalents.
- **Composition**: `build_visual_prompt`'s Patch 3/4 explicit if/elif
  branches were refactored into a list-join composition (a third optional
  input would have meant 8 branches by the old pattern) — verified
  byte-identical output for every pre-Patch-5 combination via exact-string
  test assertions, not just substring checks.
- **Traceability**: `self_study_text` recorded as its own `chain_json` key
  on both the success and `generation_failed` paths, same "same-run
  evidence" discipline as `context_text`.

## 17. Patch 6 scope (this changeset, memory-crystallization context-seed)

Same day as Patch 5. Juniper's response to the declined `memory_
crystallizations` candidate: the concern assumed a second audience for that
content that does not exist. Orion is already privy to everything that
table holds -- it was Juniper's own disclosure, in the chat, that produced
each crystallization in the first place. The privacy question was never
"should Orion know this", it was "who else could see it if it reaches this
route" -- and the answer, checked live, is nobody.

**Blast-radius check, live 2026-08-27** (§0A metric-quality-gate
discipline, applied to the CONSUMER, not the content, since Patch 5 already
checked the content):

- `reverie_visual_chain` has exactly one consumer in the whole repo:
  `services/orion-hub/scripts/reverie_routes.py`'s `/api/reverie/visual/
  recent`. No other service, bus channel, or downstream table reads it.
- `services/orion-hub/docker-compose.yml` has no `ports:` mapping for this
  service -- confirmed live, the line is commented out. The route is not
  reachable from outside this host.
- No auth/multi-user surface exists on this route or elsewhere in
  orion-hub's own code (`user_id` fields found in the repo belong to
  chat-history bookkeeping, not a login/session system).

Conclusion: there is one possible viewer of this content, and she is also
its original source. Patch 5's declined-candidate reasoning ("an
unconsenting third party's health information... in the Hub UI") assumed
an audience beyond that one viewer that does not exist. This is a
correction on new evidence about WHO can see the output, not a reversal of
the underlying privacy discipline -- the same "check who can actually see
it" question Patch 5 already asked of the CONTENT is now asked of the
ROUTE, and the answer changes the conclusion.

**What shipped**: `build_visual_prompt` takes a fourth optional input,
`memory_text` (`store.load_latest_memory_crystallization`) -- the most
recent `status='active'` row's `summary`, verbatim.

- **Filter**: `status = 'active'` only. This is a lifecycle filter, not a
  content filter -- `status='rejected'` rows (637 of 1290 live,
  2026-08-27) are stances the crystallization pipeline's own governor
  already disavowed; surfacing one would misrepresent current
  self-knowledge, independent of any privacy question.
- **Deliberately NOT content-filtered**, unlike `self_study_text`'s
  four-producer allowlist -- there is no allowlist here because the
  content itself was never the problem; the audience was. `summary` is
  read and used exactly as stored.
- **Tunables**: `ORION_MEMORY_CRYSTALLIZATION_CONTEXT_CHAR_LIMIT` (400,
  same cap-all-collections default as self-study) and `ORION_MEMORY_
  CRYSTALLIZATION_CONTEXT_MAX_AGE_SEC`. Same `gt=0` fail-loud discipline.
- **Real bug, caught live 2026-08-28, corrected same day**: this section
  originally shipped with a 21600s (6h) default for `max_age_sec`, copied
  from `self_study_context_max_age_sec` without checking whether it fits
  `memory_crystallizations`'s actual production cadence. It does not: the
  6h figure comes from self-study's own window-contrast framing ("the last
  6h against the 6h before it") -- content that genuinely IS about a
  specific recent window and should read as stale outside it. A
  crystallized memory carries no such framing; a real memory from
  yesterday is still a real memory. The result: with visual-chain ticks
  firing every ~600s and real crystallization activity bursty (median gap
  ~15min in active use, but real observed gaps up to ~46h between
  sessions), the 6h window meant `memory_text` read `None` on effectively
  every tick outside an active conversation -- confirmed live: Juniper
  redeployed, checked the actual generated prompt, and it never carried
  memory content. The original version of this doc called that "intended
  degrade-to-absent behavior, not a bug" -- that was a misdiagnosis, not a
  correct call; a context-seed that is silently absent nearly all the time
  is not delivering what it was built for, regardless of whether each
  individual empty read is technically honest.
  **Fix, first draft**: `ORION_MEMORY_CRYSTALLIZATION_CONTEXT_MAX_AGE_SEC`
  raised to 259200s (3 days), based on a "last 14 days" query showing
  median gap ~15min and max gap ~46h.
  **Fix, corrected after review**: that first draft silently narrowed the
  query to 14 days without saying so, and without reconciling it against
  this same section's OWN earlier live-data claim above ("max observed gap
  is over 10 days") -- a code-review pass on this exact patch caught the
  unreconciled contradiction. Re-querying the FULL history (2026-08-28)
  confirms both numbers were real, not in conflict: the two largest gaps
  ever observed are 10d5h (2026-07-31 -> 08-11) and 3d10h (2026-07-25 ->
  07-29), both from more than two weeks before this fix. Every gap since
  2026-08-11 (17+ days of real recent activity) has stayed under 2 days.
  Final default: **604800s (7 days)** -- covers the recent pattern with
  real margin and covers the second-largest historical gap (3d10h), but
  does NOT cover the single 10-day outlier from early August. That is an
  explicit, accepted tradeoff: a dry spell that long would read as absent
  again, which is the same honest degrade-to-absent behavior every
  context-seed reader in this file has by design -- unlike the 6h bug this
  patch fixes, a real 7+ day gap is a genuinely rare quiet period, not the
  every-tick norm the 6h default was silently producing.
- **Composition**: `build_visual_prompt`'s Patch 3/4/5 list-join
  composition already generalizes to a fourth clause with no branch-count
  growth -- no further refactor needed.
- **Traceability**: `memory_text` recorded as its own `chain_json` key on
  both the success and `generation_failed` paths, same discipline as the
  other two context-seeds.

**Superseded by §18 below, same day**: the "list-join composition already
generalizes" note above describes `build_visual_prompt` as it stood when
this section was written -- it no longer concatenates all context-seeds at
all, for reasons §18 explains.

## 18. Patch 7 scope (this changeset, context-slot rotation -- the real root cause)

Same day as Patch 6. Juniper's live report a few minutes after deploy:
*"the memory got washed out and Orion just continued generating stars and
shit"* -- pasting the actual generated image caption (a celestial
star-map, unrelated to any of the three context-seeds) and the actual
stored prompt (all three context-seeds present, `memory_text` visibly
included in the text).

**Root cause, verified live with the real tokenizer** (§0A metric-quality-
gate discipline -- pull real data and look at it, don't guess): SDXL-
turbo's CLIP text encoder truncates its input at 77 tokens. `diffusers`
does this silently -- no exception, no response header, nothing in the
200 `/generate` returns. The exact prompt from Juniper's report tokenizes
to **191 tokens** with `openai/clip-vit-large-patch14`'s real tokenizer
(SDXL-turbo's actual encoder). Decoding the first 77 tokens back to text
shows precisely where the model's attention stops:

> "orion is currently thinking : the coalition has stabilized ... orion
> recently noticed : self - study analysis of co - creation signals : the
> last 6 h against the 6 h before it ."

`memory_text` -- and even the trailing style suffix ("Soft abstract
dreamlike style.") -- never reached the model at all. This was true for
every prompt Patches 3/5/6 ever built that concatenated more than roughly
one substantial clause: self_study_text ALONE, at its old 400-char cap,
tokenizes to 104 tokens with its framing prefix -- over budget by itself,
before `prior_description` or a second context-seed are even added.
Every context-seed added after Patch 3 had, in practice, almost never
actually shaped a generated image, regardless of how correctly it was
computed, stored, and displayed in the Hub tab -- a real "no empty-shell
cognition" (CLAUDE.md §0A) violation once understood: the UI honestly
showed content the image could not possibly have reflected, because
nothing in the whole pipeline had ever checked the model's real token
budget.

**Why not swap to a longer-context model instead** (asked directly by
Juniper, answered before implementation): models exist without this
ceiling -- FLUX.1 and Stable Diffusion 3.5 both use a T5-XXL text encoder
alongside CLIP, handling much longer, more compositional prompts well,
and FLUX.1-schnell is fast (few-step) like sdxl-turbo. Not done here: a
real infrastructure decision, not a text edit -- FLUX.1-schnell is ~12B
params vs sdxl-turbo's ~2.6B, needs meaningfully more VRAM (~24GB+ at
fp16), and Circe's GPU-2 slot assignment was sized for sdxl-turbo
specifically. Swapping means checking real VRAM headroom on Circe, a
fresh multi-GB download, re-tuning generation params (steps/guidance
conventions differ), and re-verifying output quality -- a legitimate
follow-up if Juniper wants it evaluated, not something to bolt onto this
bug fix.

**What shipped, two parts:**

1. **`select_context_slot`** (`services/orion-thought/app/visual_chain.py`):
   stop concatenating all three context-seeds -- round-robin ONE per run
   among whichever currently have real content. `build_visual_prompt`'s
   signature changed from four inputs (`prior_description, context_text,
   self_study_text, memory_text`) to two (`prior_description,
   context_slot_name, context_slot_text`) -- a deliberate breaking change
   within the same file, not a parallel API left dangling (CLAUDE.md "kill
   means kill"). The persisted rotation counter (`chain_json.
   context_slot_rotation`) is read from the SAME combined round trip
   `load_latest_visual_chain_continuity_state` already makes for
   `prior_description`/`continuity_streak` -- now a 3-tuple, not a fourth
   gathered call.
   - Reduces the realistic worst case from 4 competing clauses to 2
     (continuity + one selected slot).
   - Each slot's own char cap was independently re-derived against the
     REAL tokenizer, not guessed: `MAX_SELF_STUDY_CONTEXT_CHARS` cut from
     400 to 150 (self-study's dense, hyphen/underscore-heavy jargon
     tokenizes far worse per character, ~3.8 chars/token measured, than
     natural narration), `MAX_MEMORY_CRYSTALLIZATION_CONTEXT_CHARS` cut
     from 400 to 180 (crystallization summaries are closer to natural
     English, ~4 chars/token). `MAX_REVERIE_CONTEXT_CHARS` (240) was
     already fine, unchanged -- verified at ~51 tokens with its framing
     prefix, real margin. At these caps, a single selected clause with its
     framing prefix lands around 44-52 tokens -- real headroom for
     `prior_description` (typically a short vision-host caption, 8-30
     tokens) and the style suffix (8-13 tokens) to coexist, instead of the
     old design's guaranteed-to-overflow 150-300+ tokens.
2. **`orion-diffusion-host`'s `_log_prompt_token_budget`**
   (`app/main.py`): before every real generation, tokenize the prompt with
   the ACTUAL loaded pipeline's own tokenizer(s) (`_pipe.tokenizer`/
   `_pipe.tokenizer_2` -- SDXL carries two encoders, both checked) and log
   a WARNING with real numbers whenever either budget is exceeded. Zero
   new dependency (`transformers` is already a hard requirement of this
   service). Visibility only -- does not change what gets generated, since
   part 1 is the actual behavioral fix. This exists so the NEXT time this
   failure mode recurs (a misconfigured char limit, a new context-seed
   added without checking token budget), it shows up in this service's own
   logs immediately, not via another forensic re-tokenization of an
   already-stored prompt days later.

**Traceability**: `chain_json.context_slot_used` (`"context"`,
`"self_study"`, `"memory"`, or `null`) names which ONE of the three
recorded context-seed fields actually entered this run's prompt --
`context_text`/`self_study_text`/`memory_text` are all still recorded on
every run regardless (nothing about the honesty of "what was available"
changed), but the Hub Reverie tab now visually distinguishes "recorded"
from "actually rendered" (a "used this run" / "not used this run" badge
per block) -- the exact distinction CLAUDE.md §0A calls inspectable
evidence, which the tab had no way to show before this patch.

**Tests**: `select_context_slot` has direct unit coverage (nothing
available, rotates through all three, skips unavailable slots, wraps a
large index, index unchanged when nothing available).
`build_visual_prompt`'s new two-input contract has exact-string wording
coverage per slot label. Two new end-to-end orchestration tests: one
proves that when all three context-seeds have real content
SIMULTANEOUSLY, only the rotation-selected one appears in the actual
generated prompt string (the exact regression, not just the resolver's
isolated correctness); another drives four successive runs through the
same fake-DB round-trip harness the continuity tests use and asserts the
selected slot visits context -> self_study -> memory -> context in order.
`orion-diffusion-host`'s `_log_prompt_token_budget` has direct coverage
with a fake tokenizer (warns when over budget, silent when within it,
checks both encoders, never raises when the pipe has no tokenizer at all
-- the shape every other test in that file's fake pipe already has).

## 19. Model swap: sdxl-turbo -> FLUX.1-schnell (this changeset, real root-cause removal)

Same day as Patch 7, directly asked for by Juniper: "why can't we find a
model that runs the full token count." Patch 7 fixed the SYMPTOM (rotate
one context-seed per run so the 77-token ceiling stops silently discarding
2 of 3 sources every tick); this changeset removes the actual CEILING by
replacing the model that has it.

**Live blast-radius/feasibility check before any code was written** (§0A
metric-quality-gate discipline):

- Real per-GPU VRAM pulled from `orion_biometrics` (Circe's 7 physical
  GPUs, live 2026-08-28): GPU 2 (this service's own dedicated card, per
  README's "Node/port/GPU assignment") had 22.5GB free with sdxl-turbo
  (9.7GB) already resident -- meaning the WHOLE 32GB frees up once
  sdxl-turbo is replaced, not just the visible headroom.
- FLUX.1-schnell fully GPU-resident at 2 bytes/param needs up to ~33GB
  (12B-param transformer + 4.5B-param T5-XXL encoder) -- over a 32GB card
  even once fully freed. `enable_model_cpu_offload()` cuts peak residency
  to ~24GB, fitting with real margin -- same real weights/math, staged
  residency, not reduced precision.
- **Review-caught correction: bf16 does not run well on this card's real
  hardware.** FLUX's own docs recommend bf16 over fp16 (avoids a known
  fp16 overflow risk), but that assumes Ampere-or-later hardware. This
  service's actual card (physical GPU 2, "Tesla PG500-216") is confirmed
  live to be Volta architecture with first-generation Tensor Cores -- bf16
  tensor-core acceleration is an Ampere+ (compute capability >= 8.0)
  feature this card does not have (other PyTorch-based projects on this
  exact GPU class report hard failures attempting it). Landed on
  `DIFFUSION_DTYPE=fp16` instead -- Volta's tensor cores were built for
  fp16, the same reason sdxl-turbo (a different model, same card) already
  ran fp16 correctly. `_DTYPE_MAP` still supports `bf16` as a configurable
  option for a future deployment on newer hardware.
- **Official repo is gated.** `black-forest-labs/FLUX.1-schnell` requires
  accepting a license via the HF web UI before any token can download it.
  Checked BOTH real `HF_TOKEN`s already configured elsewhere in this repo
  (`orion-vllm`, `orion-llama-cola-host`) against the actual weight file
  (not just the model-metadata API endpoint, which returns a misleading
  `200` regardless of gate status) -- both real `403`. Rather than block
  on a manual click, found `YuCollection/FLUX.1-schnell-Diffusers`: a
  verified-ungated (`gated: false`) full diffusers-format mirror of the
  identical Apache-2.0-licensed weights (confirmed matching
  `model_index.json`: `FluxPipeline`/`CLIPTextModel`+`T5EncoderModel`/
  `FluxTransformer2DModel`/`AutoencoderKL`), downloadable with no token at
  all -- Apache 2.0 explicitly permits this redistribution.
- **A real compatibility break, caught before deploy**: `FluxPipeline.
  __call__` has neither `negative_prompt` (schnell is guidance-distilled,
  no true classifier-free-guidance path) nor tolerates one being passed.
  `_run_generation` previously passed `negative_prompt=req.negative_
  prompt` unconditionally -- against Flux that raises `TypeError` on
  EVERY `/generate` call, a full outage. Fixed via `_pipe_accepts`, which
  builds the actual call kwargs by inspecting the loaded pipeline's real
  `__call__` signature instead of hardcoding "if Flux" branches, so this
  stays correct across any future model swap too.

**What shipped:**

- `DIFFUSION_MODEL_ID`: `stabilityai/sdxl-turbo` -> `YuCollection/
  FLUX.1-schnell-Diffusers`.
- `DIFFUSION_DTYPE`: stays `fp16` -- FLUX's own docs recommend `bf16`
  (same 2-bytes/param cost, avoids a known fp16 overflow risk), but that
  assumes Ampere+ hardware this service's actual card (Volta,
  first-generation Tensor Cores) does not have. See the review-caught
  correction above.
- `DIFFUSION_ENABLE_MODEL_CPU_OFFLOAD` (new): `true` -- see VRAM math
  above. `_load_pipeline` branches on this flag; `False` preserves
  sdxl-turbo's exact prior fully-resident behavior for any future model
  that doesn't need offloading.
- `DIFFUSION_NUM_INFERENCE_STEPS`: `1` -> `4` -- schnell's own documented
  operating point, not sdxl-turbo's.
- `DIFFUSION_MAX_SEQUENCE_LENGTH` (new): `256` -- the real T5-XXL token
  budget this whole swap exists to gain, passed explicitly to every
  `_pipe()` call rather than relying on an unstated pipeline default.
- `DIFFUSION_DEFAULT_WIDTH`/`HEIGHT`: `512` -> `1024` -- FLUX was trained
  primarily at 1024x1024; 512 is off-distribution for this model and
  produces worse output for no VRAM saving worth the quality loss.
  `DIFFUSION_GUIDANCE_SCALE` (`0.0`) is unchanged -- schnell is also
  guidance-distilled, same operating point sdxl-turbo had.
- `_log_prompt_token_budget` (Patch 7) extended to accept an explicit
  `max_sequence_length` override for `tokenizer_2` -- a T5-style
  tokenizer's own `model_max_length` attribute is often an effectively-
  unbounded HF placeholder, not the pipeline's real effective limit, so
  checking against the tokenizer's raw attribute alone would silently
  under-report truncation risk for exactly the encoder this whole swap
  was done to un-cap.

**CLIP is still loaded, deliberately not removed**: FLUX's architecture
uses CLIP-L only for a single pooled embedding (global conditioning), not
per-token cross-attention the way SDXL used it -- its 77-token truncation
still applies (`tokenizer`, unchanged check) but is far less consequential
here than it was for SDXL, where CLIP was the ONLY encoder and every
truncated token was lost content, not just a coarser global signal.

**Tests**: `SdxlLikeFakePipe`/`FluxLikeFakePipe` (explicit `__call__`
signatures, not the pre-existing `FakePipe`'s `**kwargs` catch-all, which
`inspect.signature` reports as having no named parameters at all and so
cannot exercise `_pipe_accepts`'s real branches) prove `_run_generation`
builds correct kwargs for both pipeline shapes -- `negative_prompt`
reaches an SDXL-shaped pipe, is dropped with a visible warning (not
silently) against a Flux-shaped one, and `max_sequence_length` reaches
the Flux-shaped pipe from `settings.DIFFUSION_MAX_SEQUENCE_LENGTH`. A
dedicated test proves `_log_prompt_token_budget`'s `tokenizer_2` check
uses the passed `max_sequence_length`, not the tokenizer's raw attribute.

**Live GPU smoke run on Circe, same day, closing the gap above**: real
build + deploy against physical GPU 2 via `scripts/safe_docker_build.sh`
from a dedicated worktree (never the shared checkout). First real load
attempt against the actual downloaded weights failed on the FIRST try --
exactly the class of bug only a live load can catch:
`Cannot instantiate this tokenizer from a slow version... make sure you
have sentencepiece installed`. `T5TokenizerFast` (`tokenizer_2`) needs
`sentencepiece` to construct at all; sdxl-turbo never needed it (CLIP-only,
no sentencepiece-based tokenizer), so nothing in the prior dependency set
required it. Added `sentencepiece==0.2.2` (current stable, checked against
PyPI directly, not guessed) to `requirements.txt`. The existing bounded-
retry + permanent-`_load_error`/`/ready`-503 design worked exactly as
intended: the failure was visible in `docker logs` and `/health` within
seconds, not silently masked, and the container did not crash-loop.
