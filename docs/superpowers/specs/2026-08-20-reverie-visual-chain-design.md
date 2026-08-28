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
  **Fix**: `ORION_MEMORY_CRYSTALLIZATION_CONTEXT_MAX_AGE_SEC` raised to
  259200s (3 days) -- comfortable margin over the ~46h real max gap
  observed in the last 14 days of live data, without being unbounded.
- **Composition**: `build_visual_prompt`'s Patch 3/4/5 list-join
  composition already generalizes to a fourth clause with no branch-count
  growth -- no further refactor needed.
- **Traceability**: `memory_text` recorded as its own `chain_json` key on
  both the success and `generation_failed` paths, same discipline as the
  other two context-seeds.
