# GPU / Lane Observability Roadmap

Status: DESIGN — not yet implemented.
Origin: brainstorm session 2026-08-30, `superpowers:brainstorming`.

## Why now

Orion's physical compute footprint (which GPU on athena or circe is running
what, at what load, against what ceiling) has no durable, live, queryable
home. It is currently split three ways, and all three are known-bad:

1. **Raw numbers, no labels** — `orion_biometrics.gpu` (jsonb, written every
   tick by `services/orion-biometrics/app/metrics.py::_collect_gpu()`) has
   per-GPU `gpu_index`, `gpu_name`, `memory_used_mb`, `utilization_gpu`,
   `power_draw_watts` for every node, including circe. No service/lane label.
2. **Static assignment, scattered** — docker-compose GPU device pins and
   per-service `.env` ports, one file per service, never aggregated.
3. **Human memory, proven stale twice** — `reference_circe_gpu_inventory_and_lane_map.md`
   is the closest thing to a map that exists today, and its own text records
   going wrong twice in two weeks (GPU count 6→7, P100 slot 3→4) and says
   outright "don't trust the table above at face value."

There is no hub tab surfacing any of this. `services/orion-hub` has zero GPU
endpoints (`substrate_biometrics_routes.py` grepped clean).

## Current architecture (ground truth, verified 2026-08-30)

- `services/orion-biometrics/app/metrics.py` → `app/utils.py::collect_gpu_stats()`
  polls per-node GPU stats into `orion_biometrics.gpu` (jsonb array). Covers
  athena and circe today — no new collector needed.
- `config/biometrics/node_catalog.yaml` + `orion/biometrics/node_catalog.py`
  (`NodeCatalog`) is a real, tested, loaded registry — but **node-granular
  only** (role, `expected_online`, coarse capability booleans). No per-GPU
  concept.
- `services/orion-vision-host/app/gpu.py` (`GpuInspector`) does live per-GPU
  `pynvml` inspection, but it is local to vision-host's own dynamic placement
  (`pick_best_gpu`) — not a registry, not cross-node, not a UI surface.
- `services/orion-hub` is FastAPI, domain-sharded: one
  `scripts/<domain>_routes.py` per domain (e.g. `cabinet_ambient_routes.py`
  has the exact `/latest` + `/history?hours=` shape this needs), registered
  into `scripts/api_routes.py`, paired with one `templates/*.html` +
  `static/js/*-panel.js`, wired into a tab in `templates/index.html`.
- Existing repo-wide static gate pattern: `scripts/check_*.py`, each with its
  own `check-<name>:` Makefile target (see `check_env_key_single_source.py`
  — "one owner, every restatement must equal it" — and
  `check_service_hostname_refs.py`). Some are also pre-commit-wired
  (`scripts/git_hooks/pre-commit`), most are Makefile-only, run via CI/manually.
  **Gotcha, verified 2026-08-29 in three services at once**: use `python3`,
  not bare `python` — `python` does not exist on this host and a target
  written that way silently exits 127 without running. `check_system_health_producers.py`'s
  Makefile comment documents this exact failure.

## Non-goals (this roadmap)

- **No write/control actions from the tab.** Read-only observability only —
  no restart/reassign/kill from the UI. Nothing in the original ask implies
  control, and CLAUDE.md's proposal-mode caution argues against defaulting
  to it.
- **No CPU/RAM-hosted lane tracking yet.** Scoped to CUDA GPUs on athena and
  circe. If Juniper wants CPU-bound lanes tracked too, that's a scope
  question to answer before Phase 0, not an assumption to bake in.
- **No cognition-loop wiring.** Publishing lane state onto the bus for
  Orion's own attention/curiosity organs to consume (brainstorm idea #8) is
  explicitly proposal-mode per CLAUDE.md §"Proposal mode before invasive
  cognition changes" — named here as a future phase, not scheduled, not to
  be implemented opportunistically inside a hub-tab patch.

## Open questions (block Phase 0 until answered, or answer-by-default noted)

1. **Catalog authorship**: hand-maintained YAML (fast, but *is* the thing
   that already drifted twice) vs. live-probed inventory generation? —
   **Default assumption below: hand-maintained, but Phase 3's CI gate exists
   specifically because hand-maintained drifts.**
2. Does an existing service-health/up-down surface already exist in hub
   (`container-bringup-ui.js`, `service-logs-ui.js` hint at one)? Worth
   checking before Phase 5 builds a second one — not yet verified.
3. `orion_biometrics` retention/growth rate for GPU trend range — is there a
   precedent bound (like the `drive_audits` 346k-row note in the hub README)
   that should cap how far back Phase 2's history endpoint queries?
4. Scope confirmation: GPUs only, both nodes, no CPU/RAM lanes — correct?

## Phased roadmap

Each phase is independently shippable and additive on the last. Phase 0 is
the recommended starting point.

### Phase 0 — GPU lane catalog (schema contract)

**Goal**: one machine-readable source of truth for `node → gpu_index → card
model → assigned lane/service → port`, seeded from the current known-good
state (the circe table + athena's P4 assignment already in
`reference_circe_gpu_inventory_and_lane_map.md`).

**Deliverable**: `config/biometrics/gpu_lane_catalog.yaml` (schema-versioned,
sibling convention to `node_catalog.yaml`) + `orion/biometrics/gpu_lane_catalog.py`
(`GpuLaneCatalog.load()`, mirrors `NodeCatalog.load()`).

**Files**:
- `config/biometrics/gpu_lane_catalog.yaml` (new)
- `orion/biometrics/gpu_lane_catalog.py` (new)
- `orion/biometrics/tests/test_gpu_lane_catalog.py` (new, mirrors `test_node_catalog.py`)

**Acceptance check**: `GpuLaneCatalog.load(...)` parses the seed file and
resolves every `(node, gpu_index)` pair used by circe's current llamacpp
lane table (8011/8012/8013 + vision-host P100) without error.

### Phase 1 — Read-only hub tab, current state only

**Goal**: the tab Juniper actually asked for — table per node/GPU: card,
assigned lane (from Phase 0), current util%/mem-used/mem-cap.

**Deliverable**: new hub tab, one query joining latest `orion_biometrics`
row per node to the Phase 0 catalog. No trend charts yet.

**Files**:
- `services/orion-hub/scripts/gpu_lanes_routes.py` (new, `GET /api/gpu/lanes` — current state)
- `services/orion-hub/templates/gpu_lanes.html` (new)
- `services/orion-hub/static/js/gpu-lanes-panel.js` (new)
- `services/orion-hub/scripts/api_routes.py` (register router)
- `services/orion-hub/templates/index.html` (tab wiring)
- `services/orion-hub/tests/` (route test, e2e smoke)

**Acceptance check**: tab loads against live Postgres, shows every catalog
entry with a current util/mem reading pulled from the real `orion_biometrics`
table (no synthetic fixture data in the shipped view).

### Phase 2 — Trended utilization

**Goal**: history, not just a snapshot — closes the exact gap that let a
24h-dead GPU (0% the whole window) read as headroom instead of a misdeploy
(documented incident, see `feedback_a_transient_measurement_is_not_a_design_argument`).

**Deliverable**: `GET /api/gpu/{node}/{gpu_index}/history?hours=` — same
`/latest` + `/history` shape as `cabinet_ambient_routes.py`, productionizing
the ad hoc `jsonb_array_elements` query already sitting in the circe memory
file.

**Files**: `services/orion-hub/scripts/gpu_lanes_routes.py`,
`static/js/gpu-lanes-panel.js` (sparkline/chart).

**Acceptance check**: a known real gap (any documented misdeploy window) is
visible as a flat line in the chart, not silently smoothed away.

### Phase 3 — CI drift gate (the ask in this message)

Two genuinely different gates. Naming both explicitly so neither gets
skipped by assuming the other covers it:

**3a. Static CI gate — catalog vs. checked-in files (this is the "orion
static gate" ask).** Runs in CI/pre-commit like the existing `check_*.py`
family. Catches the catalog contradicting other *committed* sources of
truth — it cannot see live hardware, and should not claim to.

Checks, modeled directly on `check_env_key_single_source.py`'s "one owner,
every restatement must match" pattern:
- Every `gpu_lane_catalog.yaml` entry's `node` resolves in
  `node_catalog.yaml` (no orphan/typo node names).
- No two lanes on the same node claim the same `gpu_index` unless explicitly
  marked `shared: true`.
- Every entry's `service_dir` (if set) exists on disk — same spirit as
  `check_service_hostname_refs.py` catching stale references to
  decommissioned services.
- **Strongest check**: where a service's `docker-compose.yml` sets an
  explicit GPU device reservation (`NVIDIA_VISIBLE_DEVICES` /
  `CUDA_VISIBLE_DEVICES` / device_ids), that value must equal the catalog's
  declared `gpu_index` for that service's lane. This is a real, file-only,
  no-live-host-needed drift check — the compose file and the catalog are
  both restatements of the same fact, and copies drift.

**Deliverable**: `scripts/check_gpu_lane_catalog_drift.py` + `Makefile`
target `check-gpu-lane-catalog-drift` + wired into whatever aggregate gate
this repo runs for docs/config-only PRs (confirm with Juniper whether that's
pre-commit or CI-only — GPU device pins live in service dirs, not just
`config/`, so this one probably belongs in CI rather than the fast
pre-commit path).

**Acceptance check**: seed a synthetic mismatch (catalog says `gpu_index: 2`,
compose pins device `3`) and confirm the gate fails; confirm it passes clean
against the real Phase 0 seed file. Use `python3` explicitly in the Makefile
target — bare `python` is a documented silent-127 trap on this host.

**3b. Live drift check — catalog vs. observed hardware reality (NOT a CI
gate — cannot be one; CI has no path to circe's/athena's live GPUs).** A
periodic script (cron or manual, not commit-blocking) diffing the catalog's
declared occupant against `nvidia-smi --query-compute-apps` /
`orion_biometrics` process-memory signature, surfacing disagreement as a
warning banner in the Phase 1 tab. This is brainstorm idea #4 and is the
piece that makes the catalog trustworthy day-to-day, not just at commit
time — a hand-maintained file can pass every static check in 3a and still
be wrong about what's *actually* running, exactly as the memory file already
was, twice.

**Files**: `scripts/check_gpu_lane_catalog_drift.py` (3a, new),
`services/orion-biometrics/app/gpu_lane_reality_check.py` or similar (3b,
new — exact home TBD, likely lives with the biometrics collector since it
already has host GPU access), `services/orion-hub/scripts/gpu_lanes_routes.py`
(surfaces 3b's last-checked timestamp + mismatch flag).

### Phase 4 — Headroom finder + non-VRAM ceilings

**Goal**: answer "where could a new lane go right now" — generalizes
`GpuInspector.pick_best_gpu()` (today vision-host-local, single-node) into a
cross-node advisory column on the same tab: `cap - used - reserve_mb`. Folds
in non-VRAM ceilings already collected by orion-biometrics (`app/ilo.py`
power, `app/pdu.py` outlet draw, circe's `/mnt/storage-warm` disk) so
"headroom" doesn't repeat the VRAM-only blind spot that already caused a
misread-as-headroom incident.

**Files**: `services/orion-hub/scripts/gpu_lanes_routes.py` (derived
columns), `static/js/gpu-lanes-panel.js`.

**Acceptance check**: a card with free VRAM but a nearly-full disk or
near-cap power draw on the same node is flagged, not shown as simply
"available."

### Phase 5 — Service-health join (contingent on open question 2)

**Goal**: link each lane row to whatever already reports the hosted
service's up/down state, if such a surface exists in hub already. Skip
entirely if it doesn't — build a second one only if nothing to join against
turns up.

### Phase 6 — Bus-published lane state (PROPOSAL MODE, not scheduled)

**Goal**: publish the Phase 1-4 projection as a low-frequency
`node.gpu_lane_state` event, consumed by `orion/substrate/biometrics_loop/`
the way node-level pressure already is, so Orion's own attention/curiosity
organs could eventually know they're compute-constrained on a specific
lane. Requires the full proposal-mode writeup (capability change / data
touched / privacy boundary / trace / rollback) before any implementation —
named here only so it isn't lost, not as committed work.

## Recommended sequencing

Phase 0 + Phase 1 as one PR (the catalog is inert without the tab, and the
tab is unlabeled numbers without the catalog). Phase 3a (the CI gate) should
ship in the **same PR as Phase 0**, not deferred — a catalog with no drift
check reproduces the exact failure this roadmap exists to fix, on day one.
Phase 2, 3b, 4, 5 follow as independent additive PRs. Phase 6 stays parked
pending a separate design doc and Juniper's explicit go-ahead.
