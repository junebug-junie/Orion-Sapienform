# Retro Play P0 preflight

Implementation source: [design PR #2751](https://github.com/junebug-junie/Orion-Sapienform/pull/2751).
This patch implements the **offline measurement tool**, not the complete Retro Play
proposal. P0 remains **UNVERIFIED** until actual core/game runs pass. P1–P6
(service authority, Hub A/V/control, motor, cognition, training and transfer) are
not implemented. Do not deploy this module as a service.

## Ready-made local diagnostic setup

No existing ROMs or cores are required for the authored NES diagnostic. Follow
[Local NES diagnostic kit](retro-local-setup.md) to build a pinned core, generate
the original ROM and run it in a restricted container. The instructions below
also support separately supplied content.

## Operator inputs and execution

Supply an operator-owned NES/SNES ROM and a trusted libretro ABI-v1 shared library
(FCEUmm/Snes9x are candidates, not yet validated or selected dependencies). Record
its upstream source revision, build flags and license alongside the result. The
CLI requires exact expected SHA-256 values before loading either input. No ROM,
core binary, save state or gameplay image belongs in Git. The authored C test
fixture is deliberately **not an emulator** and proves no game compatibility.

Native code executes with the caller's privileges. The Python probe is **not a
sandbox**. Run it in a disposable unprivileged environment with no credentials,
no network, read-only core/ROM mounts and resource limits. Do not mount Docker's
socket or production volumes. A crash/hang must be contained by that environment;
use an external timeout. Sandbox packaging and malformed-ROM containment are
still required before P1. There are no remote endpoints here.

From the checkout, with Python 3.11+:

```bash
timeout 40m python3 -m orion.retro.preflight \
  --core /inputs/pinned-core.so --core-sha256 <expected-core-sha256> \
  --rom /inputs/operator-game.nes --rom-sha256 <expected-rom-sha256> \
  --frames 110000 --replay-frames 1000 --output /evidence/baseline.json
```

The output path must not exist. Input hash mismatch exits 2 before native loading.
Exit 0 means this baseline's measured gates passed; exit 1 means incomplete/failed
or a caught runtime error. Native crashes or external timeouts may leave an empty
result: those are failed runs, never evidence of success. A short `--frames 120`
run is a wiring smoke and must not pass the 30-minute gate. No hostname, paths,
ROM bytes, images or save states are deliberately included in the JSON.

Run this separately under idle, ordinary Orion load, CPU contention, an ordinary
chat turn, and GPU queue/swap conditions; label conditions in private evidence
metadata. Do not induce production contention without operator authorization.
Host CPU count/affinity/load are context, not measurements of *free physical*
cores or thermal headroom. Hardware inventory, alternative runner comparison,
core revision selection and the load matrix remain P0 work.

## What the probe measures

`CoreProbe` installs actual native callbacks for video, PCM and independent
joypads. Unsupported environment requests return false and are reported. Legacy
core options take their first/default choice; modern options, VFS, hardware
rendering, BIOS directories and runtime option changes are unsupported. A core
needing them must fail validation, not be declared compatible by assumption.

`benchmark` warms up 120 frames, then uses the core's reported nominal FPS with
absolute monotonic deadlines. Late frames are counted, not silently dropped.
The measurement includes video hashing overhead. Audio counts prove callbacks,
not audible/correct PCM or synchronization. The first 1,000-frame continuation runs uninterrupted after a snapshot; the
second restores that snapshot and issues the same deterministic independent
port masks. Both rolling video hashes and final serialized state must match.
No state is saved to disk and no cross-core compatibility is implied. Distinct
input values do not establish that a game supports two players.

The PASS verdict covers only cadence ±1%, deadline misses <1%, some A/V output,
matching replay, ≥30 minutes of nominal frames and ≥1,000 replay frames. It does
not pass P0 in full or any later milestone. Unknowns are explicit in the report.

## Measurement quality gate

1. Provenance: `orion/retro/probe.py:CoreProbe._video`, `_audio`, `_audio_batch`
   count native callbacks; `orion/retro/preflight.py:benchmark` captures
   `perf_counter` around `retro_run` and compares against absolute frame deadlines.
2. Dependence: actual FPS, elapsed time and deadline misses share the same clock;
   median/p95 share the same durations. They are redundant timing views, not
   independent cognitive signals. Replay state/video hashes share one core run.
3. Anchor: periodic-task deadline accounting and deterministic state-machine
   replay. Audio callback count establishes delivery only. No agency, learning,
   affect or sentience detector is introduced.
4. Live sanity: **UNVERIFIED**. Synthetic gates include absent A/V, overload and
   nondeterministic replay. Ideal paced timing has zero missed deadlines. There
   is no pause/rest classification and no unknown-to-zero cognitive mapping.
5. Existing mechanism: no retro emulator interface was found; existing bus and
   health metrics remain untouched.
6. Reversibility: local JSON reports only; no bus/schema registry, database,
   training defaults or cognition wiring. Do not promote measurements into a
   cognition pipeline before the live quality gate passes.

## Existing integration owners to revisit after P0

- Selection: `orion/substrate/endogenous_curiosity.py:endogenous_curiosity_candidates`
  produces curiosity candidates; it is not an unattended-play grant authority.
- Admission: `services/orion-durable-runs/app/admission_runtime.py` and
  `app/admitted_graph.py` own admitted workflows/holds. Retro leisure is not in
  their existing workflow contract; do not bypass it with a new scheduler.
- Artifact candidate: `orion/schemas/mind/artifact.py:MindRunArtifactV1` is produced
  by `services/orion-cortex-orch/app/mind_runtime.py` and
  `services/orion-thought/app/mind_enrichment.py`, consumed by
  `services/orion-sql-writer/app/worker.py` as `MindRunSQL`. It represents Mind
  runs, not a verified generic gameplay-memory store. A lesson producer and
  retrieval consumer still need explicit contracts.
- Hub: `services/orion-hub/scripts/main.py` and `api_routes.py` expose the app and
  operator routes. An authenticated session-owner/CSRF/origin/WS signaling path
  has **not** been established for Retro. P2 must resolve this prerequisite.

## Checks and next gate

```bash
python3 -m pytest tests/retro -q
python3 -m orion.retro.preflight --help
```

Tests require pytest and `cc`; the tool itself uses only the standard library.
The native fixture proves ABI callback wiring, separate input effects and
restore; simulated timing tests prove fail-closed verdicts. Neither substitutes
for the real-ROM eval command above. GitHub CI runs these checks.

Next required inputs: pinned trusted core/build provenance, first legal
single-player ROM, genuine two-player ROM, and isolated execution environment.
Then collect the actual P0 baseline and resolve the runner choice before P1.
No environment templates changed, no ORION_BUS_URL used, no restart required.
Rollback removes this offline tool; it cannot affect running Orion services.
