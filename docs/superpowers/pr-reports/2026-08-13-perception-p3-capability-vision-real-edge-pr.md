# PR report — perception P3: give `capability:vision` a real edge

Implements **P3** of the perception frontier design (PR #1590), following P0
(PR #1602). P4 (`PerceptionContextV1`) is not in this patch.

## Summary

- `capability:vision` was declared in the field topology with **no inbound edge
  at all**. Its live vector was fabricated: `pressure=0.0, confidence=1.0,
  available_capacity=1.0`, provenance empty. Orion's self-model asserted perfect
  vision whether or not a camera was running.
- Added `node:substrate.vision`, written by a substrate-runtime tick, and the
  `node_capability` edge that fills it.
- **The signal was designed twice.** The first version derived vision health
  from the bus synaptic graph's `gap_zscore` (EWMA over message inter-arrival
  gaps). Juniper questioned whether EWMA was the right instrument. It was not,
  and it was deleted rather than tuned.
- The shipped version reads the detector's own output and uses **no EWMA
  anywhere** — a bus listener feeding a time-triggered tick.
- Root-caused a channel-loss bug that presented as an encoder fault and was not:
  the dynamics tick was clobbering the new channels every 30s.

## Outcome moved

Live on `orion-athena-substrate-runtime`, 82 ticks, zero failures:

```text
substrate_vision_channel_tick_completed age_sec=4.3 staleness=0.000 yield=6.97 samples=60
```

`node:substrate.vision` in FalkorDB carries `perception_staleness=0`,
`perception_yield=6.966667` — matching the 6–8 objects/frame measured by hand
against the live camera.

## Why EWMA was the wrong instrument

Recorded because the reasoning generalises past this node.

`gap_zscore` measures **inter-arrival regularity** of bus messages. Two
problems, neither fixable by a threshold:

1. **It z-scores a metronome.** The vision pipeline's cadence is set by a fixed
   scheduler (`config/vision_frame_router.yaml`'s
   `min_seconds_between_tasks_per_camera: 5`), so the quantity being
   z-scored is near-constant by construction. It measures scheduler jitter, not
   perception.
2. **It is structurally blind to a blinded camera.** Measured, not argued —
   posting synthetic frames to the live vision host:

   | frame | result | objects |
   | --- | --- | --- |
   | pure black | `ok=True` | **0** |
   | flat grey | `ok=True` | **0** |
   | live cam0 | `ok=True` | **6** (max score 0.72) |

   A capped lens, a dark room, or a frozen stream all produce perfectly
   regular, perfectly successful bus traffic carrying no information. Cadence
   cannot see that. Yield sees it instantly.

A third property made it unsafe rather than merely weak:
`orion-bus-mirror`'s `compute_ewma_update` runs **once per observed message**.
When a channel goes silent the function is never called, so `gap_zscore` does
not rise — it **freezes at its last healthy value**. A dead camera would have
read as calm, reproducing the exact falsehood this edge exists to delete, and
matching the `node:substrate.route` stale-read-as-calm incident.

Deleted per "kill means kill": `vision_channel_prediction_error` and its edge
query are gone, not left excluded from one consumer.

## Architecture shipped

```text
orion:vision:artifacts  --(listener)-->  60-artifact window + last-seen clock
                                              |
                                     (time-triggered tick, 30s)
                                              |
                                   node:substrate.vision
                                     perception_staleness  -> capability:vision.pressure
                                     perception_yield      -> recorded, NOT mapped
```

**Time-triggered on purpose.** An event-triggered statistic is never recomputed
once the events stop, which is precisely why the EWMA version froze during an
outage. A clock-driven tick reads "nothing for 90s" correctly because it does
not need an event to run.

**`perception_yield` is deliberately not wired to pressure.** A sustained zero
is equally consistent with a blinded eye and an empty dark room at 3am.
Separating those needs the per-stream day-shape prior from Movement II of the
design doc. Wiring it now would fire nightly on a working camera — the
false-positive twin of the fabricated `confidence=1.0` this node exists to
delete. `perceptual_blindness_pressure` is implemented and tested but left
unwired for the same reason.

**Channel choice is measured, not assumed** (60s pubsub census, 2026-08-13):

| channel | cadence | verdict |
| --- | --- | --- |
| `orion:vision:frames` | 0.1s | pre-detector — stays healthy while the eye is blind |
| `orion:vision:artifacts` | **5.0s** | **chosen** — carries real detector output |
| `orion:vision:windows` | 8.6s | downstream of habituation |
| `orion:vision:events` | ~11/hour | far too sparse to read as liveness |

## Metric quality gate (AGENTS.md §0A)

1. **Provenance.** `outputs.objects` on `VisionArtifactPayload`, published by
   `orion-vision-host` to `orion:vision:artifacts`; liveness from the receipt
   clock in `_handle_vision_artifact_message`.
2. **Independence.** First exogenous-sensor signal in the field — every other
   Active-Inference domain is introspective or infrastructural. Shares no
   sensor or upstream computation with them. Notably it is *more* independent
   than the deleted version, which read the same graph as
   `node:substrate.bus_synaptic`.
3. **Theory anchor.** Watchdog/liveness with a deadband for availability; raw
   detection count for yield. No smoother, no baseline, no fitted constant —
   there is nothing to calibrate, only a clock and a count.
4. **Live-data sanity and rest point.** Rest point is exactly 0.0 by
   construction and genuinely reachable: the newest artifact is normally ~5s
   old against a 15s deadband, and 82 live ticks read `staleness=0.000` with
   `age_sec` between 0.1 and 4.3. Yield is non-degenerate: 6.83 → 7.00 → 6.97
   across a filling window, and 0.0 against a black probe frame. **Not added to
   `NODE_DECAY_CHANNELS`** — decay converges toward calm, and silence must
   converge toward alarm.
5. **Existing mechanism.** Nothing produces a per-organ capability-health
   signal. `bus_synaptic_prediction_error`'s own docstring says it "cannot
   cleanly separate a few-organ event from noise at ANY threshold" and
   concludes single-organ detection needs a per-organ signal.
   `bus_synaptic_graph_routes.py`'s `/propagate` is a Hub debug BFS for blast
   radius, not a field producer, and is not rebuilt here.
6. **Reversibility.** Node + edge + listener + tick, all behind one flag.
   Deliberately **not** added to `ACTIVE_INFERENCE_DOMAINS` or to
   `_PREDICTION_ERROR_DOMAIN_NODE_IDS` — observe a week first, per the
   transport retirement's measured ~0.09 aggregate shift.

## The bug that presented as an encoder fault

`perception_staleness` would not persist. Every layer verified correct **in
isolation** — metadata sanitizer kept it, the allowlist contained it, the
encoder returned it, the generated Cypher SET clause contained
`n.perception_staleness`, `ConceptNodeV1.metadata` is a free `Dict[str, Any]`,
the store was `FalkorSubstrateStore`, and the write logged success. The node was
re-upserted every 30s and the property was still NULL.

None of those layers was wrong. `EXTERNALLY_OWNED_METADATA_KEYS` is the set
`SubstrateDynamicsEngine.tick()` skips so it does not overwrite metadata owned
by another writer, and the two new keys were missing from it. So every 30s the
dynamics tick re-persisted the node from a metadata dict that did not carry
them and set both properties back to NULL — seconds after the vision tick wrote
them, and while that write logged success.

That is exactly why `prediction_error` survived on the same node and the new
channels did not: `prediction_error` was already in that set. This is the third
key-loss incident against this mechanism, so the note at the definition now says
any key owned by a writer other than the dynamics engine belongs there, not just
prediction-error-shaped ones.

## Files changed

- `orion/substrate/prediction_error.py`: deleted `vision_channel_prediction_error`;
  added `vision_channel_staleness_pressure`, `perceptual_yield`,
  `perceptual_blindness_pressure`.
- `services/orion-substrate-runtime/app/worker.py`: artifact listener,
  time-triggered tick, `extra_channels` on the node writer.
- `services/orion-substrate-runtime/app/settings.py`: flag, interval, channel.
- `orion/substrate/falkor_codec.py`: both channels through the allowlist, the
  encode dict, the decode dict, and `EXTERNALLY_OWNED_METADATA_KEYS`.
- `config/field/orion_field_topology.v1.yaml`: the edge.
- `config/field/field_channel_glossary.v1.yaml`: `perception_staleness`.
- `services/orion-substrate-runtime/docker-compose.yml`: env passthrough.
- Tests: `test_vision_channel_signal.py` (13),
  `test_falkor_codec_perception_staleness.py` (7), plus one expected-dict update.

## Schema / bus / API changes

- Added: `node:substrate.vision`; node channels `perception_staleness`,
  `perception_yield`; one `node_capability` edge into the existing
  `capability:vision`.
- Removed: `vision_channel_prediction_error` (never shipped to a consumer).
- Behavior changed: `capability:vision` gains a real inbound edge. No consumer
  contract changes; `orion:vision:*` is read-only here.
- **Not** added to `ACTIVE_INFERENCE_DOMAINS` / `_PREDICTION_ERROR_DOMAIN_NODE_IDS`.

## Env/config changes

- Added keys: `SUBSTRATE_VISION_CHANNEL_TICK_ENABLED` (true),
  `SUBSTRATE_VISION_CHANNEL_TICK_INTERVAL_SEC` (30.0),
  `SUBSTRATE_VISION_ARTIFACTS_CHANNEL` (`orion:vision:artifacts`).
- `.env_example` updated; local `.env` synced by hand — `scripts/sync_local_env_from_example.py`
  reads `.env_example` from the primary checkout, so it cannot see a worktree's
  edits. All three keys verified present in the live container.
- Also added to `docker-compose.yml`: that file enumerates env explicitly rather
  than passing `.env` wholesale, so the keys otherwise never reached the
  container — caught by deploying and finding the tick had logged nothing.

## Tests run

```text
$ PYTHONPATH=.:services/orion-substrate-runtime pytest services/orion-substrate-runtime/tests \
    orion/substrate/tests/test_vision_channel_signal.py \
    orion/substrate/tests/test_falkor_codec_perception_staleness.py -q \
    --ignore=services/orion-substrate-runtime/tests/test_grammar_consumer_integration.py
16 failed, 269 passed

$ pytest orion/substrate/tests/ -q -k "falkor or vision"
86 passed
```

The 16 failures are **pre-existing**: unmodified `main` gives an identical
16 failed / 249 passed on the same command. This branch adds 20 passing tests
and no failures. `test_grammar_consumer_integration.py` cannot collect outside
the container (`app.models` path collision), also pre-existing.

## Evals run

```text
None. services/orion-substrate-runtime has no evals/ directory.
```

The gate above is live runtime measurement, not an eval harness. The natural
first eval is a fault-injection one — stop `orion-vision-edge`, assert
`perception_staleness` crosses 0.0 → 1.0 and `capability:vision.pressure`
follows — deliberately not run here because it means blinding the live camera.

## Docker/build/smoke checks

```text
$ bash scripts/safe_docker_build.sh orion-substrate-runtime build   -> Built
$ bash scripts/safe_docker_build.sh orion-substrate-runtime up -d   -> Started
$ bash scripts/safe_docker_build.sh orion-field-digester build/up   -> Built/Started

82 vision ticks, 0 tick failures, 0 listener handle failures
node:substrate.vision -> perception_staleness=0, perception_yield=6.966667
detector probe: black 0 objects / grey 0 objects / live 6 objects
```

## Acceptance checks vs the design doc

P3's stated checks:

- *"`capability:vision` shows non-constant pressure traceable to a real
  receipt"* — **partially met.** The producer is real and the node carries real
  values, but see Risks: at rest the capability vector still reads 0.0.
- *"`capability_provenance` non-empty"* — **not met at rest.** See Risks.
- *"diffusion observably moves a downstream channel"* — **not yet observed**,
  for the same reason.

## Risks / concerns

- **Severity: medium. The field cannot express "measured and healthy", so at
  rest this edge is still indistinguishable from the fabricated constant.**
  `apply_diffusion()` gates provenance on `contribution > 0.0`, so a genuinely
  measured zero claims neither pressure nor provenance. Confirmed general, not
  specific to this edge — every capability with pressure > 0 has provenance,
  and both that sit at exactly 0 (`capability:vision`, `capability:memory`,
  the latter with a real `node:prometheus` edge) have `{}`. The producer half
  of P3 is done and verified; the falsehood is only fully deleted once a
  measured zero can claim provenance. *Proposed fix, not in this patch:* let a
  zero contribution claim provenance when no other edge has claimed that key,
  preserving the existing guard's purpose (not overwriting a real one with a
  later-iterated zero). It touches shared diffusion logic for every capability,
  so it wants its own patch and sign-off.
- **Severity: low.** `perception_yield` is recorded but unwired, so a blinded
  camera is *visible* on the node but does not yet raise pressure. Deliberate —
  see above.
- **Severity: low.** The 15s/60s deadband is anchored to one camera's measured
  cadence. A second, slower stream would need its own thresholds.
- **Operational note.** A concurrent session rebuilt `orion-substrate-runtime`
  at 05:53 during this work and reverted the deployment, which confounded part
  of the diagnosis. The current deployment (06:14) is intact and verified.

## Restart required

Already applied. To rebuild:

```bash
cd /mnt/scripts/Orion-Sapienform-perception-field-and-context
bash scripts/safe_docker_build.sh orion-substrate-runtime build
bash scripts/safe_docker_build.sh orion-substrate-runtime up -d
bash scripts/safe_docker_build.sh orion-field-digester build
bash scripts/safe_docker_build.sh orion-field-digester up -d
```

## What this does not deliver

P4 (`PerceptionContextV1` in the turn). Nothing in a chat turn reads perception
yet — the reader shape is mapped (`metacog_trend_reader.py` is the precedent)
but no code is written.
