# Orion Cockpit POV — Slice C follow-on stub

**Spec:** `docs/superpowers/specs/2026-09-06-orion-cockpit-pov-design.md`
**Slice B plan:** `docs/superpowers/plans/2026-09-06-orion-cockpit-pov-slice-b.md`

Invoke **writing-plans** before remaining implementation.

## Already live (not remaining Slice C)

**Pre-motor progress hops** (`pre_turn_appraisal`, early `association`, `thought_rpc`, `mind_enrichment`, `harness_dispatch`) ship on the Hub unified-turn path so Soft HUD can see dying appraisal / Mind / felt-state phases while the turn waits. That is Rank-1 boundary sighting.

**Ingress is now a real hop** (`hop_from_ingress` / `begin_cockpit_timeline(..., ingress=...)`): Soft HUD shows user text and turn intake (`user_message`, `session_id`, `mode`, attachment metadata without binary blobs). `observation_published=false` until Hub actually captures/publishes the observation molecule return rather than discarding it.

## Remaining thickness

| Stage | Deferred work | Likely capture site |
|-------|---------------|---------------------|
| **ingress extras** | Observation molecule dump on publish (capture `emit_observation` return / real bus publish); thicker attachment metadata if needed | `orion/hub/turn_orchestrator.py` ingress / `emit_observation` |
| **closure** extras | Any post-motor side-effects beyond outcome/closure already emitted in Slice A | `orion/harness/finalize.py`, offboarding paths |
| **stance_inputs** enrichment | Optional: mind_coloring / Thought-assembled `build_stance_react_context` echo if Hub-sent dict proves too thin live | `services/orion-thought/app/bus_listener.py` |
| **mind_enrichment thickness** | Mirror Mind quality onto Thought RPC reply (today Hub often only has `mind_details_unavailable`) | Thought → Hub reply contract / `mind_runs` artifact |

**Acceptance:** every canonical stage has a real hop or an owned gap with no silent omission; live + rewind still work.
