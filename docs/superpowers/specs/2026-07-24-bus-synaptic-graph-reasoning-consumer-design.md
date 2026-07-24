# Bus synaptic graph reasoning consumer — design (Idea 4)

Status: **proposed, design-only. No code in this patch.** Proposal-mode pass per this repo's
CLAUDE.md section 0A ("Proposal mode before invasive cognition changes") — this is the first idea
in the bus synaptic graph arc that touches a reasoning/recall pipeline, not just infrastructure
telemetry. "Idea 4" from `docs/superpowers/specs/2026-07-24-bus-vitality-field-signal-brainstorm.md`'s
Phase 3+ brainstorm, following Idea 5 (PR #1335, merged) and Idea 1 (PR #1337, merged).

## Arsonist summary

`orion-recall`'s worker already assembles chat-turn context from multiple fragment sources
(SQL, RDF/Falkor chatturn, Falkor neighborhood), each independently flag-gated, each returning a
list of fragments in one shared shape that `fusion.py` scores and merges. `services/orion-recall/app/storage/falkor_neighborhood_adapter.py`
is the closest live precedent for "query a Falkor graph as part of assembling reasoning context":
fixed, hand-authored, parameterized Cypher only (never free-form generation), fails open to `[]`
on any error, runs the sync Falkor client via `asyncio.to_thread`, gated by a settings flag
(`RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT`, default `False`) checked in `worker.py` alongside a
profile-level override.

The bus synaptic graph (`orion_bus_synapse`, written by `orion-bus-mirror`) already has the exact
query shapes this proposal needs, live-verified and shipped in PR #1335 as Hub debug routes:
`hot_organs`, `hot_edges`, `anomalies`. The `anomalies` query in particular — edges whose most
recent observation deviated sharply from that edge's own rolling baseline, with a `min_count`
floor guarding against cold-start z-score instability — is the one worth surfacing to reasoning,
not the hot-path ranking (which is closer to human-debug utility than something worth injecting
into a chat turn).

## User's own framing

Asked "on-demand verb vs. always-on context injection" as a genuine fork with real trade-offs.
Chose **always-on**: this should run automatically every turn (or every turn recall already
runs, which is most of them), not require an explicit trigger — Orion should have live awareness
of its own transport-layer stress as a background capability, not something it has to think to
ask about. This is a bigger, more ambitious choice than the on-demand alternative, made
deliberately, not defaulted to out of caution.

## Current architecture

- `services/orion-recall/app/worker.py` (~line 1100): per fragment source, a boolean gate
  (`settings.X_IN_CHAT and profile.get("enable_x", True)` [`and` an optional relevance
  precondition]), wrapped in `try/except Exception` degrading to `[]`, extending a shared
  `candidates` list that `fusion.py::fuse_candidates()` scores downstream. Every existing source
  follows this exact shape.
- `services/orion-recall/app/storage/falkor_neighborhood_adapter.py`: the closest sibling.
  Returns fragments shaped `{id, source, source_ref, uri, text, ts, tags, score, meta}` — the one
  shape `fusion.py` already knows how to handle, so a new source needs no fusion.py changes.
- `orion.graph.falkor_client.RedisGraphQueryClient` + `FALKORDB_BUS_GRAPH` setting: already
  wired in `orion-bus-mirror`, not yet in `orion-recall`. `orion-recall` already has its own
  `FALKORDB_URI`-shaped client getters for `orion_recall` (`get_recall_falkor_client()`); a new
  graph name (`orion_bus_synapse`, shared instance) needs its own client, same pattern as Hub's
  own `FALKORDB_BUS_GRAPH` addition in PR #1335.
- `services/orion-hub/scripts/bus_synaptic_graph_routes.py::anomalies()`: the exact Cypher this
  proposal reuses, already live-verified against real traffic (found 3 real anomalies in
  production data the day it shipped).

## Missing questions

- Does `orion-recall`'s worker actually run on (approximately) every chat turn, or only when a
  step explicitly invokes `RecallService`? This proposal's "always-on" framing assumes the
  former — needs confirming against real turn-orchestrator wiring before implementation, not
  assumed from this design pass alone.
- Should this be gated on `query_text` relevance (like the neighborhood adapter, which
  keyword-matches the user's message) or unconditional per turn? Recommended below: **unconditional**
  — self-awareness of operational state isn't naturally "about" what the user said, unlike memory
  recall. Worth confirming this reasoning holds before building.
- What z-score/count thresholds are right for "worth surfacing to a chat turn" vs. "worth
  surfacing to a human debugging via Hub"? The Hub route's defaults (`zscore_threshold=3.0,
  min_count=5`) were tuned for a human looking at a table, not for what's worth interrupting a
  reasoning turn with. Needs its own live-data pass, not inherited by default.
- How often, realistically, will this actually produce a non-empty fragment? If real anomalies
  are rare (Hub's live check found only 3 across the whole mesh at a point in time), this may
  fire empty on the overwhelming majority of turns — worth confirming that's an acceptable,
  expected steady state (a quiet transport layer most of the time) and not treated as "the
  feature isn't working."

## Proposed schema / API changes

- New adapter module (name TBD, mirroring the sibling): `fetch_bus_synaptic_anomaly_fragments()`,
  same signature shape as `fetch_falkor_neighborhood_fragments()` minus the `query_text` keyword
  argument (this is unconditional, not keyword-triggered — see Missing Questions).
- Wraps the exact Cypher already in `bus_synaptic_graph_routes.py::anomalies()` — literally the
  same two queries (publish-gap, causal-latency), not reinvented.
- Fragment shape per anomaly found: `source="bus_synaptic_anomaly"`, `text` a short, honest
  natural-language description (e.g. `"orion-feedback-runtime -> spark-concept-induction hop
  latency is 4.2x its normal baseline (count=340)"`), `meta` carrying the raw organ/channel/
  zscore/count fields for anything downstream that wants structured access, `score` TBD (needs a
  real calibration pass against fusion.py's existing score ranges, not guessed).
- New settings flag: `RECALL_BUS_SYNAPTIC_ANOMALY_IN_CHAT` (default `False`), same
  `AliasChoices` pattern as every sibling flag.
- New `FALKORDB_BUS_GRAPH`-shaped client getter in `orion-recall`, mirroring
  `get_recall_falkor_client()`'s shape but pointed at the shared `orion_bus_synapse` graph
  instead of `orion_recall`.

## Files likely to touch

- `services/orion-recall/app/storage/` — new adapter module (exact filename TBD at
  implementation time).
- `services/orion-recall/app/settings.py` — new flag.
- `services/orion-recall/app/worker.py` — new gated call site, same shape as the 3 existing ones.
- `services/orion-recall/.env_example` — new key.
- `services/orion-recall/tests/` — new adapter tests (fixed-query correctness, fail-open,
  empty-when-nothing-anomalous), matching `test_falkor_neighborhood_adapter.py`'s shape.
- No changes needed to `fusion.py` (reuses the existing fragment shape) or to
  `orion-bus-mirror`/`orion-hub` (this reads the already-live graph, doesn't change what writes
  to it).

## Non-goals

- **Not** free-form Cypher generation by the model, ever. This is the one hard constraint in this
  whole proposal — the model gets a capability, not a query language. If this is ever revisited,
  it needs its own separate, much more careful proposal-mode pass.
- **Not** changing `bus_synaptic_graph_routes.py`'s Hub-facing endpoints — this proposal reuses
  their Cypher, doesn't touch their code.
- **Not** deciding the exact score/threshold calibration here — explicitly deferred to a live-data
  pass at implementation time, per this repo's own metric quality gate (step 4, "live-data sanity
  check," applies here just as much as it did to every prior phase of this arc).
- **Not** injecting a "nothing anomalous right now" fragment when there's nothing to report — an
  empty list is the correct, honest output most of the time, not a gap to fill with filler
  content (this repo's own "no empty-shell cognition" rule).

## Acceptance checks

- Unit tests for the adapter: fixed-query shape correctness (mirrors the Hub route's already-
  tested Cypher), fail-open on Falkor error, empty-list when no anomaly exists, correct fragment
  shape when one does.
- Live-data sanity check before considering this done: force a real (or realistically simulated)
  anomaly in the live graph, confirm it actually surfaces as a fragment through a real recall
  call, not just in isolation against a mock.
- A real, inspectable trace: at least one real chat turn where this fragment appears in the
  assembled context/candidate list, logged with enough detail (turn id, fragment content) to
  verify after the fact — same standard `falkor_neighborhood_adapter.py`'s `meta.matched_entities`
  field already sets.
- Explicitly checked, not assumed: how often this actually fires non-empty against real traffic
  over a real time window (hours, not one live snapshot) — settles the "is this too noisy / too
  quiet" question with data before deciding on threshold tuning.

## Recommended next patch

Build the adapter + tests first, live-gate it standalone (call it directly against the real
running graph, confirm it produces sane fragments when a real anomaly exists, confirm empty
otherwise) — same sequencing this entire arc has used for every prior phase. Only wire it into
`worker.py` behind the new flag (default off) once the adapter itself is proven correct in
isolation. Flip the flag on and watch real chat turns before calling this shipped, matching the
"deploy, then verify against real traffic, don't assume the synthetic test is enough" lesson this
arc has already learned twice (PR #1327's crash, PR #1333's signal-quality bug).
