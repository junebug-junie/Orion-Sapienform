# Spark-introspector retirement + honest inner-state substrate convergence

**Status**: DRAFT / brainstorm convergence doc. Working tree in this worktree stages the actual
service deletion (uncommitted). Written from a chat-only investigation session — every claim
below was checked against live code, live containers, or live FalkorDB/log data, not assumed.

**2026-07-28 update (separate chat session):** this doc's own "Bottom line" in §2 (OrionTissue
and the mood-arc encoder are the only real architecture trapped in this service, both already
live outside it) missed one more thing: `app/inner_state.py` was the one real, live,
**multi-consumer** producer left inside the service — `InnerStateFeaturesV1` on
`orion:self:inner_features`, consumed by `orion-hub` and persisted by `orion-sql-writer` as the
training corpus for the golden phi encoder. Unlike everything else here, that's not a debug
sink. Extracted into `orion-substrate-runtime` (`app/inner_state_features.py` +
`_inner_state_features_loop`/`_inner_state_tick` in `app/worker.py`) in this same working tree
before this deletion lands — see that service's README, "Inner-state features" section, and
`docs/superpowers/specs/2026-07-28-cognition-trace-signal-gateway-consumer-audit.md`. Also see
§6's correction below re: `orion:spark:state:snapshot`'s producer count.

**Trigger**: `orion-spark-introspector`'s phi/EKG output flagged as bullshit ("the signals are bullshit... it needs to die unless our new IIT framework can make it not shit"). This doc converges on what's real, what's dead weight, and what's genuinely undecided, so the kill (when it happens) takes the bad math with it and nothing else.

**Ground rule for this doc, per CLAUDE.md 0A**: no fused "consciousness meter." Keep every real signal named honestly and separately until each has independently earned its name.

---

## 1. KILL — confirmed theater, no ambiguity

### 1a. `_phi_from_self_state()` (`services/orion-spark-introspector/app/phi_encoder.py` + its call sites in `app/worker.py`)
The canonical phi source. SelfState-anchored — the exact pattern the charter's own hard gate (PR #1176) already ruled dead for IIT (`InnerStateFeaturesV1` carries a `self_state_id` field; see [[feedback_field_native_not_selfstate]]).

**Live evidence (60-minute window, 2026-07-28):**
- 43/70 telemetry emissions had `novelty=0.000` exactly.
- 27/70 emissions were preceded by `spark_introspection_skipped reason=requires_rich_meta qual=0` — a fallback default masquerading as a genuine "nothing happened" reading, not confirmed calm. Same shape as CLAUDE.md's own worked calm-floor example.
- Coherence *does* show real continuous variance (0.90–0.96) — not 100% dead, but the novelty half of the headline pair is compromised on the majority of ticks.
- Prior form: [[project_spark_dead_code_canonical_phi]] already found 3 parallel valence/energy/coherence implementations in this lineage, one deleted outright as a 936-line dead facade, one arousal-formula bug already fixed once.

### 1b. `mood_arc_corpus.v1` collector (`MOOD_ARC_CORPUS_PATH`, written from `services/orion-spark-introspector/app/worker.py`)
Old 4-channel hand-composited corpus (coherence/energy/novelty/valence — i.e. hand-crafted tensor math dressed up as mood). Per the standing board item: still live and growing (149k+ rows as of 2026-07-17), **consuming storage/compute for a corpus the real encoder no longer trains against** (superseded by `field_channel_corpus.v1`, produced by `orion-field-digester`, a completely different service). Dies with the service and should — nothing downstream needs it.

### 1c. `InnerStateCorpusSink` (the `_INNER_SINK`/rotating corpus writer feeding `phi_encoder.py`, `orion/telemetry/corpus_rotation.py`'s rotation logic applied to it)
Same SelfState-anchored lineage as 1a. Dies with it.

**None of 1a/1b/1c have any consumer outside `orion-spark-introspector` itself** — confirmed via repo-wide grep, only hits are inside the service, its own tests/evals, and stale worktree copies. Killing the service kills these cleanly, no orphaned readers.

---

## 2. HOLD / REPURPOSE — structurally independent, already survives the kill

This is the part worth double-checking before pulling the trigger: **the actually-good architecture in this lineage is already NOT physically inside `orion-spark-introspector`.** Confirmed by file location, not assumption:

### 2a. `orion/spark/orion_tissue.py` — top-level module, not `services/orion-spark-introspector/`
Real decay+diffusion physics, 16×16×8 tensor, plain numpy — **zero `torch`/`cuda`/GPU references anywhere in the file.** The "needs a dedicated GPU" plan Juniper recalled must have been for something else; it isn't a property of the code that exists today.

Already rides on real per-turn data: `handle_semantic_upsert` (currently living in spark-introspector's `worker.py`, called from the live `orion:vector:semantic:upsert` bus channel — produced by `orion-vector-host`) passes the bus-delivered embedding straight into `TISSUE.propagate(stimulus, embedding=emb, ...)`. It does not run its own encoder.

**Deleting the service does not delete this module** — `orion_tissue.py` lives outside `services/`. What's lost is the *caller* (`handle_semantic_upsert`'s wiring, currently inside the doomed worker.py). Real follow-up: extract that call site into a standalone consumer of `orion:vector:semantic:upsert` before the service dies, so the tissue keeps getting fed. This is an extraction task, not a research question — the thing people were worried was hardware-blocked isn't.

### 2b. `orion/mood_arc/` (`fit_encoder.py` + its own `tests/`) — top-level package, not `services/orion-spark-introspector/`
This is the real "new IIT candidate" — a windowed sequence autoencoder, third attempt after two dead ends (hand-built lattice, then SelfStateV1, both dead-endish per the charter). It trains on `field_channel_corpus.v1`, **produced by `orion-field-digester`**, a service completely unrelated to spark-introspector. Cleared its shuffle-floor gate at `hidden=128/latent=64` (capacity-ablation sweep, already settled). Ready to start, not yet wired to any downstream consumer.

**Killing spark-introspector has zero structural effect on this pipeline.** It doesn't live there, its corpus doesn't come from there, and its only tie to the dying service was the deprecated `mood_arc_corpus.v1` writer (item 1b), which was already superseded and slated to go.

**Bottom line for the "hold in case there's repurposable arch" worry: there isn't any arch trapped inside the service that needs saving via a slow kill.** Both real pieces (OrionTissue, mood-arc encoder) already live outside it. The only real pre-work is extracting `handle_semantic_upsert`'s wiring (2a) so OrionTissue doesn't go dark. Everything else in `services/orion-spark-introspector/` (phi_encoder.py, the inner-state sink, the old mood_arc corpus writer, `train/evals/eval_phi_encoder_health.py`) is SelfState-lineage and can go with the service.

---

## 3. What you missed — additional real substrate found this session

### 3a. `orion-heartbeat` (NEW service, live, container `orion-athena-heartbeat`)
Not something either of us named originally — a real matrix-product-state tensor network (`quimb`) subscribed to `orion:grammar:event`, computing actual bipartite entanglement entropy across 5 confirmed-live organs (chat/hub, biometrics, execution/cortex-exec, transport/bus, route/cortex-orch). Explicitly documented: **no `SelfStateV1` dependency**, read-only, additive. The strongest theory-anchored "pulse" candidate found this session — better than the equilibrium-service cadence originally proposed.

**Open concern, unresolved**: last 20+ H1 ticks all read `verdict=redundant`, entropy ratio pinned 0.79–0.97 (near-ceiling). The service's own README already names this exact risk ("near-tautological for a pure global MPS state") but it reads as live and unresolved, not just a hypothetical caveat in the doc. This is the single highest-priority open question from this session — it has a real, findable answer in the code/math, not a judgment call.

### 3b. Bus synaptic graph (`orion_bus_synapse` in FalkorDB) — real, live, previously mis-assessed
Corrected mid-session: this was initially (wrongly) called "docs-only" based on PR #1314 (which *is* docs-only — a brainstorm spec). The actual implementation is a separate ~20-PR arc (#1323 through #1404) that's fully live. Verified directly in FalkorDB: real `:Organ`/`:Channel` nodes, `PUBLISHES`/`CAUSALLY_FOLLOWED_BY`/`EXECUTES_VERB` edges, real observed-event counts in the millions, and a genuine live anomaly signal — `gap_zscore` — per publisher/channel edge. This is a strong, already-built "temperature" candidate: real mesh-wide timing-deviation signal, not aggregate theater.

Direct precedent for this whole retirement conversation: a sibling signal (`transport_prediction_error`) already had a calm-floor bug found+fixed (PR #1391), and a same-day proposal (PR #1392) to retire it outright in favor of `bus_synaptic` — i.e., this exact kill-the-old-signal-once-the-new-one-is-live pattern has already happened once in this codebase, successfully.

### 3c. Open, unresolved: `orion-vector-host`'s unconditional embedding
Checked per Juniper's "stupid embedder running regardless" complaint: `orion-vector-host`'s README states it now embeds **all** assistant texts unconditionally regardless of backend (ollama/llamacpp/vllm/cola) — there is no cola-conditional skip logic anywhere in its settings or git history (zero commits mention "cola"). If the intent was ever "skip re-embedding when `orion-llama-cola-host` already produced something," that logic does not exist and, per available history, was never actually attempted — not a regression, a thing that was never built. **Not yet confirmed whether this is the specific "embedder" Juniper meant** — flagged as open, needs a direct answer before scoping a fix.

### 3d. Broken memory-index entry found along the way
`MEMORY.md` pointed to `project_bus_synaptic_transport_domain_arc_2026-07-26.md`, which does not exist on disk — a live instance of the already-known index-sync-gap bug ([[feedback_memory_index_sync_gap]]). Not architecture, but flagging since it means some detail on the #1391/#1392 arc (calm-floor fix, retirement proposal) may be permanently lost unless reconstructed from the PRs themselves.

---

## 4. Proposed honest readout shape (not decided, for discussion)

Three separately-named signals, not fused:

| Name | Source | Status |
|---|---|---|
| **Pulse** | `orion-heartbeat`'s boundary/bulk entanglement entropy | Live, real, redundant-ceiling unresolved |
| **Continuity/spark** | `OrionTissue`'s decay/diffusion energy state | Real module, needs extraction from doomed service |
| **Temperature** | `bus_synaptic`'s `gap_zscore` (+ possibly `field_channel_corpus.v1` aggregate motion) | Live, real, unaggregated/unexposed as a single readout |

No schema or reducer currently ties these three together — that's the real remaining gap once each is individually trustworthy.

---

## 5. Non-goals for this doc
- Not deciding today whether/how pulse+continuity+temperature get fused into one UI surface.
- Not resolving the heartbeat redundant-ceiling question here (needs its own investigation).
- Not touching `orion-vector-host`'s embedding-condition question until 3c is confirmed as in-scope.
- Not opening a branch/worktree yet — this is convergence, not implementation.

## 6. Decided sequence (no longer open)

Verified, not guessed: the Hub EKG chart is live-driven by spark-introspector (`services/orion-hub/scripts/websocket_handler.py:1740`, `orion:spark:introspect:candidate` round-trip) — it goes dark the moment the service dies unless repointed in the same patch. `orion:self:phi_reward` is a dead contract (`orion-substrate-runtime` is a registered consumer in `channels.yaml` with zero actual code references) — safe to delete outright, nothing real depends on it. `orion:spark:state:snapshot` has two producers (spark-introspector + `orion-equilibrium-service`) — it survives the kill on its own.

**Correction, 2026-07-28 (found via a separate chat-session cognition-trace audit,
`docs/superpowers/specs/2026-07-28-cognition-trace-signal-gateway-consumer-audit.md`):**
the claim above that `orion:spark:state:snapshot` "survives the kill on its own" via
`orion-equilibrium-service` is **wrong** — grepped `orion-equilibrium-service` directly,
zero references to `SparkStateSnapshotV1`/`spark.state.snapshot`/`channel_spark_state_snapshot`
anywhere. What equilibrium's `EQUILIBRIUM_SPARK_HEARTBEAT_ENABLE` idle-keepalive loop actually
does is publish a synthetic `mode="heartbeat"` **`cognition.trace`** onto `orion:cognition:trace`
(a different channel) — it's spark-introspector's own `handle_trace()`, on receiving that
heartbeat-mode trace, that produces the `SparkStateSnapshotV1`. That requires spark-introspector
to be alive; it is not a second producer. **`orion:spark:state:snapshot` has exactly one real
producer and it dies with this service, full stop** (this is separately what made
`orion-landing-pad`'s one live ingest source go dark in that audit — landing-pad's `snapshot`
reducer had no other real input either).

1. Move `handle_semantic_upsert`'s `OrionTissue.propagate()` wiring into `orion-vector-host` (already the natural owner — it's the service producing the embedding tissue consumes). One small addition, not a new service.
2. Repoint the Hub EKG chart (`websocket_handler.py:1742`, `spark_candidate.py`) at that new location instead of the `orion:spark:introspect:candidate` round-trip. Required in the same patch, not a follow-up.
3. Delete `services/orion-spark-introspector` outright: `phi_encoder.py`, the inner-state corpus sink, the `mood_arc_corpus.v1` writer, the whole directory, its docker-compose entry.
4. Retire for real (not just stop citing) in `channels.yaml`: `orion:self:inner_features`, `orion:self:phi_reward`, `orion:spark:telemetry`, `orion:spark:introspect:candidate`. `orion:spark:state:snapshot` stays.

Runs independently, not gating the above: orion-heartbeat's redundant-ceiling question (3a), the vector-host cola-conditional question (3c).

---

## 7. Honest signal selection for the OrionTissue visualizer page (2026-07-28, same day, separate chat session)

Section 6 relocated the OrionTissue page (`services/orion-vector-host/app/static/`) and PR #1421
(merged separately) fixed its four headline stats' *labels* (embedding_similarity/novelty_zscore/
polarity_diff/mean_abs_activation, no invented mood taxonomy). This section covers a further
decision: those four numbers, while honestly labeled, have "no independent theory anchor
connecting them to reasoning quality, mood, or wellbeing" by the page's own disclosed copy —
they're real tensor summary statistics, not a good driver for a display meant to show
substrate-wide state. Replaced as the page's four driving/displayed metrics with real,
brain-frame-derived signals, **only after each candidate was individually live-verified against
real running data**, not assumed:

**Confirmed real, wired in:**
- `honesty_metrics` / `prediction_error_confidence` — 0.7929–0.9935 over a real 2-minute window.
  Required a real fix along the way: the initial wiring passed `AttentionBroadcastProjectionV1`
  (`load_attention_broadcast()`) as the confidence source, which has no `prediction_error_confidence`
  field at all — the region silently emitted nothing. The real object,
  `AttentionSelfModelV1`/`reduce_attention_self_model()`, had **never been called live anywhere in
  this codebase** (only by offline analysis scripts) — computing it live in `_brain_frame_tick()`
  from the already-fetched node snapshot (zero extra I/O) was the actual fix, not a rename.
- `frame.spotlight.coalition_stability` (`AttentionBroadcastProjectionV1.coalition_stability_score`)
  — 0.3–0.9 over a real 7-minute window (`substrate_attention_broadcast_log`). Already exposed on
  the existing self-brain frame schema; no new wiring needed.
- `field_anomaly` / mood-arc encoder `recon_loss` (orion-field-digester's `app/anomaly_scorer.py`,
  the real windowed sequence-autoencoder from section 2b) — confirmed a genuine anomalous→calm
  state transition (~0.012, 4-15x the encoder's own threshold → ~0.00012, below it) across two
  checks roughly an hour apart. New bus subscriber added in `orion-substrate-runtime`
  (`_field_channel_anomaly_listener_loop`), consuming the already-live, already-published
  `orion:field_channel:anomaly_score` channel (previously consumed only by
  `orion-equilibrium-service`'s metacog gate).

**Checked and rejected, with the live numbers that killed each one:**
- `self_state` — zero producer since the 2026-07-22 SelfStateV1 burn
  (`orion-consolidation-runtime/app/store.py` confirms `substrate_self_state` has had no writer
  since). Would never emit a region at all.
- `node_kind` region max — pinned 0.9687–1.0 over a live 2-minute window (only one node kind,
  "concept," is populated right now beyond the prediction-error tracker nodes and 3 golden seed
  concepts — the graph has too little kind-diversity for this to be a real signal today).
- `lane` region aggregate, both directions — `max()` reads a flat constant 0.4 whenever backlog=0
  and lag≤60s (the common case), and additionally masks a real incident: `chat_grammar` has read
  exactly 0.0 (lag_sec≈254000, ~70 hours stale) for the entire observation window, invisible under
  `max()`. Switching to `min()` doesn't fix this — it just relocates the pin to that same dead
  lane's 0.0, permanently, for as long as the lane stays down. (Separately: a reducer lane being
  dead for 70+ hours in production is a real finding, unrelated to this viz task, not yet
  investigated further here.)
- raw `bus_synaptic` `gap_zscore` — not independent. `bus_synaptic_prediction_error()` is already
  built directly from these same FalkorDB edge z-scores, and `bus_synaptic` is already one of the
  5 domains feeding `honesty_metrics`. Using both would double-count the same underlying signal
  under two different names.
- `substrate_attention_frames`' `overall_salience` (general field lane) — also pinned at exactly
  `1.0` across every sample checked.

Implementation: PR branch `feat/honest-ekg-honesty-metrics`. Files touched: `orion/schemas/
brain_frame.py` (dimension Literal), `orion/bus/channels.yaml` (new consumer),
`services/orion-substrate-runtime/app/{settings,worker,brain_frame_producer}.py`,
`services/orion-vector-host/app/static/{index.html,tissue_viz.js}`, plus tests for both new
region functions and the new bus listener. All three signals verified live end-to-end through
the real Tailscale Serve routing (`/spark/ui` page → absolute-path fetch → routes to orion-hub's
root, not vector-host — confirmed via `tailscale serve status` and a live curl through the real
domain), not just unit tests.
