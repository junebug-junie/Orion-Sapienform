# Phase 4 research — CLUSTER_ROLE_WEIGHTS / orion-state-service node-weighting — 2026-07-12

Context: Phase 4 ("Drive unification") of
`docs/superpowers/plans/2026-07-12-self-state-mesh-substrate-redesign.md` requires resolving
whether `CLUSTER_ROLE_WEIGHTS` (`services/orion-biometrics/.env`) and
`orion-state-service`'s aggregation are a legitimate separate "ops health" concern or a
fourth duplicate of the field-topology node-weighting question (per the arsonist doc's mesh
addendum). Answered here from live code, live `.env` values, and live `docker logs`/`docker
exec` evidence on the running mesh — not from README claims.

**Verdict: neither answer in the arsonist doc's framing is quite right.** `CLUSTER_ROLE_WEIGHTS`
is architecturally a duplicate of the field-topology node-weighting question (same question:
"how much should each node's health count"), computed a different, undocumented way, with
weights that are internally inconsistent across three separate implementations. But unlike
the L7-L11 ladder, its output is *not* Hub-display-only when it fires — it's wired into a real
LLM prompt (`orion-cortex-exec`'s metacognition draft/enrich templates). Right now, however,
it never fires at all: `orion-biometrics` runs in `BIOMETRICS_MODE=agent` live, so the
`BiometricsHub`/`publish_cluster` code path that reads `CLUSTER_ROLE_WEIGHTS` never starts.
Confirmed over a 6-hour live log window: zero `biometrics.cluster.v1` messages ingested by
`orion-state-service`, despite it actively subscribing. This is a "loaded gun, safety on"
situation, not a duplicate that's already firing, and not a legitimate separate concern with
its own clean rationale either — it's an unfired, three-times-reimplemented weighting scheme
whose real reach (when enabled) is narrower and different in kind from field-topology's.

---

## Q1 — Where is `CLUSTER_ROLE_WEIGHTS` used, and what does the aggregate represent?

`services/orion-biometrics/app/main.py:282-328`, method `BiometricsHub.publish_cluster`:

```python
weights = settings.role_weights                                    # main.py:285
for node, summary in self._latest_summary.items():
    role = "atlas" if "atlas" in node else "athena" if "athena" in node else "other"  # :293
    weight = float(weights.get(role, 1.0))                          # :294
    weight_total += weight
    for k, v in summary.pressures.items():
        weighted_pressures[k] = weighted_pressures.get(k, 0.0) + weight * float(v)   # :298
    ...
pressures = {k: min(1.0, v / weight_total) for k, v in weighted_pressures.items()}   # :307
```

This is a **weighted average of each node's `BiometricsSummaryV1.pressures/headroom/
composites` dict**, bucketed by a crude substring match on node name into `atlas`/`athena`/
`other` roles, then normalized by total weight (`main.py:293-309`). It is genuinely a
"how much should each node's health count" computation — same question as the field-topology
edges, computed completely differently (role-bucket weighted mean vs. per-capability edge
diffusion).

The result feeds two things:
1. `BiometricsClusterV1` published to `orion:biometrics:cluster` (`main.py:317-328`).
2. A `spark.signal.v1` envelope with `signal_type="resource"`, `intensity=composites["strain"]`
   published to `orion:spark:signal` (`main.py:330-340`) — this looks like a DriveEngine
   feed, but **it is a dead end**: `orion/signals/adapters/spark.py:37-107` converts any
   `spark.signal.v1` envelope into an `OrionSignalV1` with `signal_kind="spark_signal"` and
   dimensions `{level, valence, arousal, coherence, novelty, confidence}` — `intensity` lands
   on `level` (`spark.py:58,94`). But `config/autonomy/signal_drive_map.yaml`'s `spark_signal`
   entry only maps `coherence`, `valence`, `novelty` to drives (lines 33-39) — **`level` is
   unmapped**, and the map's own header comment says unmapped `(signal_kind, dimension)` pairs
   "contribute nothing" (`signal_drive_map.yaml:9-10`). Since the cluster's `SparkSignalV1`
   never sets `valence_delta`/`coherence_delta`/`novelty_delta`, those three dims default to a
   constant neutral `0.5` (`spark.py:59-62`, `_delta_dim` default) every tick — no deviation,
   no tension event, no drive contribution. **The one place `CLUSTER_ROLE_WEIGHTS`'s number
   touches the DriveEngine/signal_drive_map pipeline is a wire that's connected to nothing.**

## Q2 — What does `orion-state-service`'s "per-node/global... authoritative node if configured" mean, and is it configured?

`services/orion-state-service/app/store.py:81-118` (`StateStore.ingest_snapshot`):

```python
if node == self.primary_node:
    self._latest_global = snap                     # store.py:104 — authoritative node wins
else:
    if self._latest_global is None:
        self._latest_global = snap
    else:
        if snap.snapshot_ts >= self._latest_global.snapshot_ts:
            self._latest_global = snap              # store.py:110-111 — else most-recent wins
```

`primary_state_node` is configured live: `services/orion-state-service/.env:34` →
`PRIMARY_STATE_NODE=athena` (settings default is also `athena`,
`services/orion-state-service/app/settings.py:53`). **This is answering a different question
than "node health weighting."** It applies only to `SparkStateSnapshotV1` (the Layer-6/spark
self-state snapshot object, one per producer), not to biometrics pressures/composites — it's
"which node's *self-state snapshot* is truthful right now," a single-writer-vs-most-recent
policy, not a weighted blend. In practice this is close to moot today: only Athena runs
`orion-self-state-runtime`/`spark-introspector` (confirmed live —
`docker logs orion-athena-state-service` shows every `spark.state.snapshot.v1` ingestion is
`node=athena`), so the "authoritative vs most-recent" branch never diverges from a trivial
single-node case.

Separately, `store.py:171-177` (`_pick_latest`, used only for **biometrics summary/induction**,
not for the spark snapshot rollup) picks `primary_node`'s biometrics summary if present, else
most-recent-by-timestamp across nodes — this is used in `get_latest` for `scope="global"`
(`store.py:212-221`). This, too, is "pick one node's raw reading," not a weighted blend — it
does not use `CLUSTER_ROLE_WEIGHTS` or any weight at all. `orion-state-service` does not
compute its own node-weighted aggregate anywhere; it only caches/serves whatever
`BiometricsClusterV1` the biometrics service already computed (`store.py:167-169`,
`ingest_biometrics_cluster` — pure passthrough), plus this pick-one-or-newest logic for
summary/induction.

## Q3 — Real consumers of `orion:biometrics:cluster`

- **`services/orion-state-service/app/main.py:191-213`** (`_handle_biometrics`) and
  **`app/store.py:167-169`** — ingest and cache the cluster payload verbatim (no
  transformation), exposed via `state.latest.reply.v1`'s `biometrics.cluster` field and the
  HTTP `/state/latest` debug route. Confirmed: `orion-state-service/app/settings.py:36-39`
  only defines the channel name constant — no weighting logic of its own.
- **`services/orion-hub/scripts/biometrics_cache.py`** — subscribes directly to
  `orion:biometrics:cluster` (line 110) and, when a cluster message *is* present, uses its
  `composites`/`constraint` verbatim (`_cluster_composite`: lines 267-273, `_cluster_constraint`:
  lines 347-350). **When no cluster message has arrived** (the live condition today), it falls
  back to computing its *own*, third independent role-weighting scheme keyed by literal node
  name (not role-bucket substring matching) from `BIOMETRICS_ROLE_WEIGHTS_JSON`
  (`services/orion-hub/.env:70` → `{"atlas":0.6,"athena":0.4}` — yet a third, different set of
  numbers from `CLUSTER_ROLE_WEIGHTS`'s `{"atlas":0.7,"athena":0.3,"other":0.5}`) —
  `biometrics_cache.py:262-307,309-340,375-389`. This cache's `get_snapshot()` output is
  consumed **exclusively** via `enriched["biometrics"] = await cache.get_snapshot()`
  (`websocket_handler.py:438`) inside `_with_biometrics(...)`, whose only call sites are
  `websocket.send_json(...)` to the browser (`websocket_handler.py:464,527,678,970,1582,1853`,
  etc.) — **this path is genuinely Hub-UI-display-only**, no gating/control-flow use found.
- **`orion/signals/registry.py`** — lists `orion:biometrics:cluster` as a bus channel in the
  static `biometrics` organ's catalog entry (line 21). This is metadata only — grepping
  `orion/signals/`, `orion/autonomy/`, `orion/spark/` for `BiometricsClusterV1`/
  `biometrics.cluster` finds **zero** real adapters or consumers beyond this declarative
  listing. It does not drive behavior.
- **`services/orion-cortex-exec/app/executor.py`** (`MetacogContextService`, ~lines 3712-3935)
  — the one consumer that is **not** display-only. It calls the `state.get_latest.v1` bus RPC
  (line 3714-3735), pulls `state_res.biometrics.cluster` (the `CLUSTER_ROLE_WEIGHTS`-weighted
  composite/constraint, lines 3763-3770), and formats it via `_metacog_biometrics_cue`
  (lines 848-885) into `ctx["metacog_biometrics_cue"]` / `ctx["metacog_biometrics_cue_enrich"]`
  (lines 3918-3919). These are rendered directly into the metacognition LLM prompt templates:
  `{{ metacog_biometrics_cue }}` appears in both
  `orion/cognition/prompts/log_orion_metacognition_draft.j2:49` and
  `log_orion_metacognition_enrich.j2:40`. **This is real reach — it shapes actual LLM prompt
  content for Orion's metacognition lane, not a debug/display sink.** Note, however: the
  `biometrics_line` text used in the human-readable `context_summary` (lines 3883-3897, e.g.
  "Biometrics: status=..., strain=...") reads `biometrics_context["summary"]`, which for
  `scope="global"` is the **pick-one-node raw summary** (`_pick_latest`, unweighted), not the
  cluster composite — only the `constraint` field and the JSON cue's `cluster.composite`
  numbers come from the actual `CLUSTER_ROLE_WEIGHTS`-weighted `BiometricsClusterV1`.

## Q4 — Live mesh check: is any of this actually firing?

Confirmed via `docker ps` (real, running containers) and `docker logs`/`docker exec`
(read-only):

- **`orion-athena-biometrics` runs with `BIOMETRICS_MODE=agent`**
  (`services/orion-biometrics/.env:35`, matches live container). Per
  `services/orion-biometrics/app/main.py:352-376` (`lifespan`), the `BiometricsHub` /
  `BiometricsHubWorker` / `publish_cluster` machinery — the *only* code path that reads
  `CLUSTER_ROLE_WEIGHTS` — is gated behind `settings.BIOMETRICS_MODE in {"hub", "both"}`
  (line 365). **In `agent`-only mode this never starts.** `docker logs orion-athena-biometrics`
  shows no cluster-related output at all in the observed window.
- **`orion-athena-state-service` subscribes to `orion:biometrics:cluster`** (confirmed:
  `Hunter subscribing patterns=['orion:biometrics:summary', 'orion:biometrics:induction',
  'orion:biometrics:cluster']`, log line at service startup) **but received zero
  `biometrics.cluster.v1` messages over a 6-hour log window** (`grep -ci
  "biometrics.cluster"` → `1`, and that one hit was the subscription-announcement log line,
  not an ingested message). Meanwhile `orion:biometrics:summary`/`orion:biometrics:induction`
  messages from `node=athena` arrive continuously (~every 30s), and only ever from `athena` —
  no `atlas`/`circe` node in this mesh's biometrics traffic today (`docker ps` shows no
  atlas/circe containers on this host at all, consistent with Phase 3 of the plan — enabling
  Atlas/Circe biometrics — not yet done).
- **`state.get_latest.v1` RPC is genuinely live and called repeatedly** by real services —
  confirmed in `orion-athena-state-service` logs: `source=name='cortex-exec'` and
  `source=name='cortex-orch'` requests firing roughly every 20-30s, each getting a
  `state.latest.reply.v1` with a `biometrics` key in the payload. This corroborates Q3: the
  RPC path is not a dead/test-only surface, cortex-exec's metacog step really calls it on a
  live cadence.
- **Even with zero live nodes to weight, the math would currently be a no-op anyway.** With
  only Athena reporting, `publish_cluster`'s weighted average degenerates to
  `weight × raw_value / weight` = `raw_value` — the weight cancels out identically regardless
  of what `CLUSTER_ROLE_WEIGHTS` says. So even if `BIOMETRICS_MODE` were flipped to `hub`
  today, `CLUSTER_ROLE_WEIGHTS` would have **zero numerical effect** until a second node
  (Atlas/Circe, per Phase 3) starts reporting — the weighting question is entirely
  future-facing, not a live discrepancy today.
- **Bonus finding, incidental but relevant to "has this ever been exercised":** the HTTP
  debug endpoint this service exposes is currently broken —
  `GET /state/latest` throws `TypeError: StateStore.get_latest() missing 1 required
  keyword-only argument: 'biometrics_stale_after_sec'` (confirmed live via `docker exec
  orion-athena-state-service curl ...`, traceback rooted at
  `services/orion-state-service/app/main.py:292`, `http_get_latest` calling
  `STORE.get_latest(req)` without the `biometrics_stale_after_sec` kwarg that
  `_handle_get_latest`, the bus RPC path, does pass at `main.py:181`). This confirms the HTTP
  surface is dead/unexercised — only the bus RPC path is real — and is a separate, small,
  pre-existing bug worth a one-line fix independent of this research question.

## Verdict table

| System | Weights | Fires today? | Real reach when it fires | Same question as field-topology? |
|---|---|---|---|---|
| Field-topology edges (`config/field/orion_field_topology.v1.yaml`) | Per-capability, e.g. `atlas→llm_inference: 0.85`, `athena→orchestration: 0.90` | Yes (`apply_diffusion`) | Layer 6 `SelfStateV1` dimension scores → drives/policy | — (this is the reference) |
| `CLUSTER_ROLE_WEIGHTS` (`services/orion-biometrics/.env:37`) | `{"atlas":0.7,"athena":0.3,"other":0.5}`, uniform across all pressure/headroom/composite keys | **No** — `BIOMETRICS_MODE=agent` live, `BiometricsHub`/`publish_cluster` never starts; confirmed 0 `biometrics.cluster.v1` messages ingested in 6h of live logs | When enabled: `constraint` + `composite` numbers reach `orion-cortex-exec`'s metacognition LLM prompt (`log_orion_metacognition_{draft,enrich}.j2`) via `metacog_biometrics_cue` — real, not display-only. The `spark.signal.v1`/DriveEngine path is a dead end (unmapped `level` dimension). | **Yes, same underlying question** ("how much should each node's health count"), computed a structurally different way (uniform per-node scalar vs. per-capability edge weight), with **opposite implied emphasis** from the topology edges for the same two nodes (topology: athena=0.90 orchestration / atlas=0.85 llm_inference, i.e. both weighted high in their own domain; `CLUSTER_ROLE_WEIGHTS`: atlas=0.7 > athena=0.3 uniformly, undocumented reasoning) |
| `orion-hub`'s own fallback (`BIOMETRICS_ROLE_WEIGHTS_JSON`, `services/orion-hub/.env:70`) | `{"atlas":0.6,"athena":0.4}` — a **third**, different set of numbers, keyed by literal node name not role-bucket | Yes, active right now (fallback branch is exactly what's exercised while no cluster message ever arrives) | Hub WebSocket JSON only (`websocket_handler.py:438`, `_with_biometrics` → `send_json`) — confirmed display-only, no gating logic found | Same question again — **a third independent reimplementation**, not previously flagged in the arsonist doc's mesh addendum table |
| `orion-state-service` "authoritative node / most-recent" (`store.py:102-113`) | N/A — single-writer-vs-newest policy, not a weighted blend | Yes, but the branch is currently trivial (only Athena ever produces `spark.state.snapshot.v1`) | Feeds `state.get_latest.v1` RPC replies to `cortex-exec`/`cortex-orch` (confirmed live, real reach) | **No** — different question. This concerns "whose *self-state snapshot* wins," not "how much does each node's health count toward a blended metric." Its biometrics-specific `_pick_latest` (`store.py:171-177`) is "pick one node's raw reading," also not a weighted blend, and also not connected to `CLUSTER_ROLE_WEIGHTS`. |

## Recommendation

This isn't a clean "keep both" or "retire" — it's two different findings bundled in the
original question:

1. **`orion-state-service`'s aggregation is not the same question as field-topology weighting
   at all** — it's single-writer/most-recent selection for spark snapshots, and pick-one/
   newest for biometrics summaries. Nothing to merge or retire here; it's correctly scoped as
   a "which reading is current" cache, not a "how much does each node count" blend. Its
   README's phrase "authoritative node if configured" is accurate but describes a narrower
   mechanism than the arsonist doc's framing implied.
2. **`CLUSTER_ROLE_WEIGHTS` is the real duplicate** — same underlying question as the
   field-topology edges, computed independently with numbers that don't obviously relate to
   the topology edges' reasoning, **and it isn't even the only duplicate**: `orion-hub` has its
   own third, silently-different implementation (`BIOMETRICS_ROLE_WEIGHTS_JSON`) that's the one
   actually active today (as the fallback), while `CLUSTER_ROLE_WEIGHTS`'s own code path is
   currently dark (`BIOMETRICS_MODE=agent`). Given (a) it does have one real non-display
   consumer (the metacog prompt cue) so it can't be dismissed as pure theater the way L7-L11
   was, but (b) it's dark today, computed three inconsistent ways across two services, and
   duplicates a question the topology edges already answer more principled-ly — the
   recommended sequencing is: **before Atlas/Circe come online (Phase 3), decide once whether
   node-weighting for the Hub/metacog "ops health" cue should derive from
   `config/field/orion_field_topology.v1.yaml`'s existing per-node/per-capability weights
   (single source of truth) rather than a bespoke role-weight scalar, and delete the two
   redundant reimplementations (`CLUSTER_ROLE_WEIGHTS` in orion-biometrics,
   `BIOMETRICS_ROLE_WEIGHTS_JSON` in orion-hub) in favor of it.** Don't fix this by merely
   reconciling the three numeric weight sets — that keeps three code paths computing the same
   thing. If keeping a separate "ops health" concern is still desired after that comparison,
   it should derive its per-node weight from the topology file's already-declared per-node
   nodes list (`config/field/orion_field_topology.v1.yaml:5-7` — `atlas`, `athena`, `circe` are
   already named there) rather than a new hand-picked scalar.

## Incidental bug found (not fixed here, flagged for a separate small patch)

`services/orion-state-service/app/main.py`'s HTTP `GET /state/latest` route (`http_get_latest`,
~line 292) calls `STORE.get_latest(req)` without the `biometrics_stale_after_sec` keyword
argument that `StateStore.get_latest()` requires and that the bus-RPC path
(`_handle_get_latest`, ~line 181) does pass — confirmed live via `docker exec
orion-athena-state-service curl localhost:.../state/latest`, which throws
`TypeError: StateStore.get_latest() missing 1 required keyword-only argument:
'biometrics_stale_after_sec'`. The HTTP debug endpoint is dead; only the bus RPC path is
exercised in practice.
