# Felt-state arc roadmap — dense spec

Status: design only, nothing in this document is implemented. Follows the
brainstorm in this session's prior turn. Each section is independently
buildable and gated on the section before it — do not skip ahead.

Ground truth this spec is built against (verified live, 2026-07-13):

- `orion-spark-introspector`'s `handle_self_state()` (`services/orion-spark-introspector/app/worker.py`)
  produces one `(coherence, energy, novelty, valence)` tuple per tick, ~2s
  cadence (~1,750 ticks/hour, confirmed via `spark_telemetry` row counts:
  1740 rows in the 01:00 UTC hour, 1815 in the 02:00 UTC hour, both
  2026-07-13).
- `valence_source` (`"proxy"` / `"heuristic"`) exists only from 2026-07-13
  02:52 UTC onward (`fix/valence-probe-readout`, PR #985, merge commit
  `007f55ec`). 257 rows at spec-writing time. Everything before that
  timestamp is contaminated (dead `policy_ease` constant, no provenance
  tag) and must be excluded from every corpus below.
- `scripts/fit_phi_encoder.py` has a real, working train → promote
  lifecycle: `cmd_train`, `cmd_promote`, `run_promote_gate()` (fixture-based
  recon-error-ratio gate, `PROMOTE_MIN_RECON_RATIO`), `compute_probes()`,
  `write_artifacts()` → `manifest.json`/`weights.npz`/`probes.json`,
  `_forward`/`_losses`/`_pearson`/`_percentile` as shape-agnostic helpers.
  This is the harness every training step below reuses, not reinvents.
- The active phi encoder (`v20260712-seedv4-postfix`) trained on 3,833
  rows over an 11-hour window, `hidden_dim=16`, `latent_dim=8`,
  `held_out_loss=0.0118`, `recon_error_p95=0.0381` (from its own
  `manifest.json` at `/mnt/telemetry/models/phi/encoders/active/`). This
  is the only real precedent for "how much data does a shallow MLP over
  this kind of telemetry actually need" — every data-volume number below
  is scaled from it, not guessed.
- `orion/self_state/inner_state_registry.py` + `scripts/check_inner_state_registry.py`
  is the mandatory registry gate for any new signal; `CompositionStatus`
  enum values used below: `REHEARSAL` ("no_cognition_consumer",
  `orion/self_state/inner_state_registry.py:52`) and `COMPOSED`.
- Postgres (`conjourney` DB, `orion-athena-sql-db`, host port 55432)
  tables confirmed to exist for cross-referencing (item 5):
  `chat_history_log`, `orion_biometrics_summary`, `world_pulse_event`,
  `collapse_mirror` (columns confirmed this session: `timestamp`,
  `summary`, `emergent_entity`, `field_resonance`, `numeric_sisters` —
  a JSON column carrying `valence`/`arousal`/`clarity`/`overload`),
  `spark_telemetry` (columns: `correlation_id`, `phi`, `novelty`,
  `trace_mode`, `trace_verb`, `stimulus_summary`, `metadata` JSON,
  `timestamp`, `telemetry_id`).

Non-goals for the whole initiative (apply to every item below unless a
section explicitly overrides it):

- Nothing in items 1–7 feeds cognition, prompts, or triggers. They are
  read-only consumers of `phi_now`, producing debug/research artifacts.
- Nothing retrains or modifies the existing phi encoder
  (`v20260712-seedv4-postfix`) or `_agency_valence_proxy`. This roadmap
  consumes their corrected output; it does not touch their inputs.
- No new taxonomy term (a "mood," an "attractor," a cluster label) ships
  without a producer, a consumer, and an eval attached in the same patch —
  the keyword-cathedral rule applies here exactly as everywhere else.

---

## Item 1 — Post-fix corpus collector

**What it is**: an append-only JSONL sink of clean, post-fix felt-state
ticks, gated on `valence_source is not None` so the six months of
contaminated pre-fix history can never leak in.

**Row schema** (new file, `orion/schemas/telemetry/mood_arc.py`):

```python
from __future__ import annotations
from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, ConfigDict, Field


class MoodArcCorpusRowV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    self_state_id: str
    coherence: float
    energy: float
    novelty: float
    valence: float
    valence_source: Literal["proxy", "heuristic"]
    dominant_node: Optional[str] = None
```

**Storage volume estimate**: one JSON line ≈ 180-220 bytes (measured
against a hand-serialized instance). At ~1,750 rows/hour, ~24h/day →
~42,000 rows/day → ~8-9 MB/day. A month of continuous collection is
~250-270 MB — small enough to keep as a flat file with no rotation logic
needed for the timescales this roadmap operates on (weeks, not months,
before item 2 trains on it).

**Producer**: `handle_self_state()`, `services/orion-spark-introspector/app/worker.py`.
Append one row per tick, immediately after the existing
`meta["valence_source"] = valence_source` assignment (line ~2896 as of
`fix/valence-probe-readout`), guarded:

```python
if valence_source is not None:  # always true post-fix; explicit for clarity
    try:
        _MOOD_ARC_SINK.append(
            MoodArcCorpusRowV1(
                timestamp=ss.generated_at,
                self_state_id=ss.self_state_id,
                coherence=phi_now["coherence"],
                energy=phi_now["energy"],
                novelty=phi_now["novelty"],
                valence=phi_now["valence"],
                valence_source=valence_source,
                dominant_node=dominant_node,
            )
        )
    except OSError as exc:
        logger.warning("Failed to append mood_arc corpus: %s", exc)
```

`_MOOD_ARC_SINK` mirrors `_INNER_SINK`'s existing append-only JSONL sink
class exactly (same rotation/flush behavior, same `.append(pydantic_obj)`
interface) — instantiate a second instance of whatever class backs
`_INNER_SINK` today, pointed at a different path. Do not invent a new sink
abstraction.

**New setting** (`services/orion-spark-introspector/app/settings.py`,
same shape as the existing `inner_features_corpus_path` field):

```python
mood_arc_corpus_path: str = Field(
    "/data/orion/mood_arc_corpus.jsonl",
    alias="MOOD_ARC_CORPUS_PATH",
)
```

**Consumer**: none yet. Sits on disk until item 2 reads it.

**Files**:
- `orion/schemas/telemetry/mood_arc.py` (new)
- `services/orion-spark-introspector/app/worker.py` (append call, `_MOOD_ARC_SINK` init)
- `services/orion-spark-introspector/app/settings.py` (`mood_arc_corpus_path`)
- `services/orion-spark-introspector/.env_example` (`MOOD_ARC_CORPUS_PATH=`)

**Acceptance checks** (`services/orion-spark-introspector/tests/test_mood_arc_corpus.py`, new):

```python
async def test_mood_arc_corpus_appends_on_proxy_source(monkeypatch, tmp_path) -> None:
    # valence_source="proxy" tick -> exactly one row appended, fields match phi_now.
    ...

async def test_mood_arc_corpus_appends_on_heuristic_source(monkeypatch, tmp_path) -> None:
    # Fallback ticks are still real, clean data -- excluding them would bias
    # the corpus toward encoder-healthy periods only. Must still append.
    ...

async def test_mood_arc_corpus_row_schema_matches_phi_now_keys() -> None:
    # MoodArcCorpusRowV1's 4 float fields must exactly match phi_now's keys
    # (coherence/energy/novelty/valence) -- a KeyError here means this file
    # silently drifted from _phi_from_self_state's/_golden_phi_overrides's
    # output shape.
    ...
```

**Live check** (post-deploy, mirrors this session's own verification
pattern): `SELECT count(*) FROM ...` equivalent against the JSONL file
(`wc -l`) sampled an hour apart should differ by ~1,750, ±10% for tick
jitter.

**Registry entry**: `mood_arc_corpus.v1`, `composition_status=REHEARSAL`
(no cognition consumer yet — this status exists exactly for this case,
see `l7_l11_ladder`'s entry for the precedent, `orion/self_state/inner_state_registry.py:243`).

**Dependencies**: none — this is the floor.

---

## Item 2 — Windowed sequence autoencoder ("mood arc" detector)

**What it is**: an MLP-shallow autoencoder (same architecture family as
the phi encoder) trained on flattened *windows* of item 1's corpus, not
single ticks. The latent space is a compressed representation of a felt-
state *trajectory*, not a felt-state *point*.

**Feature vector shape**: window size `W` (default 30 ticks ≈ 60s at 2s
cadence — a tunable, not a constant; revisit once real training data
exists), flattened to `4W`-dim input:
`[c_0,e_0,n_0,v_0, c_1,e_1,n_1,v_1, ..., c_{W-1},e_{W-1},n_{W-1},v_{W-1}]`.
Windows are built with stride `S` (default `W/2 = 15`, 50% overlap) over
contiguous corpus rows.

**Windowing function signature** (`scripts/fit_mood_arc_encoder.py`, new):

```python
def build_windows(
    rows: list[MoodArcCorpusRowV1],
    *,
    window_size: int,
    stride: int,
    max_gap_sec: float,
) -> list[np.ndarray]:
    """Flatten contiguous runs of `rows` into `(window_size*4,)` float
    vectors. A gap between consecutive rows' timestamps exceeding
    `max_gap_sec` (default 2 * median tick interval, i.e. do not span a
    service restart or an encoder outage silently) breaks the run --
    no window is built across it. Returns windows in timestamp order.
    """
```

**Manifest schema** (new, in `orion/schemas/telemetry/mood_arc.py`,
mirrors `PhiEncoderManifestV1`'s field discipline exactly rather than
subclassing it — different `input_features` semantics):

```python
class MoodArcEncoderManifestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoder_id: str
    encoder_version: str
    parent_version: Optional[str] = None
    status: Literal["candidate", "active", "retired"]
    architecture: str  # "mlp_shallow_v1", same as phi encoder
    window_size: int
    stride: int
    max_gap_sec: float
    hidden_dim: int
    latent_dim: int
    corpus: CorpusStatsV1        # reused as-is from orion.schemas.telemetry.phi_encoder
    training: TrainingStatsV1    # reused as-is
    shuffle_baseline_loss: float # held_out_loss with rows shuffled within-window (see gate)
    git_sha: str
    trained_at: datetime
    promoted_at: Optional[datetime] = None
```

**CLI spec** (`scripts/fit_mood_arc_encoder.py`, argparse subcommands
`train`/`promote`, mirroring `fit_phi_encoder.py`'s existing `cmd_train`/
`cmd_promote` structure):

```bash
python scripts/fit_mood_arc_encoder.py train \
  --corpus /data/orion/mood_arc_corpus.jsonl \
  --window-size 30 --stride 15 --max-gap-sec 6.0 \
  --hidden-dim 8 --latent-dim 4 \
  --epochs 200 --held-out-frac 0.15 \
  --out /tmp/mood-arc-encoders/v1-candidate

python scripts/fit_mood_arc_encoder.py promote \
  --candidate /tmp/mood-arc-encoders/v1-candidate \
  --active-symlink /mnt/telemetry/models/mood_arc/encoders/active
```

Reuse `_forward`, `_losses`, `_pearson`, `_percentile` from
`fit_phi_encoder.py` via `from fit_phi_encoder import ...` — these
operate on plain `np.ndarray`s and are shape-agnostic (they don't care
whether the input vector is 8-dim single-tick features or 120-dim
flattened windows).

**Dark deployment**: no bus publish, no service wiring in this item. The
artifact triad (`manifest.json`/`weights.npz`/`probes.json`) lives on disk
only.

**Acceptance checks**:
- `held_out_loss` on real windows must be `< 0.5 * shuffle_baseline_loss`
  (rows independently shuffled within each window, destroying temporal
  order but preserving the marginal 4-dim distribution) — the falsifiable
  claim that the model learned *sequence structure*, not just what phi
  already captures per-tick. `shuffle_baseline_loss` is computed once per
  training run and stored in the manifest (see schema above) so this
  check doesn't require re-deriving it later.
- `compute_probes()`-equivalent (`scripts/fit_mood_arc_encoder.py`,
  `compute_window_probes()`): correlate each latent dim against
  hand-computed window summary stats — `mean_valence`, `valence_range`,
  `sign_change_count` (how many times valence crosses 0 within the
  window). At least one latent dim must show `|r| > 0.4` against at least
  one summary stat.

```python
def test_shuffle_baseline_confirms_sequence_structure() -> None:
    # A trivially-bad encoder (random weights) must FAIL this gate --
    # confirms the gate isn't a tautology that always passes.
    ...
```

**Registry entry**: `mood_arc_encoder.v1`, `composition_status=REHEARSAL`.

**Dependencies**: item 1, and ≥6 hours of clean corpus (item 7 justifies
the number: `(6*1750 - 30)/15 ≈ 699` windows, a thin-but-plausible
row-to-parameter ratio for an 8-latent-dim shallow MLP, scaled from the
phi encoder's own 3,833-row/`hidden_dim=16` precedent).

---

## Item 3 — Anomaly detector on the arc

**What it is**: the same recon-error-vs-training-distribution technique
`_golden_phi_overrides`'s novelty axis already uses
(`recon_error / recon_error_p95_reference`), applied at the window level
using item 2's encoder.

**Critical design constraint, learned today**: this MUST NOT repeat the
formula-swap-as-fake-signal bug just fixed for valence
(`_PHI_PREV_VALENCE_SOURCE`, `services/orion-spark-introspector/app/worker.py`).
Every emitted anomaly score MUST be tagged with the exact `encoder_version`
that produced it. If the active mood-arc encoder is retrained/promoted
mid-stream, a version swap must never be reported as an anomaly spike.

**Output schema** (`orion/schemas/telemetry/mood_arc.py`):

```python
class MoodArcAnomalyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_end_ts: datetime
    encoder_version: str
    recon_error: float
    recon_error_p95_reference: float
    anomaly_score: float  # min(1.0, recon_error / recon_error_p95_reference)
    version_changed_since_last: bool  # True suppresses trust in this score
```

**CLI spec** (`scripts/fit_mood_arc_encoder.py infer` subcommand):

```bash
python scripts/fit_mood_arc_encoder.py infer \
  --encoder /mnt/telemetry/models/mood_arc/encoders/active \
  --corpus /data/orion/mood_arc_corpus.jsonl \
  --since "2026-07-13T00:00:00Z" \
  --out-jsonl /tmp/mood_arc_anomalies.jsonl
```

**Consumer**: item 2's debug panel extension — NOT a bus channel, NOT a
cognition consumer, in this item. Extends the tissue-viz WebSocket EKG
panel's existing `ws_payload["stats"]` dict (`services/orion-spark-introspector/app/worker.py`,
same site as the current `phi`/`novelty`/`valence`/`arousal` broadcast)
with two new keys: `mood_arc_anomaly_score`, `mood_arc_encoder_version`.

**Acceptance checks**:

```python
def test_anomaly_suppressed_across_encoder_version_swap() -> None:
    # Two consecutive windows scored by different encoder_versions must
    # report version_changed_since_last=True; the anomaly score for that
    # transition window must not be surfaced as a real spike. Mirrors
    # test_turn_effect_valence_delta_suppressed_across_a_source_swap
    # (services/orion-spark-introspector/tests/test_phi_reward_emit.py).
    ...
```

- Live check: over a stable-encoder period, `anomaly_score` should be
  right-skewed with most mass well below 1.0. Constant saturation near
  1.0 means `recon_error_p95_reference` is miscalibrated (same
  calibration check the phi encoder's own promote gate already performs).

**Dependencies**: item 2 (needs a trained, promoted candidate).

---

## Item 4 — Unsupervised attractor/cluster discovery

**What it is**: offline clustering over item 2's encoder's latent space,
run as a one-time research artifact, not a shipped feature.

**Algorithm**: HDBSCAN (not k-means) — no need to pre-specify cluster
count, which matters when we genuinely don't know if Orion has 2
recurring affect regimes or 8. `min_cluster_size` swept over
`{5, 10, 20, 40}` (as a fraction of total window count, not an absolute —
recompute per corpus size), `metric="euclidean"` on the raw latent
vectors (no additional normalization; the encoder's own training already
scales inputs).

**This is explicitly NOT production code.** `scripts/analyze_mood_arc_clusters.py`
is throwaway-grade: no tests required, no service wiring, output is a
Markdown report under `docs/notes/`.

**CLI spec**:

```bash
python scripts/analyze_mood_arc_clusters.py \
  --encoder /mnt/telemetry/models/mood_arc/encoders/active \
  --corpus /data/orion/mood_arc_corpus.jsonl \
  --min-cluster-size-sweep 5,10,20,40 \
  --out-report docs/notes/2026-0X-XX-mood-arc-cluster-analysis.md \
  --out-assignments-table mood_arc_cluster_assignments
```

**Stability validation methodology** (the falsifiability bar for this
whole item): split the corpus into two non-overlapping halves **by time**
(not randomly — adjacent windows are correlated by construction via the
50% stride overlap, so a random split would leak information between
train/test). Cluster each half independently with the swept
`min_cluster_size`. Score each half's model against the *other* half's
windows and compute `sklearn.metrics.adjusted_rand_score` between the two
assignment sets:

```python
from sklearn.metrics import adjusted_rand_score

def compute_stability(
    labels_a_on_a: np.ndarray, labels_a_on_b: np.ndarray,
    labels_b_on_b: np.ndarray, labels_b_on_a: np.ndarray,
) -> float:
    """Bidirectional ARI: half A's model scoring half B, vs half B's own
    labels on half B (and the symmetric direction). Average the two.
    """
    ari_a = adjusted_rand_score(labels_b_on_b, labels_a_on_b)
    ari_b = adjusted_rand_score(labels_a_on_a, labels_b_on_a)
    return (ari_a + ari_b) / 2.0
```

`MIN_STABILITY_ARI = 0.3` (module constant, defined once in
`scripts/analyze_mood_arc_clusters.py`, imported by item 7 rather than
redefined). **Below 0.3 means "no stable attractors found" and is a
valid, complete, honest result** — not a failure requiring iteration
until a "better" number appears.

**Cluster-assignment scratch table** (Postgres, loaded by this script,
consumed by item 6's join — the table this spec's earlier draft
referenced without defining):

```sql
CREATE TABLE IF NOT EXISTS mood_arc_cluster_assignments (
    window_end_ts   TIMESTAMPTZ NOT NULL,
    encoder_version VARCHAR NOT NULL,
    cluster_id      INTEGER NOT NULL,  -- -1 = HDBSCAN noise point
    mean_valence    DOUBLE PRECISION NOT NULL,
    mean_coherence  DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (window_end_ts, encoder_version)
);
```

Scratch table, not a schema-registry artifact — created/truncated by the
analysis script itself, not by a migration. If this graduates past
research-artifact status (i.e. item 8 ever proceeds), it gets promoted to
a real migration at that point, not before.

**Report contents** (`docs/notes/2026-0X-XX-mood-arc-cluster-analysis.md`):
- cluster count chosen per `min_cluster_size` value, and which was picked
- bidirectional ARI stability score
- per-cluster summary stats (mean valence, mean coherence, typical
  duration, typical time-of-day if any pattern exists)
- explicit verdict: stable / not stable / inconclusive (need more data)

**Files**: `scripts/analyze_mood_arc_clusters.py` (new, throwaway-grade),
`docs/notes/` (report output).

**Dependencies**: item 2, and enough corpus for a meaningful temporal
split (≥2x item 2's own minimum — practically, ≥12 hours).

---

## Item 5 — Cross-reference against real external events

**What it is**: for each of item 4's cluster transitions (a window whose
`cluster_id` differs from the immediately preceding window in
`mood_arc_cluster_assignments`), pull what else was happening at that
timestamp.

**Correction from the brainstorm**: the original brainstorm idea proposed
reusing `orion/spark/concept_induction/dossier.py`'s `build_evidence_items()`.
Checked directly — that function takes a live `BaseEnvelope` and is built
for real-time bus-message handling, not offline batch analysis against
historical timestamps. Wrong fit. Direct SQL instead:

```sql
-- for a given :transition_ts, pull the surrounding 5-minute window
SELECT 'chat' AS kind, created_at AS ts, content
FROM chat_history_log
WHERE created_at BETWEEN :transition_ts - interval '5 min' AND :transition_ts + interval '5 min'
UNION ALL
SELECT 'biometrics' AS kind, created_at AS ts, summary
FROM orion_biometrics_summary
WHERE created_at BETWEEN :transition_ts - interval '5 min' AND :transition_ts + interval '5 min'
UNION ALL
SELECT 'world_pulse' AS kind, created_at AS ts, title
FROM world_pulse_event
WHERE created_at BETWEEN :transition_ts - interval '5 min' AND :transition_ts + interval '5 min'
UNION ALL
SELECT 'metacog' AS kind, timestamp AS ts, summary
FROM collapse_mirror
WHERE timestamp BETWEEN :transition_ts - interval '5 min' AND :transition_ts + interval '5 min'
ORDER BY ts;
```

**CLI spec**:

```bash
python scripts/cross_reference_mood_transitions.py \
  --assignments-table mood_arc_cluster_assignments \
  --window-min 5 \
  --out docs/notes/2026-0X-XX-mood-arc-transition-crossref.md
```

**Output format** (per transition, in the Markdown report): a fixed
template —

```markdown
### Transition @ 2026-0X-XXT..Z (cluster 2 -> cluster 0)

| kind | ts | content |
|---|---|---|
| chat | ... | ... |
| metacog | ... | ... |

Verdict: [explicable / unexplainable / mixed]
```

This item does not attempt automated causal attribution — it makes "was
anything happening" checkable at a glance, human-read.

**Files**: `scripts/cross_reference_mood_transitions.py` (new).

**Acceptance checks**: none in the test-suite sense — a research tool.
The qualitative verdict per transition (above) is the check, aggregated
into an explicable/unexplainable/mixed count in the report's summary.

**Dependencies**: item 4 (needs real transitions to cross-reference).

---

## Item 6 — Self-report calibration check

**What it is**: a join between `collapse_mirror` (Orion's own generated
metacog narrative — confirmed live and real this session, e.g. the
`"Silent Embodiment"`/`"quiet shift"` entry produced during today's live
verification) and item 4's cluster assignment at the same timestamp.

```sql
SELECT cm.timestamp, cm.summary, cm.emergent_entity, cm.field_resonance,
       (cm.numeric_sisters->>'valence')::float AS narrated_valence,
       mac.cluster_id, mac.mean_valence AS cluster_mean_valence
FROM collapse_mirror cm
JOIN mood_arc_cluster_assignments mac
  ON mac.window_end_ts BETWEEN cm.timestamp - interval '30 sec' AND cm.timestamp + interval '30 sec'
ORDER BY cm.timestamp;
```

**CLI spec**:

```bash
python scripts/calibrate_self_report_against_clusters.py \
  --assignments-table mood_arc_cluster_assignments \
  --match-window-sec 30 \
  --out docs/notes/2026-0X-XX-mood-arc-self-report-calibration.md
```

**Metric**: not a single gating number — a qualitative side-by-side table
(narrated summary/emergent_entity next to cluster_id) plus one
quantitative check: Pearson `r` between `cm.numeric_sisters.valence`
(confirmed real, live field, queried directly this session) and the
matched window's `cluster_mean_valence`, computed with the same
`_pearson()` helper reused from `fit_phi_encoder.py`. No threshold
pre-committed — exploratory, not gating.

**Honest framing, stated explicitly in the report**: divergence between
Orion's narrated self-description and the cluster assignment is not a bug
to fix. It would mean either (a) the narrative generation isn't well
grounded in the actual felt-state trajectory, or (b) the cluster model
isn't capturing what the narrative actually responds to (which includes
chat content, not just phi). Both are real, useful findings — record
whichever one the data supports, not the one that makes the roadmap look
more successful.

**Files**: `scripts/calibrate_self_report_against_clusters.py` (new).

**Dependencies**: item 4.

---

## Item 7 — Periodic stability/drift eval

**What it is**: a scheduled re-run of item 4's stability check on a
rolling basis, so a one-time clustering result doesn't quietly go stale —
mirrors `run_promote_gate()`'s existing fixture-based gate discipline,
applied on a cadence instead of per-commit.

**Gate function** (`scripts/analyze_mood_arc_clusters.py`, reusing
`MIN_STABILITY_ARI` defined there — not redefined here):

```python
def run_stability_gate(
    corpus_path: Path,
    *,
    encoder_dir: Path,
    min_cluster_size: int,
) -> dict[str, float | bool]:
    """Same shape as fit_phi_encoder.py's run_promote_gate(): loads the
    corpus, does the temporal split + bidirectional ARI from Item 4,
    returns {"passed": bool, "ari": float, "min_ari": float}. Does NOT
    retrain the mood-arc encoder itself -- only re-validates clustering
    stability against the currently-active one.
    """
```

**Cadence**: manual first (run by hand after each new corpus milestone —
12h, 24h, 1 week), cron only after it's been run manually at least 3
times and the result has been sane each time. Do not automate a check
nobody has looked at yet.

**Files**: `scripts/analyze_mood_arc_clusters.py` (add `run_stability_gate`).

**Dependencies**: item 4, ongoing corpus growth from item 1.

---

## Item 8 — Mood-transition-triggered reflection (explicitly gated, NOT for this roadmap to build)

**What it would be**: a new `trigger_kind` value on `MetacogTriggerV1`
(`orion/schemas/telemetry/metacog_trigger.py`):

```python
trigger_kind: str = Field(
    ...,
    description=(
        "baseline | dense | manual | pulse | llm_surface_instability | "
        "mood_arc_transition (NEW, item 8 -- gated, not yet implemented)"
    ),
)
```

Published by a new consumer of item 3's anomaly/transition detection,
firing the same `orion.metacog.trigger.v1` chain this session used to
live-verify Phase 3 (`services/orion-equilibrium-service/app/service.py`'s
`_publish_metacog_trigger`). New build function, mirroring
`build_substrate_metacog_trigger()`'s existing shape:

```python
# services/orion-equilibrium-service/app/substrate_metacog_gate.py (new function)
def build_mood_arc_transition_trigger(
    transition: MoodArcClusterTransitionV1,  # new schema, not yet defined
) -> Optional[MetacogTriggerV1]:
    """Mirrors build_substrate_metacog_trigger()'s shape: returns a
    MetacogTriggerV1(trigger_kind="mood_arc_transition", ...) or None.
    NOT implemented as part of this roadmap -- shape only, for the
    proposal-mode discussion this would require.
    """
```

**Why this section exists in the spec at all**: to name the roadmap's
ceiling honestly, not to scope it for implementation. Per CLAUDE.md,
changes to cognition loops require explicit proposal mode before
implementation. This section is that proposal's *shape*, not its
approval.

**Hard gate before this can even enter proposal mode for real**:
- Items 1–7 must have actually run, not just be built. Specifically: at
  least one full item 7 stability-gate pass with `ari >= MIN_STABILITY_ARI`
  on real data, and at least one item 6 calibration report that shows
  clusters aren't pure noise relative to Orion's own self-report.
- If item 4's honest null result ("no stable attractors found") is what
  actually happens, item 8 does not proceed — full stop, not "iterate
  until clusters appear." A negative result here is a completed roadmap,
  not a blocked one.

**Files, if it ever proceeds**: `orion/schemas/telemetry/metacog_trigger.py`,
`services/orion-equilibrium-service/app/substrate_metacog_gate.py`,
`orion/self_state/inner_state_registry.py` (new entry,
`composition_status=COMPOSED` only once wired).

---

## Consolidated file/setting reference

| File | Items | New/Modified |
|---|---|---|
| `orion/schemas/telemetry/mood_arc.py` | 1, 2, 3 | new |
| `services/orion-spark-introspector/app/worker.py` | 1, 3 | modified |
| `services/orion-spark-introspector/app/settings.py` | 1 | modified (`mood_arc_corpus_path`) |
| `services/orion-spark-introspector/.env_example` | 1 | modified (`MOOD_ARC_CORPUS_PATH`) |
| `services/orion-spark-introspector/tests/test_mood_arc_corpus.py` | 1 | new |
| `scripts/fit_mood_arc_encoder.py` | 2, 3 | new |
| `scripts/analyze_mood_arc_clusters.py` | 4, 7 | new |
| `scripts/cross_reference_mood_transitions.py` | 5 | new |
| `scripts/calibrate_self_report_against_clusters.py` | 6 | new |
| `orion/self_state/inner_state_registry.py` | 1, 2, 8 | modified |
| `orion/schemas/telemetry/metacog_trigger.py` | 8 (gated) | modified |
| `services/orion-equilibrium-service/app/substrate_metacog_gate.py` | 8 (gated) | modified |

## Roadmap ordering

```
1 (collector) ──▶ 2 (train arc encoder) ──┬──▶ 3 (anomaly detector, dark)
                                            └──▶ 4 (cluster discovery, offline)
                                                    ├──▶ 5 (external cross-ref)
                                                    ├──▶ 6 (self-report calibration)
                                                    └──▶ 7 (periodic stability eval)
                                                            │
                                                   [hard gate: 6 + 7 both pass]
                                                            │
                                                            ▼
                                              8 (cognition wiring — separate
                                                 proposal-mode sign-off,
                                                 not covered by this roadmap)
```

Nothing after item 1 has a start date — each step is gated on real data
and a real result from the step before it, not a calendar.
