# orion/mood_arc — felt-state-trajectory autoencoder

Offline research pipeline testing whether Orion has recurring **trajectory**
structure in its felt-state substrate — not just a single-tick `phi` value,
but a pattern across a *window* of ticks. It is a windowed autoencoder
trained over `field_channel_corpus.v1`'s raw, per-node/per-capability channel
pressures (`orion.schemas.telemetry.field_channel_corpus.FieldChannelCorpusRowV1`,
produced by `services/orion-field-digester`).

This is Item 2 (+3) of
`docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md`. For the
full technical history — why the original corpus was invalidated, what was
tried and rejected, the real incidents this pipeline has already been through,
and the open design question blocking the rest of the roadmap — see
`orion/mood_arc/docs/DESIGN.md`. This README is the practical "how do I run
it" reference; the docs directory is the "why does it look like this" one.

**Status: dark deployment.** Everything here is an offline, manually-invoked
CLI. No bus publish, no service process, no container, no cognition consumer.
Registered `REHEARSAL` in `orion/inner_state_registry.py` (moved from
`orion/self_state/inner_state_registry.py` during the 2026-07-22 SelfStateV1
module deletion — path corrected here, no behavior change)
(`mood_arc_encoder.v1`, `mood_arc_corpus.v1`, `field_channel_corpus.v1`) —
that status is correct and intentional, not a gap to close as part of this
patch.

## What's in this module

- `fit_encoder.py` — the CLI. Four subcommands: `train` (fit a candidate
  encoder against a corpus slice), `detect-anomalies` (score a corpus slice
  against an already-trained encoder), `promote` (copy a candidate that
  passed its floor gate to the durable models root and mark it active), and
  `enrich-corpus` (join the phi-v2 audit's dense-enough live signals onto an
  existing corpus before training — see the `v4` section below).
- `corpus_enrichment.py` — the Postgres-reading half of `enrich-corpus`,
  kept separate from `fit_encoder.py` so the training/scoring/promotion path
  stays DB-free and importable with no `psycopg2` installed.
- `tests/` — covers windowing, field selection/pruning, the purged temporal
  split (single- and multi-block), the two-tier gate, the AR(1) surrogate,
  the anomaly detector, and the corpus-enrichment asof/forward-fill join.
- `docs/DESIGN.md` — the technical history and the open valence-replacement
  design question.

The corpus row schema (`FieldChannelCorpusRowV1`) and the manifest/anomaly
schemas (`MoodArcCorpusRowV1`, `MoodArcEncoderManifestV1`) stay in
`orion/schemas/telemetry/` — this module trains against them, it does not own
them, per this repo's shared-schema convention.

## Running `train`

Trains a shallow MLP autoencoder over flattened windows of the corpus, gated
by a two-tier check: a hard **floor** gate (must beat a within-window-shuffle
baseline by 2x — the falsifiable claim that the model learned real sequence
structure, not just per-tick statistics) and a diagnostic-only **ceiling**
comparison against an AR(1) surrogate (rules out "the encoder just relearned
the field-digester's own decay filter" — see `docs/DESIGN.md` for why that
check exists at all).

Current best-known-good configuration (validated 2026-07-18, see
`docs/DESIGN.md`'s "the real production run" section):

```bash
python orion/mood_arc/fit_encoder.py train \
  --corpus /mnt/telemetry/field_channels/corpus/field_channels.jsonl \
  --min-generated-at 2026-07-17T04:32:14Z \
  --hidden-dim 128 --latent-dim 64 \
  --epochs 250 \
  --out /tmp/mood-arc-encoders/v1-candidate
```

- `--min-generated-at 2026-07-17T04:32:14Z` is the first data-quality
  cutoff: rows before this timestamp reflect known-broken channel behavior
  from before a 7-PR fix sprint (PRs #1108-#1113, #1115). Training on the
  full unfiltered corpus produces contaminated field selection and fails the
  gate — see `docs/DESIGN.md`. This cutoff is also documented in
  `services/orion-field-digester/README.md`'s "`field_channel_corpus.v1`
  training-data quality cutoff" section; keep the two in sync if it's ever
  revised.
- **Second cutoff, `--min-generated-at 2026-07-22T04:35:01Z` (PR #1248,
  merged + deployed):** `stream_backlog_pressure`/`catalog_drift_pressure`/
  `observer_failure_pressure`/`reliability_pressure`/`contract_pressure`
  could get permanently stuck at a stale value (an `add`-mode perturbation
  bug in `services/orion-field-digester/app/ingest/state_deltas.py`, fixed
  in PR #1248, merged 2026-07-22T04:32:27Z, `orion-field-digester` restarted
  2026-07-22T04:35:01Z) — confirmed live as the cause of
  `catalog_drift_pressure` alone driving ~66% of average reconstruction
  error against `field_channel_anomaly.v2`. Unlike the first cutoff, this
  contamination window has no known start — `catalog_drift_pressure` was in
  fact stuck for the *entire* span of `v2`'s training corpus
  (2026-07-17T04:32:14Z-2026-07-22T01:30:24Z), confirmed by the fact that
  post-fix, with the real value correctly reading `0.0`, `v2` (trained on
  the stuck ~`0.135` reading) flipped to flagging the *correct* value as
  anomalous instead: `telemetry_anomaly` still fired 20 times in the 41
  minutes after the restart, same channel, opposite direction. **`v2` was
  not a valid baseline** — see the third cutoff below for what actually
  shipped as its replacement. If both cutoffs apply, use whichever is
  later.
- **Third cutoff, `--min-generated-at 2026-07-22T08:29:48Z` (PR #1262,
  merged + deployed):** two bugs in `orion/substrate/biometrics_loop`'s
  active-node-pressure reducer, upstream of this service. (1) `availability`
  one-way ratchet — a transient staleness blip could permanently flag a node
  unavailable, with no rule able to clear it; not one of `v2`'s 16 trained
  channels, but its exclusion from training was itself likely an artifact of
  this bug (permanently-stuck looks like no-signal). (2) merge-window dedup
  was a no-op across ticks — `node:atlas` accepted 767 "reinforce" deltas in
  2 hours instead of the ~24 a working 5-minute window should allow,
  inflating `pressure_score` and therefore `cpu_pressure` (`mode="add"`, one
  of `v2`'s 16 trained channels) via `active_node_pressure` deltas'
  `"strain"` pressure kind. See `services/orion-field-digester/README.md`'s
  "third training-data quality cutoff" section for the full detail.
  `2026-07-22T08:29:48Z` is `orion-substrate-runtime`'s restart time (the
  later of the two services this fix spans, and the binding one).
  Confirmed live: `node:atlas`'s `availability` recovered to `1.0`
  immediately post-restart, and its reinforce-delta rate dropped from
  ~1/9s to quiet within the first minute.

  **`v3` (currently deployed) trained against exactly this cutoff**:
  18,377 rows / 10.3h clean data, `floor_ratio=0.210` (pass, CI
  0.174-0.231), `ceiling_ratio=0.190` — within 0.001 of `v2`'s 0.189
  despite the much smaller/different corpus (an early n=2 signal this
  number may be stable, not yet the full multi-seed calibration the
  roadmap wants). `availability` survived field selection for the first
  time (`std=0.0398`), confirming its prior exclusion was the ratchet bug,
  not a real absence of signal. See `services/orion-field-digester/
  README.md`'s "Deployed model history" table for the full `v1`/`v2`/`v3`
  comparison.
- **Fourth cutoff, `--min-generated-at 2026-07-22T19:18:31Z` (commit
  `a98854a2` + PR #1267, both merged + deployed):** found while auditing
  chat/route prediction-error instruments — **`v3`, the currently deployed
  encoder, is also contaminated by this cutoff**, since `v3` trained on the
  third cutoff's window (2026-07-22T08:29:48Z onward), which predates this
  fix. `prediction_error` is confirmed as one of `v2`'s (and, by field
  selection carrying over, presumably `v3`'s) trained channels via
  `docs/DESIGN.md`'s "15 channels survived selection + pruning" list — a
  `max()`-merge across five nodes' shadow prediction-error instruments
  (`orion/substrate/prediction_error.py`, wired in `services/
  orion-substrate-runtime`). Two of those five were broken until today:
  `execution_prediction_error()`/`route_prediction_error()` matched on an
  exact `trace_id` that structurally never recurs (permanently `0.0`, fixed
  in `a98854a2`), and `chat_prediction_error()` skipped every brand-new turn
  (also permanently `0.0` in production despite 241 real accumulated chat
  turns, fixed in PR #1267). Confirmed directly against the training corpus
  file: its earliest available rows (2026-07-18T20:41Z) already read
  `prediction_error = 3e-323` — the `apply_decay()` floor a stale
  `NODE_DECAY_CHANNELS` entry settles to — consistent with this channel
  sitting at or near that floor for its entire history in this corpus,
  including throughout `v3`'s own training window, not carrying real
  learnable variance the way field selection presumably assumed.
  `2026-07-22T19:18:31Z` is `orion-substrate-runtime`'s restart time (the
  later of the two fixes' merges, both landing in that one service — the
  binding cutoff). Confirmed live post-restart, directly against the corpus
  file: rows minutes after read `prediction_error = 0.0671`-`0.1171`,
  varying across ticks rather than stuck at the old floor. See `services/
  orion-field-digester/README.md`'s "fourth training-data quality cutoff"
  section for the full detail, including the honest caveat that exact
  per-node attribution of the new reading hasn't been traced further.
  **A `v4` retrain against this cutoff would give `prediction_error` its
  first-ever real signal** (or whichever prior cutoff is later — currently
  this one); do not retrain yet, only minutes of clean data exist as of
  this writing. `v3` remains the right model to keep serving until then —
  it is not invalidated on the channels it actually gates on
  (`floor_ratio`/`ceiling_ratio` don't single out `prediction_error`), just
  carrying one contaminated-but-not-dominant input feature the same way
  `v2` carried `catalog_drift_pressure` before the second cutoff.
- **Fifth cutoff, `--min-generated-at 2026-07-23T06:10:08Z` (commit `5b1cc0fa`, merged +
  deployed):** found while checking whether the FCC-motor signal bundle
  (`docs/superpowers/specs/2026-07-23-fcc-motor-field-digester-signals-design.md`) endangered
  `v3` — its 5 brand-new channels don't (confirmed: `app/anomaly_scorer.py`'s `score_latest()`
  selects strictly by `self._manifest.channel_names`, and `build_windows()` defaults any field
  not present in a row to `0.0` by design — a genuinely new channel name is silently ignored,
  not a contamination risk). But two of `v3`'s own 15 *trained* channels were live-changed by
  that same investigation's fix: `execution_load` (`min(1.0, started_step_count/8.0)` over a
  blended cortex-exec+harness-governor counter, hard-capped) and `reasoning_load` (a boolean
  `0.35`/`0.05` wearing a magnitude's name) both replaced with real, continuously-varying
  formulas — see `services/orion-field-digester/README.md`'s entries for the full mechanism.
  `v3` was trained on the *old* distributions for both. Every score computed since
  `orion-field-digester`'s `2026-07-23T06:10:08Z` restart compares new-distribution live data
  against a stale-distribution-trained model — the same contamination class as cutoffs two
  through four. **Resolved 2026-07-24**: exactly 2 `field_channel_anomaly_flagged` events
  fired, both 11-14 minutes post-restart, none since (re-checked at 11+ hours post-restart).
  Timing matches `app/anomaly_scorer.py`'s own documented cold-start artifact (reconcile-seeded
  defaults still in the buffer right after a restart), not a sustained distribution-shift
  problem — a real shift would keep firing as the new formulas kept flowing, not fire twice and
  go silent. Read as noise. **Second shift on `reasoning_load` the same day (2026-07-24)**:
  the FCC-motor/harness-governor path gained its own real magnitude too
  (`reasoning_output_tokens`, real provider token counts from the FCC CLI's own result-event
  usage object — previously that path had no magnitude at all, always the `0.35`/`0.05` flag).
  Same cutoff timestamp already covers it (no new cutoff needed), but noted here since it's a
  second, later change to an already-cutoff channel, not just the one `5b1cc0fa` fix. **Do not
  retrain yet** — matching every prior cutoff's own caveat, only minutes of clean post-fix data
  exist as of this writing. `v3` remains the model to keep serving until a `v4` retrain against
  this cutoff (or whichever is later) is warranted.
  **Materialized same day as a sixth cutoff:** `transport_pressure`/`bus_health` renamed to
  `stream_backlog_pressure`/`stream_backlog_health` 2026-07-24 (Juniper's explicit naming
  choice). Both are 2 of `v3`'s 15 trained channels; `build_windows()` now finds them genuinely
  absent by name and silently defaults both to `0.0` (per its missing-field convention) on every
  post-rename row — a different failure mode than a distribution shift, but still cutoff-class,
  not the "safe, no-training-impact" rename it might look like. See
  `services/orion-field-digester/README.md`'s matching sixth-cutoff entry for the
  `--min-generated-at` command and full reasoning. Juniper's explicit call: don't block the
  rename on retrain timing — `v3` keeps serving with 2 of 15 inputs dark until a `v4` retrain,
  same posture as every prior cutoff.
- **Seventh cutoff, same day (2026-07-24):** `execution_load` renamed to
  `cortex_exec_step_load` (the cortex-exec-scoped sibling of `harness_step_load`), same
  genuinely-absent-by-name failure mode as the sixth cutoff. Only 1 of `v3`'s 15 trained
  channels affected here (vs. 2 for the sixth cutoff). The same patch also fixed a live
  cross-lane stomp bug on this channel's values (a `harness_motor`-lane delta's
  structurally-always-`0.0` reading was overwriting the real cortex-exec-lane value via
  `mode="replace"` on the shared node key) — a behavior change to the channel's values,
  not just its name, landing in the same restart. See
  `services/orion-field-digester/README.md`'s matching seventh-cutoff entry and
  `cortex_exec_step_load` glossary entry for the full mechanism and `--min-generated-at`
  command. Same posture as every prior cutoff: don't block on retrain timing, `v3` keeps
  serving with 1 of 15 inputs dark until a `v4` retrain.
- **Scope caveat on `prediction_error`'s "transport" contributor (found 2026-07-22, not a new
  cutoff — no code changed, just what the channel actually means):** `prediction_error` is a
  `max()`-merge across five nodes; one of them, `node:substrate.transport`, is fed entirely by
  whatever streams `orion-bus`'s bus-observer role watches (`BUS_OBSERVER_STREAMS`,
  `services/orion-bus/.env_example`) — currently `orion:stream:world_pulse:run:result` and its
  DLQ, **the only two real Redis Streams anywhere in the architecture** (everything else is
  pub/sub, which has no depth/backlog concept to measure). This is not general bus/transport
  health across services, despite the name — it's whether one specific service's result queue
  backs up. Confirmed live: that queue has sat at a constant 91 messages for the entire
  post-second-cutoff corpus window (zero variance) — **not** an unconsumed backlog (corrected
  2026-07-23: `XINFO GROUPS` shows the real consumer group at `pending=0, lag=0`, fully caught
  up; the Stream is simply never trimmed, so `XLEN` reflects lifetime message count since
  2026-07-07, not a depth). Either way, this contributor to the merge is essentially always the
  smallest/least-surprising of the five, structurally, not because transport is calm. See
  `services/orion-substrate-runtime/README.md`'s "transport domain is one queue" note for the
  full trace, including this correction and the redesign now in progress
  (`docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md`, PR #1290) —
  not yet wired to feed `prediction_error` or any mood-arc training channel; flagged here so a
  future retrain's field-selection results for `prediction_error` aren't misread as "the whole
  bus is healthy" when they're really "one queue is quiet, structurally the only thing that can
  ever show up here."
- **`catalog_drift_pressure` is now a structurally dead channel going forward (found
  2026-07-23):** unlike `prediction_error`, this one is an actual member of `v2`'s (and
  presumably `v3`'s) 15 selected channels. It counts streams the bus-observer watches that
  aren't in `orion/bus/channels.yaml`'s catalog — since the same 2026-07-18 fix above made the
  observer only ever watch cataloged streams, that count, and this channel, is now permanently
  `0.0` by construction, not by observation (confirmed `MISSING` from every recent corpus row —
  the `PRESSURE_CHANNELS`/`value > 0` inclusion gate no longer lets it through at all). A future
  retrain's field-selection pass will likely drop it outright; worth noting in the retrain
  writeup as an expected, not anomalous, absence. Full detail:
  `services/orion-field-digester/README.md`'s `catalog_drift_pressure` channel-catalog entry.
- `--hidden-dim 128 --latent-dim 64` are the defaults as of this patch
  (`DEFAULT_HIDDEN_DIM`/`DEFAULT_LATENT_DIM` in `fit_encoder.py`) — sized for
  `field_channel_corpus.v1`'s ~16-26-channel width, not the old 4-channel
  corpus's 32/16.
- `--epochs 250` is also the current default (`DEFAULT_EPOCHS`) — changed
  down from 500 after a 2026-07-18 overfitting incident, see
  `docs/DESIGN.md`.
- Correlation pruning (`prune_correlated_fields()`, `--corr-threshold 0.9` by
  default) runs automatically after field selection — no flag needed to
  enable it.

Every `train` run writes `manifest.json` / `weights.npz` / `probes.json`
under `--out`. `probes.json` is intentionally empty right now —
`compute_window_probes()` raises `NotImplementedError` rather than guessing a
replacement for the old `valence` probe target (see `docs/DESIGN.md`'s open
question section). `cmd_train` catches this and writes the empty file with a
clear skip message; this is not a bug.

## Running `detect-anomalies`

Scores a corpus slice's reconstruction loss against an already-trained
encoder and flags windows whose loss exceeds
`manifest.training.recon_error_p95 * --threshold-multiplier` (default `3.0`).

```bash
python orion/mood_arc/fit_encoder.py detect-anomalies \
  --corpus /mnt/telemetry/field_channels/corpus/field_channels.jsonl \
  --encoder-dir /tmp/mood-arc-encoders/v1-candidate \
  --min-generated-at 2026-07-13T23:46:00Z \
  --max-generated-at 2026-07-17T04:32:14Z
```

`--min-generated-at`/`--max-generated-at` scope scoring to a specific
historical window independently of the encoder's own training window — the
example above scores the known pre-fix period against the production
encoder (see `docs/DESIGN.md`'s "the anomaly detector" section for the real
result of running exactly this).

**`detect-anomalies` is currently a dark deployment: a manual CLI tool
only.** Nothing runs it automatically — no scheduled job, no bus channel, no
live cognition consumer reads its output. Practical uses today:

1. **A pre-training QA gate** — before training on a new corpus slice, score
   it against a known-good encoder to catch contamination before wasting a
   training run on it.
2. **Retrospective incident scoring** — given a suspected-bad time period,
   get a quantified anomaly rate instead of eyeballing logs.

If it needs to run automatically or feed a live consumer, that's unbuilt
follow-on work (a scheduler + a destination for the output), not something
this patch does.

## `v4`: phi-v2 clean-metrics retrain (2026-09-02)

`v4` is the current live model (promoted via `cmd_promote`, `/mnt/telemetry/models/mood_arc/v4`
+ `active.json` — the first time this actually happened; `v1`-`v3` were config-documented as
"the model to use," never formally promoted through this mechanism). It started as a request to
use `docs/superpowers/specs/2026-08-21-phi-v2-design.md`'s live-audited "clean metrics" as `v3`'s
input feature set, deliberately unsupervised (no predictive target — that's phi-v2's own
still-open, deliberately deferred second half). What it actually took to get there is worth
recording in full, since two of the three problems found were genuine methodology gaps, not
mistakes specific to this one retrain.

**What's new in `v4` vs `v3`'s 15 channels:** `action_warrant` (`substrate_proposal_frames`),
`heartbeat_mean_ratio` + per-domain `prediction_error_{execution,chat,biometrics,bus_synaptic}`
(`substrate_attention_self_model`) — joined onto the existing `field_channel_corpus.v1` corpus by
`orion/mood_arc/corpus_enrichment.py` (new module) via `fit_encoder.py enrich-corpus` (new
subcommand), using last-observation-carried-forward on each source's own real *occurrence-time*
column (`generated_at`, never `created_at` — every source table here has both, and they're
different clocks). Field-digester's cabinet sensors, including the mic
(`cabinet_ambient_audio_activity`), needed no new plumbing at all — they already reach
`field_channel_corpus.v1` through the same `node_vectors` merge every other channel does, so
`select_fields()` was already evaluating them.

**Deliberately deferred, not wired in:** `git_delta`/`pr_lifecycle_delta`/`graph_delta`
(`substrate_codebase_delta_log`), `dev_economics` (`dev_economics_ledger_log`),
`doc_semantic_drift` (`doc_semantic_drift_log`), `swear_frequency`
(`juniper_affective_state_log`). Real update cadence for these is ~16min-11.6h (live-checked
2026-09-02) — far coarser than `fit_encoder.py`'s ~60s default window (`window_size=30` at
~2s/tick), so forward-filling them in makes them constant across nearly every window: added input
dimensionality with no real per-window trajectory signal at this timescale. `corpus_enrichment.py`
fully implements and tests the fetch functions for all four (they're real, reusable, live-verified
SQL); `fetch_all_series()` just doesn't call them by default. Wiring them back in is real
follow-up work needing a different representation (a per-window static "context" feature, not a
repeated-30x trajectory slot) — not a bigger window or more epochs on the current shape.

**Metric-shape finding, not fixed here, worth reading before trusting a "calm" reading on these
channels:** `prediction_error_*`/`git_delta`/etc. and the mic channel are all **baseline-relative
deviation** measures (bounded [0,1] via `min(1.0, max(0.0, zscore) / saturation)` clamps,
`orion/substrate/prediction_error.py`), not absolute level — a structurally different family from
`v3`'s core channels (`cpu_pressure` etc.), which are absolute magnitudes that decay toward 0 on
producer silence. Hand-verified for the mic specifically:
`orion/telemetry/ambient_audio.py`'s own docstring proves `cabinet_ambient_audio_activity == 0.0`
for a raw RMS level that is *exactly constant tick to tick* — a sustained loud environment reads
as "calm" by construction, because the channel measures change-from-adaptive-baseline, not
loudness. Read any of these channels' low values as "not currently changing," not "currently
quiet."

**Four real problems found and fixed in sequence, in the order they were found:**

1. **Scale-dominance bug.** `dev_economics_total_tokens` (live range 0-59,290,459, avg ~4.07M) was
   briefly included before the sparse-signal scope-narrowing above — its raw magnitude
   (`channel_variance` ≈ 9.2e13) completely dominated the shared per-channel MSE reconstruction
   loss against every other ~[0,1]-scaled channel, producing a floor_ratio (0.081) that looked like
   a spectacular pass but was really "the model learned to reconstruct token counts." Fixed by
   dropping `total_tokens` from `fetch_dev_economics()` (the whole `dev_economics` group was later
   deferred entirely per the cadence-mismatch finding above, but the scale lesson generalizes:
   check live min/max before joining any new raw-magnitude signal into a shared reconstruction
   scale).
2. **Cadence/dimensionality mismatch** (see "deliberately deferred" above) — narrowing to the
   dense-enough subset and doubling `hidden_dim`/`latent_dim` (128/64 → 256/128, proportional to
   the larger channel count) was tried next. It did **not** fix the floor gate on its own
   (`floor_ratio` went from 0.683 to 0.774 — worse, not better) — see finding 3.
3. **Non-stationary held-out split — the actual root cause.** `purged_temporal_split()`'s
   single-trailing-15%-block held-out design was validated on `v3`'s short (10.3h), largely
   stationary window, where the trailing block stays inside the same regime as train. Confirmed
   live: comparing `v4`'s corpus's temporally-first-85% slice against its temporally-last-15%
   slice showed `prediction_error`'s mean shifted **+1.73 standard deviations** and
   `thermal_pressure`'s **+1.21** — the held-out block was a genuinely different operating regime,
   not a random draw from train's distribution. Since `floor_ratio` and `ceiling_ratio` share the
   same numerator (real held-out reconstruction loss), this inflated both regardless of whether
   real trajectory structure existed, and got *worse* with more model capacity (a more expressive
   model overfits train's regime harder, generalizing worse to the shifted tail) — exactly the
   pattern found in step 2. Fixed with `block_purged_temporal_split()` (new, `fit_encoder.py`,
   opt-in via `--held-out-blocks`, standard blocked/purged cross-validation practice for
   non-stationary time series): spreads held-out across N evenly-spaced, doubly-embargoed time
   blocks instead of one trailing chunk. `--held-out-blocks 1` (the default) is byte-identical to
   the original single-block method — `v3`'s already-validated 0.210 result is untouched by this
   change.
4. **Block-boundary leakage in the fix for #3 — caught by code review, not by the training
   numbers.** The first `block_purged_temporal_split()` divided windows into N segments and
   applied the trailing-embargo logic *within* each segment independently, but never embargoed a
   segment's own *leading* edge — so a held-out window at the end of segment `i` sat directly
   adjacent (zero gap, physically overlapping raw ticks under the 50%-overlap stride) to a
   training window at the start of segment `i+1`, at every one of the N-1 internal block
   boundaries. The promoted candidate's first "passing" `floor_ratio=0.415` was trained under this
   leak. A related bug in the same rewrite (`block_purge_excluded_ar1_intervals()`'s upper bound
   under-covering a run's orphaned trailing rows — `_build_windows_with_span()` drops any run's
   tail once fewer than `window_size` rows remain) could separately re-admit rows into the AR(1)
   baseline fit. Fixed: `_segment_train_held_ranges()` now embargoes both edges of every internal
   segment, and `block_purge_ar1_training_rows()` replaced the interval-upper-bound approach with
   direct inclusion-checking against real train-window spans (a row is safe for AR(1) fitting only
   if it falls inside some actual train window — no upper-bound estimate to get wrong). Both fixes
   are covered by new regression tests
   (`test_block_purged_temporal_split_embargoes_internal_block_boundaries`,
   `test_block_purge_ar1_training_rows_excludes_orphaned_trailing_rows`). `--held-out-blocks 1`
   remains byte-identical to `v3`'s original methodology throughout.

**Results, holding channels (37) and capacity (256/128) constant, varying only the split:**

| | `--held-out-blocks` | `floor_ratio` | `floor_pass` | `ceiling_ratio` |
|---|---|---|---|---|
| Attempt 3 | 1 (single trailing block) | 0.774 | **FAIL** | 1.494 |
| Attempt 4 (leaked at block boundaries) | 5 | 0.415 (CI 0.396-0.436) | PASS (invalid — see #4) | 0.660 |
| **Attempt 5 (`v4`, promoted, leak-fixed)** | 5 | **0.406** (CI 0.383-0.430) | **PASS** | **0.733** |

The leak-fixed result (attempt 5) landed close to the leaked one (0.406 vs. 0.415) — reassuring
that the internal-boundary leak wasn't actually doing much of the work, but it was still a real
defect worth fixing before trusting the number, not a rounding difference. `ceiling_ratio` (0.733)
is notably worse than `v3`'s (0.190) — real reconstruction still clearly beats the AR(1) surrogate
(ratio well under 1), just by a smaller margin. Plausibly genuine added entropy in a 3.4-day
multi-regime corpus vs. `v3`'s stationary 10.3h slice, not a red flag — `ceiling_ratio` has no
calibrated pass/fail threshold (diagnostic only, same as every prior version). Treat `v4` the same
way `v3`'s own writeup treated itself: real evidence, n=1 on this exact corpus/config, not yet
independently re-confirmed with a second seed.

**Independence check (CLAUDE.md's metric quality gate, step 2), recorded here per that gate's own
requirement:** the new per-domain `prediction_error_{execution,chat,biometrics,bus_synaptic}`
channels are components of the pre-existing `prediction_error` channel (a `max()`-merge across the
same underlying node instruments), a short, direct causal chain that would normally flag them as
redundant. Checked against real training-run evidence rather than asserted: `prune_correlated_fields()`'s
full printed output (`--corr-threshold 0.9`, the same run that produced the promoted `v4`) does
**not** flag `prediction_error` against any of its four components — none appear in its "dropping
X (kept Y, r=...)" list, meaning pairwise Pearson correlation between the max-merge and each
component stays under 0.9 in practice. Consistent with `max()` being nonlinear: the merge and any
one component diverge whenever a *different* component is currently the maximum. Read as
"empirically not redundant by the pruning gate's own linear-correlation test," not as a from-first-
principles proof of independence — a true multicollinearity check (e.g. VIF) hasn't been run.

**Follow-up work, explicitly not done here:**

- Wire the four deferred sparse signals in via a per-window context representation, not the
  current per-tick trajectory slot.
- phi-v2's actual point (a named, falsifiable predictive target — "predict a near-future
  prediction-error spike" — replacing pure reconstruction-only training) is still fully open. `v4`
  only consumed phi-v2's *clean-metrics audit*, not its target-design proposal.
- A second training seed against the same corpus/config, to move past n=1 the way `v3`'s own
  `ceiling_ratio`-stability observation was itself only n=2.

## Semantic taxonomy of the channels

The raw channels `field_channel_corpus.v1` carries (38 as of 2026-07-25 --
this count has already drifted from this doc more than once as channels were
added; check `services/orion-field-digester/README.md`'s "Field channel
glossary" section header for the current number rather than trusting a
hardcoded figure here) (and the 15-16 that typically survive
selection/pruning) have a real-world-meaning grouping already written up in
`services/orion-field-digester/README.md`'s `### Semantic categories`
section. That is the canonical reference — this module links to it rather
than duplicating it, since the taxonomy applies to the corpus/channel
producer, not to this training pipeline specifically.

## Related PRs

`#989` (original `mood_arc_corpus.v1` collector, since superseded),
`#1018`/`#1019` (manifest schema + original CLI, pre-corpus-swap), `#1022`
(the `field_channel_corpus.v1` replacement collector), `#1172` (channel
glossary correction: `reasoning_pressure` is single-source not dual-source),
`#1177` (fix: `reasoning_load` now attributes to the node that actually
served the LLM call, not the static orchestrator identity), `#1182`
(corpus-swap CLI rework + epochs fix), `#1185` (the anomaly detector). Full
narrative in `docs/DESIGN.md`.
