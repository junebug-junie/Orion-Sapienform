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
Registered `REHEARSAL` in `orion/self_state/inner_state_registry.py`
(`mood_arc_encoder.v1`, `mood_arc_corpus.v1`, `field_channel_corpus.v1`) —
that status is correct and intentional, not a gap to close as part of this
patch.

## What's in this module

- `fit_encoder.py` — the CLI. Two subcommands: `train` (fit a candidate
  encoder against a corpus slice) and `detect-anomalies` (score a corpus
  slice against an already-trained encoder). `promote` (item 2's eventual
  candidate → active promotion path) is not built yet.
- `tests/` — 33 tests covering windowing, field selection/pruning, the
  purged temporal split, the two-tier gate, the AR(1) surrogate, and the
  anomaly detector.
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
- **Second cutoff, 2026-07-22 (PR #1248, pending deploy):** `transport_pressure`/
  `catalog_drift_pressure`/`observer_failure_pressure`/`reliability_pressure`/
  `contract_pressure` could get permanently stuck at a stale value (an
  `add`-mode perturbation bug in `services/orion-field-digester/app/ingest/
  state_deltas.py`, fixed in PR #1248) — confirmed live as the cause of
  `catalog_drift_pressure` alone driving ~66% of average reconstruction
  error against `field_channel_anomaly.v2`. Unlike the first cutoff, this
  contamination window has no known start (a channel only breaks once it
  picks up a nonzero value and then fails to correct/decay), so **do not
  train against any corpus data until PR #1248 is merged and deployed, and
  a second `--min-generated-at` cutoff is set to that deploy timestamp** —
  see `services/orion-field-digester/README.md`'s "second training-data
  quality cutoff" section for the exact mechanism to determine it once
  known. If both cutoffs apply, use whichever is later.
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

## Semantic taxonomy of the channels

The 29 raw channels `field_channel_corpus.v1` carries (and the 15-16 that
typically survive selection/pruning) have a real-world-meaning grouping
already written up in
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
