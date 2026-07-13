# Spark train / evals

Offline training-adjacent evals for phi encoder artifacts under
`${TELEMETRY_ROOT}/models/phi/encoders/`.

These are **not** the same as `services/orion-spark-introspector/evals/`
(CI-style synthetic gates). This folder is for corpus-backed health reports
across **all** training runs so operators can compare and trace promotions.

## `eval_phi_encoder_health.py`

Primary health metrics (do **not** treat φ↔headline correlation as success):

1. **recon** — p50 / p95 / mean reconstruction MSE on matching corpus rows
2. **residual-after-headline-fit** — fit `φ ≈ a·headline + b`, report residual
   std and fraction of rows with `|φ − headline| < 1e-4` (identity collapse)
3. φ↔headline corr / MAE — reported as *supervised target diagnostics only*

Every version directory under `--encoders-root` is included with manifest
metadata (`git_sha`, `trained_at`, corpus span, training losses, status).

**Rotation-aware (2026-07-13):** `--corpus` no longer has to point at a
single ever-growing file. `InnerStateCorpusSink`
(`services/orion-spark-introspector/app/inner_state_sink.py`) now rotates
the live corpus at `CORPUS_SINK_MAX_BYTES` (default 200MB); this script's
`load_corpus_rows()` resolves the given `--corpus` path plus any rotated
siblings (`orion/telemetry/corpus_rotation.py`) so a health report always
covers the full retained history, not just the slice written since the
last rotation.

```bash
# Live corpus + all encoder runs
python services/orion-spark-introspector/train/evals/eval_phi_encoder_health.py \
  --corpus /mnt/telemetry/phi/corpus/inner_state.jsonl \
  --encoders-root /mnt/telemetry/models/phi/encoders

# JSON only
python services/orion-spark-introspector/train/evals/eval_phi_encoder_health.py \
  --corpus /mnt/telemetry/phi/corpus/inner_state.jsonl \
  --encoders-root /mnt/telemetry/models/phi/encoders \
  --json
```

Exit non-zero when the **active** encoder fails the collapse gate:

- `near_identity_frac > 0.5` (`|φ − headline| < 1e-4`), or
- `residual_std < 1e-3` **and** `slope ≈ 1` (true identity copy; constant-φ alone does not fail)
