# Orion Vision Window

Aggregates vision artifacts into rolling time windows and publishes `VisionWindowPayload` summaries for Council.

## Role in the pipeline

```text
orion:vision:artifacts  ──►  orion-vision-window  ──►  orion:vision:windows
   (edge + host artifacts)         (aggregate)              (VisionWindowPayload)
```

## Evidence tiers

Window summaries include an `evidence` block used by Council for grounded interpretation:

| Field | Meaning |
|-------|---------|
| `hard_labels` | Detection labels above score threshold (factual tier) |
| `soft_labels` | Tokens from captions matching stoplist (YouTube, google, …) |
| `edge_person_hits` | Always 0 (edge artifacts excluded from pipeline evidence) |
| `host_person_hits` | Person detections from host `retina_fast` artifacts |
| `caption_count` | Number of captioned artifacts in the window |

Council treats `hard_labels` as admissible evidence; captions are soft hints only.

## Configuration

```bash
cd services/orion-vision-window
cp .env_example .env
```

Key channels: `CHANNEL_WINDOW_INTAKE` (artifacts in), `CHANNEL_WINDOW_PUB` (windows out).

## Tests

From repo root:

```bash
PYTHONPATH=services/orion-vision-window:. pytest services/orion-vision-window/tests -q
```

See also: `docs/vision_services.md` for cross-service contracts.
