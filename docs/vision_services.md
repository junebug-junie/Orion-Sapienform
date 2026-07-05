# Vision Services Documentation

This document outlines the architecture and contracts for the Orion Vision subsystem, comprising the Host and downstream workers.

**Per-service READMEs:** `services/orion-vision-edge/`, `orion-vision-frame-router/`, `orion-vision-host/`, `orion-vision-window/`, `orion-vision-council/` (operator checklists and smoke commands).

**Grounded pipeline design:** `docs/plans/vision/2026-07-02-vision-grounded-pipeline-design.md`

## Architecture

The vision pipeline is a distributed system using the Titanium Contract Stack over Redis PubSub.

1.  **Vision Edge (`orion-vision-edge`)**: On-device capture + YOLO/motion detectors. Publishes frame pointers, slim edge-detection artifacts, and compact activity trigger signals.
2.  **Vision Frame Router (`orion-vision-frame-router`)**: Subscribes to frames; dispatches `retina_fast` tasks to Host using baseline vs triggered policy tiers. Trigger TTL is refreshed from host task replies (person detections), not edge activity.
3.  **Vision Host (`orion-vision-host`)**: GPU-accelerated inference service. Executes tasks (Embed, Detect, Caption, Retina) and broadcasts artifacts.
4.  **Vision Retina (`orion-vision-retina`)**: Capture service that publishes frame pointers (legacy/alternate capture path).
5.  **Vision Window (`orion-vision-window`)**: Aggregates artifacts into time-based windows with evidence tiers.
6.  **Vision Council (`orion-vision-council`)**: Performs high-level cognitive analysis on windows using LLMs. Council V2 parses `VisionSceneInterpretationV1` internally (strict validation, then salvage/coercion for common malformed nested fields, then legacy fallback), then projects `event_candidates` to the legacy `VisionEventPayload` published on `orion:vision:events` (Scribe contract unchanged).
7.  **Vision Scribe (`orion-vision-scribe`)**: Persists events to SQL, RDF, and Vector stores.

## Channels and Kinds

| Service | Channel (Env Var) | Default Channel | Kind | Direction |
| :--- | :--- | :--- | :--- | :--- |
| **Host** | `CHANNEL_VISIONHOST_INTAKE` | `orion-exec:request:VisionHostService` | `vision.task.request` | In (Req) |
| **Host** | `CHANNEL_VISIONHOST_PUB` | `orion:vision:artifacts` | `vision.artifact` | Out (Broadcast) |
| **Host** | (Caller specified) | - | `vision.task.result` | Out (Reply) |
| **Retina** | `CHANNEL_RETINA_PUB` | `orion:vision:frames` | `vision.frame.pointer` | Out |
| **Edge** | `CHANNEL_VISION_EDGE_ACTIVITY` | `orion:vision:edge:activity` | `vision.edge.activity.v1` | Out |
| **Edge** | (artifact pub) | `orion:vision:artifacts` | `vision.artifact` | Out |
| **Frame Router** | `CHANNEL_FRAMES_IN` | `orion:vision:frames` | `vision.frame.pointer` | In |
| **Frame Router** | `CHANNEL_HOST_INTAKE` | `orion:exec:request:VisionHostService` | `vision.task.request` | Out (Req) |
| **Window** | `CHANNEL_WINDOW_INTAKE` | `orion:vision:artifacts` | `vision.artifact` | In |
| **Window** | `CHANNEL_WINDOW_PUB` | `orion:vision:windows` | `vision.window` | Out |
| **Council** | `CHANNEL_COUNCIL_INTAKE` | `orion:vision:windows` | `vision.window` | In |
| **Council** | `CHANNEL_COUNCIL_PUB` | `orion:vision:events` | `vision.event.bundle` | Out |
| **Scribe** | `CHANNEL_SCRIBE_INTAKE` | `orion:vision:events` | `vision.event.bundle` | In |
| **Scribe** | `CHANNEL_SCRIBE_PUB` | `orion:vision:scribe:pub` | `vision.scribe.ack` | Out |

## Envelopes

All messages MUST be `BaseEnvelope` objects.

-   `schema_id`: Dot-separated schema identifier (e.g., `vision.task.request`).
-   `kind`: Functional type (e.g., `vision.task.request`).
-   `source`: Service identifier (`name:version`).
-   `correlation_id`: Trace ID.
-   `causality_chain`: List of previous correlation IDs for tracing.
-   `payload`: Strictly typed Pydantic model.

## Schemas (`orion/schemas/vision.py`)

### `VisionTaskRequestPayload`
-   `task_type`: `embed_image` | `detect_open_vocab` | `caption_frame` | `retina_fast`
-   `request`: Task-specific dictionary (e.g., `image_path`, `prompts`).
-   `meta`: Optional metadata.

### `VisionArtifactPayload`
-   `artifact_id`: Unique ID.
-   `outputs`:
    -   `objects`: List of `{label, score, box_xyxy}`.
    -   `caption`: `{text, confidence}`.
    -   `embedding`: `{ref, path, dim}`.
-   `timing`: Performance metrics.

### `VisionWindowPayload`
-   `window_id`: Unique ID.
-   `start_ts`, `end_ts`: Window duration.
-   `summary`: Aggregated stats (counts, top labels).
-   `summary.evidence`: `hard_labels` (observed), `believed_hard_labels` (habituated gate input), `belief` metadata, soft labels, person hit counts.
-   `artifact_ids`: List of artifacts in this window.

### `VisionEdgeActivityPayload`
-   `stream_id`: Operator-facing stream key (e.g. `cam0`).
-   `camera_id`: Stable capture identifier (RTSP URL on edge deployments).
-   `labels`: Trigger labels detected (`person`, `motion`, …).
-   `max_score`: Highest detector score in the triggering frame.
-   `frame_ts`, `image_path`, `artifact_id`: Provenance for downstream dispatch.

### `VisionEventPayload`

## Frame router dispatch tiers

Policy file: `config/vision_frame_router.yaml`. Merge order: `defaults` base, then `streams[stream_id]`, then `cameras[camera_id]` (camera-specific wins).

The frame router maintains per-`stream_id` trigger TTL from **host task replies** (person labels in GroundingDINO output). Each frame dispatch selects one tier:

| Tier | When | `retina_fast` request | Purpose |
| :--- | :--- | :--- | :--- |
| **baseline** | No active trigger labels within TTL | `want_caption: false`, `want_embeddings: false` | Keep GroundingDINO fresh without VLM on every frame |
| **triggered** | Host recently detected configured `trigger_labels` (default: `person`) within `trigger_ttl_seconds` | `want_caption: true`, `want_embeddings: true` | Rich caption + embed when a person was detected on the host pipe |

Triggered tier settings merge over baseline (rate limits, `every_n_frames`, etc.). Task meta includes `dispatch_tier` (`baseline` or `triggered`) for observability.

Per-stream overrides use the `streams:` block (e.g. `streams.cam0`). Legacy per-camera overrides remain under `cameras:` keyed by `camera_id`.

## Deployment

### GPU Scheduling
`orion-vision-host` manages GPU resources via `VisionScheduler`.
-   **Env**: `VISION_MAX_INFLIGHT`, `VISION_MAX_INFLIGHT_PER_GPU`.
-   **VRAM**: Monitors free memory using `pynvml`.

### Caching
-   Models are cached in `MODEL_CACHE_DIR` (default: `/mnt/telemetry/models/vision`).
-   HuggingFace cache: `HF_HOME`.

## Testing

### CLI Scripts
Run from `services/orion-vision-host/scripts/`:

1.  **Publish Test Task**:
    ```bash
    python publish_test_task.py --image /path/to/img.jpg --task retina_fast
    ```

2.  **Tap Artifacts**:
    ```bash
    python tap_artifacts.py
    ```

### Verification
1.  Start services via `docker-compose up`.
2.  Run `tap_artifacts.py`.
3.  Send a request via `publish_test_task.py`.
4.  Observe artifact broadcast on tap script.
5.  Check downstream logs (`orion-vision-window`, etc.) for processing.

## Cortex Enablement

The vision services are registered with `cortex-exec` and expose RPC endpoints for one-shot cognitive verbs.

### Verbs

- `perceive_retina_fast`: Host -> Artifact (Embed + Detect + Caption)
- `perceive_caption_frame`: Host -> Artifact (Caption)
- `perceive_detect_open_vocab`: Host -> Artifact (Detect)
- `perceive_embed_image`: Host -> Artifact (Embed)
- `perceive_vision_events`: Host -> Window -> Council -> Events
- `perceive_vision_memory`: Host -> Window -> Council -> Scribe -> Memory

### RPC Channels

Each service listens on a request channel for `orion-exec`:

| Service | RPC Channel (Env) | Schema (Request) | Schema (Result) |
| :--- | :--- | :--- | :--- |
| **Window** | `CHANNEL_WINDOW_REQUEST` | `VisionWindowRequestPayload` | `VisionWindowResultPayload` |
| **Council** | `CHANNEL_COUNCIL_REQUEST` | `VisionCouncilRequestPayload` | `VisionCouncilResultPayload` |
| **Scribe** | `CHANNEL_SCRIBE_REQUEST` | `VisionScribeRequestPayload` | `VisionScribeResultPayload` |

**Note**: `VisionHostService` uses its existing intake channel for RPC requests from Cortex.
