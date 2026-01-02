# Vision Services Documentation

This document outlines the architecture and contracts for the Orion Vision subsystem, comprising the Host and downstream workers.

## Architecture

The vision pipeline is a distributed system using the Titanium Contract Stack over Redis PubSub.

1.  **Vision Host (`orion-vision-host`)**: GPU-accelerated inference service. Executes tasks (Embed, Detect, Caption, Retina) and broadcasts artifacts.
2.  **Vision Retina (`orion-vision-retina`)**: Capture service that publishes frame pointers.
3.  **Vision Window (`orion-vision-window`)**: Aggregates artifacts into time-based windows.
4.  **Vision Council (`orion-vision-council`)**: Performs high-level cognitive analysis on windows using LLMs.
5.  **Vision Scribe (`orion-vision-scribe`)**: Persists events to SQL, RDF, and Vector stores.

## Channels and Kinds

| Service | Channel (Env Var) | Default Channel | Kind | Direction |
| :--- | :--- | :--- | :--- | :--- |
| **Host** | `CHANNEL_VISIONHOST_INTAKE` | `orion-exec:request:VisionHostService` | `vision.task.request` | In (Req) |
| **Host** | `CHANNEL_VISIONHOST_PUB` | `orion:vision:artifacts` | `vision.artifact` | Out (Broadcast) |
| **Host** | (Caller specified) | - | `vision.task.result` | Out (Reply) |
| **Retina** | `CHANNEL_RETINA_PUB` | `orion:vision:frames` | `vision.frame.pointer` | Out |
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
-   `artifact_ids`: List of artifacts in this window.

### `VisionEventPayload`
-   `events`: List of high-level events (`event_type`, `narrative`, `entities`, `tags`, `confidence`).

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
