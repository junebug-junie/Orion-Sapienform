# Orion Metacognition Logging

This module implements a bus-first, machine-generated logging system ("Metacognition") triggered by `orion-equilibrium-service` and executed by `orion-cortex-exec`.

## Flow

1.  **Triggers** (`orion-equilibrium-service`):
    *   **Baseline:** Scheduled check (default every 60s).
    *   **Dense:** User creates a Collapse Mirror entry (manual trigger).
    *   **Pulse:** Landing Pad "salience" signal exceeds threshold.

2.  **Dispatch**:
    *   Equilibrium service publishes `MetacogTriggerV1` to `orion:equilibrium:metacog:trigger`.
    *   Cortex-Orch subscribes to that trigger channel and dispatches the `log_orion_metacognition` plan to Cortex-Exec.

3.  **Execution** (`orion-cortex-exec`):
    *   **Step 1: Collect Context**: Fetches latest Landing Pad frame/signal and Spark State via RPC.
    *   **Step 2: Draft (LLM)**: Generates a JSON "subjective experience" log (Collapse Mirror V2).
    *   **Step 3: Enrich (LLM)**: Adds causal analysis, scores, and tags.
    *   **Step 4: Publish**: Sends final `CollapseMirrorEntryV2` to `orion:collapse:sql-write`.

## Configuration

### Environment Variables (`orion-equilibrium-service`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `EQUILIBRIUM_METACOG_ENABLE` | `False` | Master toggle for metacognition features. |
| `EQUILIBRIUM_METACOG_BASELINE_INTERVAL_SEC` | `60.0` | Seconds between baseline "zen check" triggers. |
| `EQUILIBRIUM_METACOG_COOLDOWN_SEC` | `30.0` | Minimum seconds between any two triggers. |
| `EQUILIBRIUM_METACOG_PAD_PULSE_THRESHOLD` | `0.8` | Salience threshold (0.0-1.0) to trigger on Landing Pad signal. |

### Channels (`orion/bus/channels.yaml`)

*   **Trigger Event**: `orion:equilibrium:metacog:trigger` (Schema: `MetacogTriggerV1`)
*   **Cortex-Exec Intake**: `orion:cortex:exec:request` (Schema: `GenericPayloadV1`)
*   **Final Output**: `orion:collapse:sql-write` (Schema: `CollapseMirrorEntryV2`)

### Schemas

*   **`MetacogTriggerV1`** (`orion/schemas/telemetry/metacog_trigger.py`): Contains trigger reason, pressure scores, and upstream references.
*   **`CollapseMirrorEntryV2`** (`orion/schemas/collapse_mirror.py`): The standard logging format for machine subjective experience.

## Usage

Enable the feature in `.env` for `orion-equilibrium-service`:

```bash
EQUILIBRIUM_METACOG_ENABLE=true
```

## Trace tooling

Run a deterministic, end-to-end trace (publishes a metacog trigger and prints downstream envelopes):

```bash
python scripts/trace_metacog.py --timeout 45
```

Probe live bus traffic with optional trace filtering:

```bash
python scripts/bus_probe.py --pattern "orion:*metacog*" --trace-id <trace_id>
```

### Docker (using service images)

```bash
docker compose --env-file .env -f services/orion-cortex-orch/docker-compose.yml run --rm \
  -v "$(pwd)":/workspace -w /workspace cortex-orchestrator \
  python scripts/trace_metacog.py --timeout 45
```
