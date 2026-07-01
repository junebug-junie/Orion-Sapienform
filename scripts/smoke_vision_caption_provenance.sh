#!/usr/bin/env bash
# smoke_vision_caption_provenance.sh
#
# Verifies the Orion Vision caption + provenance chain end-to-end at the
# CONTRACT level, WITHOUT requiring the `vision-edge` capture service:
#
#   frame/task request+meta
#     -> Vision Host artifact (caption + provenance inputs)
#       -> Vision Window projection (stream/camera/uris + caption summary)
#
# The dev/CI box typically has NO torch/GPU, so real VLM inference cannot run
# here. This smoke therefore supports an explicit FAKE (contract) mode that
# exercises the schema/builder/projection contracts using a synthetic-but-
# honest VisionResult. It does NOT pretend that real inference happened.
#
# MODES (select via env; there is no implicit default):
#   VISION_SMOKE_FAKE_CAPTION=1
#       FAKE (contract) mode. Runs fully in-process using the repo's Python
#       schema/builders; no bus, no GPU, no vision-edge. This is the path that
#       is run and verified in CI on GPU-less boxes.
#
#   VISION_HOST_URL=<base-url>   (and FAKE mode NOT set)
#       REAL mode. POSTs a task to a running Vision Host and asserts the
#       response contract. Requires a real, host-visible image via
#       VISION_SMOKE_IMAGE. Will not run on a GPU-less box.
#
#   (neither set)
#       Prints a usage error to stderr and exits non-zero. Does NOT fake.
#
# ENV VARS:
#   VISION_SMOKE_FAKE_CAPTION  set to 1 to select FAKE (contract) mode.
#   VISION_HOST_URL            base URL of a running Vision Host (REAL mode).
#   VISION_SMOKE_IMAGE         real, host-visible image path (REAL mode only).
#   VISION_SMOKE_PYTHON        explicit python interpreter to use (optional).
#
# INTERPRETER SELECTION (first that exists wins):
#   $VISION_SMOKE_PYTHON, else $REPO_ROOT/orion_dev/bin/python,
#   else $REPO_ROOT/venv/bin/python, else python3.
#   The interpreter must have `pydantic` (and, for the contract imports, the
#   repo's `orion` package on PYTHONPATH, which this script sets up).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pick_python() {
    if [[ -n "${VISION_SMOKE_PYTHON:-}" ]]; then
        echo "${VISION_SMOKE_PYTHON}"
    elif [[ -x "${REPO_ROOT}/orion_dev/bin/python" ]]; then
        echo "${REPO_ROOT}/orion_dev/bin/python"
    elif [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
        echo "${REPO_ROOT}/venv/bin/python"
    else
        echo "python3"
    fi
}

PY="$(pick_python)"

run_fake_mode() {
    # Temp image (a real file, not required to be a valid image in fake mode)
    # and temp artifact JSON to pass Host -> Window across two python processes.
    SMOKE_IMG="$(mktemp --suffix=.jpg)"
    SMOKE_ART="$(mktemp --suffix=.json)"
    export SMOKE_IMG SMOKE_ART
    # shellcheck disable=SC2064
    trap "rm -f '${SMOKE_IMG}' '${SMOKE_ART}'" EXIT

    # --- Step A: Vision Host side (build artifact from synthetic result) ---
    # NOTE: orion-vision-host and orion-vision-window BOTH ship a top-level
    # package named `app`, so they cannot be imported in one process. Split
    # into two invocations, passing data via SMOKE_ART.
    PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/services/orion-vision-host" "${PY}" - <<'PYEOF'
from app.artifacts import merge_result_inputs, build_artifact_payload
from app.models import VisionResult
import json, os
request = {"image_path": os.environ["SMOKE_IMG"], "want_caption": True, "want_embeddings": True}
meta = {"camera_id": "mock-cam-01", "stream_id": "mock-stream", "frame_ts": 123.4,
        "source_frame_envelope_id": "frame-env-1", "source_frame_correlation_id": "frame-corr-1",
        "router_policy": "defaults"}
res = VisionResult(
    corr_id="smoke-1", ok=True, task_type="retina_fast", device="cuda:0",
    artifacts={
        "objects": [{"label": "screen", "score": 0.8, "box_xyxy": [0, 0, 10, 10]}],
        "caption": {"text": "A terminal window is visible.", "confidence": 0.9},
        "embedding": {"ref": "emb:smoke", "path": "/tmp/emb.npy", "dim": 8},
        "model_id": "fake-vlm",
        "_fingerprints": {"retina_detect_open_vocab": "fake-gdino", "vlm_caption": "fake-vlm"},
    },
    meta={"latency_s": 0.01},
    inputs=merge_result_inputs(request, meta),
)
art = build_artifact_payload(res)
assert art is not None, "artifact is None"
assert art.outputs.caption and art.outputs.caption.text, "caption missing"
assert art.outputs.embedding and art.outputs.embedding.ref, "embedding missing"
for k in ("image_path", "camera_id", "stream_id", "frame_ts"):
    assert k in art.inputs, f"missing provenance input: {k}"
with open(os.environ["SMOKE_ART"], "w") as fh:
    json.dump(art.model_dump(mode="json"), fh)
print(f"[host] OK caption={art.outputs.caption.text!r} inputs={sorted(art.inputs)}")
PYEOF

    # --- Step B: Vision Window side (project artifact -> stream/camera/uris) ---
    PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/services/orion-vision-window" "${PY}" - <<'PYEOF'
from orion.schemas.vision import VisionArtifactPayload
from app.projection import (stream_key_from_artifact, camera_id_from_artifact,
                            artifact_uris_from_artifact, summarize_items)
import json, os
with open(os.environ["SMOKE_ART"]) as fh:
    art = VisionArtifactPayload(**json.load(fh))
assert stream_key_from_artifact(art) == "mock-stream", stream_key_from_artifact(art)
assert camera_id_from_artifact(art) == "mock-cam-01"
assert os.environ["SMOKE_IMG"] in artifact_uris_from_artifact(art), artifact_uris_from_artifact(art)
summ = summarize_items([(art, 1.0)])
assert "A terminal window is visible." in summ["captions"], summ
print(f"[window] OK stream={stream_key_from_artifact(art)} camera={camera_id_from_artifact(art)} captions={summ['captions']}")
PYEOF

    echo "SMOKE PASS (fake mode)"
}

run_real_mode() {
    local img="${VISION_SMOKE_IMAGE:-}"
    if [[ -z "${img}" ]]; then
        echo "ERROR: REAL mode requires VISION_SMOKE_IMAGE (a real, host-visible image path)." >&2
        exit 2
    fi

    local url="${VISION_HOST_URL%/}/v1/vision/task"
    # Vision Host serves HTTP on container port 6600 (mapped to ${HOST_PORT}).
    local body
    body="$("${PY}" - "${img}" <<'PYEOF'
import json, sys
img = sys.argv[1]
print(json.dumps({
    "task_type": "retina_fast",
    "request": {"image_path": img, "want_caption": True, "want_embeddings": True},
    "meta": {"camera_id": "mock-cam-01", "stream_id": "mock-stream", "frame_ts": 123.4},
}))
PYEOF
)"

    local resp
    resp="$(curl -fsS -X POST -H 'Content-Type: application/json' -d "${body}" "${url}")"

    printf '%s' "${resp}" | "${PY}" - <<'PYEOF'
import json, sys
data = json.load(sys.stdin)
assert data.get("ok") is True, f"host did not report ok=True: {data.get('ok')!r}"
artifact = data.get("artifact") or {}
outputs = artifact.get("outputs") or {}
caption = outputs.get("caption") or {}
assert caption.get("text"), "artifact.outputs.caption.text missing/empty"
inputs = artifact.get("inputs") or {}
for k in ("image_path", "camera_id", "stream_id", "frame_ts"):
    assert k in inputs, f"missing provenance input: {k}"
print(f"[real] OK caption={caption['text']!r} inputs={sorted(inputs)}")
PYEOF

    echo "SMOKE PASS (real mode)"
    # NOTE: a fuller live-stack check (Vision Window current/recent contains
    # the caption, Council event bundle, Scribe ack) can be layered on later.
    # This script deliberately stays focused on the host -> window contract.
}

main() {
    if [[ "${VISION_SMOKE_FAKE_CAPTION:-}" == "1" ]]; then
        run_fake_mode
    elif [[ -n "${VISION_HOST_URL:-}" ]]; then
        run_real_mode
    else
        cat >&2 <<'USAGE'
ERROR: no mode selected. Choose exactly one:
  * FAKE (contract) mode  : VISION_SMOKE_FAKE_CAPTION=1 bash scripts/smoke_vision_caption_provenance.sh
      Runs in-process against the repo schema/builders (no bus, no GPU).
  * REAL mode             : VISION_HOST_URL=<base-url> VISION_SMOKE_IMAGE=<img> bash scripts/smoke_vision_caption_provenance.sh
      POSTs a task to a running Vision Host and asserts the response contract.
USAGE
        exit 2
    fi
}

main "$@"
