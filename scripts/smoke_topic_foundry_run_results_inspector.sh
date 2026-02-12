#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${TOPIC_FOUNDRY_BASE_URL:-http://127.0.0.1:8615}"
POLL_SECONDS="${TOPIC_FOUNDRY_POLL_SECONDS:-2}"
POLL_MAX="${TOPIC_FOUNDRY_POLL_MAX:-60}"

models_resp="$(curl -fsS "$BASE_URL/models")"
model_id="$(echo "$models_resp" | jq -r '.models[0].model_id // empty')"
datasets_resp="$(curl -fsS "$BASE_URL/datasets")"
dataset_id="$(echo "$datasets_resp" | jq -r '.datasets[0].dataset_id // empty')"

if [[ -z "$model_id" || -z "$dataset_id" ]]; then
  echo "[FAIL] missing model_id or dataset_id (create model+dataset first)"
  exit 1
fi

train_payload="$(jq -nc --arg model_id "$model_id" --arg dataset_id "$dataset_id" '{model_id:$model_id,dataset_id:$dataset_id,topic_mode:"standard"}')"
train_resp="$(curl -fsS -X POST "$BASE_URL/runs/train" -H 'Content-Type: application/json' -d "$train_payload")"
run_id="$(echo "$train_resp" | jq -r '.run_id // empty')"
if [[ -z "$run_id" ]]; then
  echo "[FAIL] no run_id from /runs/train"
  echo "$train_resp"
  exit 1
fi
echo "[INFO] started run_id=$run_id"

status="queued"
for ((i=0; i<POLL_MAX; i++)); do
  run_resp="$(curl -fsS "$BASE_URL/runs/$run_id")"
  status="$(echo "$run_resp" | jq -r '.status // "unknown"')"
  echo "[INFO] poll[$i] status=$status"
  if [[ "$status" == "complete" || "$status" == "trained" ]]; then
    break
  fi
  if [[ "$status" == "failed" ]]; then
    echo "[FAIL] run failed"
    echo "$run_resp"
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

if [[ "$status" != "complete" && "$status" != "trained" ]]; then
  echo "[FAIL] run did not complete in time"
  exit 1
fi

list_resp="$(curl -fsS "$BASE_URL/runs/$run_id/results?limit=5")"
count="$(echo "$list_resp" | jq -r '.items | length')"
if [[ "$count" -lt 1 ]]; then
  echo "[FAIL] expected >=1 run result"
  echo "$list_resp"
  exit 1
fi
segment_id="$(echo "$list_resp" | jq -r '.items[0].segment_id // empty')"
preview_len="$(echo "$list_resp" | jq -r '.items[0].text_preview | length')"

detail_resp="$(curl -fsS "$BASE_URL/runs/$run_id/results/$segment_id")"
full_len="$(echo "$detail_resp" | jq -r '.full_text | length')"
if [[ "$full_len" -le "$preview_len" ]]; then
  echo "[FAIL] expected full_text longer than snippet"
  echo "preview_len=$preview_len full_len=$full_len"
  exit 1
fi

echo "[PASS] run results smoke run_id=$run_id segment_id=$segment_id full_len=$full_len"
