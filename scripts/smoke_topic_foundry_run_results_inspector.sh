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

run_summary="$(curl -fsS "$BASE_URL/runs/$run_id")"
outlier_rate="$(echo "$run_summary" | jq -r '.outlier_rate')"
if [[ "$outlier_rate" == "null" ]]; then
  echo "[FAIL] outlier_rate missing on run summary"
  echo "$run_summary"
  exit 1
fi
python - <<PY2
meta_path="$(echo "$run_summary" | jq -r '.artifact_paths.run_metadata_json // empty')"
words_path="$(echo "$run_summary" | jq -r '.artifact_paths.top_words_json // empty')"
if [[ -z "$meta_path" || -z "$words_path" || ! -f "$meta_path" || ! -f "$words_path" ]]; then
  echo "[FAIL] missing run artifact files (run_metadata_json/top_words_json)"
  echo "$run_summary"
  exit 1
fi
ngram_range="$(jq -r '.vectorizer_params.ngram_range // empty | @json' "$meta_path")"
echo "[INFO] effective ngram_range=$ngram_range"
python - <<PY3
import json
from pathlib import Path
stop = {"the","and","to","you","of","in","it","is","that","for"}
words = json.loads(Path("$words_path").read_text())
keys = sorted(words.keys(), key=lambda x: int(x))
for k in keys[:3]:
    top = [str(w).lower() for w in (words.get(k) or [])[:10]]
    bad = sorted(stop.intersection(top))
    if bad:
        raise SystemExit(f"[FAIL] stopwords dominated topic {k}: {bad} in {top}")
print("[PASS] top_words stopword guard")
PY3
v = float("$outlier_rate")
assert 0.0 <= v <= 1.0, f"outlier_rate out of range: {v}"
PY2
meta_path="$(echo "$run_summary" | jq -r '.artifact_paths.run_metadata_json // empty')"
words_path="$(echo "$run_summary" | jq -r '.artifact_paths.top_words_json // empty')"
if [[ -z "$meta_path" || -z "$words_path" || ! -f "$meta_path" || ! -f "$words_path" ]]; then
  echo "[FAIL] missing run artifact files (run_metadata_json/top_words_json)"
  echo "$run_summary"
  exit 1
fi
ngram_range="$(jq -r '.vectorizer_params.ngram_range // empty | @json' "$meta_path")"
echo "[INFO] effective ngram_range=$ngram_range"
python - <<PY3
import json
from pathlib import Path
stop = {"the","and","to","you","of","in","it","is","that","for"}
words = json.loads(Path("$words_path").read_text())
keys = sorted(words.keys(), key=lambda x: int(x))
for k in keys[:3]:
    top = [str(w).lower() for w in (words.get(k) or [])[:10]]
    bad = sorted(stop.intersection(top))
    if bad:
        raise SystemExit(f"[FAIL] stopwords dominated topic {k}: {bad} in {top}")
print("[PASS] top_words stopword guard")
PY3
list_resp="$(curl -fsS "$BASE_URL/runs/$run_id/results?limit=5")"
count="$(echo "$list_resp" | jq -r '.items | length')"
if [[ "$count" -lt 1 ]]; then
  echo "[FAIL] expected >=1 run result"
  echo "$list_resp"
  exit 1
fi
missing_keys="$(echo "$list_resp" | jq -r '.items[0] | [has("topic_id"), has("topic_label"), (has("topic_prob") or has("prob")), has("representation_backend"), has("topic_repr_terms")] | all')"
if [[ "$missing_keys" != "true" ]]; then
  echo "[FAIL] missing expected run result keys"
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
