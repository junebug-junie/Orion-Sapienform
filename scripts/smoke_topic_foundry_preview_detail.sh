#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${TOPIC_FOUNDRY_BASE_URL:-http://127.0.0.1:8615}"

payload='{"name":"smoke_preview_detail","source_table":"public.smoke_topic_foundry","id_column":"id","time_column":"created_at","text_columns":["body"],"boundary_column":null,"boundary_strategy":null}'
model_resp="$(curl -fsS -X POST "$BASE_URL/datasets" -H 'content-type: application/json' -d "$payload")"
dataset_id="$(echo "$model_resp" | jq -r '.dataset_id')"
if [[ -z "$dataset_id" || "$dataset_id" == "null" ]]; then
  echo "[FAIL] dataset create missing dataset_id"
  echo "$model_resp"
  exit 1
fi

preview_payload="$(jq -n --arg ds "$dataset_id" '{dataset_id:$ds,windowing:{windowing_mode:"time_gap",time_gap_minutes:15,max_chars:120},limit:200}')"
preview_resp="$(curl -fsS -X POST "$BASE_URL/datasets/preview" -H 'content-type: application/json' -d "$preview_payload")"
doc_id="$(echo "$preview_resp" | jq -r '.samples[0].doc_id')"
snippet_len="$(echo "$preview_resp" | jq -r '.samples[0].snippet | length')"
if [[ -z "$doc_id" || "$doc_id" == "null" ]]; then
  echo "[FAIL] preview returned no doc_id"
  echo "$preview_resp"
  exit 1
fi

detail_resp="$(curl -fsS "$BASE_URL/datasets/$dataset_id/preview/docs/$doc_id?windowing_mode=time_gap&time_gap_minutes=15&max_chars=120&limit=200")"
char_count="$(echo "$detail_resp" | jq -r '.char_count')"
full_len="$(echo "$detail_resp" | jq -r '.full_text | length')"
if [[ "$char_count" -le "$snippet_len" || "$full_len" -le 120 ]]; then
  echo "[FAIL] detail full_text not longer than snippet/max threshold"
  echo "snippet_len=$snippet_len char_count=$char_count full_len=$full_len"
  echo "$detail_resp"
  exit 1
fi

echo "[PASS] preview detail smoke passed dataset_id=$dataset_id doc_id=$doc_id full_len=$full_len"
