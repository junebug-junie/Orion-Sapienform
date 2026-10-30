#!/bin/bash
# Strata (Qwen3.8-Flash-Next) on hecate's single V100-SXM2-32GB, same harness as the 2026-09-30 circe bake-off.
# usage: run_hecate.sh <data_dir> [pin|nopin] [model]   e.g. run_hecate.sh /mnt/telemetry/strata pin IQ3_S
# pin = the GPU's socket only (Juniper's single-CPU rule; read CPUS/NODE from `nvidia-smi topo -m` first).
set -u
DATA=$1; MODE=${2:-pin}; MODEL=${3:-IQ3_S}
CPUS=${CPUS:-20-39,60-79}; NODE=${NODE:-1}
IMAGE=ghcr.io/niko1221/strata:0.2.7-cuda12
PORT=8090; URL=http://127.0.0.1:$PORT/v1/chat/completions
cd "$(dirname "$0")"; export BENCH_DIR=$PWD/out-$MODE-$MODEL BENCH_STREAM=1; mkdir -p "$BENCH_DIR"
LOG=$BENCH_DIR/run.log
KEY_FILE=$DATA/api_key; [ -s "$KEY_FILE" ] || openssl rand -hex 24 > "$KEY_FILE"; export BENCH_API_KEY=$(cat "$KEY_FILE")
PIN=(); [ "$MODE" = pin ] && PIN=(--cpuset-cpus "$CPUS" --cpuset-mems "$NODE")

docker rm -f strata-bench >/dev/null 2>&1
echo "=== $MODE $MODEL $(date -u +%FT%TZ) pin='${PIN[*]}'" | tee -a "$LOG"
docker run -d --name strata-bench --gpus '"device=0"' -p 127.0.0.1:$PORT:8080 "${PIN[@]}" \
  --ulimit memlock=-1:-1 --shm-size 2g --stop-timeout 120 -v "$DATA":/data \
  -e FAMILY=qwen -e MODEL="$MODEL" -e CONTEXT=128k -e PARALLEL=1 -e API_KEY="$BENCH_API_KEY" "$IMAGE" >/dev/null || exit 1
# first start downloads the model (tens of GB); wait up to 3 h, bail if the container dies
for i in $(seq 1 1080); do
  curl -sf -H "Authorization: Bearer $BENCH_API_KEY" localhost:$PORT/v1/models >/dev/null && break
  [ "$(docker inspect -f '{{.State.Running}}' strata-bench)" = true ] || { echo "container died" | tee -a "$LOG"; docker logs --tail 50 strata-bench >> "$LOG" 2>&1; exit 1; }
  sleep 10
done
export BENCH_MODEL=$(curl -s -H "Authorization: Bearer $BENCH_API_KEY" localhost:$PORT/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
echo "model=$BENCH_MODEL vram=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader) rss_node$NODE=$(numastat -p "$(docker inspect -f '{{.State.Pid}}' strata-bench)" 2>/dev/null | tail -1)" | tee -a "$LOG"

python3 -c "
import bonsai_bench as b
b.URL='$URL'
for i in range(3):
    r,_=b.chat([{'role':'user','content':'Write a long story about lighthouse keeper #%d.'%i}],512,'none','$MODE-short')
    print(b.line(r))" 2>&1 | tee -a "$LOG"
[ "${SHORT_ONLY:-0}" = 1 ] && exit 0
python3 -u bonsai_depth.py "$URL" "$MODE-$MODEL" none 2>&1 | tee -a "$LOG"
python3 -u reason_depth.py "$URL" "$MODE-$MODEL" '{"reasoning_effort":"medium"}' 2>&1 | tee -a "$LOG"
python3 -u reason_hard.py "$URL" "$MODE-$MODEL" '{"reasoning_effort":"medium"}' 2>&1 | tee -a "$LOG"
echo "vram_after $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader) ALLDONE" | tee -a "$LOG"
