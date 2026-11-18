#!/bin/bash
# 4 slots x 32K, flash-attn on, gpu0: bonsai vs Qwen3.8-27B Q4. Restores chat at the end.
cd /tmp/bonsai-test
LOG=conc.log
PORT=8018; URL=http://127.0.0.1:$PORT/v1/chat/completions
COMMON="--host 0.0.0.0 --port 8080 --ctx-size 131072 --parallel 4 --n-gpu-layers 99 --threads 16 --batch-size 1024 --jinja --reasoning-format deepseek --no-context-shift --n-predict 16384 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --flash-attn on"
run() {
  local label=$1
  echo "=== $label $(date -u +%T)" >> $LOG
  docker rm -f ab-worker >/dev/null 2>&1
  docker run -d --name ab-worker --runtime nvidia -e CUDA_VISIBLE_DEVICES=0 -v /mnt/telemetry/llm-cache:/models -p $PORT:8080 \
    --entrypoint /app/llama-server $2 -m /models/gguf/$3 $COMMON >/dev/null
  for i in $(seq 1 90); do [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:$PORT/health)" = 200 ] && break; sleep 5; done
  echo "vram_loaded $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)" >> $LOG
  (nvidia-smi -i 0 --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 1 > vram-conc-$label.csv 2>&1 &) 
  python3 -u -c "
import bonsai_bench as b
b.URL='$URL'; b.OUT=open('conc-$label.jsonl','w')
b.phase_concurrency((1, 2, 3, 4))
b.phase_concurrency((1, 2, 3, 4))
b.phase_agent4()" >> $LOG 2>&1
  for i in 0 1 2 3; do DEPTH_OFFSET=$((i * 23000)) DEPTH_MAXTURN=3 python3 -u bonsai_depth.py $URL $label-cd$i none > cdepth-$label-$i.log 2>&1 & done; wait
  for i in 0 1 2 3; do sed "s/^/cd$i /" cdepth-$label-$i.log >> $LOG; done
  pkill -f "vram-conc" ; pkill -f "query-gpu=timestamp,memory.used,utilization.gpu"
  echo "vram_after $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)" >> $LOG
  docker rm -f ab-worker >/dev/null
}
run bonsai llamacpp-bonsai-prism:server-local-volta Ternary-Bonsai-2-27B-PQ2_0.gguf
run q4-27b orion-llamacpp-host:0.1.0 Qwen3.8-27B-UD-Q4_K_XL.gguf
docker start orion-circe-atlas-llamacpp-chat >/dev/null && echo "chat restarted $(date -u +%T)" >> $LOG
echo CONCDONE >> $LOG
