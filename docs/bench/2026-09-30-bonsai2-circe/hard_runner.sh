#!/bin/bash
# Waits for ab_runner, then the hard reasoning test with flash-attn on, then restores chat.
cd /tmp/bonsai-test
LOG=hard.log
while ! grep -q ALLDONE ab.log 2>/dev/null; do sleep 15; done
PORT=8018; URL=http://127.0.0.1:$PORT/v1/chat/completions
COMMON="--host 0.0.0.0 --port 8080 --ctx-size 65536 --parallel 1 --n-gpu-layers 99 --threads 16 --batch-size 1024 --jinja --reasoning-format deepseek --no-context-shift --n-predict 16384 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --flash-attn on"
run() {
  echo "=== $1 $(date -u +%T)" >> $LOG
  docker rm -f ab-worker >/dev/null 2>&1
  docker run -d --name ab-worker --runtime nvidia -e CUDA_VISIBLE_DEVICES=0 -v /mnt/telemetry/llm-cache:/models -p $PORT:8080 \
    --entrypoint /app/llama-server $2 -m /models/gguf/$3 $COMMON >/dev/null
  for i in $(seq 1 90); do [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:$PORT/health)" = 200 ] && break; sleep 5; done
  python3 -u reason_hard.py $URL $1 "$4" >> $LOG 2>&1
  docker rm -f ab-worker >/dev/null
}
run bonsai $(echo llamacpp-bonsai-prism:server-local-volta) Ternary-Bonsai-2-27B-PQ2_0.gguf '{"reasoning_effort":"medium"}'
run q4-27b orion-llamacpp-host:0.1.0 Qwen3.8-27B-UD-Q4_K_XL.gguf '{"reasoning_effort":"medium"}'
run q5-35b orion-llamacpp-host:0.1.0 Qwen3.6-35B-A3B-UD-Q5_K_M.gguf '{"enable_thinking":true}'
docker start orion-circe-atlas-llamacpp-chat >/dev/null && echo "chat restarted $(date -u +%T)" >> $LOG
echo HARDDONE >> $LOG
