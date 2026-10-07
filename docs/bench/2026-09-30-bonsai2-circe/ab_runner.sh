#!/bin/bash
# Apples-to-apples on circe gpu0: 1 slot x 65536, same harness; flash-attn off vs on.
cd /tmp/bonsai-test
LOG=ab.log
PORT=8018
URL=http://127.0.0.1:$PORT/v1/chat/completions
COMMON="--host 0.0.0.0 --port 8080 --ctx-size 65536 --parallel 1 --n-gpu-layers 99 --threads 16 --batch-size 1024 --jinja --reasoning-format deepseek --no-context-shift --n-predict 16384 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0"
run() {  # label image model fa reasoning_kwargs
  local label=$1 image=$2 model=$3 fa=$4 kw=$5
  echo "=== $label $(date -u +%T)" >> $LOG
  docker rm -f ab-worker >/dev/null 2>&1
  docker run -d --name ab-worker --runtime nvidia -e CUDA_VISIBLE_DEVICES=0 -v /mnt/telemetry/llm-cache:/models -p $PORT:8080 \
    --entrypoint /app/llama-server $image -m /models/gguf/$model $COMMON --flash-attn $fa >/dev/null
  for i in $(seq 1 90); do [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:$PORT/health)" = 200 ] && break; sleep 5; done
  echo "vram_loaded $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)" >> $LOG
  python3 -c "
import bonsai_bench as b
b.URL='$URL'; b.OUT=open('ab-short-$label.jsonl','w')
for i in range(2):
    r,_=b.chat([{'role':'user','content':'Write a long story about lighthouse keeper #0.'}],512,'none','$label-short',ignore_eos=True)
print(b.line(r))" >> $LOG 2>&1
  python3 -u bonsai_depth.py $URL $label none >> $LOG 2>&1
  if [ -n "$kw" ]; then python3 -u reason_depth.py $URL $label "$kw" >> $LOG 2>&1; fi
  echo "vram_after $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)" >> $LOG
  docker rm -f ab-worker >/dev/null
}
BON=llamacpp-bonsai-prism:server-local-volta
STOCK=orion-llamacpp-host:0.1.0
Q38='{"reasoning_effort":"medium"}'
Q36='{"enable_thinking":true}'
run bonsai-faoff $BON Ternary-Bonsai-2-27B-PQ2_0.gguf off ""
run bonsai-faon  $BON Ternary-Bonsai-2-27B-PQ2_0.gguf on  "$Q38"
run q4-27b-faoff $STOCK Qwen3.8-27B-UD-Q4_K_XL.gguf off ""
run q4-27b-faon  $STOCK Qwen3.8-27B-UD-Q4_K_XL.gguf on  "$Q38"
run q5-35b-faoff $STOCK Qwen3.6-35B-A3B-UD-Q5_K_M.gguf off ""
run q5-35b-faon  $STOCK Qwen3.6-35B-A3B-UD-Q5_K_M.gguf on  "$Q36"
echo ALLDONE >> $LOG
