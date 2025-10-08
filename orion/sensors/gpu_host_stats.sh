#!/bin/bash

#!/bin/bash

# Run once, collect GPU stats, save to /mnt/telemetry/gpu_stats

OUTDIR="/mnt/telemetry/gpu_stats"
mkdir -p "$OUTDIR"

TIMESTAMP=$(date -Iseconds)
OUTFILE="$OUTDIR/${TIMESTAMP}.csv"

echo "timestamp,gpu_index,gpu_name,utilization_gpu,memory_used_mb,memory_total_mb,power_draw_watts" > "$OUTFILE"

nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw \
           --format=csv,noheader,nounits \
| while IFS=',' read -r index name util mem_used mem_total power; do
    echo "$TIMESTAMP,$index,$name,$util,$mem_used,$mem_total,$power" >> "$OUTFILE"
done
