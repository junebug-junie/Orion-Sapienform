#!/bin/bash

#!/bin/bash

# Run once, collect GPU stats, save to /mnt/telemetry/gpu_stats

OUTDIR="/mnt/telemetry/gpu_stats"
mkdir -p "$OUTDIR"

TIMESTAMP=$(date -Iseconds)
OUTFILE="$OUTDIR/${TIMESTAMP}.csv"
PROCS_OUTFILE="$OUTDIR/${TIMESTAMP}.procs.csv"

echo "timestamp,gpu_index,gpu_uuid,gpu_name,utilization_gpu,memory_used_mb,memory_total_mb,power_draw_watts" > "$OUTFILE"

nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw \
           --format=csv,noheader,nounits \
| while IFS=',' read -r index uuid name util mem_used mem_total power; do
    echo "$TIMESTAMP,$index,$uuid,$name,$util,$mem_used,$mem_total,$power" >> "$OUTFILE"
done

# Per-GPU compute-process list, joined back to the row above by gpu_uuid (not
# index -- --query-compute-apps does not report an index on every driver
# version, only gpu_uuid). Sibling filename deliberately does NOT end in a
# bare ".csv" pattern that the main-file glob in utils.py:collect_gpu_stats()
# would also match -- it's filtered out there explicitly by suffix, but the
# ".procs.csv" naming keeps that filter legible rather than relying on it alone.
echo "gpu_uuid,pid,process_name,used_memory_mb" > "$PROCS_OUTFILE"

nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
           --format=csv,noheader,nounits \
| while IFS=',' read -r gpu_uuid pid process_name used_memory; do
    echo "$gpu_uuid,$pid,$process_name,$used_memory" >> "$PROCS_OUTFILE"
done
