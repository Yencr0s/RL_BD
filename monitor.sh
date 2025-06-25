#!/bin/bash
OUTPUT="/vol/dis/rriano/monitor.log"

# Write header line
echo "Timestamp,GPU_Info,CPU_Utilization(%),Memory_Used_MB" > "$OUTPUT"

while true; do
  # Get timestamp
#   TS=$(date +"%Y-%m-%d %H:%M:%S")

  # Get GPU metrics (index, memory used, GPU utilization) and join multiple GPUs with a semicolon
  GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | paste -sd " |-| " -)

  # Get overall CPU utilization: using mpstat to extract the "all" line and computing 100 - idle
  CPU=$(mpstat -P ALL 1 1 | awk '/^Average:/ && $2=="all" {print 100 - $12}')

  # Get memory usage in MB (used memory from free -m)
  MEM=$(free -m | awk '/^Mem:/ {print $3}')

  # Write CSV row (quote the GPU field in case it contains commas)
  echo "$TS,\"$GPU\",$CPU,$MEM" >> "$OUTPUT"

  sleep 60
done
