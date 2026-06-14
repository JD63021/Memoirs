#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage:"
  echo "  tools/run_gpu_peak.sh <log_file> <command...>"
  exit 2
fi

LOG="$1"
shift

MEMLOG="${LOG%.log}_gpu_mem.csv"

mkdir -p "$(dirname "$LOG")"

base_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"

echo "gpuMemoryBaseMiB = ${base_mib}" | tee "$LOG"
echo "time_ms,memory_used_mib" > "$MEMLOG"

(
  while true; do
    ts="$(date +%s%3N)"
    mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    echo "${ts},${mem}" >> "$MEMLOG"
    sleep 0.10
  done
) &
sampler_pid=$!

set +e
"$@" 2>&1 | tee -a "$LOG"
cmd_status=${PIPESTATUS[0]}
set -e

kill "$sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true

end_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
peak_mib="$(awk -F, 'NR>1 { if ($2>max) max=$2 } END { if (max=="") max=0; print max }' "$MEMLOG")"
delta_mib="$(( peak_mib - base_mib ))"

{
  echo "gpuMemoryEndMiB  = ${end_mib}"
  echo "gpuMemoryPeakMiB = ${peak_mib}"
  echo "gpuMemoryDeltaMiB = ${delta_mib}"
  echo "gpuMemorySamples = ${MEMLOG}"
  echo "commandExitCode = ${cmd_status}"
} | tee -a "$LOG"

exit "$cmd_status"
