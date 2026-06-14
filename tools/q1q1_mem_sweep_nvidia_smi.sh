#!/usr/bin/env bash
set -euo pipefail

APP="./build_dp_cuda_gmg/memoirs_test_structured_q1_stokes_pspg_mms_cuda"

OUTDIR="mem_sweep_q1q1_${SOLVER}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

GPU_ID="${GPU_ID:-0}"
TOL="${TOL:-1e-6}"
MAXIT="${MAXIT:-50}"
RESTART="${RESTART:-50}"
SOLVER="${SOLVER:-gmres}"
PRINT_EVERY="${PRINT_EVERY:-25}"

NS="${NS:-32 48 64 80 96 112 128}"

CSV="$OUTDIR/summary.csv"
echo "solver,N,unknowns,iterations,converged,checkedRel,baseMiB,peakMiB,deltaMiB,MiB_per_Munknown,rc,log" > "$CSV"

echo "Output directory: $OUTDIR"
echo "SOLVER=$SOLVER TOL=$TOL MAXIT=$MAXIT RESTART=$RESTART NS=$NS"
echo

for N in $NS; do
  echo "============================================================"
  echo "Running Q1/Q1 memory sweep N=$N"
  echo "============================================================"

  LOG="$OUTDIR/N${N}.log"
  MEM="$OUTDIR/N${N}_nvidia_smi_mem.csv"

  base=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')

  "$APP" \
    -solver "$SOLVER" \
    -gmresRestart "$RESTART" \
    -n "$N" \
    -tol "$TOL" \
    -maxit "$MAXIT" \
    -nu 1.0 \
    -tauScale 1.0 \
    -tauMode metric \
    -tauC 4.0 \
    -pGradSign 1 \
    -pspgRhsSign 1 \
    -pspgForceMode fullF \
    -rhsMode 0 \
    -printEvery "$PRINT_EVERY" \
    -repeats 1 \
    > "$LOG" 2>&1 &
  pid=$!

  (
    while kill -0 "$pid" 2>/dev/null; do
      ts=$(date +%s.%N)
      mem=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
      echo "$ts,$mem"
      sleep 0.20
    done
  ) > "$MEM" &
  monpid=$!

  set +e
  wait "$pid"
  rc=$?
  set -e

  wait "$monpid" 2>/dev/null || true

  peak=$(awk -F, -v b="$base" 'BEGIN{m=b} {if($2+0>m)m=$2+0} END{print m}' "$MEM")
  delta=$((peak - base))

  unknowns=$(grep -m1 "unknowns" "$LOG" | awk '{print $3}' || echo "NA")
  iterations=$(grep -m1 "iterations" "$LOG" | awk '{print $3}' || echo "NA")
  converged=$(grep -m1 "converged" "$LOG" | awk '{print $3}' || echo "NA")
  checkedRel=$(grep -m1 "checkedRelativeResidual" "$LOG" | awk '{print $3}' || echo "NA")

  mib_per_munknown="NA"
  if [[ "$unknowns" != "NA" && "$unknowns" -gt 0 ]]; then
    mib_per_munknown=$(awk -v d="$delta" -v u="$unknowns" 'BEGIN{printf "%.3f", d/(u/1000000.0)}')
  fi

  echo "$SOLVER,$N,$unknowns,$iterations,$converged,$checkedRel,$base,$peak,$delta,$mib_per_munknown,$rc,$LOG" >> "$CSV"

  echo "N=$N unknowns=$unknowns it=$iterations conv=$converged rel=$checkedRel"
  echo "baseMiB=$base peakMiB=$peak deltaMiB=$delta MiB/Munknown=$mib_per_munknown"
  echo "log=$LOG"
  echo

  if [[ "$rc" -ne 0 ]]; then
    if grep -qiE "out of memory|cuda.*error|allocation|bad_alloc" "$LOG"; then
      echo "Run appears to have hit allocation/CUDA failure for N=$N with rc=$rc. Stopping sweep."
      exit "$rc"
    elif [[ "$unknowns" != "NA" ]]; then
      echo "Run returned rc=$rc, likely nonconvergence/breakdown. Continuing because memory data was captured."
    else
      echo "Run failed for N=$N with rc=$rc and no unknown count. Stopping sweep."
      exit "$rc"
    fi
  fi
done

echo "============================================================"
echo "Summary:"
column -s, -t "$CSV" || cat "$CSV"
echo "============================================================"
echo "CSV: $CSV"
