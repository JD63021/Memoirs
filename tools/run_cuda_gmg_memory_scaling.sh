#!/usr/bin/env bash
set -u

PROJECT=/home/jd/memoirs_v0_modular_patch014

OUTDIR="$PROJECT/logs/cuda_gmg_memory_scaling_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

SUMMARY="$OUTDIR/summary.csv"
echo "prec,N,status,it,finalRel,checkedRel,xL2,baselineMiB,peakMiB,netPeakMiB,log" > "$SUMMARY"

extract_field() {
  local key="$1"
  local file="$2"
  awk -v k="$key" '$1 == k {print $3}' "$file" | tail -n 1
}

run_case() {
  local prec="$1"
  local N="$2"

  local exe tol maxit
  if [[ "$prec" == "dp" ]]; then
    exe="$PROJECT/build_dp_cuda_gmg/memoirs_test_structured_gmg_cuda"
    tol="1e-10"
    maxit="180"
  else
    exe="$PROJECT/build_sp_cuda_gmg/memoirs_test_structured_gmg_cuda"
    tol="1e-6"
    maxit="180"
  fi

  local log="$OUTDIR/${prec}_N${N}.log"

  echo
  echo "============================================================"
  echo "CUDA GMG $prec N=$N"
  echo "============================================================"

  if [[ ! -x "$exe" ]]; then
    echo "$prec,$N,NO_EXE,,,,,,,,$log" >> "$SUMMARY"
    echo "Missing executable: $exe"
    return
  fi

  export MEMOIRS_GPU_MEM_DIAG=1
  export MEMOIRS_GPU_MEM_INTERVAL_MS=10

  set +e
  "$exe" \
    -n "$N" \
    -tol "$tol" \
    -maxit "$maxit" \
    -pre 2 \
    -post 2 \
    -omega 0.70 \
    -coarseMaxDofs 256 \
    -printEvery 20 \
    -repeats 1 \
    2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e

  local status="OK"
  if [[ "$rc" -ne 0 ]]; then
    status="FAIL_${rc}"
  fi

  local it finalRel checkedRel xL2 base peak net
  it="$(extract_field iterations "$log")"
  finalRel="$(extract_field finalRelativeResidual "$log")"
  checkedRel="$(extract_field checkedRelativeResidual "$log")"
  xL2="$(extract_field xErrorL2Discrete "$log")"
  base="$(extract_field gpuMemBaselineMiB "$log")"
  peak="$(extract_field gpuMemPeakMiB "$log")"
  net="$(extract_field gpuMemNetPeakMiB "$log")"

  echo "$prec,$N,$status,${it:-NA},${finalRel:-NA},${checkedRel:-NA},${xL2:-NA},${base:-NA},${peak:-NA},${net:-NA},$log" >> "$SUMMARY"

  echo "RESULT prec=$prec N=$N status=$status it=${it:-NA} netPeakMiB=${net:-NA}"
  sleep 3
}

cd "$PROJECT"

# DP: test through and beyond old HYPRE DP memory wall.
for N in 64 96 126 140 160 180 216; do
  run_case dp "$N"
done

# SP: should go farther.
for N in 64 96 126 140 160 180 216 256; do
  run_case sp "$N"
done

echo
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
column -s, -t "$SUMMARY" || cat "$SUMMARY"
echo
echo "Saved to: $OUTDIR"
