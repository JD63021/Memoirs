#!/usr/bin/env bash
set -u

PROJECT=/home/jd/memoirs_v0_modular_patch014
HEX_ROOT=/home/jd/Desktop/meshes/unitcube/blockmesh
HYPRE_DOUBLE_ROOT=/home/jd/opt/hypre-cuda-double-clean
HYPRE_SINGLE_ROOT=/home/jd/opt/hypre-cuda-single-clean

OUTDIR="$PROJECT/logs/hex_gpu_net_peak_compare_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

SUMMARY="$OUTDIR/summary.csv"
echo "prec,assembly,N,status,baseline_mib,peak_mib,net_peak_mib,it,finalRel,log" > "$SUMMARY"

extract_field() {
  local key="$1"
  local file="$2"
  awk -v k="$key" '$1 == k {print $3}' "$file" | tail -n 1
}

run_case() {
  local prec="$1"
  local assembly="$2"
  local N="$3"

  local exe hroot tol
  if [[ "$prec" == "dp" ]]; then
    exe="$PROJECT/build_dp_hypre/memoirs_q1_poisson_oneshot"
    hroot="$HYPRE_DOUBLE_ROOT"
    tol="1e-10"
  else
    exe="$PROJECT/build_sp_hypre/memoirs_q1_poisson_oneshot"
    hroot="$HYPRE_SINGLE_ROOT"
    tol="1e-6"
  fi

  local mesh="$HEX_ROOT/${N}cube/constant/polyMesh"
  local tag="${prec}_${assembly}_${N}"
  local log="$OUTDIR/${tag}.log"

  echo
  echo "=================================================================="
  echo "RUN $tag"
  echo "=================================================================="

  if [[ ! -x "$exe" ]]; then
    echo "$prec,$assembly,$N,NO_EXE,,,,,,$log" >> "$SUMMARY"
    echo "NO_EXE: $exe"
    return
  fi

  if [[ ! -d "$mesh" ]]; then
    echo "$prec,$assembly,$N,NO_MESH,,,,,,$log" >> "$SUMMARY"
    echo "NO_MESH: $mesh"
    return
  fi

  export LD_LIBRARY_PATH="$hroot/lib:${LD_LIBRARY_PATH:-}"

  export MEMOIRS_GPU_MEM_DIAG=1
  export MEMOIRS_GPU_MEM_INTERVAL_MS=20
  export MEMOIRS_GPU_MEM_DEVICE=0

  export MEMOIRS_STAGE_TIMERS=1
  export MEMOIRS_COMPUTE_ERROR=0
  export MEMOIRS_SOLVE_REPEATS=1
  export MEMOIRS_RHS_UPDATE_MODE=none

  export MEMOIRS_AMG_COARSEN=8
  export MEMOIRS_AMG_INTERP=6
  export MEMOIRS_AMG_RELAX=18
  export MEMOIRS_AMG_AGG_LEVELS=0
  export MEMOIRS_AMG_KEEP_TRANSPOSE=1
  export MEMOIRS_AMG_PMAX=4
  export MEMOIRS_AMG_SWEEPS=1
  export MEMOIRS_AMG_STRONG=-1
  export MEMOIRS_AMG_TRUNC=0
  export MEMOIRS_AMG_RAP2=0
  export MEMOIRS_IJ_ROW_SIZES=1
  export MEMOIRS_IJ_BULK_INSERT=1

  if [[ "$assembly" == "sumfact" ]]; then
    export MEMOIRS_ASSEMBLY_MODE=structured_hex_q1_sumfact
  else
    unset MEMOIRS_ASSEMBLY_MODE
  fi

  set +e
  "$exe" \
    -polyMeshDir "$mesh" \
    -space cg_hex_q1 \
    -mms sin \
    -solve 1 \
    -solver pcg \
    -precond amg \
    -hypreMemory device \
    -tol "$tol" \
    -maxit 1000 \
    -diagLevel 0 \
    -hyprePrint 0 \
    2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e

  local status="OK"
  if [[ "$rc" -ne 0 ]]; then
    status="FAIL_${rc}"
  fi

  local baseline peak net it rel
  baseline="$(extract_field gpuMemBaselineMiB "$log")"
  peak="$(extract_field gpuMemPeakMiB "$log")"
  net="$(extract_field gpuMemNetPeakMiB "$log")"
  it="$(extract_field iterations "$log")"
  rel="$(extract_field finalRelativeResidual "$log")"

  echo "$prec,$assembly,$N,$status,${baseline:-NA},${peak:-NA},${net:-NA},${it:-NA},${rel:-NA},$log" >> "$SUMMARY"

  echo "RESULT $tag status=$status baseline=${baseline:-NA} peak=${peak:-NA} net=${net:-NA}"
  sleep 3
}

cd "$PROJECT"

# DP: around known limit.
for N in 96 115 126 140 147; do
  run_case dp generic "$N"
  run_case dp sumfact "$N"
done

# SP: around known limit and beyond.
for N in 140 147 152 160 180; do
  run_case sp generic "$N"
  run_case sp sumfact "$N"
done

echo
echo "=================================================================="
echo "SUMMARY"
echo "=================================================================="
column -s, -t "$SUMMARY" || cat "$SUMMARY"
echo
echo "Saved to: $OUTDIR"
