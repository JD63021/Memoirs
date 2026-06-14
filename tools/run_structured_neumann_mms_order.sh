#!/usr/bin/env bash
set -u

PROJECT=/home/jd/memoirs_v0_modular_patch014
OUTDIR="$PROJECT/logs/structured_neumann_mms_order_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

SUMMARY="$OUTDIR/summary.csv"
echo "prec,N,DOFs,status,it,finalRel,l2DiscreteError,solveTime,netPeakMiB,log" > "$SUMMARY"

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
    exe="$PROJECT/build_dp_cuda_gmg/memoirs_test_structured_q1_poisson_neumann_mms_cuda"
    tol="1e-10"
    maxit="400"
  else
    exe="$PROJECT/build_sp_cuda_gmg/memoirs_test_structured_q1_poisson_neumann_mms_cuda"
    tol="1e-6"
    maxit="400"
  fi

  local log="$OUTDIR/${prec}_N${N}.log"

  echo
  echo "============================================================"
  echo "Neumann MMS $prec N=$N"
  echo "============================================================"

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
    -coarseMaxDofs 64 \
    -printEvery 20 \
    -repeats 1 \
    2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e

  local status="OK"
  if [[ "$rc" -ne 0 ]]; then
    status="FAIL_${rc}"
  fi

  local dofs it rel l2 t net
  dofs=$(( (N + 1) * (N + 1) * (N + 1) ))
  it="$(extract_field iterations "$log")"
  rel="$(extract_field finalRelativeResidual "$log")"
  l2="$(extract_field l2DiscreteError "$log")"
  t="$(extract_field cudaGmgSolveAvgSeconds "$log")"
  net="$(extract_field gpuMemNetPeakMiB "$log")"

  echo "$prec,$N,$dofs,$status,${it:-NA},${rel:-NA},${l2:-NA},${t:-NA},${net:-NA},$log" >> "$SUMMARY"
}

cd "$PROJECT"

for prec in dp sp; do
  for N in 16 24 32; do
    run_case "$prec" "$N"
  done
done

python3 - "$SUMMARY" <<'PY'
import csv
import math
import sys
from pathlib import Path

summary = Path(sys.argv[1])

rows = list(csv.DictReader(summary.open()))
print()
print("Orders based on l2DiscreteError:")
for prec in ["dp", "sp"]:
    rs = [r for r in rows if r["prec"] == prec and r["status"] == "OK" and r["l2DiscreteError"] != "NA"]
    rs.sort(key=lambda r: int(r["N"]))

    print(f"\n{prec.upper()}:")
    for r in rs:
        print(f"  N={r['N']:>3}  DOFs={r['DOFs']:>8}  L2disc={float(r['l2DiscreteError']):.9e}  it={r['it']}")

    for a, b in zip(rs, rs[1:]):
        Na = int(a["N"])
        Nb = int(b["N"])
        ea = float(a["l2DiscreteError"])
        eb = float(b["l2DiscreteError"])
        order = math.log(ea / eb) / math.log(Nb / Na)
        print(f"  {Na} -> {Nb}: L2disc order = {order:.3f}")
PY

echo
echo "Saved to: $OUTDIR"
echo "Summary: $SUMMARY"
