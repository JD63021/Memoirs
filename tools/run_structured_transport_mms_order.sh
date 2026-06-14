#!/usr/bin/env bash
set -u

PROJECT=/home/jd/memoirs_v0_modular_patch014
OUTDIR="$PROJECT/logs/structured_transport_mms_order_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

SUMMARY="$OUTDIR/summary.csv"
echo "prec,N,DOFs,status,it,finalRel,xErrorL2Discrete,solveTime,netPeakMiB,log" > "$SUMMARY"

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
    exe="$PROJECT/build_dp_cuda_gmg/memoirs_test_structured_q1_transport_mms_cuda"
    tol="1e-10"
    maxit="500"
  else
    exe="$PROJECT/build_sp_cuda_gmg/memoirs_test_structured_q1_transport_mms_cuda"
    tol="1e-7"
    maxit="700"
  fi

  local log="$OUTDIR/${prec}_N${N}.log"

  echo
  echo "============================================================"
  echo "Transport MMS $prec N=$N"
  echo "============================================================"

  export MEMOIRS_GPU_MEM_DIAG=1
  export MEMOIRS_GPU_MEM_INTERVAL_MS=10

  set +e
  "$exe" \
    -n "$N" \
    -tol "$tol" \
    -maxit "$maxit" \
    -kappa 0.1 \
    -ax 1 -ay 2 -az 3 \
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
  l2="$(extract_field xErrorL2Discrete "$log")"
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
print("Orders based on xErrorL2Discrete:")
for prec in ["dp", "sp"]:
    rs = [
        r for r in rows
        if r["prec"] == prec
        and r["status"] == "OK"
        and r["xErrorL2Discrete"] != "NA"
    ]
    rs.sort(key=lambda r: int(r["N"]))

    print(f"\n{prec.upper()}:")
    for r in rs:
        print(
            f"  N={r['N']:>3}  DOFs={r['DOFs']:>8}  "
            f"L2disc={float(r['xErrorL2Discrete']):.9e}  "
            f"it={r['it']}  rel={r['finalRel']}"
        )

    for a, b in zip(rs, rs[1:]):
        Na = int(a["N"])
        Nb = int(b["N"])
        ea = float(a["xErrorL2Discrete"])
        eb = float(b["xErrorL2Discrete"])
        order = math.log(ea / eb) / math.log(Nb / Na)
        print(f"  {Na} -> {Nb}: L2disc order = {order:.3f}")
PY

echo
echo "Saved to: $OUTDIR"
echo "Summary: $SUMMARY"
