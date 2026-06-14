#!/usr/bin/env bash
set -u

PROJECT="${PROJECT:-/home/jd/memoirs_v0_modular_patch014}"
NS="${NS:-16 24 32}"
OUTDIR="$PROJECT/logs/structured_stokes_pspg_mms_order_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

SUMMARY="$OUTDIR/summary.csv"
echo "prec,N,unknowns,status,it,finalRel,checkedRel,velocityL2Quadrature,pressureL2Quadrature,solveTime,netPeakMiB,log" > "$SUMMARY"

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
    exe="$PROJECT/build_dp_cuda_gmg/memoirs_test_structured_q1_stokes_pspg_mms_cuda"
    tol="${DP_TOL:-1e-10}"
    maxit="${DP_MAXIT:-2000}"
  else
    exe="$PROJECT/build_sp_cuda_gmg/memoirs_test_structured_q1_stokes_pspg_mms_cuda"
    tol="${SP_TOL:-1e-7}"
    maxit="${SP_MAXIT:-2500}"
  fi

  local log="$OUTDIR/${prec}_N${N}.log"

  echo
  echo "============================================================"
  echo "Stokes PSPG MMS $prec N=$N"
  echo "============================================================"

  export MEMOIRS_GPU_MEM_DIAG="${MEMOIRS_GPU_MEM_DIAG:-1}"
  export MEMOIRS_GPU_MEM_INTERVAL_MS="${MEMOIRS_GPU_MEM_INTERVAL_MS:-10}"

  set +e
  "$exe" \
    -n "$N" \
    -tol "$tol" \
    -maxit "$maxit" \
    -nu "${NU:-1.0}" \
    -tauScale "${TAU_SCALE:-0.25}" \
    -rhsMode "${RHS_MODE:-0}" \
    -printEvery "${PRINT_EVERY:-25}" \
    -repeats 1 \
    2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e

  local status="OK"
  if [[ "$rc" -ne 0 ]]; then
    status="FAIL_${rc}"
  fi

  local unknowns it rel checked vl2 pl2 t net
  unknowns=$(( 4 * (N + 1) * (N + 1) * (N + 1) ))
  it="$(extract_field iterations "$log")"
  rel="$(extract_field finalRelativeResidual "$log")"
  checked="$(extract_field checkedRelativeResidual "$log")"
  vl2="$(extract_field velocityL2Quadrature "$log")"
  pl2="$(extract_field pressureL2Quadrature "$log")"
  t="$(extract_field cudaGmgSolveAvgSeconds "$log")"
  net="$(extract_field gpuMemNetPeakMiB "$log")"

  echo "$prec,$N,$unknowns,$status,${it:-NA},${rel:-NA},${checked:-NA},${vl2:-NA},${pl2:-NA},${t:-NA},${net:-NA},$log" >> "$SUMMARY"
}

cd "$PROJECT" || exit 1

for prec in ${PRECS:-dp sp}; do
  for N in $NS; do
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

for field, label in [("velocityL2Quadrature", "velocity L2quad"),
                     ("pressureL2Quadrature", "pressure L2quad")]:
    print()
    print(f"Orders based on {label}:")
    for prec in ["dp", "sp"]:
        rs = [r for r in rows if r["prec"] == prec and r["status"] == "OK" and r[field] != "NA"]
        rs.sort(key=lambda r: int(r["N"]))
        print(f"\n{prec.upper()}:")
        for r in rs:
            print(
                f"  N={int(r['N']):>3} unknowns={int(r['unknowns']):>9} "
                f"err={float(r[field]):.9e} it={r['it']} rel={r['finalRel']}"
            )
        for a, b in zip(rs, rs[1:]):
            Na = int(a["N"])
            Nb = int(b["N"])
            ea = float(a[field])
            eb = float(b[field])
            order = math.log(ea / eb) / math.log(Nb / Na)
            print(f"  {Na} -> {Nb}: order = {order:.3f}")
PY

echo
echo "Saved to: $OUTDIR"
echo "Summary: $SUMMARY"
