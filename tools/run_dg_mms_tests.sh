#!/usr/bin/env bash
set -euo pipefail

ROOT="${MEMOIRS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TET_ROOT="${TET_ROOT:-/home/jd/Desktop/meshes/unitcube/tetmesh}"
HYPRE_DOUBLE_ROOT="${HYPRE_DOUBLE_ROOT:-/home/jd/opt/hypre-cuda-double-clean}"
HYPRE_SINGLE_ROOT="${HYPRE_SINGLE_ROOT:-/home/jd/opt/hypre-cuda-single-clean}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/dg_mms}"
BUILD_JOBS="${BUILD_JOBS:-8}"

mkdir -p "$LOG_DIR"

build_one() {
  local prec="$1"
  local hroot="$2"
  local bdir="$ROOT/build_${prec}_hypre"

  cmake -S "$ROOT" -B "$bdir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMEMOIRS_PRECISION="$([[ "$prec" == dp ]] && echo double || echo single)" \
    -DMEMOIRS_USE_HYPRE=ON \
    -DMEMOIRS_BUILD_EXPERIMENTS="${MEMOIRS_BUILD_EXPERIMENTS:-OFF}" \
    -DHYPRE_ROOT="$hroot" \
    -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-86}"

  cmake --build "$bdir" --target memoirs_dg_poisson_mms -j "$BUILD_JOBS"
}

run_case() {
  local prec="$1"
  local hroot="$2"
  local N="$3"
  local tol="$4"

  local bdir="$ROOT/build_${prec}_hypre"
  local exe="$bdir/memoirs_dg_poisson_mms"
  local mesh="$TET_ROOT/${N}cube/constant/polyMesh"
  local log="$LOG_DIR/dg_tet_${N}_${prec}_mms.log"

  if [[ ! -f "$mesh/points" ]]; then
    echo "SKIP ${prec} dg_tet ${N}cube: missing $mesh/points"
    return 0
  fi

  echo "============================================================"
  echo "${prec^^} DG tet P1 SIPG MMS ${N}cube"
  echo "============================================================"

  export LD_LIBRARY_PATH="$hroot/lib:${LD_LIBRARY_PATH:-}"
  export MEMOIRS_SPARSE_MODE="${MEMOIRS_SPARSE_MODE:-fixed_csr}"
  export MEMOIRS_DG_PENALTY_SIGMA="${MEMOIRS_DG_PENALTY_SIGMA:-20}"

  export MEMOIRS_AMG_COARSEN="${MEMOIRS_AMG_COARSEN:-8}"
  export MEMOIRS_AMG_INTERP="${MEMOIRS_AMG_INTERP:-6}"
  export MEMOIRS_AMG_RELAX="${MEMOIRS_AMG_RELAX:-18}"
  export MEMOIRS_AMG_AGG_LEVELS="${MEMOIRS_AMG_AGG_LEVELS:-0}"
  export MEMOIRS_AMG_KEEP_TRANSPOSE="${MEMOIRS_AMG_KEEP_TRANSPOSE:-1}"
  export MEMOIRS_AMG_PMAX="${MEMOIRS_AMG_PMAX:-4}"
  export MEMOIRS_AMG_SWEEPS="${MEMOIRS_AMG_SWEEPS:-1}"
  export MEMOIRS_AMG_STRONG="${MEMOIRS_AMG_STRONG:--1}"
  export MEMOIRS_AMG_TRUNC="${MEMOIRS_AMG_TRUNC:-0}"
  export MEMOIRS_AMG_RAP2="${MEMOIRS_AMG_RAP2:-0}"

  "$exe" \
    -polyMeshDir "$mesh" \
    -mms sin \
    -solve 1 \
    -solver pcg \
    -precond amg \
    -hypreMemory device \
    -tol "$tol" \
    -maxit 3000 \
    -diagLevel 0 \
    -hyprePrint 0 \
    -penaltySigma "$MEMOIRS_DG_PENALTY_SIGMA" \
    2>&1 | tee "$log"
}

summarize() {
  python3 - "$LOG_DIR" <<'PY'
import glob
import math
import re
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])

def get_float(txt, key):
    m = re.search(rf"{re.escape(key)}\s*=\s*([-+0-9.eE]+)", txt)
    return float(m.group(1)) if m else None

def parse(prec):
    rows = []
    for fn in sorted(glob.glob(str(log_dir / f"dg_tet_*_{prec}_mms.log"))):
        path = Path(fn)
        mN = re.search(r"dg_tet_(\d+)_" + re.escape(prec) + r"_mms\.log$", path.name)
        if not mN:
            continue

        txt = path.read_text(errors="replace")
        l2 = get_float(txt, "L2Error")
        h1 = get_float(txt, "H1SemiError")
        it = get_float(txt, "iterations")
        res = get_float(txt, "finalRelativeResidual")

        if l2 is None or h1 is None:
            continue

        rows.append((int(mN.group(1)), l2, h1, it, res))

    rows.sort()
    return rows

for prec in ["dp", "sp"]:
    rows = parse(prec)
    print(f"\n==== {prec.upper()} DG tet P1 SIPG MMS order ====")

    if not rows:
        print("No valid logs found.")
        continue

    print("N        L2Error          H1SemiError       it   finalRel")
    for N, l2, h1, it, res in rows:
        it_s = str(int(it)) if it is not None else "NA"
        res_s = f"{res:.3e}" if res is not None else "NA"
        print(f"{N:<8d} {l2:<16.9e} {h1:<16.9e} {it_s:<4s} {res_s}")

    print("\nOrders:")
    for a, b in zip(rows, rows[1:]):
        pL2 = math.log(a[1] / b[1]) / math.log(b[0] / a[0])
        pH1 = math.log(a[2] / b[2]) / math.log(b[0] / a[0])
        print(f"{a[0]} -> {b[0]}:  L2 order = {pL2:.3f}, H1 order = {pH1:.3f}")
PY
}

build_one dp "$HYPRE_DOUBLE_ROOT"
build_one sp "$HYPRE_SINGLE_ROOT"

for prec in dp sp; do
  if [[ "$prec" == dp ]]; then
    tol="1e-10"
    hroot="$HYPRE_DOUBLE_ROOT"
  else
    tol="1e-6"
    hroot="$HYPRE_SINGLE_ROOT"
  fi

  for N in ${DG_N_LIST:-16 24 32}; do
    run_case "$prec" "$hroot" "$N" "$tol"
  done
done

summarize
