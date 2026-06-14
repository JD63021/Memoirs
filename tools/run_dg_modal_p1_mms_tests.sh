#!/usr/bin/env bash
set -euo pipefail

ROOT="${MEMOIRS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HEX_ROOT="${HEX_ROOT:-/home/jd/Desktop/meshes/unitcube/blockmesh}"
TET_ROOT="${TET_ROOT:-/home/jd/Desktop/meshes/unitcube/tetmesh}"
HYPRE_DOUBLE_ROOT="${HYPRE_DOUBLE_ROOT:-/home/jd/opt/hypre-cuda-double-clean}"
HYPRE_SINGLE_ROOT="${HYPRE_SINGLE_ROOT:-/home/jd/opt/hypre-cuda-single-clean}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/dg_modal_p1_mms}"
BUILD_JOBS="${BUILD_JOBS:-8}"

TET_N_LIST="${TET_N_LIST:-16 24 32}"
HEX_N_LIST="${HEX_N_LIST:-16 24 32 64}"
PENALTY_MODAL="${PENALTY_MODAL:-10}"

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

  cmake --build "$bdir" --target memoirs_dg_modal_p1_mms -j "$BUILD_JOBS"
}

run_case() {
  local prec="$1"
  local hroot="$2"
  local kind="$3"
  local N="$4"
  local tol="$5"

  local bdir="$ROOT/build_${prec}_hypre"
  local exe="$bdir/memoirs_dg_modal_p1_mms"

  if [[ "$kind" == tet ]]; then
    mesh="$TET_ROOT/${N}cube/constant/polyMesh"
  else
    mesh="$HEX_ROOT/${N}cube/constant/polyMesh"
  fi

  local log="$LOG_DIR/modal_${kind}_N${N}_${prec}.log"

  if [[ ! -f "$mesh/points" ]]; then
    echo "SKIP ${prec} modal ${kind} ${N}cube: missing $mesh/points"
    return 0
  fi

  echo
  echo "============================================================"
  echo "${prec^^} modal DG P1 ${kind} MMS ${N}cube sigma=${PENALTY_MODAL}"
  echo "============================================================"

  export LD_LIBRARY_PATH="$hroot/lib:${LD_LIBRARY_PATH:-}"
  export MEMOIRS_SPARSE_MODE="${MEMOIRS_SPARSE_MODE:-fixed_csr}"

  "$exe" \
    -polyMeshDir "$mesh" \
    -mms sin \
    -bc sipg \
    -penaltySigma "$PENALTY_MODAL" \
    -solve 1 \
    -solver pcg \
    -precond amg \
    -hypreMemory device \
    -tol "$tol" \
    -maxit 5000 \
    -diagLevel 0 \
    -hyprePrint 0 \
    2>&1 | tee "$log"
}

summarize() {
  python3 - "$LOG_DIR" <<'PY'
import math, re, sys
from pathlib import Path

log_dir = Path(sys.argv[1])

def get(txt, key):
    m = re.search(rf"{re.escape(key)}\s*=\s*([-+0-9.eE]+)", txt)
    return float(m.group(1)) if m else None

for prec in ["dp", "sp"]:
    for kind in ["tet", "hex"]:
        rows = []
        for p in sorted(log_dir.glob(f"modal_{kind}_N*_{prec}.log")):
            m = re.search(rf"modal_{kind}_N(\d+)_{prec}\.log$", p.name)
            if not m:
                continue
            N = int(m.group(1))
            txt = p.read_text(errors="replace")
            l2 = get(txt, "L2Error")
            h1 = get(txt, "H1SemiError")
            it = get(txt, "iterations")
            res = get(txt, "finalRelativeResidual")
            if l2 is not None and h1 is not None:
                rows.append((N,l2,h1,it,res))
        rows.sort()

        print(f"\n==== {prec.upper()} modal DG P1 {kind} MMS order ====")
        if not rows:
            print("No valid logs found.")
            continue

        print("N        L2Error          H1SemiError       it   finalRel")
        for N,l2,h1,it,res in rows:
            print(f"{N:<8d} {l2:<16.9e} {h1:<16.9e} {int(it):<4d} {res:.3e}")

        print("\nOrders:")
        for a,b in zip(rows, rows[1:]):
            pL2 = math.log(a[1]/b[1]) / math.log(b[0]/a[0])
            pH1 = math.log(a[2]/b[2]) / math.log(b[0]/a[0])
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

  for N in $TET_N_LIST; do
    run_case "$prec" "$hroot" tet "$N" "$tol"
  done

  for N in $HEX_N_LIST; do
    run_case "$prec" "$hroot" hex "$N" "$tol"
  done
done

summarize
