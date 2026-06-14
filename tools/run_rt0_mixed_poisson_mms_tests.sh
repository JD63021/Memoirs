#!/usr/bin/env bash
set -euo pipefail

ROOT="${MEMOIRS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HEX_ROOT="${HEX_ROOT:-/home/jd/Desktop/meshes/unitcube/blockmesh}"
TET_ROOT="${TET_ROOT:-/home/jd/Desktop/meshes/unitcube/tetmesh}"
HYPRE_DOUBLE_ROOT="${HYPRE_DOUBLE_ROOT:-/home/jd/opt/hypre-cuda-double-clean}"
HYPRE_SINGLE_ROOT="${HYPRE_SINGLE_ROOT:-/home/jd/opt/hypre-cuda-single-clean}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/rt0_mixed_poisson_mms}"
BUILD_JOBS="${BUILD_JOBS:-8}"

RT0_HEX_N_LIST="${RT0_HEX_N_LIST:-16 24 32}"
RT0_TET_N_LIST="${RT0_TET_N_LIST:-16 24 32}"
RT0_GMRES_KDIM="${RT0_GMRES_KDIM:-200}"
RT0_MAXIT="${RT0_MAXIT:-10000}"
RT0_QUAD_ORDER="${RT0_QUAD_ORDER:-4}"

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

  cmake --build "$bdir" --target memoirs_rt0_mixed_poisson_mms -j "$BUILD_JOBS"
}

run_case() {
  local prec="$1"
  local hroot="$2"
  local kind="$3"
  local N="$4"
  local tol="$5"

  local bdir="$ROOT/build_${prec}_hypre"
  local exe="$bdir/memoirs_rt0_mixed_poisson_mms"

  if [[ "$kind" == hex ]]; then
    mesh="$HEX_ROOT/${N}cube/constant/polyMesh"
  else
    mesh="$TET_ROOT/${N}cube/constant/polyMesh"
  fi

  local log="$LOG_DIR/rt0_${kind}_N${N}_${prec}.log"

  if [[ ! -f "$mesh/points" ]]; then
    echo "SKIP ${prec} RT0 ${kind} ${N}cube: missing $mesh/points"
    return 0
  fi

  echo
  echo "============================================================"
  echo "${prec^^} RT0 mixed Poisson ${kind} MMS ${N}cube"
  echo "============================================================"

  export LD_LIBRARY_PATH="$hroot/lib:${LD_LIBRARY_PATH:-}"
  export MEMOIRS_SPARSE_MODE="${MEMOIRS_SPARSE_MODE:-fixed_csr}"
  export MEMOIRS_QUAD_ORDER="$RT0_QUAD_ORDER"
  export MEMOIRS_GMRES_KDIM="$RT0_GMRES_KDIM"

  "$exe" \
    -polyMeshDir "$mesh" \
    -space rt0 \
    -mms sin \
    -solve 1 \
    -solver gmres \
    -precond none \
    -hypreMemory host \
    -tol "$tol" \
    -maxit "$RT0_MAXIT" \
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
    m = re.search(rf"{re.escape(key)}\s*=\s*([-+0-9.eE]+|nan|inf|-inf)", txt, re.I)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

for prec in ["dp", "sp"]:
    for kind in ["hex", "tet"]:
        rows = []
        for p in sorted(log_dir.glob(f"rt0_{kind}_N*_{prec}.log")):
            m = re.search(rf"rt0_{kind}_N(\d+)_{prec}\.log$", p.name)
            if not m:
                continue
            N = int(m.group(1))
            txt = p.read_text(errors="replace")
            scalar = get(txt, "scalarL2Error")
            flux = get(txt, "fluxL2Error")
            cons = get(txt, "cellConservationAbsMax")
            it = get(txt, "iterations")
            res = get(txt, "finalRelativeResidual")
            if scalar is not None and flux is not None:
                rows.append((N, scalar, flux, cons, it, res))
        rows.sort()

        print(f"\n==== {prec.upper()} RT0 mixed Poisson {kind} MMS order ====")
        if not rows:
            print("No valid logs found.")
            continue

        print("N        scalarL2         fluxL2           consMax      it    finalRel")
        for N,sc,fl,cons,it,res in rows:
            print(f"{N:<8d} {sc:<16.9e} {fl:<16.9e} {cons:<12.3e} {int(it):<5d} {res:.3e}")

        print("\nOrders:")
        for a,b in zip(rows, rows[1:]):
            pS = math.log(a[1]/b[1]) / math.log(b[0]/a[0])
            pF = math.log(a[2]/b[2]) / math.log(b[0]/a[0])
            print(f"{a[0]} -> {b[0]}:  scalar order = {pS:.3f}, flux order = {pF:.3f}")
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

  for N in $RT0_HEX_N_LIST; do
    run_case "$prec" "$hroot" hex "$N" "$tol"
  done

  for N in $RT0_TET_N_LIST; do
    run_case "$prec" "$hroot" tet "$N" "$tol"
  done
done

summarize
