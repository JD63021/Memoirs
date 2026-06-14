#!/usr/bin/env bash
set -euo pipefail

ROOT="${MEMOIRS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HEX_ROOT="${HEX_ROOT:-/home/jd/Desktop/meshes/unitcube/blockmesh}"
HYPRE_DOUBLE_ROOT="${HYPRE_DOUBLE_ROOT:-/home/jd/opt/hypre-cuda-double-clean}"
HYPRE_SINGLE_ROOT="${HYPRE_SINGLE_ROOT:-/home/jd/opt/hypre-cuda-single-clean}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/dg_hex_q1_mms}"
BUILD_JOBS="${BUILD_JOBS:-8}"
HEX_N_LIST="${HEX_N_LIST:-16 24 32 64}"
PENALTY_WEAK="${PENALTY_WEAK:-10}"
PENALTY_STRONG="${PENALTY_STRONG:-5}"

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

  cmake --build "$bdir" --target memoirs_dg_hex_q1_mms -j "$BUILD_JOBS"
}

run_case() {
  local prec="$1"
  local hroot="$2"
  local N="$3"
  local bc="$4"
  local sigma="$5"
  local tol="$6"

  local bdir="$ROOT/build_${prec}_hypre"
  local exe="$bdir/memoirs_dg_hex_q1_mms"
  local mesh="$HEX_ROOT/${N}cube/constant/polyMesh"
  local log="$LOG_DIR/dg_hex_q1_${bc}_N${N}_${prec}.log"

  if [[ ! -f "$mesh/points" ]]; then
    echo "SKIP ${prec} dg_hex_q1 ${bc} ${N}cube: missing $mesh/points"
    return 0
  fi

  echo
  echo "============================================================"
  echo "${prec^^} DG hex Q1 ${bc} MMS ${N}cube sigma=${sigma}"
  echo "============================================================"

  export LD_LIBRARY_PATH="$hroot/lib:${LD_LIBRARY_PATH:-}"
  export MEMOIRS_SPARSE_MODE="${MEMOIRS_SPARSE_MODE:-fixed_csr}"

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
    -bc "$bc" \
    -penaltySigma "$sigma" \
    -solve 1 \
    -solver pcg \
    -precond amg \
    -hypreMemory device \
    -tol "$tol" \
    -maxit 3000 \
    -diagLevel 0 \
    -hyprePrint 0 \
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

def parse(prefix):
    rows = []
    for fn in sorted(glob.glob(str(log_dir / f"{prefix}_N*_*.log"))):
        p = Path(fn)
        m = re.search(r"_N(\d+)_(dp|sp)\.log$", p.name)
        if not m:
            continue
        N = int(m.group(1))
        prec = m.group(2)
        txt = p.read_text(errors="replace")
        l2 = get_float(txt, "L2Error")
        h1 = get_float(txt, "H1SemiError")
        it = get_float(txt, "iterations")
        res = get_float(txt, "finalRelativeResidual")
        if l2 is None or h1 is None:
            continue
        rows.append((prec, N, l2, h1, it, res))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows

for prec in ["dp", "sp"]:
    for bc in ["sipg", "strong"]:
        prefix = f"dg_hex_q1_{bc}"
        rows = [r for r in parse(prefix) if r[0] == prec]

        print(f"\n==== {prec.upper()} DG hex Q1 {bc} MMS order ====")
        if not rows:
            print("No valid logs found.")
            continue

        print("N        L2Error          H1SemiError       it   finalRel")
        for _,N,l2,h1,it,res in rows:
            print(f"{N:<8d} {l2:<16.9e} {h1:<16.9e} {int(it):<4d} {res:.3e}")

        print("\nOrders:")
        for a,b in zip(rows, rows[1:]):
            pL2 = math.log(a[2]/b[2]) / math.log(b[1]/a[1])
            pH1 = math.log(a[3]/b[3]) / math.log(b[1]/a[1])
            print(f"{a[1]} -> {b[1]}:  L2 order = {pL2:.3f}, H1 order = {pH1:.3f}")
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

  for N in $HEX_N_LIST; do
    run_case "$prec" "$hroot" "$N" sipg "$PENALTY_WEAK" "$tol"
  done

  for N in $HEX_N_LIST; do
    run_case "$prec" "$hroot" "$N" strong "$PENALTY_STRONG" "$tol"
  done
done

summarize
