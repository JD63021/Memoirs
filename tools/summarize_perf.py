#!/usr/bin/env python3
import re
import sys
from pathlib import Path
from math import isfinite

def grab_float(text, key):
    m = re.search(rf"{re.escape(key)}\s*=\s*([-+0-9.eE]+)", text)
    return float(m.group(1)) if m else None

def grab_int(text, key):
    x = grab_float(text, key)
    return int(x) if x is not None else None

def main():
    if len(sys.argv) != 2:
        print("Usage: tools/summarize_perf.py <log_file>")
        return 2

    path = Path(sys.argv[1])
    text = path.read_text(errors="replace")

    rows = grab_int(text, "rows")
    nnz = grab_int(text, "nnz")
    iterations = grab_int(text, "iterations")
    repeats = grab_int(text, "solveRepeats") or 1

    solve_avg = grab_float(text, "hypreSolveOnlyAvgSeconds")
    solve_total = grab_float(text, "hypreSolveOnlySeconds")
    setup = grab_float(text, "hypreSetupSeconds")
    assembly = grab_float(text, "assemblySeconds")
    mat_insert = grab_float(text, "hypreMatrixInsertSeconds")
    mat_migrate = grab_float(text, "hypreMatrixMigrateSeconds")
    rhs_avg = grab_float(text, "rhsUpdateAvgSeconds")
    final_rel = grab_float(text, "finalRelativeResidual")

    base = grab_int(text, "gpuMemoryBaseMiB")
    peak = grab_int(text, "gpuMemoryPeakMiB")
    delta = grab_int(text, "gpuMemoryDeltaMiB")

    miups = None
    if rows and iterations and solve_avg and solve_avg > 0:
        miups = rows * iterations / solve_avg / 1.0e6

    spmv_like_mnzps = None
    if nnz and iterations and solve_avg and solve_avg > 0:
        spmv_like_mnzps = nnz * iterations / solve_avg / 1.0e6

    print("============== performance summary ==============")
    print(f"log                         = {path}")
    print(f"rows                        = {rows}")
    print(f"nnz                         = {nnz}")
    print(f"iterations_last_solve        = {iterations}")
    print(f"solveRepeats                = {repeats}")
    print(f"finalRelativeResidual        = {final_rel}")
    print(f"assemblySeconds             = {assembly}")
    print(f"hypreMatrixInsertSeconds     = {mat_insert}")
    print(f"hypreMatrixMigrateSeconds    = {mat_migrate}")
    print(f"hypreSetupSeconds            = {setup}")
    print(f"rhsUpdateAvgSeconds          = {rhs_avg}")
    print(f"hypreSolveOnlyAvgSeconds     = {solve_avg}")
    print(f"hypreSolveOnlySeconds        = {solve_total}")
    if miups is not None:
        print(f"MIUPS_rows_iter_per_s        = {miups:.3f}")
    if spmv_like_mnzps is not None:
        print(f"MNZIPS_nnz_iter_per_s        = {spmv_like_mnzps:.3f}")
    print(f"gpuMemoryBaseMiB            = {base}")
    print(f"gpuMemoryPeakMiB            = {peak}")
    print(f"gpuMemoryDeltaMiB           = {delta}")
    print("=================================================")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
