#!/usr/bin/env bash
set -u

cd /home/jd/memoirs_v0_modular_patch014
mkdir -p logs

# Fixed problem settings
N=16
SOLVER=bicgstab
LIN_TOL=1e-8
LIN_MAXIT=2000
PIC_TOL=1e-5
MAX_PICARD=8

NU=1.0
DT=1e-3
ADV_SCALE=1.0
TAU_MODE=metric
TAU_SCALE=1.0
TAU_C=4.0
SUPG=0

HYPRE_MEMORY=host
AMG_NUM_FUNCTIONS=4
AMG_NODAL=0
AMG_AGG_LEVELS=0
AMG_KEEP_TRANSPOSE=1
GMRES_RESTART=80

# Sweep settings
for AMG_INTERP in 6 15 17 18; do
for AMG_STRONG in -1 0.25 0.5 0.7; do
for AMG_TRUNC in 0.0 0.1 0.2; do
for AMG_PMAX in 4 8; do

  TAG="N${N}_interp${AMG_INTERP}_strong${AMG_STRONG}_trunc${AMG_TRUNC}_pmax${AMG_PMAX}"
  TAG=${TAG//./p}
  TAG=${TAG//-/m}
  LOG="logs/sweep_${TAG}.log"

  echo
  echo "============================================================"
  echo "RUN $TAG"
  echo "============================================================"

  stdbuf -oL -eL env \
    MEMOIRS_COMPUTE_ERROR=1 \
    MEMOIRS_Q1Q1_RHS_MODE=picard \
    MEMOIRS_Q1Q1_MAX_PICARD="$MAX_PICARD" \
    MEMOIRS_Q1Q1_PICARD_TOL="$PIC_TOL" \
    MEMOIRS_Q1Q1_BETA_INITIAL=zero \
    MEMOIRS_Q1Q1_DT="$DT" \
    MEMOIRS_Q1Q1_ADV_SCALE="$ADV_SCALE" \
    MEMOIRS_Q1Q1_TAU_MODE="$TAU_MODE" \
    MEMOIRS_Q1Q1_TAU_SCALE="$TAU_SCALE" \
    MEMOIRS_Q1Q1_TAU_C="$TAU_C" \
    MEMOIRS_Q1Q1_SUPG="$SUPG" \
    MEMOIRS_AMG_NUM_FUNCTIONS="$AMG_NUM_FUNCTIONS" \
    MEMOIRS_AMG_NODAL="$AMG_NODAL" \
    MEMOIRS_AMG_COARSEN=8 \
    MEMOIRS_AMG_INTERP="$AMG_INTERP" \
    MEMOIRS_AMG_RELAX=18 \
    MEMOIRS_AMG_AGG_LEVELS="$AMG_AGG_LEVELS" \
    MEMOIRS_AMG_KEEP_TRANSPOSE="$AMG_KEEP_TRANSPOSE" \
    MEMOIRS_AMG_PMAX="$AMG_PMAX" \
    MEMOIRS_AMG_TRUNC="$AMG_TRUNC" \
    MEMOIRS_AMG_STRONG_THRESHOLD="$AMG_STRONG" \
    MEMOIRS_GMRES_RESTART="$GMRES_RESTART" \
    MEMOIRS_Q1Q1_NU="$NU" \
    ./build_dp_hypre_q1q1/memoirs_test_structured_q1q1_nse_mms_hypre_ij \
      -polyMeshDir /home/jd/Desktop/meshes/unitcube/blockmesh/${N}cube/constant/polyMesh \
      -space cg_hex_q1 \
      -mms sin \
      -solver "$SOLVER" \
      -precond boomeramg \
      -hypreMemory "$HYPRE_MEMORY" \
      -tol "$LIN_TOL" \
      -maxit "$LIN_MAXIT" \
      -diagLevel 0 \
    2>&1 | tee "$LOG" | grep -E "amgSettings|q1q1PicardSummary|q1q1PicardConverged|FATAL|HYPRE error|Segmentation" || true

done
done
done
done

echo
echo "Sweep complete."
