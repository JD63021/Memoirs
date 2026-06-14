#!/usr/bin/env bash
set -u

cd /home/jd/memoirs_v0_modular_patch014

# ============================================================
# EDIT SETTINGS HERE ONLY
# ============================================================

N="${N:-32}"

SOLVER="${SOLVER:-"${SOLVER:-"${SOLVER:-bicgstab}"}"}"
LIN_TOL="${LIN_TOL:-1e-8}"
LIN_MAXIT="${LIN_MAXIT:-"${LIN_MAXIT:-"${LIN_MAXIT:-2000}"}"}"

PIC_TOL="${PIC_TOL:-1e-5}"
MAX_PICARD="${MAX_PICARD:-"${MAX_PICARD:-"${MAX_PICARD:-8}"}"}"

NU="${NU:-"${NU:-"${NU:-0.1}"}"}"
DT="${DT:-"${DT:-"${DT:-1e-3}"}"}"
ADV_SCALE="${ADV_SCALE:-"${ADV_SCALE:-"${ADV_SCALE:-1.0}"}"}"
PROBLEM="${PROBLEM:-mms}"
CAVITY_STEPS="${CAVITY_STEPS:-100}"
CAVITY_TIME_SCHEME="${CAVITY_TIME_SCHEME:-bdf1}"
CAVITY_VTU_EVERY="${CAVITY_VTU_EVERY:-10}"
CAVITY_LID_UX="${CAVITY_LID_UX:-1.0}"
CAVITY_LID_UY="${CAVITY_LID_UY:-0.0}"
CAVITY_LID_UZ="${CAVITY_LID_UZ:-0.0}"
CAVITY_VTU_DIR="${CAVITY_VTU_DIR:-vtu/q1q1_cavity_N${N}_dt${DT}}"
CAVITY_BC_DIAG="${CAVITY_BC_DIAG:-0}"

TAU_MODE="${TAU_MODE:-"${TAU_MODE:-"${TAU_MODE:-tezduyar}"}"}"
TAU_SCALE="${TAU_SCALE:-"${TAU_SCALE:-"${TAU_SCALE:-2.0}"}"}"
TAU_C="${TAU_C:-"${TAU_C:-"${TAU_C:-4.0}"}"}"
TAU_CT="${TAU_CT:-"${TAU_CT:-2.0}"}"
TAU_ADV_SCALE="${TAU_ADV_SCALE:-"${TAU_ADV_SCALE:-1.0}"}"
SUPG="${SUPG:-"${SUPG:-"${SUPG:-1}"}"}"
SUPG_TAU_SCALE="${SUPG_TAU_SCALE:-"${SUPG_TAU_SCALE:-1.0}"}"
GRAD_DIV="${GRAD_DIV:-0}"
GRAD_DIV_SCALE=0.0
GRAD_DIV_COEFF=0.0

HYPRE_MEMORY="${HYPRE_MEMORY:-device}"
FIXED_PATTERN="${FIXED_PATTERN:-1}"
SLOT_ASSEMBLY="${SLOT_ASSEMBLY:-1}"
FLAT_SLOT_ASSEMBLY="${FLAT_SLOT_ASSEMBLY:-1}"
CACHE_IJ_PATTERN="${CACHE_IJ_PATTERN:-1}"
DIRECT_IJ_VALUES="${DIRECT_IJ_VALUES:-1}"
CUDA_AUDIT_ASSEMBLY="${CUDA_AUDIT_ASSEMBLY:-0}"
CUDA_ASSEMBLY="${CUDA_ASSEMBLY:-0}"
CUDA_CACHE_BUFFERS="${CUDA_CACHE_BUFFERS:-1}"
CUDA_QP_KERNEL="${CUDA_QP_KERNEL:-1}"
COMPUTE_ERROR="${COMPUTE_ERROR:-1}"
DIAG_EVERY="${DIAG_EVERY:-1}"

AMG_NUM_FUNCTIONS="${AMG_NUM_FUNCTIONS:-"${AMG_NUM_FUNCTIONS:-"${AMG_NUM_FUNCTIONS:-4}"}"}"
AMG_NODAL_LEVELS="${AMG_NODAL_LEVELS:-"${AMG_NODAL_LEVELS:-0}"}"
AMG_NODAL="${AMG_NODAL:-0}"
AMG_COARSEN="${AMG_COARSEN:-"${AMG_COARSEN:-"${AMG_COARSEN:-8}"}"}"
AMG_INTERP="${AMG_INTERP:-"${AMG_INTERP:-"${AMG_INTERP:-15}"}"}"
AMG_RELAX="${AMG_RELAX:-"${AMG_RELAX:-"${AMG_RELAX:-18}"}"}"
AMG_AGG_LEVELS="${AMG_AGG_LEVELS:-"${AMG_AGG_LEVELS:-"${AMG_AGG_LEVELS:-0}"}"}"
AMG_KEEP_TRANSPOSE="${AMG_KEEP_TRANSPOSE:-"${AMG_KEEP_TRANSPOSE:-"${AMG_KEEP_TRANSPOSE:-1}"}"}"
AMG_PMAX="${AMG_PMAX:-"${AMG_PMAX:-"${AMG_PMAX:-4}"}"}"
AMG_TRUNC="${AMG_TRUNC:-"${AMG_TRUNC:-"${AMG_TRUNC:-0.2}"}"}"
AMG_STRONG="${AMG_STRONG:-"${AMG_STRONG:-"${AMG_STRONG:--1}"}"}"

GMRES_RESTART="${GMRES_RESTART:-"${GMRES_RESTART:-"${GMRES_RESTART:-80}"}"}"

# ============================================================

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/q1q1_N${N}_${SOLVER}_lin${LIN_TOL}_pic${PIC_TOL}_interp${AMG_INTERP}_strong${AMG_STRONG}_${STAMP}.log"
LATEST="logs/q1q1_latest.log"

echo "============================================================"
echo "Q1/Q1 NSE IJ/ParCSR RUN"
echo "============================================================"
echo "N                  = $N"
echo "SOLVER             = $SOLVER"
echo "LIN_TOL            = $LIN_TOL"
echo "LIN_MAXIT          = $LIN_MAXIT"
echo "PIC_TOL            = $PIC_TOL"
echo "MAX_PICARD         = $MAX_PICARD"
echo "NU                 = $NU"
echo "DT                 = $DT"
echo "ADV_SCALE          = $ADV_SCALE"
echo "PROBLEM            = $PROBLEM"
echo "CAVITY_STEPS       = $CAVITY_STEPS"
echo "CAVITY_TIME_SCHEME = $CAVITY_TIME_SCHEME"
echo "CAVITY_VTU_EVERY   = $CAVITY_VTU_EVERY"
echo "CAVITY_VTU_DIR     = $CAVITY_VTU_DIR"
echo "CAVITY_BC_DIAG    = $CAVITY_BC_DIAG"
echo "TAU_MODE           = $TAU_MODE"
echo "TAU_SCALE          = $TAU_SCALE"
echo "TAU_C              = $TAU_C"
echo "TAU_CT             = $TAU_CT"
echo "TAU_ADV_SCALE      = $TAU_ADV_SCALE"
echo "SUPG               = $SUPG"
echo "SUPG_TAU_SCALE    = $SUPG_TAU_SCALE"
echo "GRAD_DIV          = $GRAD_DIV"
echo "GRAD_DIV_SCALE    = $GRAD_DIV_SCALE"
echo "GRAD_DIV_COEFF    = $GRAD_DIV_COEFF"
echo "HYPRE_MEMORY       = $HYPRE_MEMORY"
echo "SLOT_ASSEMBLY      = $SLOT_ASSEMBLY"
echo "FLAT_SLOT_ASSEMBLY = $FLAT_SLOT_ASSEMBLY"
echo "CACHE_IJ_PATTERN  = $CACHE_IJ_PATTERN"
echo "DIRECT_IJ_VALUES  = $DIRECT_IJ_VALUES"
echo "CUDA_AUDIT_ASSEMBLY = $CUDA_AUDIT_ASSEMBLY"
echo "CUDA_ASSEMBLY       = $CUDA_ASSEMBLY"
echo "CUDA_CACHE_BUFFERS  = $CUDA_CACHE_BUFFERS"
echo "CUDA_QP_KERNEL      = $CUDA_QP_KERNEL"
echo "FIXED_PATTERN      = $FIXED_PATTERN  # cached graph inside assembler"
echo "COMPUTE_ERROR      = $COMPUTE_ERROR"
echo "DIAG_EVERY         = $DIAG_EVERY"
echo "AMG_NODAL          = $AMG_NODAL"
echo "AMG_KEEP_TRANSPOSE = $AMG_KEEP_TRANSPOSE"
echo "AMG_AGG_LEVELS     = $AMG_AGG_LEVELS"
echo "AMG_NODAL_LEVELS   = $AMG_NODAL_LEVELS"
echo "AMG_NUM_FUNCTIONS  = $AMG_NUM_FUNCTIONS"
echo "AMG_COARSEN        = $AMG_COARSEN"
echo "AMG_INTERP         = $AMG_INTERP"
echo "AMG_RELAX          = $AMG_RELAX"
echo "AMG_STRONG         = $AMG_STRONG"
echo "AMG_PMAX           = $AMG_PMAX"
echo "AMG_TRUNC          = $AMG_TRUNC"
echo "LOG                = $LOG"
echo "============================================================"
echo

stdbuf -oL -eL env \
  MEMOIRS_COMPUTE_ERROR=1 \
  MEMOIRS_Q1Q1_RHS_MODE=picard \
  MEMOIRS_Q1Q1_PROBLEM="$PROBLEM" \
  MEMOIRS_Q1Q1_CAVITY_STEPS="$CAVITY_STEPS" \
  MEMOIRS_Q1Q1_CAVITY_TIME_SCHEME="$CAVITY_TIME_SCHEME" \
  MEMOIRS_Q1Q1_CAVITY_VTU_EVERY="$CAVITY_VTU_EVERY" \
  MEMOIRS_Q1Q1_CAVITY_VTU_DIR="$CAVITY_VTU_DIR" \
  MEMOIRS_Q1Q1_LID_UX="$CAVITY_LID_UX" \
  MEMOIRS_Q1Q1_LID_UY="$CAVITY_LID_UY" \
  MEMOIRS_Q1Q1_LID_UZ="$CAVITY_LID_UZ" \
  MEMOIRS_Q1Q1_CAVITY_BC_DIAG="$CAVITY_BC_DIAG" \
  MEMOIRS_Q1Q1_MAX_PICARD="$MAX_PICARD" \
  MEMOIRS_Q1Q1_PICARD_TOL="$PIC_TOL" \
  MEMOIRS_Q1Q1_BETA_INITIAL=zero \
  MEMOIRS_Q1Q1_DT="$DT" \
  MEMOIRS_Q1Q1_ADV_SCALE="$ADV_SCALE" \
  MEMOIRS_Q1Q1_TAU_MODE="$TAU_MODE" \
  MEMOIRS_Q1Q1_TAU_SCALE="$TAU_SCALE" \
  MEMOIRS_Q1Q1_TAU_C="$TAU_C" \
  MEMOIRS_Q1Q1_TAU_CT="$TAU_CT" \
  MEMOIRS_Q1Q1_TAU_ADV_SCALE="$TAU_ADV_SCALE" \
  MEMOIRS_Q1Q1_SUPG="$SUPG" \
  MEMOIRS_Q1Q1_SUPG_TAU_SCALE="$SUPG_TAU_SCALE" \
  MEMOIRS_Q1Q1_GRAD_DIV="$GRAD_DIV" \
  MEMOIRS_Q1Q1_GRAD_DIV_SCALE="$GRAD_DIV_SCALE" \
  MEMOIRS_Q1Q1_GRAD_DIV_COEFF="$GRAD_DIV_COEFF" \
  MEMOIRS_AMG_NUM_FUNCTIONS="$AMG_NUM_FUNCTIONS" \
  MEMOIRS_AMG_NODAL="$AMG_NODAL" \
  MEMOIRS_AMG_NODAL_LEVELS="$AMG_NODAL_LEVELS" \
  MEMOIRS_AMG_COARSEN="$AMG_COARSEN" \
  MEMOIRS_AMG_INTERP="$AMG_INTERP" \
  MEMOIRS_AMG_RELAX="$AMG_RELAX" \
  MEMOIRS_AMG_AGG_LEVELS="$AMG_AGG_LEVELS" \
  MEMOIRS_AMG_KEEP_TRANSPOSE="$AMG_KEEP_TRANSPOSE" \
  MEMOIRS_AMG_PMAX="$AMG_PMAX" \
  MEMOIRS_AMG_TRUNC="$AMG_TRUNC" \
  MEMOIRS_AMG_STRONG="$AMG_STRONG" \
  MEMOIRS_GMRES_RESTART="$GMRES_RESTART" \
  MEMOIRS_Q1Q1_FIXED_PATTERN="$FIXED_PATTERN" \
  MEMOIRS_Q1Q1_SLOT_ASSEMBLY="$SLOT_ASSEMBLY" \
  MEMOIRS_Q1Q1_FLAT_SLOT_ASSEMBLY="$FLAT_SLOT_ASSEMBLY" \
  MEMOIRS_Q1Q1_CACHE_IJ_PATTERN="$CACHE_IJ_PATTERN" \
  MEMOIRS_Q1Q1_DIRECT_IJ_VALUES="$DIRECT_IJ_VALUES" \
  MEMOIRS_Q1Q1_CUDA_AUDIT_ASSEMBLY="$CUDA_AUDIT_ASSEMBLY" \
  MEMOIRS_Q1Q1_CUDA_ASSEMBLY="$CUDA_ASSEMBLY" \
  MEMOIRS_Q1Q1_CUDA_CACHE_BUFFERS="$CUDA_CACHE_BUFFERS" \
  MEMOIRS_Q1Q1_CUDA_QP_KERNEL="$CUDA_QP_KERNEL" \
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
  2>&1 | tee "$LOG"

STATUS=${PIPESTATUS[0]}
cp "$LOG" "$LATEST"

echo
echo "============================================================"
echo "RUN FINISHED"
echo "exit status = $STATUS"
echo "full log    = $LOG"
echo "latest log  = $LATEST"
echo "============================================================"

echo
echo "==== Picard summaries ===="
grep "q1q1PicardSummary" "$LOG" || true

echo
echo "==== Krylov iteration total ===="
awk '
/q1q1PicardSummary/ {
  for (i=1;i<=NF;i++) {
    if ($i=="linIters") {
      total += $(i+1)
      print "Picard solve linIters = " $(i+1) "    cumulative = " total
    }
  }
}
END {
  print "TOTAL_KRYLOV_ITERATIONS = " total
}
' "$LOG"

echo
echo "==== Last 40 lines ===="
tail -40 "$LOG"

exit "$STATUS"
