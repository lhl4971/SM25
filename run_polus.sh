#!/bin/bash
# ============================================================
# Poisson solver run script (sequential + OpenMP + MPI)
# Author: Hailin Liu
# This script executes all experiment stages step-by-step.
# Usage:
#   ./run.sh --stage 1 --stop_stage 8
# ============================================================

# ============================================================
# Environment bootstrap (MPI setup)
# ============================================================
echo "[INFO] Purging all previously loaded modules..."
module purge
echo "[INFO] Loading OpenMPI/4.0.0 module..."
module load OpenMPI/4.0.0
ulimit -s 10240

stage=1
stop_stage=8

# Parse optional arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      stage="$2"
      shift 2
      ;;
    --stop_stage)
      stop_stage="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

mkdir -p solution
mkdir -p log
echo "[INFO] Cleaning previous outputs..."
rm -f debug.log
rm -rf solution/*
rm -rf log/*

# ============================================================
# Stage 1: Compile sequential version and run on (10x10), (20x20), (40x40)
# ============================================================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "========== Stage 1: Sequential version =========="

  echo "[Compile] Building sequential version..."
  g++ -o task_seq task.cpp -lm -std=c++11
  chmod u+x task_seq

  mkdir -p log/stage_1

  for M in 10 20 40; do
    N=$M
    echo "---- Submitting job for Grid: ${M}x${N} ----"

    JOB_NAME="seq_${M}x${N}"
    OUT_FILE="log/stage_1/${JOB_NAME}_cout.log"
    ERR_FILE="log/stage_1/${JOB_NAME}_cerr.log"

    JOB_ID=$(bsub -J "${JOB_NAME}" -n 1 -W 1 \
            -oo "${OUT_FILE}" -eo "${ERR_FILE}" \
            ./task_seq ${M} ${N} | awk '{print $2}' | tr -d '<>')

    echo "[INFO] Job ${JOB_ID} submitted, waiting for it to complete..."
    bwait -w "ended(${JOB_ID})"
    echo "[INFO] Job ${JOB_ID} finished."
  done
fi

# ============================================================
# Stage 2: Compile OpenMP version and run on 1, 4, 16 threads (grid 40x40)
# ============================================================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "========== Stage 2: OpenMP scalability test =========="

  echo "[Compile] Building OpenMP version..."
  g++ -o task_omp task.cpp -lm -std=c++11 -fopenmp
  chmod u+x task_omp

  mkdir -p log/stage_2

  for THREADS in 1 4 16; do
    echo "---- Submitting job: Threads=${THREADS}, Grid=40x40 ----"

    JOB_NAME="omp_40_40_${THREADS}"
    OUT_FILE="log/stage_2/${JOB_NAME}_cout.log"
    ERR_FILE="log/stage_2/${JOB_NAME}_cerr.log"

    JOB_ID=$(bsub -J "${JOB_NAME}" -W 1 \
         -oo "${OUT_FILE}" -eo "${ERR_FILE}" \
         -R "affinity[core(${THREADS})]" \
         OMP_NUM_THREADS=${THREADS} ./task_omp 40 40 | awk '{print $2}' | tr -d '<>')
    
    echo "[INFO] Job ${JOB_ID} submitted, waiting for it to complete..."
    bwait -w "ended(${JOB_ID})"
    echo "[INFO] Job ${JOB_ID} finished."
  done
fi

# ============================================================
# Stage 3: OpenMP version — performance test on large grids
# ============================================================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "======== Stage 3: OpenMP hybrid parallel test ========"

  mkdir -p log/stage_3

  # ---------- 400x600 ----------
  for THREADS in 1 2 4 8 16; do
    echo "---- Submitting job: Threads=${THREADS}, Grid=400x600 ----"

    JOB_NAME="omp_400_600_${THREADS}"
    OUT_FILE="log/stage_3/${JOB_NAME}_cout.log"
    ERR_FILE="log/stage_3/${JOB_NAME}_cerr.log"

    JOB_ID=$(bsub -J "${JOB_NAME}" -W 10 \
         -oo "${OUT_FILE}" -eo "${ERR_FILE}" \
         -R "affinity[core(${THREADS})]" \
         OMP_NUM_THREADS=${THREADS} ./task_omp 400 600 | awk '{print $2}' | tr -d '<>')

    echo "[INFO] Job ${JOB_ID} submitted, waiting for it to complete..."
    bwait -w "ended(${JOB_ID})"
    echo "[INFO] Job ${JOB_ID} finished."
  done

  # ---------- 800x1200 ----------
  for THREADS in 1 4 8 16; do
    echo "---- Submitting job: Threads=${THREADS}, Grid=800x1200 ----"

    JOB_NAME="omp_800_1200_${THREADS}"
    OUT_FILE="log/stage_3/${JOB_NAME}_cout.log"
    ERR_FILE="log/stage_3/${JOB_NAME}_cerr.log"

    JOB_ID=$(bsub -J "${JOB_NAME}" -W 30 \
         -oo "${OUT_FILE}" -eo "${ERR_FILE}" \
         -R "affinity[core(${THREADS})]" \
         OMP_NUM_THREADS=${THREADS} ./task_omp 800 1200 | awk '{print $2}' | tr -d '<>')

    echo "[INFO] Job ${JOB_ID} submitted, waiting for it to complete..."
    bwait -w "ended(${JOB_ID})"
    echo "[INFO] Job ${JOB_ID} finished."
  done

  # --- 800x1200 (32 threads) ----
  echo "---- Submitting job: Threads=32, Grid=800x1200 ----"

  JOB_NAME="omp_800_1200_32"
  OUT_FILE="log/stage_3/${JOB_NAME}_cout.log"
  ERR_FILE="log/stage_3/${JOB_NAME}_cerr.log"

  JOB_ID=$(bsub -J "${JOB_NAME}" -W 30 \
       -oo "${OUT_FILE}" -eo "${ERR_FILE}" \
       -R "affinity[core(16)]" \
       OMP_NUM_THREADS=32 /polusfs/lsf/openmp/launchOpenMP.py ./task_omp 800 1200 | awk '{print $2}' | tr -d '<>')

  echo "[INFO] Job ${JOB_ID} submitted, waiting for it to complete..."
  bwait -w "ended(${JOB_ID})"
  echo "[INFO] Job ${JOB_ID} finished."
fi

# ============================================================
# Stage 5: Compile MPI version and run on 1, 4, 16 cores (grid 40x40)
# ============================================================
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "=========== Stage 5: MPI scalability test ============"

  echo "[Compile] Building MPI version..."
  mpicxx -o task_mpi task_MPI.cpp -lm -std=c++11
  chmod u+x task_mpi

  mkdir -p log/stage_5

  for PROCS in 1 2 4; do
    echo "---- Submitting job: MPI Processes=${PROCS}, Grid=40x40 ----"

    JOB_NAME="mpi_40_40_${PROCS}"
    OUT_FILE="log/stage_5/${JOB_NAME}_cout.log"
    ERR_FILE="log/stage_5/${JOB_NAME}_cerr.log"

    JOB_ID=$(bsub -J "${JOB_NAME}" -n ${PROCS} -W 1 \
         -oo "${OUT_FILE}" -eo "${ERR_FILE}" \
         mpiexec ./task_mpi 40 40 | awk '{print $2}' | tr -d '<>')

    echo "[INFO] Job ${JOB_ID} submitted, waiting for it to complete..."
    bwait -w "ended(${JOB_ID})"
    echo "[INFO] Job ${JOB_ID} finished."
  done
fi

# ============================================================
# Stage 6: MPI version — performance test on large grids
# ============================================================
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "========= Stage 6: MPI hybrid parallel test =========="

  mkdir -p log/stage_6

  # ---------- 400x600 ----------
  for PROCS in 1 2 4 8 16; do
    echo "---- Submitting job: MPI Processes=${PROCS}, Grid=400x600 ----"

    JOB_NAME="mpi_400_600_${PROCS}"
    OUT_FILE="log/stage_6/${JOB_NAME}_cout.log"
    ERR_FILE="log/stage_6/${JOB_NAME}_cerr.log"

    JOB_ID=$(bsub -J "${JOB_NAME}" -n ${PROCS} -W 10 \
         -oo "${OUT_FILE}" -eo "${ERR_FILE}" \
         mpiexec ./task_mpi 400 600 | awk '{print $2}' | tr -d '<>')
    
    echo "[INFO] Job ${JOB_ID} submitted, waiting for it to complete..."
    bwait -w "ended(${JOB_ID})"
    echo "[INFO] Job ${JOB_ID} finished."
  done

  # ---------- 800x1200 ----------
  for PROCS in 1 4 8 16 32; do
    echo "---- Submitting job: MPI Processes=${PROCS}, Grid=800x1200 ----"

    JOB_NAME="mpi_800_1200_${PROCS}"
    OUT_FILE="log/stage_6/${JOB_NAME}_cout.log"
    ERR_FILE="log/stage_6/${JOB_NAME}_cerr.log"

    JOB_ID=$(bsub -J "${JOB_NAME}" -n ${PROCS} -W 30 \
         -oo "${OUT_FILE}" -eo "${ERR_FILE}" \
         mpiexec ./task_mpi 800 1200 | awk '{print $2}' | tr -d '<>')

    echo "[INFO] Job ${JOB_ID} submitted, waiting for it to complete..."
    bwait -w "ended(${JOB_ID})"
    echo "[INFO] Job ${JOB_ID} finished."
  done
fi

# ============================================================
# Stage 7: Compile MPI+OpenMP version and run on 1, 2 cores with 4 threads (grid 40x40)
# ============================================================
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "======== Stage 7: MPI+OpenMP scalability test ========"

  echo "[Compile] Building MPI+OpenMP hybrid version..."
  mpicxx -o task_mpi_omp task_MPI.cpp -lm -std=c++11 -fopenmp
  chmod u+x task_mpi_omp

  mkdir -p log/stage_7

  for PROCS in 1 2; do
    JOB_NAME="mpi_omp_40_40_${PROCS}_4"
    LSF_FILE="config/stage_7/${JOB_NAME}.lsf"

    echo "---- Submitting hybrid job via ${LSF_FILE} ----"

    JOB_ID=$(bsub < "${LSF_FILE}" | awk '{print $2}' | tr -d '<>')
    echo "[INFO] Job ${JOB_ID} submitted, waiting for it to complete..."
    bwait -w "ended(${JOB_ID})"
    echo "[INFO] Job ${JOB_ID} finished."
  done
fi

# ============================================================
# Stage 8: MPI+OpenMP version — performance test on large grids
# ============================================================
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "====== Stage 8: MPI+OpenMP hybrid parallel test ======"

  mkdir -p log/stage_8

  # ---------- 400x600: 2 MPI процессов, 1/2/4/8 нитей ----------
  for THREADS in 1 2 4 8; do
    JOB_NAME="mpi_omp_400_600_2_${THREADS}"
    LSF_FILE="config/stage_8/${JOB_NAME}.lsf"

    echo "---- Submitting job via ${LSF_FILE} ----"

    JOB_ID=$(bsub < "${LSF_FILE}" | awk '{print $2}' | tr -d '<>')
    echo "[INFO] Job ${JOB_ID} submitted, waiting for it to complete..."
    bwait -w "ended(${JOB_ID})"
    echo "[INFO] Job ${JOB_ID} finished."
  done

  # ---------- 800x1200: 4 MPI процессов, 1/2/4/8 нитей ----------
  for THREADS in 1 2 4 8; do
    JOB_NAME="mpi_omp_800_1200_4_${THREADS}"
    LSF_FILE="config/stage_8/${JOB_NAME}.lsf"

    echo "---- Submitting job via ${LSF_FILE} ----"

    JOB_ID=$(bsub < "${LSF_FILE}" | awk '{print $2}' | tr -d '<>')
    echo "[INFO] Job ${JOB_ID} submitted, waiting for it to complete..."
    bwait -w "ended(${JOB_ID})"
    echo "[INFO] Job ${JOB_ID} finished."
  done
fi

# ============================================================
# Cleanup
# ============================================================
echo "======================================================"
echo "[INFO] All experiment stages completed successfully."
echo "[INFO] Unloading OpenMPI/4.0.0 module and cleaning environment..."
module unload OpenMPI/4.0.0
echo "[INFO] Done."
echo "======================================================"

