#!/bin/bash
# ============================================================
# Poisson solver run script (sequential + OpenMP + MPI)
# Author: Hailin Liu
# This script executes all experiment stages step-by-step.
# Usage:
#   ./run.sh --stage 1 --stop_stage 8
# ============================================================

stage=1
stop_stage=3

# Parse optional arguments
for arg in "$@"; do
  case $arg in
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

# ============================================================
# Stage 1: Compile sequential version and run on (10x10), (20x20), (40x40)
# ============================================================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "========== Stage 1: Sequential version =========="

  echo "[Compile] Building sequential version..."
  g++ -o task_seq task.cpp -lm -std=c++11
  chmod u+x task_seq

  for M in 10 20 40; do
    N=$M
    echo "---- Grid: ${M}x${N} ----"
    ./task_seq $M $N
  done
fi

# ============================================================
# Stage 2: Compile OpenMP version and run on 1, 4, 16 threads (grid 40x40)
# ============================================================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "========== Stage 2: OpenMP parallel version =========="

  echo "[Compile] Building OpenMP version..."
  g++ -o task_openmp task.cpp -lm -std=c++11 -fopenmp
  chmod u+x task_openmp

  for THREADS in 1 4 16; do
    echo "---- Threads: ${THREADS}, Grid: 40x40 ----"
    export OMP_NUM_THREADS=$THREADS
    ./task_openmp 40 40
  done
fi

# ============================================================
# Stage 3: OpenMP version — performance test on large grids
# ============================================================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "========== Stage 3: OpenMP parallel version =========="

  for THREADS in 1 2 4 8 16; do
    echo "---- Threads: ${THREADS}, Grid: 400x600 ----"
    export OMP_NUM_THREADS=$THREADS
    ./task_openmp 400 600
  done

  for THREADS in 1 4 8 16 32; do
    echo "---- Threads: ${THREADS}, Grid: 800x1200 ----"
    export OMP_NUM_THREADS=$THREADS
    ./task_openmp 800 1200
  done
fi
