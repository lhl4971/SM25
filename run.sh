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
echo "[INFO] Cleaning previous outputs..."
rm -f debug.log
rm -rf solution/*

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
  echo "========== Stage 2: OpenMP scalability test =========="

  echo "[Compile] Building OpenMP version..."
  g++ -o task_omp task.cpp -lm -std=c++11 -fopenmp
  chmod u+x task_omp

  for THREADS in 1 4 16; do
    echo "---- Threads: ${THREADS}, Grid: 40x40 ----"
    export OMP_NUM_THREADS=$THREADS
    ./task_omp 40 40
  done
fi

# ============================================================
# Stage 3: OpenMP version — performance test on large grids
# ============================================================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "======== Stage 3: OpenMP hybrid parallel test ========"

  for THREADS in 1 2 4 8 16; do
    echo "---- Threads: ${THREADS}, Grid: 400x600 ----"
    export OMP_NUM_THREADS=$THREADS
    ./task_omp 400 600
  done

  for THREADS in 1 4 8 16; do
    echo "---- Threads: ${THREADS}, Grid: 800x1200 ----"
    export OMP_NUM_THREADS=$THREADS
    ./task_omp 800 1200
  done
fi

# ============================================================
# Stage 5: Compile MPI version and run on 1, 4, 16 cores (grid 40x40)
# ============================================================
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "=========== Stage 5: MPI scalability test ============"

  echo "[Compile] Building MPI version..."
  mpicxx -o task_mpi task_MPI.cpp -lm -std=c++11
  chmod u+x task_mpi

  for PROCS in 1 2 4; do
    echo "---- MPI Processes: ${PROCS}, Grid: 40x40 ----"
    mpirun -np ${PROCS} ./task_mpi 40 40
  done
fi

# ============================================================
# Stage 6: MPI version — performance test on large grids
# ============================================================
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "========= Stage 6: MPI hybrid parallel test =========="

  for PROCS in 1 2 4 8 16; do
    echo "---- MPI Processes: ${PROCS}, Grid: 400x600 ----"
    mpirun -np ${PROCS} ./task_mpi 400 600
  done

  for PROCS in 1 4 8 16; do
    echo "---- MPI Processes: ${PROCS}, Grid: 800x1200 ----"
    mpirun -np ${PROCS} ./task_mpi 800 1200
  done

  # The HPC server only contains 19 physical cores, allowing MPI 
  # to allocate logical cores when executing 32-process tasks.
  for PROCS in 32; do
    echo "---- MPI Processes: ${PROCS}, Grid: 800x1200 ----"
    mpirun --use-hwthread-cpus -np ${PROCS} ./task_mpi 800 1200
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

  export OMP_NUM_THREADS=4
  for PROCS in 1 2; do
    echo "[Hybrid] MPI procs = ${PROCS}, OMP threads = ${OMP_NUM_THREADS}, Grid 40x40"
    mpirun -np ${PROCS} ./task_mpi_omp 40 40
  done
fi

# ============================================================
# Stage 8: MPI+OpenMP version — performance test on large grids
# ============================================================
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "====== Stage 8: MPI+OpenMP hybrid parallel test ======"

  for THREADS in 1 2 4 8; do
    echo "---- MPI Processes: 2, OMP Threads: ${THREADS}, Grid: 400x600 ----"
    export OMP_NUM_THREADS=${THREADS}
    mpirun -np 2 ./task_mpi_omp 400 600
  done

  for THREADS in 1 2 4 8; do
    echo "---- MPI Processes: 4, OMP Threads: ${THREADS}, Grid: 800x1200 ----"
    export OMP_NUM_THREADS=${THREADS}
    mpirun -np 4 ./task_mpi_omp 800 1200
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
