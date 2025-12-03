#ifndef CUDA_OPERATORS_HPP
#define CUDA_OPERATORS_HPP

#include <cuda_runtime.h>
#include "cuda_kernels.cuh"

#ifndef CUDA_BLOCK_X
#define CUDA_BLOCK_X 16
#endif

#ifndef CUDA_BLOCK_Y
#define CUDA_BLOCK_Y 16
#endif

#ifndef NUM_PARTIALS
#define NUM_PARTIALS 1024
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK (CUDA_BLOCK_X * CUDA_BLOCK_Y)
#endif

void cuda_check_error(const char *msg);

void cuda_apply_A(const double *d_v,
                  const double *d_k,
                  double *d_out,
                  int M, int N,
                  double hx2, double hy2,
                  cudaStream_t stream = 0);

void cuda_update_u(double *d_u,
                   const double *d_p,
                   double alpha,
                   int M, int N,
                   cudaStream_t stream = 0);

void cuda_update_r(double *d_r,
                   const double *d_Ap,
                   double alpha,
                   int M, int N,
                   cudaStream_t stream = 0);

void cuda_update_p(double *d_p,
                   const double *d_z,
                   double beta,
                   int M, int N,
                   cudaStream_t stream = 0);

void cuda_apply_preconditioner(double *d_z,
                               const double *d_r,
                               const double *d_M_inv,
                               int M, int N,
                               cudaStream_t stream = 0);

void cuda_reduce_r2(double *d_partial,
                    const double *d_r,
                    int M, int N,
                    cudaStream_t stream = 0);

void cuda_reduce_p_Ap(double *d_partial,
                      const double *d_p,
                      const double *d_Ap,
                      int M, int N,
                      cudaStream_t stream = 0);

void cuda_reduce_rz(double *d_partial,
                    const double *d_r,
                    const double *d_z,
                    int M, int N,
                    cudaStream_t stream = 0);

#endif // CUDA_OPERATORS_HPP