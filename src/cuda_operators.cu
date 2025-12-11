#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include "cuda_operators.hpp"
#include "cuda_kernels.cuh"

void cuda_check_error(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "[CUDA ERROR] " << msg
                  << " : " << cudaGetErrorString(err)
                  << std::endl;
        std::exit(1);
    }
}

void cuda_apply_A(const double *d_v,
                  const double *d_k,
                  double *d_out,
                  int M, int N,
                  double hx2, double hy2,
                  cudaStream_t stream)
{
    dim3 block(CUDA_BLOCK_X, CUDA_BLOCK_Y);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    apply_A_kernel<<<grid, block, 0, stream>>>(d_v, d_k, d_out, M, N, hx2, hy2);
    cuda_check_error("apply_A_kernel");
}

void cuda_update_u(double *d_u,
                   const double *d_p,
                   double alpha,
                   int M, int N,
                   cudaStream_t stream)
{
    dim3 block(CUDA_BLOCK_X, CUDA_BLOCK_Y);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    update_u_kernel<<<grid, block, 0, stream>>>(d_u, d_p, alpha, M, N);
    cuda_check_error("update_u_kernel");
}

void cuda_update_r(double *d_r,
                   const double *d_Ap,
                   double alpha,
                   int M, int N,
                   cudaStream_t stream)
{
    dim3 block(CUDA_BLOCK_X, CUDA_BLOCK_Y);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    update_r_kernel<<<grid, block, 0, stream>>>(d_r, d_Ap, alpha, M, N);
    cuda_check_error("update_r_kernel");
}

void cuda_update_p(double *d_p,
                   const double *d_z,
                   double beta,
                   int M, int N,
                   cudaStream_t stream)
{
    dim3 block(CUDA_BLOCK_X, CUDA_BLOCK_Y);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    update_p_kernel<<<grid, block, 0, stream>>>(d_p, d_z, beta, M, N);
    cuda_check_error("update_p_kernel");
}

void cuda_apply_preconditioner(double *d_z,
                               const double *d_r,
                               const double *d_M_inv,
                               int M, int N,
                               cudaStream_t stream)
{
    dim3 block(CUDA_BLOCK_X, CUDA_BLOCK_Y);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    apply_preconditioner_kernel<<<grid, block, 0, stream>>>(d_z, d_r, d_M_inv, M, N);
    cuda_check_error("apply_preconditioner_kernel");
}

void cuda_reduce_r2_partials(
        double *d_partial,
        const double *d_r,
        int M, int N,
        cudaStream_t stream)
{
    int total = (M + 1) * (N + 1);

    dim3 block(DOT_THREADS, 1, 1);
    dim3 grid(DOT_BLOCKS, 1, 1);

    compute_r2_partials_kernel<<<grid, block, 0, stream>>>(d_partial, d_r, total);
    cuda_check_error("compute_r2_partials_kernel");
}

void cuda_reduce_p_Ap_partials(
        double *d_partial,
        const double *d_p,
        const double *d_Ap,
        int M, int N,
        cudaStream_t stream)
{
    int total = (M + 1) * (N + 1);

    dim3 block(DOT_THREADS, 1, 1);
    dim3 grid(DOT_BLOCKS, 1, 1);

    compute_p_Ap_partials_kernel<<<grid, block, 0, stream>>>(d_partial, d_p, d_Ap, total);
    cuda_check_error("compute_pAp_partials_kernel");
}

void cuda_reduce_rz_partials(
        double *d_partial,
        const double *d_r,
        const double *d_z,
        int M, int N,
        cudaStream_t stream)
{
    int total = (M + 1) * (N + 1);

    dim3 block(DOT_THREADS, 1, 1);
    dim3 grid(DOT_BLOCKS, 1, 1);

    compute_rz_partials_kernel<<<grid, block, 0, stream>>>(d_partial, d_r, d_z, total);
    cuda_check_error("compute_rz_partials_kernel");
}

double cuda_reduce_sum(double* d_partial, int n, cudaStream_t stream)
{
    thrust::device_ptr<double> begin(d_partial);
    thrust::device_ptr<double> end = begin + n;

    double init = 0.0;
    double result = thrust::reduce(
        thrust::cuda::par.on(stream),
        begin, end,
        init,
        thrust::plus<double>()
    );

    return result;
}