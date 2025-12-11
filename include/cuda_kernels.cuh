#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>

__device__ __host__
inline int idx(int i, int j, int Np1) {
    return i * Np1 + j;
}

__global__
void apply_A_kernel(const double * __restrict__ v,
                    const double * __restrict__ k,
                    double * __restrict__ out,
                    int M, int N,
                    double hx2, double hy2);

__global__
void update_u_kernel(double* __restrict__ u,
                     const double* __restrict__ p,
                     double alpha,
                     int M, int N);

__global__
void update_r_kernel(double* __restrict__ r,
                     const double* __restrict__ A_p,
                     double alpha,
                     int M, int N);

__global__
void update_p_kernel(double* __restrict__ p,
                     const double* __restrict__ z,
                     double beta,
                     int M, int N);

__global__
void apply_preconditioner_kernel(double* __restrict__ z, 
                                 const double* __restrict__ r,
                                 const double* __restrict__ M_inv, 
                                 int M, int N);

__global__
void compute_r2_partials_kernel(double* __restrict__ d_partial,
                                const double* __restrict__ r,
                                int total);

__global__
void compute_rz_partials_kernel(double* __restrict__ d_partial,
                                const double* __restrict__ r,
                                const double* __restrict__ z,
                                int total);
                        
__global__
void compute_p_Ap_partials_kernel(double* __restrict__ d_partial,
                                  const double* __restrict__ p,
                                  const double* __restrict__ Ap,
                                  int total);

#endif // CUDA_KERNELS_CUH
