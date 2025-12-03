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
void update_u_kernel(double *u,
                     const double *p,
                     double alpha,
                     int M, int N);

__global__
void update_r_kernel(double *r,
                     const double *A_p,
                     double alpha,
                     int M, int N);

__global__
void update_p_kernel(double *p,
                     const double *z,
                     double beta,
                     int M, int N);

__global__
void apply_preconditioner_kernel(double *z,
                                 const double *r,
                                 const double *M_inv,
                                 int M, int N);

__global__
void reduce_r2_kernel(double *partial,
                      const double *r,
                      int M, int N,
                      int num_partials);

__global__
void reduce_p_Ap_kernel(double *partial,
                        const double *p,
                        const double *A_p,
                        int M, int N,
                        int num_partials);

__global__
void reduce_rz_kernel(double *partial,
                      const double *r,
                      const double *z,
                      int M, int N,
                      int num_partials);

#endif // CUDA_KERNELS_CUH
