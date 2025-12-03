#include <cuda_runtime.h>
#include "cuda_kernels.cuh"

__global__
void apply_A_kernel(const double * __restrict__ v,
                    const double * __restrict__ k,
                    double * __restrict__ out,
                    int M, int N, double hx2, double hy2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < 1 || i >= M) return;
    if (j < 1 || j >= N) return;

    int id_c = idx(i, j, N+1);
    int id_l = idx(i-1, j, N+1);
    int id_r = idx(i+1, j, N+1);
    int id_d = idx(i, j-1, N+1);
    int id_u = idx(i, j+1, N+1);

    double k_c  = k[id_c];
    double k_l  = k[id_l];
    double k_r  = k[id_r];
    double k_d  = k[id_d];
    double k_u  = k[id_u];

    double kx_plus  = 0.5 * (k_c + k_r);
    double kx_minus = 0.5 * (k_c + k_l);
    double ky_plus  = 0.5 * (k_c + k_u);
    double ky_minus = 0.5 * (k_c + k_d);

    double v_c = v[id_c];
    double v_l = v[id_l];
    double v_r = v[id_r];
    double v_d = v[id_d];
    double v_u = v[id_u];

    out[id_c] = -1.0 * (
        (kx_plus  * (v_r - v_c) - kx_minus * (v_c - v_l)) / hx2 +
        (ky_plus  * (v_u - v_c) - ky_minus * (v_c - v_d)) / hy2
    );
}

__global__
void update_u_kernel(double *u, const double *p, double alpha, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < 1 || i >= M) return;
    if (j < 1 || j >= N) return;

    int id = idx(i, j, N + 1);
    u[id] += alpha * p[id];
}

__global__
void update_r_kernel(double *r, const double *A_p, double alpha, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < 1 || i >= M) return;
    if (j < 1 || j >= N) return;
    
    int id = idx(i, j, N + 1);
    r[id] -= alpha * A_p[id];
}

__global__
void update_p_kernel(double *p, const double *z, double beta, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < 1 || i >= M) return;
    if (j < 1 || j >= N) return;

    int id = idx(i, j, N + 1);
    p[id] = z[id] + beta * p[id];
}

__global__
void apply_preconditioner_kernel(double *z, const double *r, const double *M_inv,  int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < 1 || i >= M) return;
    if (j < 1 || j >= N) return;

    int id = idx(i, j, N + 1);
    z[id] = M_inv[id] * r[id];
}

__global__
void reduce_r2_kernel(double *partial, const double *r, int M, int N, int num_partials) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_partials) return;

    int total = (M + 1) * (N + 1);
    double sum = 0.0;

    for (int lin = tid; lin < total; lin += num_partials) {
        int i = lin / (N + 1);
        int j = lin % (N + 1);

        if (i < 1 || i >= M) continue;
        if (j < 1 || j >= N) continue;

        sum += r[lin] * r[lin];
    }
    partial[tid] = sum;
}

__global__
void reduce_p_Ap_kernel(double *partial, const double *p, const double *A_p, int M, int N, int num_partials) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_partials) return;

    int total = (M + 1) * (N + 1);
    double sum = 0.0;

    for (int lin = tid; lin < total; lin += num_partials) {
        int i = lin / (N + 1);
        int j = lin % (N + 1);

        if (i < 1 || i >= M) continue;
        if (j < 1 || j >= N) continue;

        sum += p[lin] * A_p[lin];
    }
    partial[tid] = sum;
}

__global__
void reduce_rz_kernel(double *partial, const double *r, const double *z, int M, int N, int num_partials) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_partials) return;

    int total = (M + 1) * (N + 1);
    double sum = 0.0;

    for (int lin = tid; lin < total; lin += num_partials) {
        int i = lin / (N + 1);
        int j = lin % (N + 1);

        if (i < 1 || i >= M) continue;
        if (j < 1 || j >= N) continue;

        sum += r[lin] * z[lin];
    }
    partial[tid] = sum;
}
