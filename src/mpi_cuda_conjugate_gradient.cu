#include <vector>
#include <cmath>
#include <mpi.h>
#include <iostream>
#include <fstream>

#include <cuda_runtime.h>
#include "mpi_cuda_conjugate_gradient.hpp"
#include "cuda_operators.hpp"

MPICudaPoissonSolver::MPICudaPoissonSolver(
        int M_, int N_,
        double x_min, double x_max,
        double y_min, double y_max,
        std::function<bool(double, double)> region_func,
        std::function<double(double, double)> f_func,
        int world_rank,
        MPI_Comm cart_comm_
    )
    : MPIPoissonSolver(M_, N_, x_min, x_max, y_min, y_max,
                       region_func, f_func,
                       world_rank, cart_comm_)
{
    size = (M + 1) * (N + 1);
    h_buf.resize(size);
    cudaStreamCreate(&stream);

    cudaMalloc(&d_u,    size * sizeof(double));
    cudaMalloc(&d_r,    size * sizeof(double));
    cudaMalloc(&d_z,    size * sizeof(double));
    cudaMalloc(&d_k,    size * sizeof(double));
    cudaMalloc(&d_p,    size * sizeof(double));
    cudaMalloc(&d_A_p,  size * sizeof(double));
    cudaMalloc(&d_M_inv,size * sizeof(double));
    cudaMalloc(&d_buf,  size * sizeof(double));
    cuda_check_error("cudaMalloc in MPICudaPoissonSolver ctor");
}

void MPICudaPoissonSolver::cuda_exchange_halo(double *d_field)
{
    MPI_Request reqs[8];
    int rq = 0;

    const int Ny = N + 1;
    const size_t row_bytes = Ny * sizeof(double);
    const size_t col_bytes = sizeof(double);
    const size_t pitch     = row_bytes;

    cudaStreamSynchronize(stream);

    if (left_rank != MPI_PROC_NULL) {
        cudaMemcpy(send_left.data(), d_field + idx(1, 0, Ny), row_bytes, cudaMemcpyDeviceToHost);
    }

    if (right_rank != MPI_PROC_NULL) {
        cudaMemcpy(send_right.data(), d_field + idx(M-1, 0, Ny), row_bytes, cudaMemcpyDeviceToHost);
    }

    if (bottom_rank != MPI_PROC_NULL) {
        cudaMemcpy2D(send_bottom.data(), col_bytes, d_field + idx(0, 1, Ny), pitch, col_bytes, M + 1, cudaMemcpyDeviceToHost);
    }

    if (top_rank != MPI_PROC_NULL) {
        cudaMemcpy2D(send_top.data(), col_bytes, d_field + idx(0, N-1, Ny), pitch, col_bytes, M + 1, cudaMemcpyDeviceToHost);
    }

    cuda_check_error("cudaMemcpyDeviceToHost in cuda_exchange_halo");

    if (left_rank != MPI_PROC_NULL) {
        MPI_Irecv(recv_left.data(), Ny, MPI_DOUBLE, left_rank, 0, cart_comm, &reqs[rq++]);
        MPI_Isend(send_left.data(), Ny, MPI_DOUBLE, left_rank, 1, cart_comm, &reqs[rq++]);
    }

    if (right_rank != MPI_PROC_NULL) {
        MPI_Irecv(recv_right.data(), Ny, MPI_DOUBLE, right_rank, 1, cart_comm, &reqs[rq++]);
        MPI_Isend(send_right.data(), Ny, MPI_DOUBLE, right_rank, 0, cart_comm, &reqs[rq++]);
    }

    if (bottom_rank != MPI_PROC_NULL) {
        MPI_Irecv(recv_bottom.data(), M+1, MPI_DOUBLE, bottom_rank, 2, cart_comm, &reqs[rq++]);
        MPI_Isend(send_bottom.data(), M+1, MPI_DOUBLE, bottom_rank, 3, cart_comm, &reqs[rq++]);
    }

    if (top_rank != MPI_PROC_NULL) {
        MPI_Irecv(recv_top.data(), M+1, MPI_DOUBLE, top_rank, 3, cart_comm, &reqs[rq++]);
        MPI_Isend(send_top.data(), M+1, MPI_DOUBLE, top_rank, 2, cart_comm, &reqs[rq++]);
    }

    MPI_Waitall(rq, reqs, MPI_STATUSES_IGNORE);

    if (left_rank != MPI_PROC_NULL) {
        cudaMemcpy(d_field + idx(0, 0, Ny), recv_left.data(), row_bytes, cudaMemcpyHostToDevice);
    }

    if (right_rank != MPI_PROC_NULL) {
        cudaMemcpy(d_field + idx(M, 0, Ny), recv_right.data(), row_bytes, cudaMemcpyHostToDevice);
    }

    if (bottom_rank != MPI_PROC_NULL) {
        cudaMemcpy2D(d_field + idx(0, 0, Ny), pitch, recv_bottom.data(), col_bytes, col_bytes, M + 1, cudaMemcpyHostToDevice);
    }

    if (top_rank != MPI_PROC_NULL) {
        cudaMemcpy2D(d_field + idx(0, N, Ny), pitch, recv_top.data(), col_bytes, col_bytes, M + 1, cudaMemcpyHostToDevice);
    }

    cuda_check_error("cudaMemcpyHostToDevice in cuda_exchange_halo");
}

double MPICudaPoissonSolver::compute_l2_norm() {
    cuda_compute_r2(d_buf, d_r, M, N, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpy(h_buf.data(), d_buf, size * sizeof(double), cudaMemcpyDeviceToHost);
    cuda_check_error("compute_l2_norm memcpy");

    double local_sum = 0.0;
    for (int i = 1; i < M; ++i)
        for (int j = 1; j < N; ++j)
            local_sum += h_buf[i * (N + 1) + j];

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, cart_comm);

    return std::sqrt(global_sum);
}

void MPICudaPoissonSolver::initialize_p() {
    cudaMemcpy(d_p, d_z, size * sizeof(double), cudaMemcpyDeviceToDevice);
}

double MPICudaPoissonSolver::compute_rz() {
    cuda_compute_rz(d_buf, d_r, d_z, M, N, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpy(h_buf.data(), d_buf, size * sizeof(double), cudaMemcpyDeviceToHost);
    cuda_check_error("compute_rz memcpy");

    double local_sum = 0.0;
    for (int i = 1; i < M; ++i)
        for (int j = 1; j < N; ++j)
            local_sum += h_buf[i * (N + 1) + j];

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, cart_comm);

    return global_sum;
}

double MPICudaPoissonSolver::compute_p_Ap() {
    cuda_compute_p_Ap(d_buf, d_p, d_A_p, M, N, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpy(h_buf.data(), d_buf, size * sizeof(double), cudaMemcpyDeviceToHost);
    cuda_check_error("compute_p_Ap memcpy");

    double local_sum = 0.0;
    for (int i = 1; i < M; ++i)
        for (int j = 1; j < N; ++j)
            local_sum += h_buf[i * (N + 1) + j];

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, cart_comm);

    return global_sum;
}

void MPICudaPoissonSolver::update_u(double alpha) {
    cuda_update_u(d_u, d_p, alpha, M, N, stream);
}

void MPICudaPoissonSolver::update_r(double alpha) {
    cuda_update_r(d_r, d_A_p, alpha, M, N, stream);
}

void MPICudaPoissonSolver::update_p(double beta) {
    cuda_update_p(d_p, d_z, beta, M, N, stream);
}

void MPICudaPoissonSolver::apply_preconditioner() {
    cuda_apply_preconditioner(d_z, d_r, d_M_inv, M, N, stream);
}

void MPICudaPoissonSolver::solve(int max_iter, double tolerance) {
    initialize_f();

    initialize_k();
    exchange_halo(k);
    to_device(k, d_k);

    initialize_M_inv();
    to_device(M_inv, d_M_inv);

    exchange_halo(u);
    to_device(u, d_u);

    initialize_r();
    to_device(r, d_r);

    apply_preconditioner();
    initialize_p();

    double rz_prev = compute_rz();
    double r_norm  = compute_l2_norm();

    int iter;
    for (iter = 0; r_norm > tolerance && iter < max_iter; ++iter) {
        cuda_exchange_halo(d_p);

        cuda_apply_A(d_p, d_k, d_A_p, M, N, hx2, hy2, stream);
        cudaStreamSynchronize(stream);

        double p_Ap = compute_p_Ap();
        double alpha = rz_prev / p_Ap;
        update_u(alpha);
        update_r(alpha);

        r_norm = compute_l2_norm();
        if (rank == 0 && iter % 1000 == 0) {
            std::ofstream debug_log("debug.log", std::ios::app);
            debug_log << "Iter: " << iter / 1000
                        << "k, Residual Norm: " << r_norm << std::endl;
        }

        apply_preconditioner();
        double rz = compute_rz();
        double beta = rz / rz_prev;
        update_p(beta);
        rz_prev = rz;
    }

    to_host(u, d_u);

    if (rank == 0) {
        if (iter == max_iter)
            std::cout << "[ERROR] Failed to converge after "
                        << max_iter << " iterations.\n";
        else
            std::cout << "[OK] Converged in " << iter
                        << " iterations, residual = " << r_norm << "\n";
    }
}

MPICudaPoissonSolver::~MPICudaPoissonSolver() {
    if (d_u)     cudaFree(d_u);
    if (d_r)     cudaFree(d_r);
    if (d_z)     cudaFree(d_z);
    if (d_k)     cudaFree(d_k);
    if (d_p)     cudaFree(d_p);
    if (d_A_p)   cudaFree(d_A_p);
    if (d_M_inv) cudaFree(d_M_inv);
    if (d_buf)   cudaFree(d_buf);

    cudaStreamDestroy(stream);
}


void MPICudaPoissonSolver::to_device(const std::vector<std::vector<double>> &field, double *d_field)
{
    int Nx = M + 1;
    int Ny = N + 1;

    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            h_buf[i * Ny + j] = field[i][j];

    cudaMemcpy(d_field, h_buf.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    cuda_check_error("to_device");
}

void MPICudaPoissonSolver::to_host(std::vector<std::vector<double>> &field, const double *d_field)
{
    int Nx = M + 1;
    int Ny = N + 1;

    cudaMemcpy(h_buf.data(), d_field, size * sizeof(double), cudaMemcpyDeviceToHost);
    cuda_check_error("to_host");

    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            field[i][j] = h_buf[i * Ny + j];
}
