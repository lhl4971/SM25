#ifndef MPI_CONJ_GRAD_HPP
#define MPI_CONJ_GRAD_HPP

#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include "conjugate_gradient.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

class MPIPoissonSolver : public PoissonSolver {
private:
    MPI_Comm cart_comm;
    int rank;
    int left_rank, right_rank, bottom_rank, top_rank;

public:
    MPIPoissonSolver(
        int M, int N,
        double x_min, double x_max,
        double y_min, double y_max,
        std::function<bool(double, double)> region_func,
        std::function<double(double, double)> f_func,
        int world_rank,
        MPI_Comm cart_comm_
    ): 
        PoissonSolver(M, N, x_min, x_max, y_min, y_max, region_func, f_func),
        rank(world_rank),
        cart_comm(cart_comm_)
    {
        MPI_Cart_shift(cart_comm, 0, 1, &left_rank, &right_rank);
        MPI_Cart_shift(cart_comm, 1, 1, &bottom_rank, &top_rank);

    }

    void exchange_halo(std::vector<std::vector<double>>& field) const {
        std::vector<double> send_left(N+1), recv_left(N+1);
        std::vector<double> send_right(N+1), recv_right(N+1);
        std::vector<double> send_bottom(M+1), recv_bottom(M+1);
        std::vector<double> send_top(M+1), recv_top(M+1);

        MPI_Request reqs[8];
        int rq = 0;

        if (left_rank != MPI_PROC_NULL) {
            #pragma omp parallel for schedule(static) collapse(1)
            for (int j = 0; j <= N; ++j) send_left[j] = field[1][j];
            MPI_Irecv(recv_left.data(), N+1, MPI_DOUBLE, left_rank, 0, cart_comm, &reqs[rq++]);
            MPI_Isend(send_left.data(), N+1, MPI_DOUBLE, left_rank, 1, cart_comm, &reqs[rq++]);
        }

        if (right_rank != MPI_PROC_NULL) {
            #pragma omp parallel for schedule(static) collapse(1)
            for (int j = 0; j <= N; ++j) send_right[j] = field[M-1][j];
            MPI_Irecv(recv_right.data(), N+1, MPI_DOUBLE, right_rank, 1, cart_comm, &reqs[rq++]);
            MPI_Isend(send_right.data(), N+1, MPI_DOUBLE, right_rank, 0, cart_comm, &reqs[rq++]);
        }

        if (bottom_rank != MPI_PROC_NULL) {
            #pragma omp parallel for schedule(static) collapse(1)
            for (int i = 0; i <= M; ++i) send_bottom[i] = field[i][1];
            MPI_Irecv(recv_bottom.data(), M+1, MPI_DOUBLE, bottom_rank, 2, cart_comm, &reqs[rq++]);
            MPI_Isend(send_bottom.data(), M+1, MPI_DOUBLE, bottom_rank, 3, cart_comm, &reqs[rq++]);
        }

        if (top_rank != MPI_PROC_NULL) {
            #pragma omp parallel for schedule(static) collapse(1)
            for (int i = 0; i <= M; ++i) send_top[i] = field[i][N-1];
            MPI_Irecv(recv_top.data(), M+1, MPI_DOUBLE, top_rank, 3, cart_comm, &reqs[rq++]);
            MPI_Isend(send_top.data(), M+1, MPI_DOUBLE, top_rank, 2, cart_comm, &reqs[rq++]);
        }

        MPI_Waitall(rq, reqs, MPI_STATUSES_IGNORE);

        if (left_rank != MPI_PROC_NULL) {
            #pragma omp parallel for schedule(static) collapse(1)
            for (int j = 0; j <= N; ++j) field[0][j] = recv_left[j];
        }
        if (right_rank != MPI_PROC_NULL) {
            #pragma omp parallel for schedule(static) collapse(1)
            for (int j = 0; j <= N; ++j) field[M][j] = recv_right[j];
        }
        if (bottom_rank != MPI_PROC_NULL) {
            #pragma omp parallel for schedule(static) collapse(1)
            for (int i = 0; i <= M; ++i) field[i][0] = recv_bottom[i];
        }
        if (top_rank != MPI_PROC_NULL) {
            #pragma omp parallel for schedule(static) collapse(1)
            for (int i = 0; i <= M; ++i) field[i][N] = recv_top[i];
        }
    }

    double compute_l2_norm() const override {
        double local_sum = 0.0;
        #pragma omp parallel for reduction(+:local_sum) collapse(2) schedule(static)
        for (int i = 1; i < M; ++i)
            for (int j = 1; j < N; ++j)
                local_sum += r[i][j] * r[i][j];

        double global_sum = 0.0;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
        return std::sqrt(global_sum);
    }

    double compute_alpha() const override {
        double local_rz = 0.0, local_pAp = 0.0;
        #pragma omp parallel for reduction(+:local_rz,local_pAp) collapse(2) schedule(static)
        for (int i = 1; i < M; ++i)
            for (int j = 1; j < N; ++j) {
                local_rz  += r[i][j] * z[i][j];
                local_pAp += p[i][j] * A_p[i][j];
            }

        double glob[2] = {0.0, 0.0};
        double loc[2] = {local_rz, local_pAp};
        MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, cart_comm);
        return glob[0] / glob[1];
    }

    double compute_rz() const override {
        double local_rz = 0.0;
        #pragma omp parallel for reduction(+:local_rz) collapse(2) schedule(static)
        for (int i = 1; i < M; ++i)
            for (int j = 1; j < N; ++j)
                local_rz += r[i][j] * z[i][j];

        double global_rz = 0.0;
        MPI_Allreduce(&local_rz, &global_rz, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
        return global_rz;
    }

    void solve(int max_iter = 100000, double tolerance = 1e-6) override {
        exchange_halo(k);
        initialize_M_inv();

        exchange_halo(u);
        initialize_r();
        apply_preconditioner();
        initialize_p();

        double rz_prev = compute_rz();
        double r_norm = compute_l2_norm();

        int iter;
        for (iter = 0; r_norm > tolerance && iter < max_iter; iter++) {
            exchange_halo(p);
            apply_A(p, A_p);

            double alpha = compute_alpha();
            update_u(alpha);
            update_r(alpha);

            r_norm = compute_l2_norm();
            if (rank == 0 && iter % 1000 == 0) {
                std::ofstream debug_log("debug.log", std::ios::app);
                debug_log << "Iter: " << iter / 1000 << "k, Residual Norm: " << r_norm << std::endl;
            }

            apply_preconditioner();
            double rz = compute_rz();
            double beta = rz / rz_prev;
            update_p(beta);
            rz_prev = rz;
        }

        if (rank == 0) {
            if (iter == max_iter)
                std::cout << "[ERROR] Failed to converge after " << max_iter << " iterations.\n";
            else
                std::cout << "[OK] Converged in " << iter << " iterations, residual = " << r_norm << "\n";
        }
    }
};

#endif // MPI_CONJ_GRAD_HPP