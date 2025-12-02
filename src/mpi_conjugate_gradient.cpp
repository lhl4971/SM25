#include "mpi_conjugate_gradient.hpp"

MPIPoissonSolver::MPIPoissonSolver(
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
    send_left.assign(N + 1, 0.0);
    recv_left.assign(N + 1, 0.0);
    send_right.assign(N + 1, 0.0);
    recv_right.assign(N + 1, 0.0);
    send_bottom.assign(M + 1, 0.0);
    recv_bottom.assign(M + 1, 0.0);
    send_top.assign(M + 1, 0.0);
    recv_top.assign(M + 1, 0.0);

    MPI_Cart_shift(cart_comm, 0, 1, &left_rank, &right_rank);
    MPI_Cart_shift(cart_comm, 1, 1, &bottom_rank, &top_rank);
}

void MPIPoissonSolver::exchange_halo(std::vector<std::vector<double>>& field) {
    MPI_Request reqs[8];
    int rq = 0;

    if (left_rank != MPI_PROC_NULL) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j <= N; ++j) send_left[j] = field[1][j];
        MPI_Irecv(recv_left.data(), N+1, MPI_DOUBLE, left_rank, 0, cart_comm, &reqs[rq++]);
        MPI_Isend(send_left.data(), N+1, MPI_DOUBLE, left_rank, 1, cart_comm, &reqs[rq++]);
    }

    if (right_rank != MPI_PROC_NULL) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j <= N; ++j) send_right[j] = field[M-1][j];
        MPI_Irecv(recv_right.data(), N+1, MPI_DOUBLE, right_rank, 1, cart_comm, &reqs[rq++]);
        MPI_Isend(send_right.data(), N+1, MPI_DOUBLE, right_rank, 0, cart_comm, &reqs[rq++]);
    }

    if (bottom_rank != MPI_PROC_NULL) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i <= M; ++i) send_bottom[i] = field[i][1];
        MPI_Irecv(recv_bottom.data(), M+1, MPI_DOUBLE, bottom_rank, 2, cart_comm, &reqs[rq++]);
        MPI_Isend(send_bottom.data(), M+1, MPI_DOUBLE, bottom_rank, 3, cart_comm, &reqs[rq++]);
    }

    if (top_rank != MPI_PROC_NULL) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i <= M; ++i) send_top[i] = field[i][N-1];
        MPI_Irecv(recv_top.data(), M+1, MPI_DOUBLE, top_rank, 3, cart_comm, &reqs[rq++]);
        MPI_Isend(send_top.data(), M+1, MPI_DOUBLE, top_rank, 2, cart_comm, &reqs[rq++]);
    }

    MPI_Waitall(rq, reqs, MPI_STATUSES_IGNORE);

    if (left_rank != MPI_PROC_NULL) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j <= N; ++j) field[0][j] = recv_left[j];
    }
    if (right_rank != MPI_PROC_NULL) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j <= N; ++j) field[M][j] = recv_right[j];
    }
    if (bottom_rank != MPI_PROC_NULL) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i <= M; ++i) field[i][0] = recv_bottom[i];
    }
    if (top_rank != MPI_PROC_NULL) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i <= M; ++i) field[i][N] = recv_top[i];
    }
}

double MPIPoissonSolver::compute_l2_norm() {
    double local_sum = 0.0;
    #pragma omp parallel for reduction(+:local_sum) collapse(2) schedule(static)
    for (int i = 1; i < M; ++i)
        for (int j = 1; j < N; ++j)
            local_sum += r[i][j] * r[i][j];

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
    return std::sqrt(global_sum);
}

double MPIPoissonSolver::compute_p_Ap() {
    double local_p_Ap = 0.0;
    #pragma omp parallel for reduction(+:local_p_Ap) collapse(2) schedule(static)
    for (int i = 1; i < M; ++i)
        for (int j = 1; j < N; ++j)
            local_p_Ap += p[i][j] * A_p[i][j];

    double global_p_Ap = 0.0;
    MPI_Allreduce(&local_p_Ap, &global_p_Ap, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
    return global_p_Ap;
}

double MPIPoissonSolver::compute_rz() {
    double local_rz = 0.0;
    #pragma omp parallel for reduction(+:local_rz) collapse(2) schedule(static)
    for (int i = 1; i < M; ++i)
        for (int j = 1; j < N; ++j)
            local_rz += r[i][j] * z[i][j];

    double global_rz = 0.0;
    MPI_Allreduce(&local_rz, &global_rz, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
    return global_rz;
}

void MPIPoissonSolver::solve(int max_iter, double tolerance) {
    initialize_f();
    initialize_k();

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

        double p_Ap = compute_p_Ap();
        double alpha = rz_prev / p_Ap;
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