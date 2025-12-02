#ifndef MPI_CONJ_GRAD_HPP
#define MPI_CONJ_GRAD_HPP

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <functional>
#include "conjugate_gradient.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

class MPIPoissonSolver : public PoissonSolver {
protected:
    MPI_Comm cart_comm;
    int rank;
    int left_rank, right_rank, bottom_rank, top_rank;

    std::vector<double> send_left, recv_left;
    std::vector<double> send_right, recv_right;
    std::vector<double> send_bottom, recv_bottom;
    std::vector<double> send_top, recv_top;

public:
    MPIPoissonSolver(
        int M, int N,
        double x_min, double x_max,
        double y_min, double y_max,
        std::function<bool(double, double)> region_func,
        std::function<double(double, double)> f_func,
        int world_rank,
        MPI_Comm cart_comm_
    );

    void exchange_halo(std::vector<std::vector<double>>& field);

    double compute_l2_norm() override;
    double compute_p_Ap() override;
    double compute_rz() override;

    void solve(int max_iter = 100000, double tolerance = 1e-4) override;
};

#endif // MPI_CONJ_GRAD_HPP
