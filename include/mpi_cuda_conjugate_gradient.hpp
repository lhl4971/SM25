#ifndef MPI_CUDA_CONJUGATE_GRADIENT_HPP
#define MPI_CUDA_CONJUGATE_GRADIENT_HPP

#include <vector>
#include <functional>
#include <mpi.h>
#include <cuda_runtime.h>

#include "mpi_conjugate_gradient.hpp"
#include "cuda_operators.hpp"

class MPICudaPoissonSolver : public MPIPoissonSolver {
private:
    // Device side pointers
    double *d_u   = nullptr;
    double *d_r   = nullptr;
    double *d_z   = nullptr;
    double *d_k   = nullptr;
    double *d_p   = nullptr;
    double *d_A_p = nullptr;
    double *d_M_inv = nullptr;
    
    // Device and host side buffer
    double *d_buf = nullptr;
    double *d_partial = nullptr;
    std::vector<double> h_buf;
    std::vector<double> h_partial;

    int size = 0;
    cudaStream_t stream = nullptr;

    void to_device(const std::vector<std::vector<double>> &field, double *d_field);
    void to_host(std::vector<std::vector<double>> &field, const double *d_field);

public:
    MPICudaPoissonSolver(
        int M, int N,
        double x_min, double x_max,
        double y_min, double y_max,
        std::function<bool(double,double)> region_func,
        std::function<double(double,double)> f_func,
        int world_rank,
        MPI_Comm cart_comm_
    );

    void cuda_exchange_halo(double *d_field);
    double compute_l2_norm() override;
    double compute_rz() override;
    double compute_p_Ap() override;

    void update_u(double alpha) override;
    void update_r(double alpha) override;
    void update_p(double beta) override;
    void apply_preconditioner() override;

    void initialize_p() override;

    void solve(int max_iter = 100000, double tolerance = 1e-4) override;

    ~MPICudaPoissonSolver() override;
};

#endif // MPI_CUDA_CONJUGATE_GRADIENT_HPP