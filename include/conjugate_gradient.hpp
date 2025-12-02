#ifndef CONJ_GRAD_HPP
#define CONJ_GRAD_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

class PoissonSolver {
private:
    double x_min, x_max, y_min, y_max;
    double hx, hy, eps;

    std::function<bool(double, double)> inside_region;
    std::function<double(double, double)> source_func;

protected:
    int M, N;
    double hx2, hy2;
    std::vector<std::vector<double>> u;   // solution
    std::vector<std::vector<double>> f;   // source term
    std::vector<std::vector<double>> r;   // residual
    std::vector<std::vector<double>> z;
    std::vector<std::vector<double>> k;   // diffusion coefficient
    std::vector<std::vector<double>> p;   // conjugate direction
    std::vector<std::vector<double>> A_p; // A*p temporary
    std::vector<std::vector<double>> M_inv; 

public:
    PoissonSolver(
        int M_, int N_,
        double x_min_, double x_max_,
        double y_min_, double y_max_,
        std::function<bool(double, double)> region_func,
        std::function<double(double, double)> f_func
    );

    void initialize_f();
    void initialize_k();
    void initialize_M_inv();

    virtual double compute_l2_norm();

    // A*v = -div(k grad(v))
    void apply_A(const std::vector<std::vector<double>>& v,
                 std::vector<std::vector<double>>& out) const;

    void initialize_r();

    virtual void initialize_p();

    virtual double compute_p_Ap();
    virtual double compute_rz();

    virtual void update_u(double alpha);
    virtual void update_r(double alpha);
    virtual void update_p(double beta);

    virtual void apply_preconditioner();

    virtual void solve(int max_iter = 100000, double tolerance = 1e-4);

    virtual const std::vector<std::vector<double>>& get_solution() const;

    virtual ~PoissonSolver() = default;
};

#endif // CONJ_GRAD_HPP
