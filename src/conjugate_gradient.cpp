#include "conjugate_gradient.hpp"

PoissonSolver::PoissonSolver(
    int M_, int N_,
    double x_min_, double x_max_,
    double y_min_, double y_max_,
    std::function<bool(double, double)> region_func,
    std::function<double(double, double)> f_func
) :
    M(M_), N(N_),
    x_min(x_min_), x_max(x_max_),
    y_min(y_min_), y_max(y_max_),
    inside_region(region_func),
    source_func(f_func)
{
    hx = (x_max - x_min) / M;
    hy = (y_max - y_min) / N;
    eps = std::max(hx, hy) * std::max(hx, hy);
    hx2 = hx * hx;
    hy2 = hy * hy;

    u.assign(M + 1, std::vector<double>(N + 1, 0.0));
    f.assign(M + 1, std::vector<double>(N + 1, 0.0));
    r.assign(M + 1, std::vector<double>(N + 1, 0.0));
    z.assign(M + 1, std::vector<double>(N + 1, 0.0));
    k.assign(M + 1, std::vector<double>(N + 1, 0.0));
    p.assign(M + 1, std::vector<double>(N + 1, 0.0));
    A_p.assign(M + 1, std::vector<double>(N + 1, 0.0));
    M_inv.assign(M + 1, std::vector<double>(N + 1, 1.0));
}

void PoissonSolver::initialize_f() {
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i <= M; ++i) {
        for (int j = 0; j <= N; ++j) {
            double x = x_min + i * hx;
            double y = y_min + j * hy;
            f[i][j] = inside_region(x, y) ? source_func(x, y) : 0.0;
        }
    }
}

void PoissonSolver::initialize_k() {
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i <= M; ++i) {
        for (int j = 0; j <= N; ++j) {
            double x = x_min + i * hx;
            double y = y_min + j * hy;
            k[i][j] = inside_region(x, y) ? 1.0 : 1 / eps;
        }
    }
}

void PoissonSolver::initialize_M_inv() {
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 1; i < M; ++i) {
        for (int j = 1; j < N; ++j) {
            double kx_plus  = 0.5 * (k[i][j] + k[i+1][j]);
            double kx_minus = 0.5 * (k[i][j] + k[i-1][j]);
            double ky_plus  = 0.5 * (k[i][j] + k[i][j+1]);
            double ky_minus = 0.5 * (k[i][j] + k[i][j-1]);

            double diag = (kx_plus + kx_minus) / hx2 + (ky_plus + ky_minus) / hy2;
            M_inv[i][j] = 1.0 / diag;
        }
    }
}

double PoissonSolver::compute_l2_norm() {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static) collapse(2)
    for (int i = 1; i < M; ++i)
        for (int j = 1; j < N; ++j)
            sum += r[i][j] * r[i][j];
    return std::sqrt(sum);
}

// A*v = -div(k grad(v))
void PoissonSolver::apply_A(const std::vector<std::vector<double>>& v,
                            std::vector<std::vector<double>>& out) const {
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 1; i < M; ++i) {
        for (int j = 1; j < N; ++j) {
            double kx_plus  = 0.5 * (k[i][j] + k[i+1][j]);
            double kx_minus = 0.5 * (k[i][j] + k[i-1][j]);
            double ky_plus  = 0.5 * (k[i][j] + k[i][j+1]);
            double ky_minus = 0.5 * (k[i][j] + k[i][j-1]);

            out[i][j] = -1.0 * (
                (kx_plus  * (v[i+1][j] - v[i][j]) - kx_minus * (v[i][j] - v[i-1][j])) / hx2 +
                (ky_plus  * (v[i][j+1] - v[i][j]) - ky_minus * (v[i][j] - v[i][j-1])) / hy2
            );
        }
    }
}

void PoissonSolver::initialize_r() {
    std::vector<std::vector<double>> A_u(M + 1, std::vector<double>(N + 1, 0.0));
    apply_A(u, A_u);
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 1; i < M; ++i)
        for (int j = 1; j < N; ++j)
            r[i][j] = f[i][j] - A_u[i][j];
}

void PoissonSolver::initialize_p() {
    p = z;
}

double PoissonSolver::compute_p_Ap() {
    double p_Ap = 0.0;
    #pragma omp parallel for reduction(+:p_Ap) schedule(static) collapse(2)
    for (int i = 1; i < M; ++i)
        for (int j = 1; j < N; ++j)
            p_Ap += p[i][j] * A_p[i][j];
    return p_Ap;
}

double PoissonSolver::compute_rz() {
    double rz = 0.0;
    #pragma omp parallel for reduction(+:rz) schedule(static) collapse(2)
    for (int i = 1; i < M; ++i)
        for (int j = 1; j < N; ++j)
            rz += r[i][j] * z[i][j];
    return rz;
}

void PoissonSolver::update_u(double alpha) {
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 1; i < M; ++i)
        for (int j = 1; j < N; ++j)
            u[i][j] += alpha * p[i][j];
}

void PoissonSolver::update_r(double alpha) {
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 1; i < M; ++i)
        for (int j = 1; j < N; ++j)
            r[i][j] -= alpha * A_p[i][j];
}

void PoissonSolver::update_p(double beta) {
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 1; i < M; ++i)
        for (int j = 1; j < N; ++j)
            p[i][j] = z[i][j] + beta * p[i][j];
}

void PoissonSolver::apply_preconditioner() {
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 1; i < M; ++i)
        for (int j = 1; j < N; ++j)
            z[i][j] = M_inv[i][j] * r[i][j];
}

void PoissonSolver::solve(int max_iter, double tolerance) {
    initialize_f();
    initialize_k();
    initialize_M_inv();
    initialize_r();
    apply_preconditioner();
    initialize_p();

    double rz_prev = compute_rz();
    double r_norm = compute_l2_norm();

    int iter;
    for (iter = 0; r_norm > tolerance && iter < max_iter; iter++) {
        apply_A(p, A_p);

        double p_Ap = compute_p_Ap();
        double alpha = rz_prev / p_Ap;
        update_u(alpha);
        update_r(alpha);

        r_norm = compute_l2_norm();
        if (iter % 1000 == 0) {
            std::ofstream debug_log("debug.log", std::ios::app);
            debug_log << "Iter: " << iter / 1000 << "k, Residual Norm: " << r_norm << std::endl;
        }

        // z <- M^{-1} * r
        apply_preconditioner();
        double rz = compute_rz();
        double beta = rz / rz_prev;
        update_p(beta);
        rz_prev = rz;
    }

    if (iter == max_iter)
        std::cout << "[ERROR] Failed to converge after " << max_iter << " iterations.\n";
    else
        std::cout << "[OK] Converged in " << iter << " iterations, residual = " << r_norm << "\n";
}

const std::vector<std::vector<double>>& PoissonSolver::get_solution() const {
    return u;
}
