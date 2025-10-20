#ifndef SPEEDY_DESCENT_HPP
#define SPEEDY_DESCENT_HPP

#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <fstream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

class PoissonSolver {
private:
    int M, N;
    double x_min, x_max, y_min, y_max;
    double hx, hy, hx2, hy2, eps;
    double beta;

    std::vector<std::vector<double>> u;  // solution
    std::vector<std::vector<double>> f;  // Formula (5)
    std::vector<std::vector<double>> r;  // gradient
    std::vector<std::vector<double>> v; // momentum
    std::vector<std::vector<double>> k;  // Formula (4)
    std::vector<std::vector<double>> A_r; 
    std::vector<std::vector<double>> A_u;

    bool (*inside_region)(double, double);
    double (*source_func)(double, double); // f(x,y)
    double (*distance_to_boundary)(double, double);

public:
    PoissonSolver(
            int M_, int N_,
            double x_min_, double x_max_,
            double y_min_, double y_max_,
            double beta_,
            bool (*region_func)(double, double),
            double (*f_func)(double, double),
            double (*distance_func)(double, double)
        ):
        M(M_), N(N_),
        x_min(x_min_), x_max(x_max_),
        y_min(y_min_), y_max(y_max_),
        beta(beta_),
        inside_region(region_func),
        source_func(f_func),
        distance_to_boundary(distance_func)
    {
        hx = (x_max - x_min) / M;
        hy = (y_max - y_min) / N;
        eps = std::max(hx, hy) * std::max(hx, hy);
        hx2 = hx * hx;
        hy2 = hy * hy;
        

        u.assign(M + 1, std::vector<double>(N + 1, 0.0));
        f.assign(M + 1, std::vector<double>(N + 1, 0.0));
        r.assign(M + 1, std::vector<double>(N + 1, 0.0));
        v.assign(M + 1, std::vector<double>(N + 1, 0.0));
        k.assign(M + 1, std::vector<double>(N + 1, 0.0));
        A_r.assign(M + 1, std::vector<double>(N + 1, 0.0));
        A_u.assign(M + 1, std::vector<double>(N + 1, 0.0));
        
        initialize_f();
        initialize_k();
    }

    void initialize_f() {
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i <= M; ++i) {
            for (int j = 0; j <= N; ++j) {
                double x = x_min + i * hx;
                double y = y_min + j * hy;
                f[i][j] = inside_region(x, y) ? source_func(x, y) : 0.0;
            }
        }
    }

    void initialize_k() {
        double delta = 0.1;
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i <= M; ++i) {
            for (int j = 0; j <= N; ++j) {
                double x = x_min + i * hx;
                double y = y_min + j * hy;
                double d = distance_to_boundary(x, y);

                // Tanh smooth transition (cosine)
                if (d >= delta)
                    k[i][j] = 1.0;
                else if (d <= -delta)
                    k[i][j] = 1.0 / eps;
                else {
                    double s = 0.5 * (1.0 + std::tanh(d / delta));
                    k[i][j] = (1.0 / eps) + (1.0 - 1.0 / eps) * s;
                }
            }
        }
    }

    void apply_A(const std::vector<std::vector<double>>& v,
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
                    (ky_plus  * (v[i][j+1] - v[i][j]) - ky_minus * (v[i][j] - v[i][j-1])) / hy2);
            }
        }
    }

    void compute_r() {
        apply_A(u, A_u);
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j)
                r[i][j] = f[i][j] - A_u[i][j];
        }
    }

    double compute_l2_norm() {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum) schedule(static) collapse(2)
        for (int i = 1; i < M; ++i)
            for (int j = 1; j < N; ++j)
                sum += r[i][j] * r[i][j];
        return std::sqrt(sum);
    }

    void update_u(double alpha) {
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {
                v[i][j] = beta * v[i][j] + alpha * r[i][j];
                u[i][j] += v[i][j];
            }
        }
    }

    // $ \alpha = (r,r) / (r, A @ r) $
    double compute_alpha() {
        apply_A(r, A_r);
        double r_r = 0.0, r_A_r = 0.0;
        #pragma omp parallel for reduction(+:r_r,r_A_r) schedule(static) collapse(2)
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {
                r_r   += r[i][j] * r[i][j];
                r_A_r += r[i][j] * A_r[i][j];
            }
        }
        return r_r / r_A_r;
    }

    void solve(int max_iter = 1000000, double tolerance = 1e-4) {
        compute_r();
        double norm = compute_l2_norm();

        int iter;
        for (iter = 0; norm > tolerance && iter < max_iter; iter++){
            double alpha = compute_alpha();
            update_u(alpha);        
            compute_r();
            norm = compute_l2_norm();
            if (iter % 10000 == 0) {
                std::ofstream debug_log("debug.log", std::ios::app);
                debug_log << "Iter: " << iter / 1000 << "k, Residual Norm: " << norm << ", Alpha: " << alpha << std::endl;
            }
        }

        if (iter == max_iter)
            std::cout << "[ERROR] Failed to converge after " << max_iter << " iterations.\n";
        else
            std::cout << "[OK] Converged in " << iter << " iterations, residual = " << norm << "\n";
    }

    void save_to_file(const std::string& filename) const {
        std::ofstream file(filename.c_str());
        if (!file.is_open()) {
            std::cerr << "Error: cannot open " << filename << "\n";
            return;
        }

        for (int i = 0; i <= M; ++i) {
            for (int j = 0; j <= N; ++j)
                file << u[i][j] << " ";
            file << "\n";
        }
        file.close();
        std::cout << "Saved solution to " << filename << "\n";
    }
    
    const std::vector<std::vector<double>>& get_solution() const {
        return u;
    }
};

#endif // SPEEDY_DESCENT_HPP
